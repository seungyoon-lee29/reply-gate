"""FastAPI 표면 — 엔드포인트 4개와 응답 스키마 (spec "API 표면").

| `POST /inquiries`       | 문의 접수 + 동기 처리 |
| `GET  /inquiries/{id}`  | 처리 기록 조회 (저장된 값에서 재구성) |
| `GET  /`                | 최소 웹 폼 1장 |
| `GET  /health`          | 헬스 체크 |

응답 스키마의 계약은 **모든 키가 항상 존재한다** 이다: `answer`·`escalation_reason` 은
해당 없을 때 null, `claims`·`citations`·`attempts` 는 해당 없을 때 빈 배열. 초안 전
인계라도 그때까지 수집된 근거는 `citations` 에 남는다(감사 목적). `metrics.tokens` 는
**생성 LLM 합산**이고 임베딩 토큰은 여기 섞지 않는다 — 처리 기록에만 별도 필드로 남는다.

접수 검증(내용 필수·주문번호 형식)은 **파이프라인에 들어가기 전에** 422 로 끝난다.
형식이 틀린 주문번호는 인계가 아니라 요청 오류다.

생성·임베딩 클라이언트는 **첫 호출 직전에** 만든다(`_LazyGenerationClient`). 의존성 해석
시점에 만들면 API 키가 없는 환경에서 LLM 을 전혀 쓰지 않는 경로 — 조회 전용
`GET /inquiries/{id}`, 접수 거부 422 — 까지 500 으로 무너진다. 자격 증명 부재는 업무
판정이 아니므로 인계 사유(`llm_call_failed`)로 기록하지 않고 **설정 오류(503)** 로 끝낸다.

핸들러는 동기(`def`)다 — psycopg 는 동기 드라이버이고, FastAPI 가 스레드풀에서 돌린다.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Annotated, Any, Self

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from psycopg.rows import DictRow
from pydantic import BaseModel, Field, model_validator

from reply_gate.config import Settings, get_settings
from reply_gate.db import connect, readonly_connect
from reply_gate.llm import (
    EmbeddingClient,
    EmbeddingResult,
    GenerationClient,
    JsonCompletion,
    OpenAIEmbeddingClient,
    OpenAIGenerationClient,
)
from reply_gate.pipeline import (
    InquiryPipeline,
    ProcessedInquiry,
    ReceiptError,
    accept_inquiry,
    build_pipeline,
    new_inquiry_id,
)
from reply_gate.records import load_inquiry, save_inquiry

__all__ = [
    "InquiryRequest",
    "InquiryResponse",
    "InquiryService",
    "MissingCredentialsError",
    "app",
    "build_embedding_client",
    "build_generation_client",
    "get_service",
]

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

app = FastAPI(
    title="Reply-Gate",
    description="근거 없는 답변을 스스로 기각하는 이커머스 CS 답변 에이전트",
    version="0.1.0",
)


# ── 요청 ────────────────────────────────────────────────────────────────────


class InquiryRequest(BaseModel):
    """접수 입력. 검증은 `pipeline.accept_inquiry` 한 곳에서만 한다.

    주문번호 형식 정의는 `reply_gate.order_ref` 가 단독 소유한다 — 여기서 정규식을
    다시 쓰지 않는다. 검증을 통과하면 정규화된 값으로 갈아 끼워 아래로 흘린다.
    """

    content: str = Field(description="문의 내용 (필수)")
    order_no: str | None = Field(default=None, description="주문번호 ORD-YYYYMMDD-NNNN (선택)")

    @model_validator(mode="after")
    def _accept(self) -> Self:
        try:
            accepted = accept_inquiry(content=self.content, order_no=self.order_no)
        except ReceiptError as exc:
            raise ValueError(str(exc)) from exc
        self.content = accepted.content
        self.order_no = accepted.order_no
        return self


# ── 응답 ────────────────────────────────────────────────────────────────────


class ClaimOut(BaseModel):
    text: str
    citation_ids: list[str]


class CitationOut(BaseModel):
    id: str
    source: str
    content: str


class AttemptOut(BaseModel):
    verdict: str
    reject_reasons: list[str]


class TokensOut(BaseModel):
    input: int
    output: int


class MetricsOut(BaseModel):
    latency_ms: int
    tokens: TokensOut


class InquiryResponse(BaseModel):
    """spec "API 표면" 의 공통 골격. 값이 없을 때도 키는 사라지지 않는다."""

    inquiry_id: str
    status: str
    answer: str | None
    claims: list[ClaimOut]
    citations: list[CitationOut]
    attempts: list[AttemptOut]
    escalation_reason: str | None
    metrics: MetricsOut

    @classmethod
    def of(cls, processed: ProcessedInquiry) -> InquiryResponse:
        return cls(
            inquiry_id=processed.inquiry_id,
            status=processed.status.value,
            answer=processed.answer,
            claims=[
                ClaimOut(text=claim.text, citation_ids=list(claim.citation_ids))
                for claim in processed.claims
            ],
            citations=[
                CitationOut(id=item.id, source=item.source.value, content=item.content)
                for item in processed.evidence
            ],
            attempts=[
                AttemptOut(
                    verdict=attempt.verdict.value,
                    reject_reasons=[reason.value for reason in attempt.reject_reasons],
                )
                for attempt in processed.attempts
            ],
            escalation_reason=(
                None if processed.escalation_reason is None else processed.escalation_reason.value
            ),
            metrics=MetricsOut(
                latency_ms=processed.latency_ms,
                tokens=TokensOut(input=processed.input_tokens, output=processed.output_tokens),
            ),
        )


class HealthResponse(BaseModel):
    status: str


# ── 서비스 (접수 → 파이프라인 → 처리 기록) ─────────────────────────────────


class InquiryService:
    """요청 1건의 수명 동안 살아 있는 조립품 — 파이프라인 + 커넥션 2개.

    문의 ID 는 **근거 수집 전에** 확정한다: SQL 근거 ID 가 그 값을 품고, DB CHECK 가
    둘의 일치를 강제하기 때문이다.
    """

    def __init__(
        self,
        *,
        pipeline: InquiryPipeline,
        app_conn: psycopg.Connection[DictRow],
        readonly_conn: psycopg.Connection[DictRow],
    ) -> None:
        self._pipeline = pipeline
        self._app_conn = app_conn
        self._readonly_conn = readonly_conn

    def process(self, *, content: str, order_no: str | None) -> ProcessedInquiry:
        processed = self._pipeline.run(
            inquiry_id=new_inquiry_id(),
            content=content,
            order_no=order_no,
            app_conn=self._app_conn,
            readonly_conn=self._readonly_conn,
        )
        save_inquiry(conn=self._app_conn, processed=processed)
        return processed

    def fetch(self, inquiry_id: str) -> ProcessedInquiry | None:
        return load_inquiry(conn=self._app_conn, inquiry_id=inquiry_id)


class MissingCredentialsError(RuntimeError):
    """OpenAI 자격 증명이 없다 — **설정 오류**이지 인계 사유가 아니다.

    `LLMCallError` 를 상속하지 않는 것이 핵심이다: 상속하면 근거 수집기가 이것을 잡아
    `llm_call_failed` 인계로 기록해 버리고, 키를 안 넣고 돌린 실행이 평가 지표에
    "전송 오류 인계"로 섞여 들어간다.
    """


def _require_api_key(settings: Settings) -> str:
    if not settings.openai_api_key:
        raise MissingCredentialsError(
            "OPENAI_API_KEY 가 설정되지 않았다. `.env` 또는 환경 변수에 키를 넣고 다시 실행한다."
        )
    return settings.openai_api_key


class _LazyGenerationClient:
    """실제 클라이언트를 **첫 호출 때** 만드는 `GenerationClient` 어댑터.

    조립 시점에 만들면 자격 증명이 없는 환경에서 LLM 을 쓰지 않는 경로까지 무너진다
    (모듈 docstring 참조). 만들어진 클라이언트는 요청 수명 동안 재사용한다.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: GenerationClient | None = None

    def _resolve(self) -> GenerationClient:
        if self._client is None:
            self._client = OpenAIGenerationClient(
                api_key=_require_api_key(self._settings),
                model=self._settings.generation_model,
            )
        return self._client

    def complete_json(self, **kwargs: Any) -> JsonCompletion:
        return self._resolve().complete_json(**kwargs)


class _LazyEmbeddingClient:
    """`EmbeddingClient` 의 지연 생성판. `dimensions` 는 설정만으로 답한다."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: EmbeddingClient | None = None

    @property
    def dimensions(self) -> int:
        return self._settings.embedding_dimensions

    def _resolve(self) -> EmbeddingClient:
        if self._client is None:
            self._client = OpenAIEmbeddingClient(
                api_key=_require_api_key(self._settings),
                model=self._settings.embedding_model,
                dimensions=self._settings.embedding_dimensions,
            )
        return self._client

    def embed(self, *, stage: str, texts: Sequence[str]) -> EmbeddingResult:
        return self._resolve().embed(stage=stage, texts=texts)


def build_generation_client(settings: Settings) -> GenerationClient:
    return _LazyGenerationClient(settings)


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    return _LazyEmbeddingClient(settings)


def get_service() -> Iterator[InquiryService]:
    """요청마다 커넥션 2개를 열고 닫는다 (text-to-SQL 은 반드시 read-only 계정으로).

    테스트는 `app.dependency_overrides[get_service]` 로 이 함수를 통째로 갈아 끼운다 —
    LLM 목과 트랜잭션 커넥션을 주입하는 지점이다.
    """
    settings = get_settings()
    pipeline = build_pipeline(
        generation_client=build_generation_client(settings),
        embedding_client=build_embedding_client(settings),
        settings=settings,
    )
    with connect(settings=settings) as app_conn, readonly_connect(settings=settings) as ro_conn:
        yield InquiryService(pipeline=pipeline, app_conn=app_conn, readonly_conn=ro_conn)


ServiceDep = Annotated[InquiryService, Depends(get_service)]


@app.exception_handler(MissingCredentialsError)
def missing_credentials_handler(request: Request, exc: Exception) -> JSONResponse:
    """설정 오류는 503 — 처리 기록을 남기지 않는다 (인계로 집계되면 지표가 오염된다)."""
    del request
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# ── 라우트 ──────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def web_form(request: Request) -> HTMLResponse:
    """최소 웹 폼 1장. 대시보드가 아니라 **기각 순간을 보여주는 화면**이다."""
    return templates.TemplateResponse(request, "form.html")


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/inquiries")
def create_inquiry(payload: InquiryRequest, service: ServiceDep) -> InquiryResponse:
    """문의 접수 + 동기 처리. 접수 검증 실패는 이 함수에 닿기 전에 422 로 끝난다."""
    processed = service.process(content=payload.content, order_no=payload.order_no)
    return InquiryResponse.of(processed)


@app.get("/inquiries/{inquiry_id}")
def read_inquiry(inquiry_id: str, service: ServiceDep) -> InquiryResponse:
    """처리 기록 조회 — **DB 에 저장된 값**에서 같은 골격을 재구성한다."""
    processed = service.fetch(inquiry_id)
    if processed is None:
        raise HTTPException(status_code=404, detail="그런 문의가 없다")
    return InquiryResponse.of(processed)
