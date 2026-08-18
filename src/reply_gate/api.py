"""FastAPI 표면 — 엔드포인트 4개와 응답 스키마 (docs/contracts.md).

| `POST /inquiries`       | 문의 접수 + 동기 처리 |
| `GET  /inquiries/{id}`  | 처리 기록 조회 (저장된 값에서 재구성) |
| `GET  /`                | 최소 웹 폼 1장 |
| `GET  /health`          | 헬스 체크 |

응답 스키마의 계약은 **모든 키가 항상 존재한다** 이다: `answer`·`escalation_reason` 은
해당 없을 때 null, `claims`·`citations`·`attempts` 는 해당 없을 때 빈 배열. 초안 전
인계라도 그때까지 수집된 근거는 `citations` 에 남는다(감사 목적).

`attempts[]` 는 **종합 판정과 층별 판정을 함께** 싣는다: `verdict`/`reject_reasons` 는
기존 키 그대로 종합이고(값 집합에만 L2 사유 2종이 더해졌다), `l1`/`l2` 가 층별 내역이다.
`l2` 는 **미실행이면 `null`** 이고(L1 reject · 스위치 꺼짐 · L2 호출 실패) 키는 사라지지
않는다 — `null` 은 "통과"가 아니라 "판정이 없었다"는 뜻이며, 그 시도의 진실은 인계 사유가
들고 있다.

`metrics.tokens` 의 `input`/`output` 은 **생성 LLM 합산**이고 임베딩·판정·검색 토큰을 여기
섞지 않는다. 판정 토큰은 `judge_input`/`judge_output` 으로(L2 미실행이면 0), 검색 단계
재작성 토큰은 `retrieval_input`/`retrieval_output` 으로 **분리 신설**한다 — provider·단가가
다르고, 검색 계열은 특히 "초안 없이 끝난 문의"의 비용을 초안 생성 칸에 찍지 않기 위한
분리다. 임베딩 토큰은 지금도 응답에 싣지 않고 처리 기록에만 남는다.

`metrics.retrieval_fallback_reason` 은 검색 단계가 폴백한 사유다. `null` 이 기본이고
**인계 사유가 아니다** — 폴백한 문의도 그대로 답변될 수 있다.

접수 검증(내용 필수·주문번호 형식)은 **파이프라인에 들어가기 전에** 422 로 끝난다.
형식이 틀린 주문번호는 인계가 아니라 요청 오류다.

**DB 커넥션도 첫 사용 직전에 연다** — 의존성 해석 시점에 열면 DB 가 없을 때 요청 오류와
설정 오류가 모두 500 에 가린다(422 도 503 도 커넥션이 먼저 터진다). 라우트가 자기 선검사를
끝낸 뒤 `with open_service() as service` 로 여는 형태라, 이 순서는 프레임워크가 의존성을
어떤 차례로 푸는지와 무관하게 성립한다. DB 가 **진짜** 필요한 경로(조회·처리)는 그대로
500 이다 — DB 장애를 업무 판정으로 바꾸지 않는다(docs/standards.md).

생성·임베딩 클라이언트는 **첫 호출 직전에** 만든다(`_LazyGenerationClient`). 의존성 해석
시점에 만들면 API 키가 없는 환경에서 LLM 을 전혀 쓰지 않는 경로 — 조회 전용
`GET /inquiries/{id}`, 접수 거부 422 — 까지 500 으로 무너진다. 자격 증명 부재는 업무
판정이 아니므로 인계 사유(`llm_call_failed`)로 기록하지 않고 **설정 오류(503)** 로 끝낸다
(`MissingCredentialsError` 의 정의는 `pipeline.py` 가 소유한다 — 파이프라인 내부의 지연
클라이언트도 같은 예외를 던져야 이 핸들러가 잡는다).

판정 키(`ANTHROPIC_API_KEY`)만 예외적으로 **eager** 다: L2 가 켜져 있으면
`POST /inquiries` 는 처리에 진입하자마자 키를 선검사해 503 을 낸다(`_require_judge_credentials`).
검증하지 못할 답변을 만들기 시작하지 않기 위해서다. 선검사는 **POST 경로에만** 걸고
조회·422 경로는 건드리지 않는다.

핸들러는 동기(`def`)다 — psycopg 는 동기 드라이버이고, FastAPI 가 스레드풀에서 돌린다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Annotated, Any, Self

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from psycopg.rows import DictRow
from pydantic import BaseModel, Field, model_validator

from reply_gate.config import Settings, get_settings
from reply_gate.contracts import GateResult, JudgeResult
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
    AttemptRecord,
    InquiryPipeline,
    MissingCredentialsError,
    ProcessedInquiry,
    ReceiptError,
    accept_inquiry,
    build_pipeline,
    new_inquiry_id,
)
from reply_gate.policy_index import PolicyIndexProvenanceError
from reply_gate.records import load_inquiry, save_inquiry

__all__ = [
    "InquiryRequest",
    "InquiryResponse",
    "InquiryService",
    "MissingCredentialsError",
    "ServiceOpener",
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


class ClaimJudgmentOut(BaseModel):
    """L2 가 claim 1개에 내린 판정. 데모의 "왜 기각됐는지" 장면이 여기서 나온다.

    `claim_text` 는 판정 대상 claim 의 text 참조다 — 답변 계약의 claim 에는 별도 ID 가
    없으므로 text 로 가리킨다(`contracts.ClaimJudgment` 와 같은 규칙).
    """

    claim_text: str
    verdict: str
    explanation: str


class ContradictionOut(BaseModel):
    """L2 가 찾은 근거쌍 모순 1건.

    모순은 **근거쌍 단위**라 특정 claim 에 귀속되지 않을 수 있어 claim 판정과 별도 배열이다.
    기록됐다고 곧 기각은 아니다 — 초안이 모순을 명시하고 두 기준을 모두 안내했으면
    `reject_reasons` 에 오르지 않고 기록만 남는다.
    """

    evidence_id_a: str
    evidence_id_b: str
    explanation: str


class GateOut(BaseModel):
    """시도 1건의 **L1 층** 판정 — 기계 검사(LLM 호출 0회)의 결과."""

    verdict: str
    reject_reasons: list[str]


class JudgeOut(BaseModel):
    """시도 1건의 **L2 층** 판정 — 층 판정 + claim 단위 판정 + 근거쌍 모순."""

    verdict: str
    reject_reasons: list[str]
    claim_judgments: list[ClaimJudgmentOut]
    contradictions: list[ContradictionOut]


class AttemptOut(BaseModel):
    """초안 1건에 대한 판정. **종합과 층별을 함께** 싣는다.

    `verdict`/`reject_reasons` 는 기존 키이고 의미도 그대로 **종합**이다(값 집합에만
    L2 사유 2종이 더해졌다): 종합 pass ⟺ L1 pass 이고 L2 가 실행됐다면 L2 도 pass.

    `l1`/`l2` 는 층별 내역이며 **미실행은 `null`, 키는 항상 존재**한다. `l2` 가 `null` 인
    경우는 셋이다 — L1 reject(L2 는 L1 통과분에만 돈다) · 스위치 꺼짐 · L2 호출 실패.
    셋 다 "통과"가 아니라 "판정이 없었다"이므로, 종합 verdict 만 보고 통과로 읽으면 안
    된다(L2 호출 실패 시도는 층 결합 정의상 종합 pass 인데 문의는 인계된다).
    `l1` 이 `null` 인 것은 층별 컬럼이 없던 시절의 처리 기록을 복원한 경우뿐이다.
    """

    verdict: str
    reject_reasons: list[str]
    l1: GateOut | None
    l2: JudgeOut | None


class TokensOut(BaseModel):
    """토큰은 **계열별로 분리**한다 — provider 와 단가가 다르기 때문이다.

    `input`/`output` 은 기존 키·기존 의미 그대로 **생성 LLM 합산**이다(의도 해석 + SQL
    생성 + 초안 생성). `judge_input`/`judge_output` 은 판정 모델 토큰으로 **분리 신설**한
    키이며 **L2 미실행이면 0** 이다. 판정 토큰을 생성 합산에 섞으면 건당 비용 지표가
    무너진다 — 임베딩 토큰을 섞지 않는 것과 같은 이유다(임베딩은 지금도 응답에 싣지 않고
    처리 기록에만 남는다).

    `retrieval_input`/`retrieval_output` 은 **검색 단계 생성 호출**(질의 재작성)의 토큰이다.
    같은 규칙의 세 번째 계열이고 분리하는 이유가 특히 뾰족하다: 무근거 문의가 검색에서
    걸러져 **초안 없이** 끝나는 것이 이 사이클의 성공 판정인데, 재작성 토큰이 생성 칸에
    들어가면 초안을 만들지도 않은 문의가 초안 생성 토큰을 쓴 것으로 찍힌다.
    """

    input: int
    output: int
    judge_input: int
    judge_output: int
    retrieval_input: int
    retrieval_output: int


class MetricsOut(BaseModel):
    """`latency_ms` 는 파이프라인 벽시계 시간이다 — **L2 호출과 검색 단계 재작성 호출을
    포함**하고 처리 기록 저장은 포함하지 않는다."""

    latency_ms: int
    tokens: TokensOut
    #: 검색 단계가 폴백한 사유. `null` 은 "폴백하지 않았다"이고 **인계 사유가 아니다** —
    #: 폴백한 문의도 그대로 답변될 수 있다.
    retrieval_fallback_reason: str | None


def _gate_out(result: GateResult | None) -> GateOut | None:
    """L1 층 내역 → 응답. 층별 내역이 없는 기록(복원 경로)은 `None` 으로 남긴다."""
    if result is None:
        return None
    return GateOut(
        verdict=result.verdict.value,
        reject_reasons=[reason.value for reason in result.reject_reasons],
    )


def _judge_out(result: JudgeResult | None) -> JudgeOut | None:
    """L2 층 내역 → 응답. **미실행이면 `None`** 이고 키는 그대로 남는다.

    claim 판정·모순쌍이 0건인 것과 L2 가 아예 돌지 않은 것은 다른 상태다 — 전자는 빈
    배열, 후자는 `null` 이다. 저장 층도 같은 구분을 지킨다(`records._l2_columns`).
    """
    if result is None:
        return None
    return JudgeOut(
        verdict=result.verdict.value,
        reject_reasons=[reason.value for reason in result.reject_reasons],
        claim_judgments=[
            ClaimJudgmentOut(
                claim_text=judgment.claim_text,
                verdict=judgment.verdict.value,
                explanation=judgment.explanation,
            )
            for judgment in result.claim_judgments
        ],
        contradictions=[
            ContradictionOut(
                evidence_id_a=contradiction.evidence_id_a,
                evidence_id_b=contradiction.evidence_id_b,
                explanation=contradiction.explanation,
            )
            for contradiction in result.contradictions
        ],
    )


def _attempt_out(attempt: AttemptRecord) -> AttemptOut:
    return AttemptOut(
        verdict=attempt.verdict.value,
        reject_reasons=[reason.value for reason in attempt.reject_reasons],
        l1=_gate_out(attempt.l1_result),
        l2=_judge_out(attempt.l2_result),
    )


class InquiryResponse(BaseModel):
    """docs/contracts.md "공통 규약" 의 공통 골격. 값이 없을 때도 키는 사라지지 않는다.

    `POST /inquiries` 와 `GET /inquiries/{id}` 가 **같은 조립기**를 쓴다 — 조회는 저장된
    기록에서 복원한 `ProcessedInquiry` 를 넣을 뿐이라 층별 판정·판정 토큰까지 같은 모양으로
    다시 나온다.
    """

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
            attempts=[_attempt_out(attempt) for attempt in processed.attempts],
            escalation_reason=(
                None if processed.escalation_reason is None else processed.escalation_reason.value
            ),
            metrics=MetricsOut(
                latency_ms=processed.latency_ms,
                tokens=TokensOut(
                    input=processed.input_tokens,
                    output=processed.output_tokens,
                    judge_input=processed.judge_input_tokens,
                    judge_output=processed.judge_output_tokens,
                    retrieval_input=processed.retrieval_input_tokens,
                    retrieval_output=processed.retrieval_output_tokens,
                ),
                retrieval_fallback_reason=processed.retrieval_fallback_reason,
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


#: 요청 수명의 조립품을 **필요해진 순간에** 여는 것. 의존성 해석 시점이 아니다 —
#: 이유는 `get_service` 의 docstring 에 있다.
type ServiceOpener = Callable[[], AbstractContextManager[InquiryService]]


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
    """`EmbeddingClient` 의 지연 생성판. `model`·`dimensions` 는 설정만으로 답한다."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: EmbeddingClient | None = None

    @property
    def model(self) -> str:
        return self._settings.embedding_model

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


def get_service() -> ServiceOpener:
    """조립만 돌려준다 — **커넥션은 아직 열지 않는다.**

    의존성 해석 시점에 연결하면 DB 가 없을 때 **요청 오류와 설정 오류가 500 에 가린다**:
    형식이 틀린 주문번호(422)도, 판정 키 부재(503)도 커넥션이 먼저 터져 전부 500 이 됐다.
    라우트가 자기 선검사를 끝낸 뒤 `with open_service() as service` 로 여는 형태여야
    그 순서가 **프레임워크의 의존성 해석 순서와 무관하게** 성립한다.

    이 모듈은 같은 교훈을 LLM 클라이언트에는 이미 적용해 두었다(`_LazyGenerationClient`)
    — 커넥션도 같은 종류의 자원이다.

    테스트는 `app.dependency_overrides[get_service]` 로 이 함수를 통째로 갈아 끼운다 —
    LLM 목과 트랜잭션 커넥션을 주입하는 지점이다.
    """
    settings = get_settings()
    pipeline = build_pipeline(
        generation_client=build_generation_client(settings),
        embedding_client=build_embedding_client(settings),
        settings=settings,
    )

    @contextmanager
    def _open() -> Iterator[InquiryService]:
        """요청마다 커넥션 2개를 열고 닫는다 (text-to-SQL 은 반드시 read-only 계정으로)."""
        with connect(settings=settings) as app_conn, readonly_connect(settings=settings) as ro:
            yield InquiryService(pipeline=pipeline, app_conn=app_conn, readonly_conn=ro)

    return _open


ServiceDep = Annotated[ServiceOpener, Depends(get_service)]


@app.exception_handler(MissingCredentialsError)
def missing_credentials_handler(request: Request, exc: Exception) -> JSONResponse:
    """설정 오류는 503 — 처리 기록을 남기지 않는다 (인계로 집계되면 지표가 오염된다)."""
    del request
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(PolicyIndexProvenanceError)
def stale_policy_index_handler(request: Request, exc: Exception) -> JSONResponse:
    """낡은 정책 인덱스도 **설정 오류(503)** 다 — 자격 증명 부재와 같은 계열이다.

    `no_evidence` 인계로 바꾸면 "검색이 못 찾았다"로 집계되어, 재색인을 빠뜨린 운영 실수가
    검색 품질 지표로 위장한다. 처리 기록은 커넥션 롤백으로 남지 않는다.
    """
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


def _require_judge_credentials() -> None:
    """L2 가 켜져 있으면 판정 키를 **POST 처리 진입 시점에** 선검사한다.

    지연 검사(L2 에 도달해서야 실패)로 두면 두 가지가 무너진다: 생성 토큰을 태운 뒤에야
    503 이 나고, L1 이 2회 기각한 문의는 판정 키 없이도 정상 종결되어 "키 없으면 503"이
    조건부 규칙이 된다. 검증하지 못할 답변을 만들기 시작하지 않는 것이 fail-closed 다.

    **이 검사는 POST 경로에만 건다.** `get_service`(Depends)는 GET 라우트와 공유하므로
    거기에 넣으면 판정 키가 없는 환경에서 조회 전용 경로까지 죽는다
    (docs/engineering-notes.md "목만으로는 잡히지 않는 결함"). 접수 거부 422 도 판정 키와
    무관하다 — 그 경로는 이 함수에 닿기 전에 끝난다.
    """
    settings = get_settings()
    if settings.l2_enabled and not settings.anthropic_api_key:
        raise MissingCredentialsError(
            "ANTHROPIC_API_KEY 가 설정되지 않았다. L2 판정이 켜져 있으면 판정 없이 답변을 "
            "내보내지 않는다 — `.env` 또는 환경 변수에 키를 넣거나 L2_ENABLED=false 로 "
            "판정을 끄고 다시 실행한다."
        )


@app.post("/inquiries")
def create_inquiry(payload: InquiryRequest, open_service: ServiceDep) -> InquiryResponse:
    """문의 접수 + 동기 처리. 접수 검증 실패는 이 함수에 닿기 전에 422 로 끝난다.

    **커넥션은 선검사 뒤에 연다.** 먼저 열면 DB 가 없을 때 422·503 이 500 에 가린다.
    """
    # 처리 기록을 남기기 전에 끝낸다 — 설정 오류는 인계(`llm_call_failed`)가 아니다.
    _require_judge_credentials()
    with open_service() as service:
        processed = service.process(content=payload.content, order_no=payload.order_no)
    return InquiryResponse.of(processed)


@app.get("/inquiries/{inquiry_id}")
def read_inquiry(inquiry_id: str, open_service: ServiceDep) -> InquiryResponse:
    """처리 기록 조회 — **DB 에 저장된 값**에서 같은 골격을 재구성한다."""
    with open_service() as service:
        processed = service.fetch(inquiry_id)
    if processed is None:
        raise HTTPException(status_code=404, detail="그런 문의가 없다")
    return InquiryResponse.of(processed)
