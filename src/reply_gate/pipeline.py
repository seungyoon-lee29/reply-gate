"""에이전트 루프 — 접수 → 근거 수집 → 초안 생성 → L1 게이트 → 종결.

순서의 근거는 docs/architecture.md "대표 흐름" 이고, 이 모듈이 그 순서를 코드로 들고 있는
유일한 곳이다. LangGraph 같은
프레임워크를 쓰지 않고 **루프 종료 조건을 코드가 직접 통제**하는 것이 제품의 구조적 주장이다:

* **재생성 상한 1회** — 초안 생성은 최대 `MAX_DRAFT_ATTEMPTS`(=2)회다. 프롬프트나 모델의
  선의가 아니라 `for` 루프의 범위가 상한이다.
* **재생성은 같은 근거로** — 근거 수집은 문의 1건당 정확히 1회 호출된다. 기각 사유는
  **전부** 초안 생성에 피드백으로 넘어간다.
* **초안 전 인계는 초안 생성에 진입하지 않는다** — 근거 수집이 인계 사유를 돌려주면
  LLM 이 지어낼 기회 자체가 없다. 그때까지 모은 근거는 감사 목적으로 결과에 남는다.

**이 모듈은 DB 에 쓰지 않는다.** 처리 기록 저장은 `records.py` 가, HTTP 표면은 `api.py` 가
맡는다. 여기서 나오는 `ProcessedInquiry` 가 세 층의 공통 자료형이다(평가 하네스도 이것을
그대로 소비하면 된다).

인프라 장애(`psycopg.Error` 등)는 인계 사유로 바꾸지 않고 그대로 올려보낸다. 인계 사유
6종은 **업무 판정**이고, DB 가 죽은 것을 `no_evidence` 로 기록하면 평가 지표가 오염된다.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

import psycopg
from psycopg.rows import DictRow

from reply_gate.config import Settings, get_settings
from reply_gate.contracts import (
    Claim,
    EscalationReason,
    Evidence,
    GateResult,
    InquiryStatus,
    IntentSource,
    RejectReason,
    Verdict,
)
from reply_gate.draft import DraftGeneration, DraftGenerator
from reply_gate.evidence import (
    EvidenceCollection,
    EvidenceCollector,
    SqlEvidenceSnapshot,
    SqlFailure,
)
from reply_gate.gate import evaluate_draft, to_draft
from reply_gate.llm import EmbeddingClient, GenerationClient, LLMCallError
from reply_gate.order_ref import is_valid_order_no, normalize_order_no

__all__ = [
    "MAX_DRAFT_ATTEMPTS",
    "AcceptedInquiry",
    "AttemptRecord",
    "DraftGenerating",
    "EvidenceCollecting",
    "InquiryPipeline",
    "ProcessedInquiry",
    "ReceiptError",
    "accept_inquiry",
    "build_pipeline",
    "new_inquiry_id",
]

#: 초안 생성 호출의 상한 = 최초 1회 + 재생성 1회 (docs/standards.md "재시도 상한").
#: **이 숫자를 늘리는 것은 docs/standards.md "재시도 상한" 변경이다** —
#: `db/schema.sql` 의 attempt_no CHECK(1..2)도
#: 함께 깨진다.
MAX_DRAFT_ATTEMPTS: Final = 2


# ── 1. 접수 [코드] ──────────────────────────────────────────────────────────


class ReceiptError(ValueError):
    """접수 입력이 계약을 어겼다 — 파이프라인을 돌리지 않고 거부한다(인계가 아니다)."""


@dataclass(frozen=True)
class AcceptedInquiry:
    """접수 검증을 통과한 입력. 주문번호는 정규화된 형태로만 아래로 흐른다."""

    content: str
    order_no: str | None


def accept_inquiry(*, content: str, order_no: str | None = None) -> AcceptedInquiry:
    """접수 검증 — `content` 필수, `order_no` 는 선택이되 주면 형식이 맞아야 한다.

    형식 정의는 `reply_gate.order_ref` 가 단독 소유한다. 여기서 정규식을 다시 쓰지 않는다.
    **LLM 이 자유 텍스트에서 주문번호를 추출하지 않는다** — 주문번호는 접수 필드로만 들어온다.

    빈 문자열·공백만 있는 `order_no` 는 **미입력**으로 본다(웹 폼은 빈 필드를 그대로
    보내기 때문이다). 미입력이 곧 오류는 아니다 — 의도가 `order`/`both` 일 때 비로소
    `missing_order_ref` 인계가 된다.
    """
    text = content.strip()
    if not text:
        raise ReceiptError("문의 내용은 비워 둘 수 없다")

    if order_no is None or not order_no.strip():
        return AcceptedInquiry(content=text, order_no=None)

    normalized = normalize_order_no(order_no)
    if not is_valid_order_no(normalized):
        raise ReceiptError(f"주문번호 형식이 아니다: {order_no!r} (기대 형식: ORD-YYYYMMDD-NNNN)")
    return AcceptedInquiry(content=text, order_no=normalized)


def new_inquiry_id() -> str:
    """문의 ID 를 **근거 수집 전에** 확정한다.

    SQL 근거 ID 가 `sql:<문의 ID>:<순번>` 이고 `db/schema.sql` 의 CHECK 제약이
    `'sql:' || inquiry_id::text || ':' || sequence::text` 로 그것을 강제한다. 따라서
    ID 는 Postgres 의 uuid 텍스트 표현(소문자 하이픈)과 **글자 단위로** 같아야 한다 —
    `str(uuid.uuid4())` 가 정확히 그 형태다.
    """
    return str(uuid.uuid4())


# ── 처리 결과 ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AttemptRecord:
    """초안 1건에 대한 L1 판정 (docs/business-rules.md "엔티티와 관계" — 문의당 최대 2건)."""

    attempt_no: int
    verdict: Verdict
    reject_reasons: tuple[RejectReason, ...]
    #: L1 이 실제로 검사한 원시 초안. 형식 불일치였으면 모델 응답 원문 문자열이다.
    draft: Any


@dataclass(frozen=True)
class ProcessedInquiry:
    """문의 1건의 처리 결과 전부 — 처리 기록·API 응답·평가 하네스의 공통 자료형.

    `input_tokens`/`output_tokens` 는 **생성 LLM 합산**이다(의도 해석 + SQL 생성 + 초안 생성).
    `embedding_tokens` 는 건당 비용 산출용으로 **분리해서** 들고 있으며 생성 토큰에 섞지 않는다.
    """

    inquiry_id: str
    order_no: str | None
    content: str
    intent: IntentSource | None
    status: InquiryStatus
    answer: str | None
    claims: tuple[Claim, ...]
    escalation_reason: EscalationReason | None
    failed_stage: str | None
    evidence: tuple[Evidence, ...]
    sql_snapshots: tuple[SqlEvidenceSnapshot, ...]
    sql_failures: tuple[SqlFailure, ...]
    attempts: tuple[AttemptRecord, ...]
    latency_ms: int
    input_tokens: int
    output_tokens: int
    embedding_tokens: int

    @property
    def escalated(self) -> bool:
        return self.status is InquiryStatus.ESCALATED


# ── 협력자 계약 (테스트가 대역으로 갈아 끼울 수 있게 Protocol 로) ───────────


class EvidenceCollecting(Protocol):
    """`evidence.EvidenceCollector` 의 공개 표면."""

    def collect(
        self,
        *,
        inquiry_id: str,
        content: str,
        order_no: str | None,
        app_conn: psycopg.Connection[DictRow],
        readonly_conn: psycopg.Connection[DictRow],
    ) -> EvidenceCollection: ...


class DraftGenerating(Protocol):
    """`draft.DraftGenerator` 의 공개 표면."""

    def generate(
        self,
        *,
        inquiry: str,
        evidence: Sequence[Evidence],
        reject_reasons: Sequence[RejectReason] = ...,
    ) -> DraftGeneration: ...


# ── 루프 ────────────────────────────────────────────────────────────────────


@dataclass
class _Tally:
    """처리 중 쌓이는 것들 — 인계로 끝나도 그대로 결과에 실린다."""

    attempts: list[AttemptRecord]
    input_tokens: int
    output_tokens: int


class InquiryPipeline:
    """문의 1건을 종결 상태까지 끌고 간다. 커넥션은 호출자가 열어 넘긴다."""

    def __init__(self, *, collector: EvidenceCollecting, drafter: DraftGenerating) -> None:
        self._collector = collector
        self._drafter = drafter

    def run(
        self,
        *,
        inquiry_id: str,
        content: str,
        order_no: str | None,
        app_conn: psycopg.Connection[DictRow],
        readonly_conn: psycopg.Connection[DictRow],
    ) -> ProcessedInquiry:
        """docs/architecture.md "대표 흐름" 3~7단계를 순서대로 실행한다.

        1단계 접수는 `accept_inquiry`, 2단계 문의 ID 생성은 호출자의 `new_inquiry_id`,
        8단계 저장은 `records.py`, 9단계 응답 조립은 `api.py` 가 맡는다.

        `latency_ms` 는 이 메서드 전체의 벽시계 시간이다 — 처리 기록 저장(`records.py`)은
        측정에 포함하지 않는다. 평가 지표(p50/p95)가 재는 것은 **문의 처리**이지 저장이
        아니기 때문이다.
        """
        started = time.perf_counter()
        tally = _Tally(attempts=[], input_tokens=0, output_tokens=0)

        collection = self._collector.collect(
            inquiry_id=inquiry_id,
            content=content,
            order_no=order_no,
            app_conn=app_conn,
            readonly_conn=readonly_conn,
        )
        tally.input_tokens += collection.input_tokens
        tally.output_tokens += collection.output_tokens

        if collection.escalation_reason is not None:
            # 초안 전 인계 — 초안 생성에 진입하지 않는다.
            return self._finish(
                inquiry_id=inquiry_id,
                order_no=order_no,
                content=content,
                collection=collection,
                tally=tally,
                answer=None,
                claims=(),
                escalation=collection.escalation_reason,
                failed_stage=collection.failed_stage,
                started=started,
            )

        outcome = self._draft_loop(content=content, collection=collection, tally=tally)
        return self._finish(
            inquiry_id=inquiry_id,
            order_no=order_no,
            content=content,
            collection=collection,
            tally=tally,
            answer=outcome.answer,
            claims=outcome.claims,
            escalation=outcome.escalation,
            failed_stage=outcome.failed_stage,
            started=started,
        )

    # ── 5~7단계: 초안 생성 → L1 → 종결 ──────────────────────────────────────

    @dataclass(frozen=True)
    class _LoopOutcome:
        answer: str | None
        claims: tuple[Claim, ...]
        escalation: EscalationReason | None
        failed_stage: str | None

    def _draft_loop(
        self, *, content: str, collection: EvidenceCollection, tally: _Tally
    ) -> _LoopOutcome:
        """초안 생성 ↔ L1 게이트 루프. **상한은 이 for 문의 범위가 강제한다.**"""
        reject_reasons: tuple[RejectReason, ...] = ()

        for attempt_no in range(1, MAX_DRAFT_ATTEMPTS + 1):
            try:
                # 재생성도 **같은 근거**로 한다 — 근거 재수집 경로가 아예 없다.
                generation = self._drafter.generate(
                    inquiry=content,
                    evidence=collection.evidence,
                    reject_reasons=reject_reasons,
                )
            except LLMCallError as exc:
                # 전송 오류는 래퍼가 이미 1회 재시도했다 → 인계 + 실패 단계 기록.
                return self._LoopOutcome(
                    answer=None,
                    claims=(),
                    escalation=EscalationReason.LLM_CALL_FAILED,
                    failed_stage=exc.stage,
                )

            tally.input_tokens += generation.input_tokens
            tally.output_tokens += generation.output_tokens

            # L1 은 LLM 호출 0회의 기계 검사다 (gate.py 는 LLM 을 import 하지 않는다).
            result: GateResult = evaluate_draft(
                raw_draft=generation.raw, evidences=collection.evidence
            )
            tally.attempts.append(
                AttemptRecord(
                    attempt_no=attempt_no,
                    verdict=result.verdict,
                    reject_reasons=result.reject_reasons,
                    draft=generation.raw,
                )
            )

            if result.verdict is Verdict.PASS:
                draft = to_draft(generation.raw)
                return self._LoopOutcome(
                    answer=draft.answer_text,
                    claims=draft.claims,
                    escalation=None,
                    failed_stage=None,
                )

            # 기각 사유는 **전부** 다음 재생성의 피드백이 된다.
            reject_reasons = result.reject_reasons

        return self._LoopOutcome(
            answer=None,
            claims=(),
            escalation=EscalationReason.REJECTED_TWICE,
            failed_stage=None,
        )

    # ── 종결 ────────────────────────────────────────────────────────────────

    @staticmethod
    def _finish(
        *,
        inquiry_id: str,
        order_no: str | None,
        content: str,
        collection: EvidenceCollection,
        tally: _Tally,
        answer: str | None,
        claims: tuple[Claim, ...],
        escalation: EscalationReason | None,
        failed_stage: str | None,
        started: float,
    ) -> ProcessedInquiry:
        status = InquiryStatus.ESCALATED if escalation is not None else InquiryStatus.ANSWERED
        return ProcessedInquiry(
            inquiry_id=inquiry_id,
            order_no=order_no,
            content=content,
            intent=collection.intent,
            status=status,
            answer=answer,
            claims=claims,
            escalation_reason=escalation,
            failed_stage=failed_stage,
            evidence=collection.evidence,
            sql_snapshots=collection.sql_snapshots,
            sql_failures=collection.sql_failures,
            attempts=tuple(tally.attempts),
            latency_ms=max(round((time.perf_counter() - started) * 1000), 0),
            input_tokens=tally.input_tokens,
            output_tokens=tally.output_tokens,
            embedding_tokens=collection.embedding_tokens,
        )


def build_pipeline(
    *,
    generation_client: GenerationClient,
    embedding_client: EmbeddingClient,
    settings: Settings | None = None,
) -> InquiryPipeline:
    """실제 협력자로 파이프라인을 조립한다 (API·평가 하네스 공용 진입점)."""
    resolved = settings if settings is not None else get_settings()
    return InquiryPipeline(
        collector=EvidenceCollector(
            generation_client=generation_client,
            embedding_client=embedding_client,
            settings=resolved,
        ),
        drafter=DraftGenerator(client=generation_client, effort=resolved.generation_effort),
    )
