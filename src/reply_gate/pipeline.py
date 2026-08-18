"""에이전트 루프 — 접수 → 근거 수집 → 초안 생성 → L1 게이트 → (L2 판정) → 종결.

순서의 근거는 docs/architecture.md "대표 흐름" 이고, 이 모듈이 그 순서를 코드로 들고 있는
유일한 곳이다. LangGraph 같은
프레임워크를 쓰지 않고 **루프 종료 조건을 코드가 직접 통제**하는 것이 제품의 구조적 주장이다:

* **재생성 상한 1회** — 초안 생성은 최대 `MAX_DRAFT_ATTEMPTS`(=2)회다. 프롬프트나 모델의
  선의가 아니라 `for` 루프의 범위가 상한이다. 층이 늘어도 이 상한은 그대로다 —
  `rejected_twice` 는 **층 무관** 2회 연속 기각이다.
* **재생성은 같은 근거로** — 근거 수집은 문의 1건당 정확히 1회 호출된다. 기각 사유는
  **전부**(L1 4종 + L2 2종) 초안 생성에 피드백으로 넘어가고, L2 기각이면 claim 단위
  상세까지 함께 간다.
* **초안 전 인계는 초안 생성에 진입하지 않는다** — 근거 수집이 인계 사유를 돌려주면
  LLM 이 지어낼 기회 자체가 없다. 그때까지 모은 근거는 감사 목적으로 결과에 남는다.

**L2 는 L1 통과분에만 실행된다.** 스위치(`Settings.l2_enabled`)로 통째로 끌 수 있고, 끄면
사이클 1 동작(L1 pass → answered)과 완전히 같다. 다만 **스위치 켜짐 + 판정자 미배선은
조립 시점 오류**다 — 조용히 L2 를 건너뛰는 경로를 두면 fail-closed 가 배선 실수 하나로
무너진다. L2 호출이 실패하면(전송 오류·형식 불일치 소진) 검증하지 못한 답변을 내보내지
않고 `llm_call_failed` 로 인계한다.

**이 모듈은 DB 에 쓰지 않는다.** 처리 기록 저장은 `records.py` 가, HTTP 표면은 `api.py` 가
맡는다. 여기서 나오는 `ProcessedInquiry` 가 세 층의 공통 자료형이다(평가 하네스도 이것을
그대로 소비하면 된다).

인프라 장애(`psycopg.Error` 등)는 인계 사유로 바꾸지 않고 그대로 올려보낸다. 인계 사유
6종은 **업무 판정**이고, DB 가 죽은 것을 `no_evidence` 로 기록하면 평가 지표가 오염된다.
같은 이유로 **자격 증명 부재(`MissingCredentialsError`)도 인계가 아니다** — 설정 오류로
그대로 전파해 HTTP 표면이 503 으로 끝낸다.
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
    COMBINED_REASON_ORDER,
    Claim,
    Draft,
    EscalationReason,
    Evidence,
    GateResult,
    InquiryStatus,
    IntentSource,
    JudgeResult,
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
from reply_gate.judge import Judge, JudgeOutcome
from reply_gate.llm import (
    AnthropicGenerationClient,
    EmbeddingClient,
    GenerationClient,
    JsonCompletion,
    LLMCallError,
    LLMFormatError,
)
from reply_gate.order_ref import is_valid_order_no, normalize_order_no

__all__ = [
    "L2_JUDGE_STAGE",
    "MAX_DRAFT_ATTEMPTS",
    "AcceptedInquiry",
    "AttemptRecord",
    "DraftGenerating",
    "EvidenceCollecting",
    "InquiryPipeline",
    "Judging",
    "MissingCredentialsError",
    "PipelineWiringError",
    "ProcessedInquiry",
    "ReceiptError",
    "accept_inquiry",
    "build_judge",
    "build_pipeline",
    "new_inquiry_id",
]

#: 초안 생성 호출의 상한 = 최초 1회 + 재생성 1회 (docs/standards.md "재시도 상한").
#: **이 숫자를 늘리는 것은 docs/standards.md "재시도 상한" 변경이다** —
#: `db/schema.sql` 의 attempt_no CHECK(1..2)도
#: 함께 깨진다.
MAX_DRAFT_ATTEMPTS: Final = 2

#: L2 판정 호출이 실패했을 때 처리 기록에 남는 실패 단계 이름
#: (docs/business-rules.md "인계 사유 6종" — `llm_call_failed` 는 단계 이름을 함께 남긴다).
#: 판정 모듈이 LLM 래퍼에 넘기는 단계 이름(`judge.JUDGE_STAGE`)과 달리, 이 값은
#: **파이프라인 층에서 어디가 무너졌는지**를 가리킨다.
L2_JUDGE_STAGE: Final = "l2_judge"


# ── 조립·설정 오류 (업무 판정이 아니다) ─────────────────────────────────────


class PipelineWiringError(RuntimeError):
    """배선이 fail-closed 를 깨뜨린다 — 판정 없이 답변이 확정될 수 있는 상태다.

    두 자리에서 난다: **조립 시점**(L2 스위치가 켜졌는데 판정자가 없다)과 **실행 중**
    (L2 를 돌린 시도인데 판정 결과가 비어 있다). 둘 다 기본값·`None` 이 조용히 L2 를
    끄는 구현을 금지하기 위한 것이다: 배선을 빠뜨린 실행이 "L2 를 통과했다"가 아니라
    "L2 를 건너뛰었다"는 사실을 지표가 알 수 없게 되고, 검증하지 못한 답변이 검증된
    답변처럼 나간다.

    **인계 사유가 아니다** — 업무 판정이 아니라 배선 오류이므로 `llm_call_failed` 로
    삼키지 않고 그대로 올려보낸다(자격 증명 부재와 같은 취급).
    """


class MissingCredentialsError(RuntimeError):
    """외부 API 자격 증명이 없다 — **설정 오류**이지 인계 사유가 아니다.

    `LLMCallError` 를 상속하지 않는 것이 핵심이다: 상속하면 근거 수집기·파이프라인이
    이것을 잡아 `llm_call_failed` 인계로 기록해 버리고, 키를 안 넣고 돌린 실행이 평가
    지표에 "전송 오류 인계"로 섞여 들어간다. HTTP 표면은 이것을 503 으로 끝낸다
    (`api.missing_credentials_handler`).
    """


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
    """초안 1건에 대한 층별 판정 (docs/business-rules.md "엔티티와 관계" — 문의당 최대 2건).

    `verdict`/`reject_reasons` 는 **종합**이다: 종합 pass ⟺ L1 pass 이고 L2 가 실행됐다면
    L2 도 pass. 종합 사유는 두 층 사유의 합집합이며 순서는 `COMBINED_REASON_ORDER` 다.

    `l1_result`/`l2_result` 는 층별 내역이고 **둘 다 기본값 `None`** 이다 — 처리 기록
    복원(`records._load_attempts`)이 이 생성자를 직접 부르므로, 층별 컬럼을 읽기 전까지
    기본값이 없으면 복원이 깨진다. `l2_result` 가 `None` 인 경우는 셋이다:
    L1 reject(L2 미실행) · 스위치 꺼짐 · L2 호출 실패.
    """

    attempt_no: int
    verdict: Verdict
    reject_reasons: tuple[RejectReason, ...]
    #: L1 이 실제로 검사한 원시 초안. 형식 불일치였으면 모델 응답 원문 문자열이다.
    draft: Any
    l1_result: GateResult | None = None
    l2_result: JudgeResult | None = None


@dataclass(frozen=True)
class ProcessedInquiry:
    """문의 1건의 처리 결과 전부 — 처리 기록·API 응답·평가 하네스의 공통 자료형.

    토큰은 계열별로 **분리해서** 들고 있다. `input_tokens`/`output_tokens` 는
    **생성 LLM 합산**이고(의도 해석 + SQL 생성 + 초안 생성), `embedding_tokens` 와
    `judge_input_tokens`/`judge_output_tokens` 는 여기 섞지 않는다 — provider 와 단가가
    달라 섞으면 건당 비용 지표가 무너진다. 판정 토큰은 **L2 미실행이면 0** 이고,
    실행됐으나 실패한 호출이 쓴 토큰도 실비용이므로 그대로 집계한다.

    판정 토큰 필드는 기본값 0 이다 — `records.load_inquiry` 가 분리 컬럼을 읽기 전까지
    기존 복원 경로가 깨지지 않아야 한다.
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
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    #: 검색 단계(질의 재작성) 생성 호출의 토큰. 생성·임베딩·판정 어디에도 합산하지 않는다
    #: (docs/contracts.md "토큰 집계 경계"). 재작성을 쓰지 않은 문의는 0 이다.
    retrieval_input_tokens: int = 0
    retrieval_output_tokens: int = 0
    #: 검색 단계가 폴백한 사유. `None` 이면 폴백하지 않았다 — 인계 사유가 **아니다**.
    retrieval_fallback_reason: str | None = None

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
        judge_result: JudgeResult | None = ...,
    ) -> DraftGeneration: ...


class Judging(Protocol):
    """`judge.Judge` 의 공개 표면 — 시도(초안)당 1회 배치 판정."""

    def judge(self, *, draft: Draft, evidence: Sequence[Evidence]) -> JudgeOutcome: ...


# ── 루프 ────────────────────────────────────────────────────────────────────


@dataclass
class _Tally:
    """처리 중 쌓이는 것들 — 인계로 끝나도 그대로 결과에 실린다."""

    attempts: list[AttemptRecord]
    input_tokens: int
    output_tokens: int
    #: 판정(L2) 토큰은 생성 합산과 **분리**해서 센다.
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0


def _combine(
    l1_result: GateResult, l2_result: JudgeResult | None, *, l2_expected: bool
) -> tuple[
    Verdict,
    tuple[RejectReason, ...],
]:
    """층별 판정 → 종합 판정. 순서는 계약(`COMBINED_REASON_ORDER`)이 정한다.

    종합 pass ⟺ L1 pass 이고 L2 가 실행됐다면 L2 도 pass. L2 미실행(`None`)은 pass 로
    치지 않고 **판정에서 빠진다** — 실행 실패의 진실은 인계 사유와 실패 단계가 들고 있다.

    `l2_expected` 는 **이 시도에서 L2 판정이 나왔어야 하는가**다. 인자로 받는 이유는
    `l2_result is None` 하나로는 "L2 를 안 돌렸다"와 "돌렸는데 판정이 비었다"를 구분할 수
    없기 때문이다 — 구분하지 않으면 `None` 이 무조건 "L2 통과"로 접혀, 판정자가 배선된
    채 빈 판정을 돌려주는 배선 실수가 검증 없는 답변을 `answered` 로 확정시킨다(fail-open).
    fail-closed 는 배선 실수에도 성립해야 하므로 조용히 통과시키지 않고 죽는다.

    호출부는 둘이다: 정상 종결 경로(L2 를 돌렸어야 하면 `True`)와 `_judge_failure`
    (L2 호출이 실패해 판정이 없는 것은 docs/contracts.md "층별 판정 키" 가 정한 동작
    ③ 이므로 `False`).
    """
    if l2_expected and l2_result is None:
        raise PipelineWiringError(
            "L2 를 실행한 시도인데 판정 결과가 비어 있다 — 판정 없는 초안을 통과로 접지 "
            "않는다. 판정자 배선(`Judging.judge` 의 반환 계약)을 확인한다."
        )
    passed = l1_result.verdict is Verdict.PASS and (
        l2_result is None or l2_result.verdict is Verdict.PASS
    )
    reasons = set(l1_result.reject_reasons)
    if l2_result is not None:
        reasons |= set(l2_result.reject_reasons)
    ordered = tuple(reason for reason in COMBINED_REASON_ORDER if reason in reasons)
    return (Verdict.PASS if passed else Verdict.REJECT), ordered


class InquiryPipeline:
    """문의 1건을 종결 상태까지 끌고 간다. 커넥션은 호출자가 열어 넘긴다."""

    def __init__(
        self,
        *,
        collector: EvidenceCollecting,
        drafter: DraftGenerating,
        judge: Judging | None = None,
        l2_enabled: bool = True,
    ) -> None:
        """`l2_enabled` 가 켜져 있으면 판정자는 **필수**다.

        스위치 기본값은 설정과 같은 **켜짐**이고, 판정자를 빠뜨린 조립은 여기서 죽는다 —
        기본값 `None` 이 조용히 L2 를 끄면 fail-closed 가 배선 실수 하나로 무너진다.
        L2 없이 돌리려면 `l2_enabled=False` 로 **명시**한다.
        """
        if l2_enabled and judge is None:
            raise PipelineWiringError(
                "L2 스위치가 켜져 있는데 판정자가 배선되지 않았다 — 조용히 L2 를 건너뛰지 "
                "않는다. 판정자를 주입하거나 l2_enabled=False 로 명시한다."
            )
        self._collector = collector
        self._drafter = drafter
        self._judge = judge
        self._l2_enabled = l2_enabled

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

    # ── 5~7단계: 초안 생성 → L1 → L2 → 종결 ─────────────────────────────────

    @dataclass(frozen=True)
    class _LoopOutcome:
        answer: str | None
        claims: tuple[Claim, ...]
        escalation: EscalationReason | None
        failed_stage: str | None

    def _draft_loop(
        self, *, content: str, collection: EvidenceCollection, tally: _Tally
    ) -> _LoopOutcome:
        """초안 생성 ↔ L1 ↔ L2 루프. **상한은 이 for 문의 범위가 강제한다.**"""
        reject_reasons: tuple[RejectReason, ...] = ()
        judge_result: JudgeResult | None = None

        for attempt_no in range(1, MAX_DRAFT_ATTEMPTS + 1):
            try:
                # 재생성도 **같은 근거**로 한다 — 근거 재수집 경로가 아예 없다.
                generation = self._drafter.generate(
                    inquiry=content,
                    evidence=collection.evidence,
                    reject_reasons=reject_reasons,
                    judge_result=judge_result,
                )
            except LLMCallError as exc:
                # 전송 오류는 래퍼가 이미 1회 재시도했다 → 인계 + 실패 단계 기록.
                # 실패까지 과금된 토큰(예: 거절 응답의 입력 토큰)도 실비용이므로 집계한다.
                tally.input_tokens += exc.input_tokens
                tally.output_tokens += exc.output_tokens
                return self._LoopOutcome(
                    answer=None,
                    claims=(),
                    escalation=EscalationReason.LLM_CALL_FAILED,
                    failed_stage=exc.stage,
                )

            tally.input_tokens += generation.input_tokens
            tally.output_tokens += generation.output_tokens

            # L1 은 LLM 호출 0회의 기계 검사다 (gate.py 는 LLM 을 import 하지 않는다).
            l1_result: GateResult = evaluate_draft(
                raw_draft=generation.raw, evidences=collection.evidence
            )
            # L1 을 통과한 초안만 `Draft` 로 해석된다 — L2 입력이자 최종 답변의 원본이다.
            draft = to_draft(generation.raw) if l1_result.verdict is Verdict.PASS else None

            # 이 시도에서 L2 판정이 **나왔어야 하는가** — `_combine` 이 "L2 미실행"과
            # "판정이 비었다"를 구분하는 근거다(fail-open 차단).
            l2_expected = self._l2_enabled and draft is not None
            l2_result: JudgeResult | None = None
            if l2_expected and draft is not None:  # `draft` 재확인은 타입 좁히기다
                # 판정자 미배선은 조립에서 이미 막혔다. 여기서 다시 보는 것은 타입 좁히기
                # 겸 이중 잠금이다 — `assert` 로 두면 `python -O` 에서 사라져 스위치가
                # 켜진 채 L2 를 건너뛰는 fail-open 이 된다.
                if self._judge is None:  # pragma: no cover - 조립에서 이미 막힌다
                    raise PipelineWiringError(
                        "L2 스위치가 켜져 있는데 판정자가 없다 — 판정 없이 답변을 확정하지 않는다."
                    )
                try:
                    outcome = self._judge.judge(draft=draft, evidence=collection.evidence)
                except LLMFormatError as exc:
                    # 형식 불일치는 판정 모듈이 이미 1회 재시도했다. 실패한 호출이 쓴
                    # 토큰도 실비용이므로 그대로 집계한다.
                    tally.judge_input_tokens += exc.input_tokens
                    tally.judge_output_tokens += exc.output_tokens
                    return self._judge_failure(
                        tally=tally,
                        attempt_no=attempt_no,
                        l1_result=l1_result,
                        raw_draft=generation.raw,
                    )
                except LLMCallError as exc:
                    # 전송 오류는 래퍼가 이미 1회 재시도했다. 그때까지 이미 과금된 판정
                    # 토큰(예: 형식 실패한 1회차)이 예외에 실려 온다 — 형식 불일치 소진과
                    # 같은 이유로 그대로 집계한다. 여기서 버리면 값이 파이프라인 밖으로
                    # 나가지 못해, 처리 기록·API 응답이 판정 토큰을 싣는 뒤 태스크 시점에
                    # 실비용이 0 으로 굳는다.
                    tally.judge_input_tokens += exc.input_tokens
                    tally.judge_output_tokens += exc.output_tokens
                    return self._judge_failure(
                        tally=tally,
                        attempt_no=attempt_no,
                        l1_result=l1_result,
                        raw_draft=generation.raw,
                    )
                tally.judge_input_tokens += outcome.input_tokens
                tally.judge_output_tokens += outcome.output_tokens
                l2_result = outcome.result

            verdict, reasons = _combine(l1_result, l2_result, l2_expected=l2_expected)
            tally.attempts.append(
                AttemptRecord(
                    attempt_no=attempt_no,
                    verdict=verdict,
                    reject_reasons=reasons,
                    draft=generation.raw,
                    l1_result=l1_result,
                    l2_result=l2_result,
                )
            )

            # 종합 pass 는 L1 pass 를 함의하므로 `draft` 는 반드시 해석돼 있다 — 두 조건은
            # 정의상 같이 참이고, `draft` 를 함께 보는 것은 타입 좁히기다.
            if verdict is Verdict.PASS and draft is not None:
                return self._LoopOutcome(
                    answer=draft.answer_text,
                    claims=draft.claims,
                    escalation=None,
                    failed_stage=None,
                )

            # 기각 사유는 **전부** 다음 재생성의 피드백이 된다. L2 기각이면 claim 단위
            # 상세("어느 문장이 왜")가 함께 간다 — 사유 코드만으로는 고칠 지점을 모른다.
            reject_reasons = reasons
            judge_result = l2_result

        # 2회 연속 기각 — **층 무관**이다(L1 기각 후 L2 기각도 여기로 온다).
        return self._LoopOutcome(
            answer=None,
            claims=(),
            escalation=EscalationReason.REJECTED_TWICE,
            failed_stage=None,
        )

    def _judge_failure(
        self, *, tally: _Tally, attempt_no: int, l1_result: GateResult, raw_draft: Any
    ) -> _LoopOutcome:
        """L2 호출이 재시도까지 실패했다 — 검증하지 못한 답변은 내보내지 않는다.

        그 시도 행은 **L1 판정만** 남고 L2 판정은 null 이다(층 결합 정의상 종합 verdict 는
        pass 다 — 진실은 인계 사유와 실패 단계가 들고 있다). 재생성 사유가 아니므로 루프를
        더 돌리지 않는다: 인프라 실패는 초안을 고쳐서 나아지는 것이 아니다.

        `l2_expected=False` 를 **명시**한다: 여기서 판정이 없는 것은 배선 실수가 아니라
        docs/contracts.md "층별 판정 키" 가 `l2: null` 로 규정한 동작이라, fail-open 차단
        검사에 걸리면 안 된다.
        """
        verdict, reasons = _combine(l1_result, None, l2_expected=False)
        tally.attempts.append(
            AttemptRecord(
                attempt_no=attempt_no,
                verdict=verdict,
                reject_reasons=reasons,
                draft=raw_draft,
                l1_result=l1_result,
                l2_result=None,
            )
        )
        return self._LoopOutcome(
            answer=None,
            claims=(),
            escalation=EscalationReason.LLM_CALL_FAILED,
            failed_stage=L2_JUDGE_STAGE,
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
            # 판정 토큰은 생성 합산과 **분리**해서 싣는다
            # (docs/contracts.md "토큰 집계 경계"). L2 미실행이면 0 이다.
            judge_input_tokens=tally.judge_input_tokens,
            judge_output_tokens=tally.judge_output_tokens,
            # 검색 단계 토큰도 같은 규칙의 세 번째 계열이다 — 수집기가 이미 분리해서
            # 들고 있으므로 원장을 거치지 않고 그대로 옮긴다(초안 루프가 만들지 않는다).
            retrieval_input_tokens=collection.retrieval_input_tokens,
            retrieval_output_tokens=collection.retrieval_output_tokens,
            retrieval_fallback_reason=collection.retrieval_fallback_reason,
        )


class _LazyJudgeClient:
    """판정용 실제 클라이언트를 **첫 호출 때** 만드는 `GenerationClient` 어댑터.

    조립 시점에 만들면 판정 키가 없는 환경에서 L2 를 전혀 쓰지 않는 경로(조회 전용
    `GET /inquiries/{id}`, 스위치 꺼짐 실행)까지 무너진다 — 생성 클라이언트가
    `api._LazyGenerationClient` 로 같은 함정을 이미 한 번 밟았다
    (docs/engineering-notes.md "목만으로는 잡히지 않는 결함").

    키 검사는 **설정값을 명시적으로** 본다. `anthropic.Anthropic(api_key="")` 는 OpenAI SDK 와
    달리 생성자에서 예외를 던지지 않아, 클라이언트를 만들어 보는 것으로는 키 부재를 알 수 없다.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: GenerationClient | None = None

    def _resolve(self) -> GenerationClient:
        if self._client is None:
            api_key = self._settings.anthropic_api_key
            if not api_key:
                raise MissingCredentialsError(
                    "ANTHROPIC_API_KEY 가 설정되지 않았다. `.env` 또는 환경 변수에 키를 넣거나 "
                    "L2_ENABLED=false 로 판정을 끄고 다시 실행한다."
                )
            self._client = AnthropicGenerationClient(
                api_key=api_key, model=self._settings.judge_model
            )
        return self._client

    def complete_json(self, **kwargs: Any) -> JsonCompletion:
        return self._resolve().complete_json(**kwargs)


def build_judge(settings: Settings) -> Judge:
    """설정으로 판정자를 조립한다 — **모델·effort·상한은 전부 설정이 정한다**(하드코딩 금지).

    클라이언트는 지연 생성이라 이 함수는 자격 증명을 요구하지 않는다. 판정 계열을
    Anthropic 으로 고정하는 것은 하드 게이트다: 생성과 같은 계열로 판정하면
    self-judging bias 로 검출률이 오염된다.
    """
    return Judge(
        client=_LazyJudgeClient(settings),
        effort=settings.judge_effort,
        max_output_tokens=settings.judge_max_output_tokens,
    )


def build_pipeline(
    *,
    generation_client: GenerationClient,
    embedding_client: EmbeddingClient,
    settings: Settings | None = None,
) -> InquiryPipeline:
    """실제 협력자로 파이프라인을 조립한다 (API·평가 하네스 공용 진입점).

    **판정자는 스위치와 무관하게 항상 배선한다.** 스위치 켜짐 + 판정자 미배선은
    `InquiryPipeline` 조립 시점 오류이므로, 하위호환(키 없이도 조립은 성공)을 지는 것은
    이 함수다 — 지연 생성 클라이언트라 배선 자체는 키를 요구하지 않는다.

    **재작성 클라이언트도 스위치와 무관하게 항상 배선한다** — 같은 이유이고, 값은 생성
    클라이언트다. 재작성은 픽스처를 만든 조건과 같은 계열·같은 모델이어야 오프라인
    비교표가 런타임을 예측한다(`docs/tracking/decisions/0010`). 판정 계열 분리
    (결정 0004)의 대상이 아니다 — 재작성은 자기 출력을 평가하는 것이 아니라 검색어를
    만드는 일이고, 그 뒤에 L1·L2 두 층이 그대로 서 있다.
    """
    resolved = settings if settings is not None else get_settings()
    return InquiryPipeline(
        collector=EvidenceCollector(
            generation_client=generation_client,
            embedding_client=embedding_client,
            settings=resolved,
            rewrite_client=generation_client,
        ),
        drafter=DraftGenerator(client=generation_client, effort=resolved.generation_effort),
        judge=build_judge(resolved),
        l2_enabled=resolved.l2_enabled,
    )
