"""평가 하네스 — 결정론 층과 확률 층을 **분리해서** 측정한다 (spec "평가" 절).

두 측정은 서로의 수치를 오염시키지 않는다. 그것이 이 모듈의 존재 이유다.

* **측정 1 — L1 게이트 단위 정확도 (결정론).** `data/l1_fixtures.jsonl` 의 고정 "초안+근거"
  쌍에 `gate.evaluate_draft` 를 직접 적용한다. **LLM 을 호출하지 않는다** — 이 모듈은
  `reply_gate.llm` 의 실제 클라이언트를 import 하지 않으며, 측정 1 경로는 네트워크를 타지
  않는다. 100% 재현되므로 신뢰성 서사의 헤드라인 수치(구조적 오류 검출률·정상 초안 오탐률)는
  여기서 나온다.
* **측정 2 — 파이프라인 판정 일치율 (end-to-end).** 골든셋 30건을 파이프라인에 흘려
  허용 결과 집합과 대조한다. 확률 층이므로 재실행하면 값이 달라진다. **초안 전 인계
  경로가 포함되므로 L1 판정만의 지표가 아니다.**

골든셋 라벨은 단일 정답이 아니라 **허용 결과 집합**이다(`ExpectedOutcomeSet`): 허용 최종
상태 집합 + 허용 인계 사유 집합 + 기각 기대 여부 + 금지 기각 사유 집합. 초안이 확률적이라
같은 문의가 여러 정당한 결말을 가질 수 있기 때문이다.

**목표치는 이 모듈에 없다.** 첫 측정값을 보고 확정하는 것이 이번 사이클의 결정이고,
하네스가 수치를 산출하는 것까지가 완료 조건이다. 리포트는 목표치를 "미확정"으로 적는다.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol

import psycopg
from psycopg.rows import DictRow

from reply_gate.contracts import (
    EscalationReason,
    Evidence,
    EvidenceSource,
    InquiryStatus,
    RejectReason,
    Verdict,
)
from reply_gate.gate import DEFAULT_PII_PATTERNS, REASON_ORDER, evaluate_draft
from reply_gate.pipeline import ProcessedInquiry, ReceiptError, accept_inquiry, new_inquiry_id

__all__ = [
    "DEFAULT_GOLDEN_SET_PATH",
    "DEFAULT_L1_FIXTURES_PATH",
    "DEFAULT_REPORT_DIR",
    "REASON_ORDER",
    "EvaluationReport",
    "ExpectedOutcomeSet",
    "GateAccuracy",
    "GoldenCase",
    "GoldenOutcome",
    "L1Fixture",
    "L1Outcome",
    "PipelineAgreement",
    "PipelineRunning",
    "ReasonBreakdown",
    "RunConditions",
    "SkippedMeasurement",
    "StubGenerationClient",
    "build_report",
    "load_golden_set",
    "load_l1_fixtures",
    "measure_gate_accuracy",
    "measure_pipeline_agreement",
    "render_markdown",
    "report_to_json",
    "write_report",
]

_ROOT: Final = Path(__file__).resolve().parents[2]

#: 저장소의 골든셋(30건) — 허용 결과 집합 라벨이 붙은 ground truth.
DEFAULT_GOLDEN_SET_PATH: Final = _ROOT / "data" / "golden_set.jsonl"
#: 저장소의 L1 픽스처 셋 — 고정 초안+근거 쌍(LLM 호출 없음).
DEFAULT_L1_FIXTURES_PATH: Final = _ROOT / "data" / "l1_fixtures.jsonl"
#: 리포트 산출 위치. `.gitignore` 되어 있어 산출물은 커밋되지 않는다.
DEFAULT_REPORT_DIR: Final = _ROOT / "reports"


# ══ 측정 1 — L1 게이트 단위 정확도 (결정론, LLM 호출 0회) ═══════════════════


@dataclass(frozen=True)
class L1Fixture:
    """고정 "초안 + 근거 + 기대 판정" 1건. 규모는 조정 가능 기본값이다."""

    id: str
    category: str
    note: str
    raw_draft: Any
    evidences: tuple[Evidence, ...]
    expected_verdict: Verdict
    expected_reasons: tuple[RejectReason, ...]

    @property
    def is_violation(self) -> bool:
        """기각되어야 하는 픽스처인가 — 검출률의 분모."""
        return self.expected_verdict is Verdict.REJECT


@dataclass(frozen=True)
class L1Outcome:
    """픽스처 1건의 측정 결과."""

    fixture_id: str
    category: str
    expected_verdict: Verdict
    actual_verdict: Verdict
    expected_reasons: tuple[RejectReason, ...]
    actual_reasons: tuple[RejectReason, ...]

    @property
    def verdict_matched(self) -> bool:
        return self.expected_verdict is self.actual_verdict

    @property
    def reasons_matched(self) -> bool:
        """사유 목록이 **순서까지** 같은가. L1 은 결정론이므로 순서도 계약이다."""
        return self.expected_reasons == self.actual_reasons


@dataclass(frozen=True)
class ReasonBreakdown:
    """기각 사유 1종의 내역."""

    reason: RejectReason
    expected_count: int
    detected_count: int
    #: 그 사유를 기대하지 않은 픽스처에서 발화한 횟수(사유 단위 오탐).
    spurious_count: int

    @property
    def detection_rate(self) -> float | None:
        if self.expected_count == 0:
            return None
        return self.detected_count / self.expected_count


@dataclass(frozen=True)
class GateAccuracy:
    """측정 1 의 집계. 이 수치가 신뢰성 서사의 헤드라인이다."""

    total: int
    violation_total: int
    violation_detected: int
    clean_total: int
    clean_false_positive: int
    reason_set_exact: int
    breakdown: tuple[ReasonBreakdown, ...]
    outcomes: tuple[L1Outcome, ...]

    @property
    def detection_rate(self) -> float | None:
        """구조적 오류 검출률 — 기각되어야 할 픽스처 중 실제로 기각된 비율."""
        if self.violation_total == 0:
            return None
        return self.violation_detected / self.violation_total

    @property
    def false_positive_rate(self) -> float | None:
        """정상 초안 오탐률 — 통과해야 할 픽스처 중 기각된 비율."""
        if self.clean_total == 0:
            return None
        return self.clean_false_positive / self.clean_total

    @property
    def reason_set_exact_rate(self) -> float | None:
        """사유 목록까지 정확히 일치한 비율(검출률보다 엄격한 보조 지표)."""
        if self.total == 0:
            return None
        return self.reason_set_exact / self.total


def _evidence_from_json(raw: Mapping[str, Any]) -> Evidence:
    content = str(raw["content"])
    return Evidence(
        id=str(raw["id"]),
        source=EvidenceSource(raw["source"]),
        content=content,
        # 대조용 원문은 표시용과 다를 수 있다 — 없으면 표시용을 그대로 쓴다.
        evidence_text=str(raw.get("evidence_text", content)),
    )


def load_l1_fixtures(path: Path = DEFAULT_L1_FIXTURES_PATH) -> tuple[L1Fixture, ...]:
    """L1 픽스처 셋을 읽는다. 기대 판정과 사유가 없는 줄은 오류다(라벨이 ground truth)."""
    fixtures: list[L1Fixture] = []
    seen: set[str] = set()
    for row in _read_jsonl(path):
        expected = row["expected"]
        fixture_id = str(row["id"])
        if fixture_id in seen:
            raise ValueError(f"픽스처 ID 가 중복된다: {fixture_id}")
        seen.add(fixture_id)
        verdict = Verdict(expected["verdict"])
        reasons = tuple(RejectReason(value) for value in expected["reject_reasons"])
        if (verdict is Verdict.REJECT) != bool(reasons):
            raise ValueError(f"{fixture_id}: reject 는 사유가 1개 이상, pass 는 사유가 없어야 한다")
        fixtures.append(
            L1Fixture(
                id=fixture_id,
                category=str(row["category"]),
                note=str(row.get("note", "")),
                raw_draft=row["raw_draft"],
                evidences=tuple(_evidence_from_json(item) for item in row["evidences"]),
                expected_verdict=verdict,
                expected_reasons=reasons,
            )
        )
    if not fixtures:
        raise ValueError(f"픽스처가 하나도 없다: {path}")
    return tuple(fixtures)


def measure_gate_accuracy(fixtures: Sequence[L1Fixture]) -> GateAccuracy:
    """측정 1 — `gate.evaluate_draft` 를 직접 부른다. **LLM 호출 0회, 100% 재현.**"""
    outcomes: list[L1Outcome] = []
    for fixture in fixtures:
        result = evaluate_draft(raw_draft=fixture.raw_draft, evidences=fixture.evidences)
        outcomes.append(
            L1Outcome(
                fixture_id=fixture.id,
                category=fixture.category,
                expected_verdict=fixture.expected_verdict,
                actual_verdict=result.verdict,
                expected_reasons=fixture.expected_reasons,
                actual_reasons=result.reject_reasons,
            )
        )

    violations = [outcome for outcome in outcomes if outcome.expected_verdict is Verdict.REJECT]
    cleans = [outcome for outcome in outcomes if outcome.expected_verdict is Verdict.PASS]
    breakdown = tuple(
        ReasonBreakdown(
            reason=reason,
            expected_count=sum(1 for o in outcomes if reason in o.expected_reasons),
            detected_count=sum(
                1 for o in outcomes if reason in o.expected_reasons and reason in o.actual_reasons
            ),
            spurious_count=sum(
                1
                for o in outcomes
                if reason not in o.expected_reasons and reason in o.actual_reasons
            ),
        )
        for reason in REASON_ORDER
    )
    return GateAccuracy(
        total=len(outcomes),
        violation_total=len(violations),
        violation_detected=sum(1 for o in violations if o.actual_verdict is Verdict.REJECT),
        clean_total=len(cleans),
        clean_false_positive=sum(1 for o in cleans if o.actual_verdict is Verdict.REJECT),
        reason_set_exact=sum(1 for o in outcomes if o.reasons_matched),
        breakdown=breakdown,
        outcomes=tuple(outcomes),
    )


# ══ 측정 2 — 파이프라인 판정 일치율 (end-to-end, 확률 층) ═══════════════════


@dataclass(frozen=True)
class ExpectedOutcomeSet:
    """골든셋 라벨 = **허용 결과 집합**. 단일 정답이 아니다.

    * `statuses` — 허용 최종 상태 집합(비어 있을 수 없다).
    * `escalation_reasons` — `escalated` 로 끝났을 때 허용되는 인계 사유 집합.
    * `expect_reject` — 시도 중 **최소 1건**이 기각이어야 하는가(기각 재현율의 분모).
    * `forbidden_reject_reasons` — 어떤 시도에서도 나오면 안 되는 사유(오탐 감시).
    """

    statuses: frozenset[InquiryStatus]
    escalation_reasons: frozenset[EscalationReason]
    expect_reject: bool
    forbidden_reject_reasons: frozenset[RejectReason]


@dataclass(frozen=True)
class GoldenCase:
    """골든셋 문의 1건 + 허용 결과 집합 라벨."""

    id: str
    category: str
    order_no: str | None
    content: str
    expected: ExpectedOutcomeSet
    note: str


@dataclass(frozen=True)
class GoldenOutcome:
    """골든셋 1건의 end-to-end 결과 + 라벨 대조."""

    case_id: str
    category: str
    status: InquiryStatus | None
    escalation_reason: EscalationReason | None
    failed_stage: str | None
    attempt_verdicts: tuple[Verdict, ...]
    reject_reasons: tuple[RejectReason, ...]
    latency_ms: int
    input_tokens: int
    output_tokens: int
    embedding_tokens: int
    matched: bool
    mismatches: tuple[str, ...]
    #: 접수 거부처럼 파이프라인에 진입조차 못한 경우의 사유.
    error: str | None

    @property
    def rejected_at_least_once(self) -> bool:
        return any(verdict is Verdict.REJECT for verdict in self.attempt_verdicts)


@dataclass(frozen=True)
class PipelineAgreement:
    """측정 2 의 집계.

    **초안 전 인계 경로가 포함된다** — 근거 0건·주문번호 없음·주문 없음으로 끝난 건도
    분모에 들어간다. 따라서 이 일치율은 L1 판정만의 지표가 아니다.
    """

    total: int
    matched: int
    bait_total: int
    bait_reject_reproduced: int
    forbidden_watch_total: int
    forbidden_violations: int
    latency_p50_ms: int | None
    latency_p95_ms: int | None
    input_tokens_total: int
    output_tokens_total: int
    embedding_tokens_total: int
    status_counts: Mapping[str, int]
    escalation_counts: Mapping[str, int]
    outcomes: tuple[GoldenOutcome, ...]

    @property
    def match_rate(self) -> float | None:
        return None if self.total == 0 else self.matched / self.total

    @property
    def bait_reject_recall(self) -> float | None:
        """기각 유발 문의의 기각 재현율 — 데모 신뢰성의 근거 수치."""
        return None if self.bait_total == 0 else self.bait_reject_reproduced / self.bait_total

    @property
    def generation_tokens_per_inquiry(self) -> float | None:
        if self.total == 0:
            return None
        return (self.input_tokens_total + self.output_tokens_total) / self.total

    @property
    def embedding_tokens_per_inquiry(self) -> float | None:
        return None if self.total == 0 else self.embedding_tokens_total / self.total

    @property
    def total_tokens_per_inquiry(self) -> float | None:
        if self.total == 0:
            return None
        total = self.input_tokens_total + self.output_tokens_total + self.embedding_tokens_total
        return total / self.total


@dataclass(frozen=True)
class SkippedMeasurement:
    """측정을 실행하지 않았다는 **명시적 기록**.

    조용히 0 이나 빈 값을 채워 "돌았다"처럼 보이게 하지 않는다 — 미실행은 미실행으로 남는다.
    """

    reason: str


def _expected_from_json(raw: Mapping[str, Any]) -> ExpectedOutcomeSet:
    statuses = frozenset(InquiryStatus(value) for value in raw["statuses"])
    if not statuses:
        raise ValueError("허용 최종 상태 집합이 비어 있다")
    reasons = frozenset(EscalationReason(value) for value in raw["escalation_reasons"])
    if (InquiryStatus.ESCALATED in statuses) != bool(reasons):
        raise ValueError("escalated 를 허용하면 인계 사유 집합이 비어 있을 수 없다(역도 같다)")
    return ExpectedOutcomeSet(
        statuses=statuses,
        escalation_reasons=reasons,
        expect_reject=bool(raw["expect_reject"]),
        forbidden_reject_reasons=frozenset(
            RejectReason(value) for value in raw.get("forbidden_reject_reasons", ())
        ),
    )


def load_golden_set(path: Path = DEFAULT_GOLDEN_SET_PATH) -> tuple[GoldenCase, ...]:
    """골든셋을 읽는다. 라벨 구조가 어긋나면 조용히 넘기지 않고 오류로 세운다."""
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for row in _read_jsonl(path):
        case_id = str(row["id"])
        if case_id in seen:
            raise ValueError(f"골든셋 ID 가 중복된다: {case_id}")
        seen.add(case_id)
        order_no = row.get("order_no")
        try:
            expected = _expected_from_json(row["expected"])
        except ValueError as exc:
            raise ValueError(f"{case_id}: {exc}") from exc
        cases.append(
            GoldenCase(
                id=case_id,
                category=str(row["category"]),
                order_no=None if order_no is None else str(order_no),
                content=str(row["content"]),
                expected=expected,
                note=str(row.get("note", "")),
            )
        )
    if not cases:
        raise ValueError(f"골든셋이 비어 있다: {path}")
    return tuple(cases)


class PipelineRunning(Protocol):
    """`pipeline.InquiryPipeline.run` 의 공개 표면 — 테스트가 대역을 넣을 수 있게."""

    def run(
        self,
        *,
        inquiry_id: str,
        content: str,
        order_no: str | None,
        app_conn: psycopg.Connection[DictRow],
        readonly_conn: psycopg.Connection[DictRow],
    ) -> ProcessedInquiry: ...


def _compare(case: GoldenCase, processed: ProcessedInquiry) -> tuple[bool, tuple[str, ...]]:
    """허용 결과 집합 대비 대조. 어긋난 항목을 **전부** 모은다."""
    expected = case.expected
    mismatches: list[str] = []

    if processed.status not in expected.statuses:
        allowed = ", ".join(sorted(status.value for status in expected.statuses))
        mismatches.append(f"최종 상태 {processed.status.value} 가 허용 집합 {{{allowed}}} 밖이다")
    elif processed.status is InquiryStatus.ESCALATED:
        reason = processed.escalation_reason
        if reason is None or reason not in expected.escalation_reasons:
            allowed = ", ".join(sorted(item.value for item in expected.escalation_reasons))
            got = "없음" if reason is None else reason.value
            mismatches.append(f"인계 사유 {got} 가 허용 집합 {{{allowed}}} 밖이다")

    rejected = any(attempt.verdict is Verdict.REJECT for attempt in processed.attempts)
    if expected.expect_reject and not rejected:
        mismatches.append("기각을 기대했으나 어떤 시도도 기각되지 않았다")

    seen_reasons = {reason for attempt in processed.attempts for reason in attempt.reject_reasons}
    forbidden = sorted(reason.value for reason in seen_reasons & expected.forbidden_reject_reasons)
    if forbidden:
        mismatches.append(f"금지 기각 사유가 발화했다(오탐): {', '.join(forbidden)}")

    return not mismatches, tuple(mismatches)


def evaluate_case(
    *,
    case: GoldenCase,
    pipeline: PipelineRunning,
    app_conn: psycopg.Connection[DictRow],
    readonly_conn: psycopg.Connection[DictRow],
) -> GoldenOutcome:
    """골든셋 1건을 접수 → 파이프라인 → 라벨 대조까지 돌린다.

    접수 거부(`ReceiptError`)는 파이프라인 진입 전 실패이므로 **불일치로 기록**한다 —
    골든셋의 주문번호는 전부 형식이 맞아야 하기 때문이다.
    """
    try:
        accepted = accept_inquiry(content=case.content, order_no=case.order_no)
    except ReceiptError as exc:
        return GoldenOutcome(
            case_id=case.id,
            category=case.category,
            status=None,
            escalation_reason=None,
            failed_stage=None,
            attempt_verdicts=(),
            reject_reasons=(),
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
            embedding_tokens=0,
            matched=False,
            mismatches=(f"접수 거부: {exc}",),
            error=str(exc),
        )

    processed = pipeline.run(
        inquiry_id=new_inquiry_id(),
        content=accepted.content,
        order_no=accepted.order_no,
        app_conn=app_conn,
        readonly_conn=readonly_conn,
    )
    matched, mismatches = _compare(case, processed)
    return GoldenOutcome(
        case_id=case.id,
        category=case.category,
        status=processed.status,
        escalation_reason=processed.escalation_reason,
        failed_stage=processed.failed_stage,
        attempt_verdicts=tuple(attempt.verdict for attempt in processed.attempts),
        reject_reasons=tuple(
            reason for attempt in processed.attempts for reason in attempt.reject_reasons
        ),
        latency_ms=processed.latency_ms,
        input_tokens=processed.input_tokens,
        output_tokens=processed.output_tokens,
        embedding_tokens=processed.embedding_tokens,
        matched=matched,
        mismatches=mismatches,
        error=None,
    )


def measure_pipeline_agreement(
    *,
    cases: Sequence[GoldenCase],
    pipeline: PipelineRunning,
    app_conn: psycopg.Connection[DictRow],
    readonly_conn: psycopg.Connection[DictRow],
    on_outcome: Callable[[GoldenOutcome], None] | None = None,
) -> PipelineAgreement:
    """측정 2 — 골든셋 전체를 **끝까지** 흘린다.

    한 건이 실패해도 나머지를 계속 돈다: 30건 중 1건의 사고로 리포트가 통째로 사라지면
    측정이 되지 않는다. 인프라 예외(`psycopg.Error` 등)는 파이프라인이 그대로 올려보내므로
    여기서 삼키지 않고 그대로 터뜨린다 — 그건 지표가 아니라 환경 고장이다.
    """
    outcomes: list[GoldenOutcome] = []
    for case in cases:
        outcome = evaluate_case(
            case=case, pipeline=pipeline, app_conn=app_conn, readonly_conn=readonly_conn
        )
        outcomes.append(outcome)
        if on_outcome is not None:
            on_outcome(outcome)

    status_counts: dict[str, int] = {}
    escalation_counts: dict[str, int] = {}
    for outcome in outcomes:
        key = outcome.status.value if outcome.status is not None else "접수거부"
        status_counts[key] = status_counts.get(key, 0) + 1
        if outcome.escalation_reason is not None:
            name = outcome.escalation_reason.value
            escalation_counts[name] = escalation_counts.get(name, 0) + 1

    bait = [
        (case, outcome)
        for case, outcome in zip(cases, outcomes, strict=True)
        if case.expected.expect_reject
    ]
    watch = [
        (case, outcome)
        for case, outcome in zip(cases, outcomes, strict=True)
        if case.expected.forbidden_reject_reasons
    ]
    latencies = [outcome.latency_ms for outcome in outcomes if outcome.error is None]

    return PipelineAgreement(
        total=len(outcomes),
        matched=sum(1 for outcome in outcomes if outcome.matched),
        bait_total=len(bait),
        bait_reject_reproduced=sum(1 for _, outcome in bait if outcome.rejected_at_least_once),
        forbidden_watch_total=len(watch),
        forbidden_violations=sum(
            1
            for case, outcome in watch
            if set(outcome.reject_reasons) & case.expected.forbidden_reject_reasons
        ),
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        input_tokens_total=sum(outcome.input_tokens for outcome in outcomes),
        output_tokens_total=sum(outcome.output_tokens for outcome in outcomes),
        embedding_tokens_total=sum(outcome.embedding_tokens for outcome in outcomes),
        status_counts=status_counts,
        escalation_counts=escalation_counts,
        outcomes=tuple(outcomes),
    )


def _percentile(values: Sequence[int], percent: int) -> int | None:
    """nearest-rank 백분위. 30건 규모에서 보간은 없는 정밀도를 가장한다."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), _ceil_div(percent * len(ordered), 100)))
    return ordered[rank - 1]


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


# ══ 실행 조건 · 리포트 ══════════════════════════════════════════════════════


@dataclass(frozen=True)
class RunConditions:
    """무엇으로 돌렸는지. **대역으로 낸 수치를 실제 수치처럼 보고하지 않기 위한 기록이다.**"""

    started_at: str
    generation: str
    embedding: str
    similarity_threshold: float
    top_k: int
    l1_fixture_count: int
    golden_case_count: int
    l1_fixtures_path: str
    golden_set_path: str
    api_key_present: bool
    #: 측정 2 수치가 실제 모델로 낸 값인가. 대역이면 False 이고 리포트가 그렇게 적는다.
    measurement2_is_real: bool


@dataclass(frozen=True)
class EvaluationReport:
    """리포트 1건 — 사람이 읽는 마크다운과 기계가 읽는 JSON 의 공통 원본."""

    conditions: RunConditions
    gate_accuracy: GateAccuracy
    pipeline: PipelineAgreement | SkippedMeasurement


def build_report(
    *,
    conditions: RunConditions,
    gate_accuracy: GateAccuracy,
    pipeline: PipelineAgreement | SkippedMeasurement,
) -> EvaluationReport:
    return EvaluationReport(conditions=conditions, gate_accuracy=gate_accuracy, pipeline=pipeline)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _pct(value: float | None) -> str:
    return "측정 불가" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None) -> str:
    return "측정 불가" if value is None else f"{value:.1f}"


def _int(value: int | None) -> str:
    return "측정 불가" if value is None else str(value)


_LIMITS: Final = """\
## 한계 (과장하지 않는다)

- **L1 은 패턴형 PII 만 본다.** 전화번호·이메일·주민등록번호처럼 정규식으로 잡히는 값만
  검사한다. 이름·주소 등 비패턴형 개인정보는 정규식으로 잡을 수 없어 **L1 의 검사 대상이
  아니며 L2 사이클로 이월**한다. 검출률 수치를 "개인정보 전반"으로 읽으면 안 된다.
- **L1 은 내용의 진위를 보지 않는다.** citation 존재·무결성·스키마·PII 만 검사하므로,
  근거를 인용했지만 내용이 근거와 어긋나는 답변은 이번 사이클에서 통과한다.
- **측정 2 는 확률 층이다.** 초안 생성이 비결정론이므로 재실행하면 값이 달라진다.
  측정 1 만 100% 재현된다.
- **측정 2 의 일치율에는 초안 전 인계 경로가 포함된다** — 근거 0건·주문번호 없음·주문
  없음으로 끝난 건도 분모에 들어가므로 **L1 판정만의 지표가 아니다**.

## 이월 (L2 사이클)

- 내용상 hallucination율 (claim 단위 근거 대조)
- L1 필터링에 의한 L2 호출 감소율
- RAG 검색 품질 단계별 개선표
- 비패턴형 개인정보(이름·주소) 검출
"""


def render_markdown(report: EvaluationReport) -> str:
    """사람이 읽는 리포트. 서술 축은 **신뢰성 지표**(검출률·오탐률)다."""
    conditions = report.conditions
    lines: list[str] = [
        "# Reply-Gate 평가 리포트",
        "",
        "측정은 **결정론 층과 확률 층을 분리**한다 — LLM 비결정성이 게이트 자체의 정확도",
        "수치를 흔들지 못하게 하기 위해서다.",
        "",
        "## 실행 조건",
        "",
        f"- 실행 시각(UTC): `{conditions.started_at}`",
        f"- 생성 LLM: {conditions.generation}",
        f"- 임베딩: {conditions.embedding}",
        f"- 유사도 임계값: {conditions.similarity_threshold} / top k: {conditions.top_k}",
        f"- L1 픽스처: {conditions.l1_fixture_count}건 (`{conditions.l1_fixtures_path}`)",
        f"- 골든셋: {conditions.golden_case_count}건 (`{conditions.golden_set_path}`)",
        f"- OPENAI_API_KEY 설정 여부: {'설정됨' if conditions.api_key_present else '없음'}",
        "",
        "## 목표치",
        "",
        "**미확정 — 첫 측정값을 보고 결정한다(조정 가능으로 기록).** 지표 목표치를 지금 박지",
        "않는 것은 결정이며, 평가 하네스가 수치를 산출하는 것까지가 이번 사이클의 완료 조건이다.",
        "",
    ]
    lines.extend(_render_measurement_one(report.gate_accuracy))
    lines.extend(_render_measurement_two(report.pipeline, conditions))
    lines.append(_LIMITS)
    return "\n".join(lines)


def _render_measurement_one(accuracy: GateAccuracy) -> list[str]:
    lines = [
        "## 측정 1 — L1 게이트 단위 정확도 (결정론)",
        "",
        "고정 초안+근거 쌍에 `gate.evaluate_draft` 를 직접 적용했다. **LLM 호출 0회, 100% 재현.**",
        "신뢰성 서사의 헤드라인 수치는 이것이다.",
        "",
        f"- 픽스처 총수: **{accuracy.total}건** "
        f"(위반 {accuracy.violation_total} / 정상 {accuracy.clean_total})",
        f"- **구조적 오류 검출률: {_pct(accuracy.detection_rate)}** "
        f"({accuracy.violation_detected}/{accuracy.violation_total})",
        f"- **정상 초안 오탐률: {_pct(accuracy.false_positive_rate)}** "
        f"({accuracy.clean_false_positive}/{accuracy.clean_total})",
        f"- 사유 목록까지 정확히 일치: {_pct(accuracy.reason_set_exact_rate)} "
        f"({accuracy.reason_set_exact}/{accuracy.total})",
        "",
        "### 사유 4종별 내역",
        "",
        "| 사유 | 기대 픽스처 | 검출 | 검출률 | 오발화(기대하지 않은 발화) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| `{item.reason.value}` | {item.expected_count} | {item.detected_count} | "
        f"{_pct(item.detection_rate)} | {item.spurious_count} |"
        for item in accuracy.breakdown
    )
    lines.append("")

    failures = [outcome for outcome in accuracy.outcomes if not outcome.reasons_matched]
    if failures:
        lines.extend(["### 기대와 어긋난 픽스처", ""])
        lines.extend(
            f"- `{outcome.fixture_id}` ({outcome.category}): "
            f"기대 {outcome.expected_verdict.value}"
            f"{_reason_list(outcome.expected_reasons)} → "
            f"실제 {outcome.actual_verdict.value}{_reason_list(outcome.actual_reasons)}"
            for outcome in failures
        )
    else:
        lines.append("모든 픽스처가 기대 판정·기대 사유 목록과 일치했다.")
    lines.append("")
    return lines


def _reason_list(reasons: Sequence[RejectReason]) -> str:
    return "" if not reasons else "[" + ", ".join(reason.value for reason in reasons) + "]"


def _render_measurement_two(
    pipeline: PipelineAgreement | SkippedMeasurement, conditions: RunConditions
) -> list[str]:
    lines = ["## 측정 2 — 파이프라인 판정 일치율 (end-to-end)", ""]
    if isinstance(pipeline, SkippedMeasurement):
        lines.extend(
            [
                f"**미실행 (사유: {pipeline.reason})**",
                "",
                "수치를 0 이나 빈 값으로 채우지 않는다 — 미실행은 미실행으로 남긴다.",
                "",
            ]
        )
        return lines

    if not conditions.measurement2_is_real:
        lines.extend(
            [
                "> **경고 — 아래 수치는 실제 모델 수치가 아니다.** 결정론 대역(생성 대역 ·",
                "> 어휘 임베딩)으로 하네스 배관만 검증한 실행이다. 일치율·기각 재현율·지연·",
                "> 토큰 값을 제품 지표로 인용하면 안 된다. 실행 조건 절의 생성 LLM·임베딩",
                "> 항목이 무엇으로 돌았는지를 함께 읽어야 한다.",
                "",
            ]
        )

    lines.extend(
        [
            f"- 골든셋 {pipeline.total}건 처리",
            f"- **허용 결과 집합 대비 일치율: {_pct(pipeline.match_rate)}** "
            f"({pipeline.matched}/{pipeline.total}) "
            "— **초안 전 인계 경로 포함이며 L1 판정만의 지표가 아니다.**",
            f"- **기각 유발 문의의 기각 재현율: {_pct(pipeline.bait_reject_recall)}** "
            f"({pipeline.bait_reject_reproduced}/{pipeline.bait_total})",
            f"- 정상 PII 에코 감시 케이스: {pipeline.forbidden_watch_total}건 중 "
            f"금지 사유 발화 {pipeline.forbidden_violations}건",
            f"- 지연 p50: {_int(pipeline.latency_p50_ms)} ms / "
            f"p95: {_int(pipeline.latency_p95_ms)} ms "
            "(파이프라인 `run` 의 벽시계 시간 — 처리 기록 저장은 포함하지 않는다)",
            "",
            "### 문의 1건당 토큰 (생성·임베딩 구분)",
            "",
            "| 계열 | 합계 | 건당 |",
            "| --- | ---: | ---: |",
            f"| 생성 입력 | {pipeline.input_tokens_total} | "
            f"{_num(pipeline.input_tokens_total / pipeline.total if pipeline.total else None)} |",
            f"| 생성 출력 | {pipeline.output_tokens_total} | "
            f"{_num(pipeline.output_tokens_total / pipeline.total if pipeline.total else None)} |",
            f"| 생성 소계 | {pipeline.input_tokens_total + pipeline.output_tokens_total} | "
            f"{_num(pipeline.generation_tokens_per_inquiry)} |",
            f"| 임베딩 | {pipeline.embedding_tokens_total} | "
            f"{_num(pipeline.embedding_tokens_per_inquiry)} |",
            f"| **합산** | {_grand_total(pipeline)} | "
            f"**{_num(pipeline.total_tokens_per_inquiry)}** |",
            "",
            "### 종결 분포",
            "",
            f"- 최종 상태: {_counts(pipeline.status_counts)}",
            f"- 인계 사유: {_counts(pipeline.escalation_counts)}",
            "",
        ]
    )

    mismatched = [outcome for outcome in pipeline.outcomes if not outcome.matched]
    if mismatched:
        lines.extend(["### 허용 결과 집합과 어긋난 문의", ""])
        lines.extend(
            f"- `{outcome.case_id}` ({outcome.category}): " + " / ".join(outcome.mismatches)
            for outcome in mismatched
        )
    else:
        lines.append("모든 문의가 허용 결과 집합 안에서 종결했다.")
    lines.append("")
    return lines


def _grand_total(pipeline: PipelineAgreement) -> int:
    return (
        pipeline.input_tokens_total + pipeline.output_tokens_total + pipeline.embedding_tokens_total
    )


def _counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "없음"
    return ", ".join(f"{key} {value}건" for key, value in sorted(counts.items()))


def report_to_json(report: EvaluationReport) -> dict[str, Any]:
    """기계가 읽는 형식. 마크다운과 같은 원본에서 나온다."""
    conditions = report.conditions
    accuracy = report.gate_accuracy
    payload: dict[str, Any] = {
        "targets": "미확정 — 첫 측정값을 보고 결정 (조정 가능)",
        "conditions": {
            "started_at": conditions.started_at,
            "generation": conditions.generation,
            "embedding": conditions.embedding,
            "similarity_threshold": conditions.similarity_threshold,
            "top_k": conditions.top_k,
            "l1_fixture_count": conditions.l1_fixture_count,
            "golden_case_count": conditions.golden_case_count,
            "l1_fixtures_path": conditions.l1_fixtures_path,
            "golden_set_path": conditions.golden_set_path,
            "api_key_present": conditions.api_key_present,
            "measurement2_is_real": conditions.measurement2_is_real,
        },
        "measurement_1_l1_gate_accuracy": {
            "deterministic": True,
            "llm_calls": 0,
            "total": accuracy.total,
            "violation_total": accuracy.violation_total,
            "violation_detected": accuracy.violation_detected,
            "detection_rate": accuracy.detection_rate,
            "clean_total": accuracy.clean_total,
            "clean_false_positive": accuracy.clean_false_positive,
            "false_positive_rate": accuracy.false_positive_rate,
            "reason_set_exact_rate": accuracy.reason_set_exact_rate,
            "reason_breakdown": [
                {
                    "reason": item.reason.value,
                    "expected_count": item.expected_count,
                    "detected_count": item.detected_count,
                    "detection_rate": item.detection_rate,
                    "spurious_count": item.spurious_count,
                }
                for item in accuracy.breakdown
            ],
            "outcomes": [
                {
                    "fixture_id": outcome.fixture_id,
                    "category": outcome.category,
                    "expected_verdict": outcome.expected_verdict.value,
                    "actual_verdict": outcome.actual_verdict.value,
                    "expected_reasons": [r.value for r in outcome.expected_reasons],
                    "actual_reasons": [r.value for r in outcome.actual_reasons],
                    "matched": outcome.reasons_matched,
                }
                for outcome in accuracy.outcomes
            ],
        },
        "limits": {
            "pii": "L1 은 패턴형 PII 만 검사한다. 이름·주소 등 비패턴형은 L2 사이클로 이월.",
            "content": "L1 은 내용의 진위를 검사하지 않는다 (L2 이월).",
            "measurement_2": "확률 층이라 재실행하면 값이 달라진다. 일치율에 초안 전 인계 포함.",
        },
        "deferred_to_l2": [
            "내용상 hallucination율",
            "L1 필터링에 의한 L2 호출 감소율",
            "RAG 검색 품질 단계별 개선표",
            "비패턴형 개인정보(이름·주소) 검출",
        ],
    }
    payload["measurement_2_pipeline_agreement"] = _measurement_two_json(report.pipeline)
    return payload


def _measurement_two_json(pipeline: PipelineAgreement | SkippedMeasurement) -> dict[str, Any]:
    if isinstance(pipeline, SkippedMeasurement):
        return {"executed": False, "skip_reason": pipeline.reason}
    return {
        "executed": True,
        "deterministic": False,
        "includes_pre_draft_escalation": True,
        "total": pipeline.total,
        "matched": pipeline.matched,
        "match_rate": pipeline.match_rate,
        "bait_total": pipeline.bait_total,
        "bait_reject_reproduced": pipeline.bait_reject_reproduced,
        "bait_reject_recall": pipeline.bait_reject_recall,
        "forbidden_watch_total": pipeline.forbidden_watch_total,
        "forbidden_violations": pipeline.forbidden_violations,
        "latency_p50_ms": pipeline.latency_p50_ms,
        "latency_p95_ms": pipeline.latency_p95_ms,
        "tokens": {
            "generation_input_total": pipeline.input_tokens_total,
            "generation_output_total": pipeline.output_tokens_total,
            "embedding_total": pipeline.embedding_tokens_total,
            "generation_per_inquiry": pipeline.generation_tokens_per_inquiry,
            "embedding_per_inquiry": pipeline.embedding_tokens_per_inquiry,
            "total_per_inquiry": pipeline.total_tokens_per_inquiry,
        },
        "status_counts": dict(pipeline.status_counts),
        "escalation_counts": dict(pipeline.escalation_counts),
        "outcomes": [
            {
                "case_id": outcome.case_id,
                "category": outcome.category,
                "status": None if outcome.status is None else outcome.status.value,
                "escalation_reason": (
                    None if outcome.escalation_reason is None else outcome.escalation_reason.value
                ),
                "failed_stage": outcome.failed_stage,
                "attempt_verdicts": [verdict.value for verdict in outcome.attempt_verdicts],
                "reject_reasons": [reason.value for reason in outcome.reject_reasons],
                "latency_ms": outcome.latency_ms,
                "input_tokens": outcome.input_tokens,
                "output_tokens": outcome.output_tokens,
                "embedding_tokens": outcome.embedding_tokens,
                "matched": outcome.matched,
                "mismatches": list(outcome.mismatches),
                "error": outcome.error,
            }
            for outcome in pipeline.outcomes
        ],
    }


def write_report(
    report: EvaluationReport, *, out_dir: Path = DEFAULT_REPORT_DIR, stem: str = "evaluation"
) -> tuple[Path, Path]:
    """마크다운과 JSON 을 **둘 다** 쓴다. 돌려주는 것은 (마크다운 경로, JSON 경로)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = out_dir / f"{stem}.md"
    json_path = out_dir / f"{stem}.json"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report_to_json(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return markdown_path, json_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} JSON 파싱 실패: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{lineno} 각 줄은 JSON 객체여야 한다")
            rows.append(value)
    return rows


# ══ 생성 LLM 대역 (배관 검증 전용) ══════════════════════════════════════════

_EVIDENCE_ID = re.compile(r"^- 근거 ID: (?P<id>.+)$", re.MULTILINE)
_EVIDENCE_BLOCK = re.compile(r"\[근거\]\n(?P<block>.*?)(?=\n\n\[|\Z)", re.DOTALL)
_ORDER_NO_SECTION = re.compile(r"\[조회할 주문번호\]\n(?P<order_no>\S+)")
_ACCEPTED_ORDER_SECTION = re.compile(r"\[접수된 주문번호\]\n(?P<order_no>\S+)")
_INQUIRY_SECTION = re.compile(r"\[문의\]\n(?P<inquiry>.*?)(?:\n\n|\Z)", re.DOTALL)
_MAX_ROWS_SECTION = re.compile(r"\[결과 행 수 상한\]\n(?P<max_rows>\d+)")
_REGENERATION_MARKER = "[직전 초안이 기각된 사유]"

#: 주문 데이터가 있어야 답할 수 있는 문의를 가리키는 어휘 (대역의 의도 분류 규칙).
_ORDER_WORDS: Final = ("주문", "배송", "도착", "송장", "환불", "취소", "교환", "반품", "수거")
#: 문의가 **특정 자기 주문**을 가리키는 표현 — 주문번호가 없으면 missing_order_ref 로 가야 한다.
_SELF_ORDER_WORDS: Final = ("제 주문", "내 주문", "이 주문", "주문한", "도착하는지", "어디까지")
#: 정책 문서만으로 답할 수 있는 문의를 가리키는 어휘.
_POLICY_WORDS: Final = (
    "정책",
    "규정",
    "며칠",
    "언제까지",
    "얼마",
    "기간",
    "조건",
    "어떻게",
    "가능",
)
#: 미끼 조항이 겨냥하는 값 — 문의가 이걸 물으면 대역이 값을 지어낸다.
_PHONE_WORDS: Final = ("전화번호", "번호", "통화", "연락처", "고객센터")
_EMAIL_WORDS: Final = ("이메일", "메일", "주소가")

_FABRICATED_PHONE: Final = "1588-0000"
_FABRICATED_EMAIL: Final = "help@example.com"

_PHONE_PATTERN_NAMES: Final = frozenset({"mobile_phone", "landline_phone", "service_phone"})


class StubGenerationClient:
    """`GenerationClient` 대역 — **실제 모델이 아니다. 하네스 배관 검증 전용.**

    API 키 없이 골든셋 30건을 파이프라인 끝까지 흘려 "로드 → 파이프라인 → 라벨 대조 →
    집계 → 리포트 산출" 배관이 실제로 도는지 확인하기 위한 것이다. 이 대역으로 낸 측정 2
    수치는 실제 수치가 아니며, 리포트가 그렇게 명시한다(`measurement2_is_real=False`).

    의도적으로 **기각 경로를 재현한다**: 문의가 전화번호·이메일을 묻는데 근거에 그 값이
    없으면(= 미끼 조항) 값을 지어내 `pii_detected` 를 유발하고, 재생성 때는 그 문장을 빼서
    통과시킨다. 기각 재현율 계산이 실제로 도는지를 하네스가 스스로 증명하게 하려는 것이다.
    """

    #: 프롬프트 길이에서 유도하는 가짜 토큰 환산 비율(문자 → 토큰).
    CHARS_PER_TOKEN: Final = 4

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete_json(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str = "response",
        effort: str | None = None,
        max_output_tokens: int = 8000,
    ) -> Any:
        del schema, schema_name, effort, max_output_tokens
        self.calls.append(stage)
        data = self._respond(stage=stage, user=user)
        input_tokens = max(1, len(system) + len(user)) // self.CHARS_PER_TOKEN
        output_tokens = max(1, len(json.dumps(data, ensure_ascii=False))) // self.CHARS_PER_TOKEN
        return _StubCompletion(data=data, input_tokens=input_tokens, output_tokens=output_tokens)

    def _respond(self, *, stage: str, user: str) -> Any:
        if stage == "intent":
            return {"source": self._classify(user)}
        if stage == "sql_generation":
            return {"sql": self._build_sql(user)}
        if stage == "draft":
            return self._build_draft(user)
        raise AssertionError(f"대역이 모르는 단계다: {stage!r}")

    @staticmethod
    def _section(pattern: re.Pattern[str], user: str, group: str) -> str:
        match = pattern.search(user)
        return "" if match is None else match.group(group).strip()

    def _classify(self, user: str) -> str:
        inquiry = self._section(_INQUIRY_SECTION, user, "inquiry")
        order_no = self._section(_ACCEPTED_ORDER_SECTION, user, "order_no")
        has_order_no = bool(order_no) and order_no != "없음"
        wants_order = any(word in inquiry for word in _ORDER_WORDS)
        wants_policy = any(word in inquiry for word in _POLICY_WORDS)
        if has_order_no:
            return "both" if wants_policy else "order"
        # 주문번호 없이 **자기 주문**을 묻는 문의만 order 다 — 그래야 missing_order_ref 가 산다.
        # 일반 규정 문의("해외 배송 되나요")는 배송 어휘가 있어도 policy 로 남아야 한다.
        if wants_order and any(word in inquiry for word in _SELF_ORDER_WORDS):
            return "order"
        return "policy"

    def _build_sql(self, user: str) -> str:
        order_no = self._section(_ORDER_NO_SECTION, user, "order_no")
        max_rows = self._section(_MAX_ROWS_SECTION, user, "max_rows") or "50"
        columns = (
            "order_no, customer_name, customer_phone, customer_email, shipping_address, "
            "product_name, quantity, total_price_krw, status, ordered_at, shipped_at, "
            "delivered_at, courier, tracking_no"
        )
        return f"SELECT {columns} FROM orders WHERE order_no = '{order_no}' LIMIT {int(max_rows)}"

    def _build_draft(self, user: str) -> Any:
        evidence_ids = tuple(match.group("id").strip() for match in _EVIDENCE_ID.finditer(user))
        inquiry = self._section(_INQUIRY_SECTION, user, "inquiry")
        if not evidence_ids:
            return {"claims": [{"text": "안내가 어렵습니다.", "citation_ids": []}]}

        claims: list[dict[str, Any]] = [
            {
                "text": "문의하신 내용은 아래 근거를 확인해 안내드립니다.",
                "citation_ids": list(evidence_ids),
            }
        ]
        if _REGENERATION_MARKER not in user:
            evidence_block = self._section(_EVIDENCE_BLOCK, user, "block")
            fabricated = self._fabricate(inquiry=inquiry, evidence_block=evidence_block)
            if fabricated is not None:
                claims.append({"text": fabricated, "citation_ids": [evidence_ids[0]]})
        return {"claims": claims}

    @staticmethod
    def _fabricate(*, inquiry: str, evidence_block: str) -> str | None:
        """근거에 없는 패턴형 값을 채워 넣어 기각 장면을 재현한다(미끼 조항 대응)."""
        has_phone = any(
            pattern.regex.search(evidence_block)
            for pattern in DEFAULT_PII_PATTERNS
            if pattern.name in _PHONE_PATTERN_NAMES
        )
        has_email = any(
            pattern.regex.search(evidence_block)
            for pattern in DEFAULT_PII_PATTERNS
            if pattern.name == "email"
        )
        if any(word in inquiry for word in _PHONE_WORDS) and not has_phone:
            return f"고객센터 {_FABRICATED_PHONE} 으로 연락해 주십시오."
        if any(word in inquiry for word in _EMAIL_WORDS) and not has_email:
            return f"문의는 {_FABRICATED_EMAIL} 으로 보내주십시오."
        return None


@dataclass(frozen=True)
class _StubCompletion:
    """`llm.JsonCompletion` 과 같은 모양의 대역 산출 (llm 모듈을 import 하지 않기 위해)."""

    data: Any
    input_tokens: int
    output_tokens: int
