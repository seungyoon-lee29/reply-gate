"""평가 하네스 — 결정론 층과 확률 층을 **분리해서** 측정한다
(docs/tracking/decisions/0001-게이트를-2층으로-나눈다.md 의 층 분리가 측정에도 이어진다).

세 측정은 서로의 수치를 오염시키지 않는다. 그것이 이 모듈의 존재 이유다.

* **측정 1 — L1 게이트 단위 정확도 (결정론).** `data/l1_fixtures.jsonl` 의 고정 "초안+근거"
  쌍에 `gate.evaluate_draft` 를 직접 적용한다. **LLM 을 호출하지 않는다** — 이 모듈은
  `reply_gate.llm` 의 실제 클라이언트를 import 하지 않으며(판정 실패를 분류하는 예외 형만
  쓴다), 측정 1 경로는 네트워크를 타지 않는다. 100% 재현되므로 신뢰성 서사의 헤드라인
  수치(구조적 오류 검출률·정상 초안 오탐률)는 여기서 나온다.
* **측정 2 — 파이프라인 판정 일치율 (end-to-end).** 골든셋 30건을 파이프라인에 흘려
  허용 결과 집합과 대조한다. 확률 층이므로 재실행하면 값이 달라진다. **초안 전 인계
  경로가 포함되므로 L1 판정만의 지표가 아니다.**
* **측정 3 — L2 판정 단위 정확도 (확률 층·과금).** `data/judge_fixtures.jsonl` 의 고정
  "claim 집합 + 근거 집합 + 기대 판정" 쌍을 **실제 판정 모델에 직접** 흘린다. 픽스처
  단위는 claim 집합이다 — 판정이 배치 호출이라 그 단위가 곧 측정 단위다. 측정 1 과 같은
  모양의 수치(검출률·오탐률·사유 일치)를 내지만 **재현되지 않고 과금된다**. 목표치는
  **두지 않는다**: 첫 실측 뒤 무목표 관측으로 확정했다(결정 0006).

골든셋 라벨은 단일 정답이 아니라 **허용 결과 집합**이다(`ExpectedOutcomeSet`): 허용 최종
상태 집합 + 허용 인계 사유 집합 + 기각 기대 여부 + 금지 기각 사유 집합. 초안이 확률적이라
같은 문의가 여러 정당한 결말을 가질 수 있기 때문이다.

**목표치는 실측값을 보고 확정했다**(2026-08-05 · `docs/tracking/decisions/0006`). `TARGETS` 가
정의이고 리포트가 달성 여부를 함께 찍는다. 측정하지 않은 지표는 달성 여부를 `None` 으로
남긴다 — 돌지 않은 측정을 "미달"로도 "달성"으로도 적지 않는다.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import reduce
from pathlib import Path
from typing import Any, Final, Protocol, cast

import psycopg
from psycopg.rows import DictRow

from reply_gate import policy_index, retrieval_strategies, sql_guard
from reply_gate.contracts import (
    Claim,
    Draft,
    EscalationReason,
    Evidence,
    EvidenceSource,
    InquiryStatus,
    IntentSource,
    RejectReason,
    Verdict,
    is_policy_evidence_id,
)
from reply_gate.gate import (
    DEFAULT_PII_PATTERNS,
    NUMERIC_SEPARATOR_VARIANTS,
    REASON_ORDER,
    PiiPattern,
    evaluate_draft,
    fold_for_detection,
    fold_numeric_for_detection,
)
from reply_gate.judge import L2_REJECT_REASONS
from reply_gate.llm import LLMCallError, LLMFormatError, accumulate_optional_tokens
from reply_gate.pipeline import (
    L2_JUDGE_STAGE,
    AttemptDurations,
    Judging,
    ProcessedInquiry,
    ReceiptError,
    StageDurations,
    accept_inquiry,
    new_inquiry_id,
)
from reply_gate.query_rewrite import QUERY_REWRITE_STAGE
from reply_gate.regression_guard import (
    DEFAULT_PROMOTED_BASELINE_PATH,
    ConditionFingerprint,
    GuardUnavailable,
    RegressionGuard,
    build_regression_guard,
    fingerprint_from_conditions,
    guard_to_json,
    render_guard_section,
    run_summary_from_payload,
    text_digest,
)
from reply_gate.retrieval_strategies import AbstentionUndefined
from reply_gate.sql_guard import SqlGuardRejection

__all__ = [
    "DEFAULT_GOLDEN_SET_PATH",
    "DEFAULT_JUDGE_FIXTURES_PATH",
    "DEFAULT_L1_FIXTURES_PATH",
    "DEFAULT_REPORT_DIR",
    "DEFAULT_REPORT_STEM",
    "LIVE_L2_REPORT_STEM",
    "LIVE_REPORT_STEM",
    "REASON_ORDER",
    "TARGETS",
    "EvaluationReport",
    "ExpectedOutcomeSet",
    "GateAccuracy",
    "GoldenCase",
    "GoldenOutcome",
    "JudgeAccuracy",
    "JudgeFixture",
    "L1Fixture",
    "L1Outcome",
    "L2Outcome",
    "MetricTarget",
    "PipelineAgreement",
    "PipelineRunning",
    "ReasonBreakdown",
    "ReportStemError",
    "RunConditions",
    "SkippedMeasurement",
    "SpanAggregate",
    "StubGenerationClient",
    "TargetAssessment",
    "assess_targets",
    "attach_regression_guard",
    "build_report",
    "code_condition_fingerprint",
    "display_path",
    "load_golden_set",
    "load_judge_fixtures",
    "load_l1_fixtures",
    "measure_gate_accuracy",
    "measure_judge_accuracy",
    "measure_pipeline_agreement",
    "order_by_clause",
    "render_markdown",
    "report_to_json",
    "resolve_report_stem",
    "sql_guard_probe_outcomes",
    "write_report",
]

_ROOT: Final = Path(__file__).resolve().parents[2]

#: 구간 이름 아홉. **자료형에서 유도한다** — 목록을 여기에 다시 적으면 한쪽만 늘어난다.
SPAN_NAMES: Final[tuple[str, ...]] = tuple(StageDurations().as_mapping())

#: 저장소의 골든셋(30건) — 허용 결과 집합 라벨이 붙은 ground truth.
DEFAULT_GOLDEN_SET_PATH: Final = _ROOT / "data" / "golden_set.jsonl"
#: 저장소의 L1 픽스처 셋 — 고정 초안+근거 쌍(LLM 호출 없음).
DEFAULT_L1_FIXTURES_PATH: Final = _ROOT / "data" / "l1_fixtures.jsonl"
#: 저장소의 L2 판정 픽스처 셋 — 고정 claim 집합+근거 집합+기대 판정(실제 판정 모델 호출).
DEFAULT_JUDGE_FIXTURES_PATH: Final = _ROOT / "data" / "judge_fixtures.jsonl"
#: 리포트 산출 위치. `evaluation-live*` 만 저장소가 추적하고 나머지는 gitignore 된다
#: (라이브 실측은 재생성에 과금이 들어 근거로 커밋한다 — `resolve_report_stem`).
DEFAULT_REPORT_DIR: Final = _ROOT / "reports"

#: 기본 실행(측정 2 미실행)과 대역 실행이 쓰는 리포트 이름.
DEFAULT_REPORT_STEM: Final = "evaluation"

#: 라이브 실측이 쓰는 리포트 이름 접두. 이 접두로 시작하는 리포트만 저장소가 추적한다.
LIVE_REPORT_STEM: Final = "evaluation-live"

#: **L2 켜짐** 라이브 실측이 쓰는 이름 접두. L2 꺼짐 기준선(`evaluation-live-<n>`)과
#: 같은 계열에 섞이면 어느 실측이 L2 를 포함해 잰 것인지 산출물만 보고는 알 수 없다.
#: `LIVE_REPORT_STEM` 으로 시작하므로 기존 라이브 이름 불변식과 gitignore 추적 패턴을
#: 그대로 만족한다.
LIVE_L2_REPORT_STEM: Final = f"{LIVE_REPORT_STEM}-l2"


class ReportStemError(ValueError):
    """리포트 이름이 라이브 실측 보존 계약을 어긴다 — **측정을 시작하기 전에** 거부한다."""


def display_path(path: Path) -> str:
    """리포트에 남길 경로 — 저장소 안이면 상대 경로로 적는다.

    커밋되는 라이브 리포트에 절대 경로가 실리면 개발 머신의 사용자명·디렉터리 구조가
    저장소에 남는다. 측정 데이터가 아닌 환경 정보라 상대화해도 재현성을 해치지 않는다.
    """
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_ROOT))
    except ValueError:
        return str(resolved)


def _next_free_live_stem(out_dir: Path, *, prefix: str = LIVE_REPORT_STEM) -> str:
    """제안용 — 비어 있는 `<접두>-N` 이름을 찾는다. 접두는 실행의 계열을 따른다."""
    n = 1
    while any((out_dir / f"{prefix}-{n}{ext}").exists() for ext in (".md", ".json")):
        n += 1
    return f"{prefix}-{n}"


def resolve_report_stem(
    *,
    requested: str | None,
    live_requested: bool,
    billed: bool,
    l2_enabled: bool,
    out_dir: Path,
) -> str:
    """리포트 이름을 확정한다 — 라이브 실측 산출물을 잃지 않게 막는 유일한 자리다.

    규칙은 두 겹이고 **둘 다 양방향**이다.

    **(1) 라이브 이름 ⇔ 과금 실행**

    * 과금 실행이 아닌 실행(대역·선검사 미통과)은 라이브 이름을 쓸 수 없다. `--live` 를
      줬어도 키·DB 문제로 과금이 일어나지 않으면 기본 이름은 `evaluation` 으로 떨어진다.
    * 과금 실행은 라이브 이름만 쓸 수 있다 — 실측 결과가 gitignore 되는 이름으로 새어
      나가 다음 기본 실행에 덮이는 것을 막는다.
    * **자격 판정 근거는 "측정 2 실측 여부"가 아니라 "과금 실행 여부"다.** 불변식의 목적은
      "과금된 산출물을 잃지 않는다"이고 측정 2 는 그 목적의 낡은 대리 변수였다 — 측정 2 를
      건너뛴 측정 3 단독 실측이 라이브 이름을 거부당해 `evaluation` 스템으로 떨어지면
      `.gitignore` 의 `reports/*` 에 걸려 추적되지 않은 채 다음 실행에 덮인다. 이름 계열은
      `evaluation-live-l2-*` 그대로이므로 계열 불증가·덮어쓰기 금지·추적 패턴이 유지되고,
      미실행 측정은 리포트에 "미실행 + 사유"로 적힌다.

    **(2) L2 켜짐 실측 ⇔ `evaluation-live-l2` 접두**

    * L2 켜짐 실측은 l2 계열에만 쓸 수 있고, L2 꺼짐 실측은 l2 계열에 쓸 수 없다.
      기본 이름만 갈라 두면 `--report-stem evaluation-live-4` 한 줄이 꺼짐 기준선 계열을
      오염시킨다 — **명시 스템에도 같은 검사를 건다.**
    * L2 켜짐 실측의 기본 이름은 코드가 `evaluation-live-l2-<n>` 으로 자동 넘버링한다.
      재실측 프로토콜이 3회 반복이라 기본 이름 충돌로 죽으면 안 되기 때문이다.

    그 밖에:

    * 실측 실행은 **이미 존재하는 라이브 리포트를 덮어쓸 수 없다** — 비결정론이라 덮인
      수치는 재생성되지 않는다. 비어 있는 이름을 제안하되 **제안도 계열을 따른다.**
    * 이름은 경로가 아니라 **파일 이름 조각**이어야 한다 — `./`·`../`·절대경로로 검사를
      우회해 같은 파일명에 쓰는 것을 막는다. 대소문자 변형(macOS 기본 FS 는 대소문자
      무시)도 라이브 이름으로 취급한다.

    실행 전에 호출해야 한다: 여기서 거부되면 과금도 측정도 시작되지 않아야 한다.
    """
    del live_requested  # 판정 근거는 "과금됐는가"이지 "실측을 요청했는가"가 아니다.
    series = LIVE_L2_REPORT_STEM if l2_enabled else LIVE_REPORT_STEM

    if requested is None:
        if not billed:
            stem = DEFAULT_REPORT_STEM
        elif l2_enabled:
            # 3회 반복 실측이 기본 이름 충돌로 죽지 않게 빈 번호를 코드가 찾는다.
            stem = _next_free_live_stem(out_dir, prefix=LIVE_L2_REPORT_STEM)
        else:
            stem = LIVE_REPORT_STEM
    else:
        stem = requested
        if (
            not stem
            or stem != stem.strip()
            or stem.startswith(".")
            or "/" in stem
            or "\\" in stem
            or stem != Path(stem).name
        ):
            raise ReportStemError(
                f"거부: --report-stem 은 경로가 아니라 파일 이름 조각이어야 한다: {stem!r}"
            )

    #: 대소문자 무시 파일시스템에서 `Evaluation-live` 는 라이브 리포트와 같은 파일이다.
    folded = stem.casefold()
    is_live_name = folded.startswith(LIVE_REPORT_STEM)
    is_l2_name = folded.startswith(LIVE_L2_REPORT_STEM)

    if is_live_name and not billed:
        raise ReportStemError(
            f"거부: `{stem}` 은 라이브 실측 리포트 이름인데 이 실행은 과금 실행이 아니다"
            f"(대역 또는 선검사 미통과). 라이브 산출물을 덮어쓰면 문서가 인용하는 수치의 "
            f"근거가 사라진다. 다른 --report-stem 을 쓰거나 기본값(`{DEFAULT_REPORT_STEM}`)을 쓰라."
        )
    if billed and not stem.startswith(LIVE_REPORT_STEM):
        raise ReportStemError(
            f"거부: 과금 실행이 `{stem}` 에 쓰면 산출물이 gitignore 되어 다음 기본 "
            f"실행에 덮인다 — 실제로 한 번 그렇게 잃었다. `{series}` 로 시작하는 이름을 쓰라."
        )
    if billed and l2_enabled and not stem.startswith(LIVE_L2_REPORT_STEM):
        raise ReportStemError(
            f"거부: L2 켜짐 실측이 `{stem}` 에 쓰면 L2 꺼짐 기준선 계열과 섞인다 — 산출물만 "
            f"보고는 어느 쪽이 L2 를 포함해 잰 값인지 알 수 없게 된다. "
            f"`{LIVE_L2_REPORT_STEM}` 로 시작하는 이름을 쓰라 (예: "
            f"--report-stem {_next_free_live_stem(out_dir, prefix=LIVE_L2_REPORT_STEM)})."
        )
    if billed and not l2_enabled and is_l2_name:
        raise ReportStemError(
            f"거부: `{stem}` 은 L2 켜짐 실측 계열 이름인데 이 실행은 L2 가 꺼져 있다. "
            f"꺼짐 기준선은 `{LIVE_REPORT_STEM}` 계열에만 쓴다 — 계열이 뒤섞이면 두 실측을 "
            f"대조하는 근거가 사라진다."
        )
    if billed:
        candidates = (out_dir / f"{stem}{ext}" for ext in (".md", ".json"))
        existing = [path for path in candidates if path.exists()]
        if existing:
            raise ReportStemError(
                f"거부: {existing[0]} 이 이미 있다 — 실측은 비결정론이라 덮이면 그 수치는 "
                f"재생성되지 않는다. 비어 있는 이름을 쓰라 (예: "
                f"--report-stem {_next_free_live_stem(out_dir, prefix=series)})."
            )
    return stem


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

#: 미끼 범주 — 기각 재현율의 분모(결정 0008). 라벨(`expect_reject`)이 아니라 범주다.
BAIT_CATEGORY = "reject_bait"

#: 골든셋이 쓰는 범주 전체 — 로드 시 검증한다. 범주 오타는 예외 없이 재현율 분모를
#: 비워 지표를 조용히 소멸시키기 때문이다.
GOLDEN_CATEGORIES = frozenset({"normal", BAIT_CATEGORY, "no_evidence", "escalation"})


@dataclass(frozen=True)
class ExpectedOutcomeSet:
    """골든셋 라벨 = **허용 결과 집합**. 단일 정답이 아니다.

    * `statuses` — 허용 최종 상태 집합(비어 있을 수 없다).
    * `escalation_reasons` — `escalated` 로 끝났을 때 허용되는 인계 사유 집합.
    * `expect_reject` — 시도 중 **최소 1건**이 기각이어야 하는가. 케이스 단위 요구일 뿐
      기각 재현율의 분모가 아니다 — 분모는 `category == "reject_bait"` 다(결정 0008).
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
    #: 파이프라인이 실제 수집·채택해 결과에 실은 근거 ID. 검색 라벨은 이 값의 입력이 아니다.
    adopted_evidence_ids: tuple[str, ...]
    latency_ms: int
    input_tokens: int
    output_tokens: int
    embedding_tokens: int
    #: 판정(L2) 토큰은 생성·임베딩과 **분리**해서 싣는다 — provider 와 단가가 다르다.
    #: L2 미실행이면 0 이다(임베딩 관례와 같다).
    judge_input_tokens: int
    judge_output_tokens: int
    matched: bool
    mismatches: tuple[str, ...]
    #: 접수 거부처럼 파이프라인에 진입조차 못한 경우의 사유.
    error: str | None
    #: 검색 단계(질의 재작성) 토큰 — 같은 규칙의 세 번째 계열이다. 재작성을 쓰지 않은
    #: 문의는 0 이고, 실패한 호출의 토큰도 실비용이므로 그대로 들어온다.
    retrieval_input_tokens: int = 0
    retrieval_output_tokens: int = 0
    #: 캐시 계열 토큰 — **계열마다 (write, read) 한 쌍**이다. 두 계열이 같은 클라이언트를
    #: 지나므로 한 쌍으로 묶으면 어느 계열의 캐시였는지 되짚을 수 없다. 입력 칸에 접지
    #: 않는다(단가가 다르고, OpenAI 는 적중분이 입력 토큰에 이미 포함돼 있다).
    #: 보고되지 않은 칸은 0 이 아니라 `None`(미측정)이다.
    generation_cache_creation_tokens: int | None = None
    generation_cache_read_tokens: int | None = None
    retrieval_cache_creation_tokens: int | None = None
    retrieval_cache_read_tokens: int | None = None
    #: 검색 단계가 폴백한 사유. 인계 사유가 아니다 — 폴백한 문의도 답변으로 끝날 수 있다.
    retrieval_fallback_reason: str | None = None
    #: 기권 게이트의 통계량이 **미정의였던 사유**. `None` 은 세 경우를 함께 덮는다 —
    #: 게이트 꺼짐 · 정책 검색 미실행 · 통계량 정의됨. **처분만으로는 갈리지 않는다**:
    #: "2건 미만이라 열어 뒀다"와 "정의됐고 통과했다"는 채택 결과가 같다.
    abstention_undefined_reason: str | None = None
    #: 의도 해석이 고른 근거 소스(`policy`/`order`/`both`). **정책 검색이 실제로 돌았는지**를
    #: 이 값이 들고 있다 — 지금까지는 검색 토큰 0 + `sql:` 단독 채택으로 역추론해야 했다
    #: (docs/tracking/findings.md 20번 ①). 의도 해석이 무너진 문의는 `None`(미상)이다.
    intent: str | None = None
    #: 이 케이스의 구간 아홉 시간(ms). 돌지 않은 구간은 0 이 아니라 미측정이다.
    #: 접수 거부처럼 파이프라인에 진입조차 못한 케이스는 아홉 칸 전부 미측정이다.
    stage_durations: StageDurations = field(default_factory=StageDurations)
    #: 시도별 구간(초안 생성·게이트 판정·L2 판정). 재생성이 돌면 2건이 쌓이고, 합계는
    #: `stage_durations` 가 들고 있다 — **합계와 시도별 값을 함께** 알 수 있어야 한다.
    attempt_durations: tuple[AttemptDurations, ...] = ()

    @property
    def rejected_at_least_once(self) -> bool:
        """시도 중 **최소 1건**이 기각인가 — **층 무관**이다.

        종합 verdict 를 보므로 L1 기각도 L2 기각도 같은 자격으로 들어온다. 기각 재현율의
        정의가 층에 묶이면 L2 도입이 지표 정의를 바꾸게 되고, 도입 전후 비교가 깨진다.

        **이 값만으로 "게이트가 놓쳤다"고 읽으면 안 된다** — 게이트가 아예 돌지 않은
        경우에도 False 다. 두 상태의 구분은 `gate_never_ran` 이 들고 있다.
        """
        return any(verdict is Verdict.REJECT for verdict in self.attempt_verdicts)

    @property
    def gate_never_ran(self) -> bool:
        """L2 판정 호출이 무너져 **판정 자체가 없었던** 케이스인가.

        판정 호출이 실패한 시도의 종합 verdict 는 `pass` 다(docs/contracts.md "층별 판정 키"
        ③ — 그 시도의 진실은 `escalation_reason` 이 들고 있다). 그래서 `attempt_verdicts`
        만 보면 **"게이트가 돌았고 놓쳤다"와 "게이트가 안 돌았다"가 같은 값**이 되고,
        미실행이 관측 실패로 접혀 지표에 0 으로 채워진다
        (`scripts/AGENTS.md` 불변식 5 — 미실행을 0 으로 채우지 않는다).

        contracts.md 가 지정한 신호를 그대로 쓴다: 인계 사유 `llm_call_failed` +
        실패 단계 `l2_judge`.
        """
        return (
            self.escalation_reason is EscalationReason.LLM_CALL_FAILED
            and self.failed_stage == L2_JUDGE_STAGE
        )


@dataclass(frozen=True)
class SpanAggregate:
    """구간 하나의 세트 집계 — **미측정은 분모에서 뺀다.**

    0 을 섞어 평균 내면 그 구간이 실제보다 빨라 보인다. 그래서 분모는 전체 케이스 수가
    아니라 **그 구간이 실제로 돈 케이스 수**(`measured_cases`)이고, 돌지 않은 건수는
    `unmeasured_cases` 로 따로 적어 사람이 분모를 눈으로 확인할 수 있게 둔다.

    한 케이스도 재지 않은 구간의 값은 0 이 아니라 `None`(미측정)이다.
    """

    span: str
    measured_cases: int
    unmeasured_cases: int
    total_ms: float | None
    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None


def _span_aggregate(span: str, values: Sequence[float | None]) -> SpanAggregate:
    """구간 하나를 집계한다. **미측정(`None`)은 분모에도 분자에도 들어가지 않는다.**"""
    measured = [value for value in values if value is not None]
    if not measured:
        return SpanAggregate(
            span=span,
            measured_cases=0,
            unmeasured_cases=len(values),
            total_ms=None,
            mean_ms=None,
            p50_ms=None,
            p95_ms=None,
        )
    total = sum(measured)
    return SpanAggregate(
        span=span,
        measured_cases=len(measured),
        unmeasured_cases=len(values) - len(measured),
        total_ms=total,
        mean_ms=total / len(measured),
        p50_ms=_percentile_ms(measured, 50),
        p95_ms=_percentile_ms(measured, 95),
    )


def _percentile_ms(values: Sequence[float], percent: int) -> float | None:
    """nearest-rank 백분위 — 정수 지연(`_percentile`)과 **같은 규칙**의 실수판이다."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), _ceil_div(percent * len(ordered), 100)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class PipelineAgreement:
    """측정 2 의 집계.

    **초안 전 인계 경로가 포함된다** — 근거 0건·주문번호 없음·주문 없음으로 끝난 건도
    분모에 들어간다. 따라서 이 일치율은 L1 판정만의 지표가 아니다.
    """

    total: int
    matched: int
    #: 미끼 범주 중 **판정이 실제로 돈** 건수 = 재현율의 분모.
    bait_total: int
    bait_reject_reproduced: int
    #: 미끼 범주인데 L2 판정 호출이 무너져 **측정되지 않은** 건수. 분모 밖이고, 0 으로
    #: 채우지 않는다(`scripts/AGENTS.md` 불변식 5). 리포트에 사유와 함께 실린다.
    bait_unmeasured: int
    forbidden_watch_total: int
    forbidden_violations: int
    latency_p50_ms: int | None
    latency_p95_ms: int | None
    input_tokens_total: int
    output_tokens_total: int
    embedding_tokens_total: int
    #: 판정 계열은 생성·임베딩과 분리해서 센다. L2 켜짐 실측에서 이 계열을 빠뜨리면
    #: 건당 비용 지표가 판정 비용을 통째로 누락한다.
    judge_input_tokens_total: int
    judge_output_tokens_total: int
    status_counts: Mapping[str, int]
    escalation_counts: Mapping[str, int]
    outcomes: tuple[GoldenOutcome, ...]
    #: 검색 계열도 같은 자격으로 분리해서 센다.
    retrieval_input_tokens_total: int = 0
    retrieval_output_tokens_total: int = 0
    #: 캐시 계열 합계 — **계열마다 한 쌍**이다. 한 쌍으로 묶으면 두 계열 중 어느 쪽의
    #: 캐시였는지 되짚을 수 없고, 이 절감의 증상은 **두 계열 모두**의 달러가 상한이라는
    #: 것이라 한 쌍으로는 목적을 못 채운다. **한 건도 보고되지 않았으면 0 이 아니라
    #: `None`(미측정)** 이다 — "캐시가 0 토큰 적중했다"와 "캐시를 잰 적이 없다"는 다르다.
    generation_cache_creation_total: int | None = None
    generation_cache_read_total: int | None = None
    retrieval_cache_creation_total: int | None = None
    retrieval_cache_read_total: int | None = None
    #: 검색 단계가 폴백한 문의 수. **조용한 폴백을 금지하는 집계**다
    #: (docs/business-rules.md "검색 단계 실패") — 0 이 정상이고, 커지면 재작성 층이
    #: 사실상 꺼진 실행을 정상 실측으로 읽게 된다.
    retrieval_fallback_total: int = 0
    #: 기권 게이트 통계량이 미정의였던 문의 수 — **사유별로** 센다. 두 사유는 처분이
    #: 반대라(2건 미만은 열어 두고, 1위 코사인 0 이하는 기권) 한 칸으로 접으면 어느
    #: 분기를 탔는지 산출물에서 사라진다. 사유가 없던 문의는 세지 않는다(0 이 아니라
    #: 그 사유 키가 없다 — 미정의가 한 번도 없었다는 뜻이다).
    abstention_undefined_counts: Mapping[str, int] = field(default_factory=dict)
    #: 구간 아홉의 세트 집계. **미측정 케이스는 각 구간의 분모에서 빠진다.**
    #: 순서는 `evidence.SPAN_NAMES`(파이프라인 실행 순서)와 같다.
    stage_durations: tuple[SpanAggregate, ...] = ()

    @property
    def match_rate(self) -> float | None:
        return None if self.total == 0 else self.matched / self.total

    @property
    def bait_reject_recall(self) -> float | None:
        """미끼(`reject_bait`) 문의의 기각 재현율 — 목표 없는 관측값(결정 0006·0008).

        분자의 정의는 **시도 중 최소 1건 기각, 층 무관**이다(L2 도입으로 바뀌지 않는다).
        분모는 라벨(`expect_reject`)이 아니라 범주다 — 라벨을 판정 규칙에 정렬하면서
        (결정 0008) 기각 요구는 해제했지만 관측은 유지하기 때문이다.

        **판정이 돌지 못한 건(`bait_unmeasured`)은 분모에서 뺀다.** 남겨 두면 인프라
        실패가 "게이트가 미끼를 놓쳤다"와 같은 값이 되어, 미실행이 관측 실패로 둔갑한다.
        전부 미측정이면 재현율은 0.0 이 아니라 **None**(미실행)이다.
        """
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
    def judge_tokens_per_inquiry(self) -> float | None:
        if self.total == 0:
            return None
        return (self.judge_input_tokens_total + self.judge_output_tokens_total) / self.total

    @property
    def retrieval_tokens_per_inquiry(self) -> float | None:
        if self.total == 0:
            return None
        return (self.retrieval_input_tokens_total + self.retrieval_output_tokens_total) / self.total

    @property
    def generation_cache_measured(self) -> bool:
        """생성 계열의 캐시를 **잰** 실행인가. 안 잰 실행의 표기는 0 이 아니라 "미측정" 이다."""
        return (
            self.generation_cache_creation_total is not None
            or self.generation_cache_read_total is not None
        )

    @property
    def retrieval_cache_measured(self) -> bool:
        """검색 계열의 캐시를 **잰** 실행인가 — 생성 계열과 따로 묻는다."""
        return (
            self.retrieval_cache_creation_total is not None
            or self.retrieval_cache_read_total is not None
        )

    @property
    def total_tokens_per_inquiry(self) -> float | None:
        """네 계열 합산 — 한 계열이라도 빠지면 건당 비용이 거짓이 된다."""
        if self.total == 0:
            return None
        return _grand_total(self) / self.total


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
        category = str(row["category"])
        if category not in GOLDEN_CATEGORIES:
            raise ValueError(f"{case_id}: 알 수 없는 범주다: {category}")
        try:
            expected = _expected_from_json(row["expected"])
        except ValueError as exc:
            raise ValueError(f"{case_id}: {exc}") from exc
        cases.append(
            GoldenCase(
                id=case_id,
                category=category,
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
        # 판정이 아예 돌지 못한 경우까지 "기각되지 않았다"로 적으면 거짓이다 — 게이트가
        # 통과시킨 것이 아니라 게이트에 닿지 못했다. 불일치인 것은 같지만 사유가 다르고,
        # 사람이 이 문자열을 읽고 게이트 품질을 판단한다.
        if (
            processed.escalation_reason is EscalationReason.LLM_CALL_FAILED
            and processed.failed_stage == L2_JUDGE_STAGE
        ):
            mismatches.append("기각을 기대했으나 L2 판정 호출이 실패해 판정이 없었다(미측정)")
        else:
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
            adopted_evidence_ids=(),
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
            embedding_tokens=0,
            judge_input_tokens=0,
            judge_output_tokens=0,
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
        adopted_evidence_ids=tuple(item.id for item in processed.evidence),
        latency_ms=processed.latency_ms,
        input_tokens=processed.input_tokens,
        output_tokens=processed.output_tokens,
        embedding_tokens=processed.embedding_tokens,
        judge_input_tokens=processed.judge_input_tokens,
        judge_output_tokens=processed.judge_output_tokens,
        matched=matched,
        mismatches=mismatches,
        error=None,
        retrieval_input_tokens=processed.retrieval_input_tokens,
        retrieval_output_tokens=processed.retrieval_output_tokens,
        generation_cache_creation_tokens=processed.generation_cache_creation_tokens,
        generation_cache_read_tokens=processed.generation_cache_read_tokens,
        retrieval_cache_creation_tokens=processed.retrieval_cache_creation_tokens,
        retrieval_cache_read_tokens=processed.retrieval_cache_read_tokens,
        retrieval_fallback_reason=processed.retrieval_fallback_reason,
        abstention_undefined_reason=(
            None
            if processed.abstention_undefined_reason is None
            else processed.abstention_undefined_reason.value
        ),
        intent=None if processed.intent is None else processed.intent.value,
        stage_durations=processed.stage_durations,
        attempt_durations=tuple(attempt.durations for attempt in processed.attempts),
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

    # 미끼 범주를 "판정이 돈 것"과 "돌지 못한 것"으로 가른다 — 뒤엣것을 분모에 남기면
    # 인프라 실패가 게이트 실패로 둔갑한다(`GoldenOutcome.gate_never_ran`).
    bait_all = [
        (case, outcome)
        for case, outcome in zip(cases, outcomes, strict=True)
        if case.category == BAIT_CATEGORY
    ]
    bait = [(case, outcome) for case, outcome in bait_all if not outcome.gate_never_ran]
    bait_unmeasured = len(bait_all) - len(bait)
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
        bait_unmeasured=bait_unmeasured,
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
        judge_input_tokens_total=sum(outcome.judge_input_tokens for outcome in outcomes),
        judge_output_tokens_total=sum(outcome.judge_output_tokens for outcome in outcomes),
        status_counts=status_counts,
        escalation_counts=escalation_counts,
        outcomes=tuple(outcomes),
        retrieval_input_tokens_total=sum(outcome.retrieval_input_tokens for outcome in outcomes),
        retrieval_output_tokens_total=sum(outcome.retrieval_output_tokens for outcome in outcomes),
        # 캐시 계열은 `sum` 이 아니라 누적기로 접는다 — 미측정을 0 으로 만들지 않는다
        # (판정 계열의 같은 자리와 같은 규칙).
        generation_cache_creation_total=_fold_optional(
            outcome.generation_cache_creation_tokens for outcome in outcomes
        ),
        generation_cache_read_total=_fold_optional(
            outcome.generation_cache_read_tokens for outcome in outcomes
        ),
        retrieval_cache_creation_total=_fold_optional(
            outcome.retrieval_cache_creation_tokens for outcome in outcomes
        ),
        retrieval_cache_read_total=_fold_optional(
            outcome.retrieval_cache_read_tokens for outcome in outcomes
        ),
        retrieval_fallback_total=sum(
            1 for outcome in outcomes if outcome.retrieval_fallback_reason is not None
        ),
        abstention_undefined_counts=_abstention_undefined_counts(outcomes),
        stage_durations=_aggregate_stage_durations(outcomes),
    )


def _abstention_undefined_counts(outcomes: Sequence[GoldenOutcome]) -> dict[str, int]:
    """기권 미정의 사유별 건수. **정본 순서**(`AbstentionUndefined` 선언 순서)로 담는다.

    사유가 한 건도 없으면 빈 맵이다 — 있지도 않은 사유를 0 으로 채우면 "이 실행에서 그
    분기를 쟀다"로 읽힌다. 사전 순회 순서에 기대지 않는 것은 산출물이 실행마다 같은
    모양이어야 하기 때문이다.
    """
    counts: dict[str, int] = {}
    for reason in AbstentionUndefined:
        hits = sum(1 for outcome in outcomes if outcome.abstention_undefined_reason == reason.value)
        if hits:
            counts[reason.value] = hits
    return counts


def _fold_optional(values: Iterable[int | None]) -> int | None:
    """캐시 계열 합계 — **미측정(`None`)을 0 으로 접지 않는다.**

    한 번이라도 측정값이 있으면 합계는 측정값이고, 끝까지 없으면 합계도 미측정이다.
    `sum()` 을 쓰면 미측정이 0 이 되어 재지도 않은 축이 "적중 0" 으로 신고된다.
    """
    return reduce(accumulate_optional_tokens, values, cast(int | None, None))


def _aggregate_stage_durations(outcomes: Sequence[GoldenOutcome]) -> tuple[SpanAggregate, ...]:
    """구간 아홉을 세트 단위로 집계한다 — 순서는 구간 이름의 정본 순서 그대로다."""
    per_case = [outcome.stage_durations.as_mapping() for outcome in outcomes]
    return tuple(
        _span_aggregate(span, [values[span] for values in per_case]) for span in SPAN_NAMES
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


# ══ 측정 3 — L2 판정 단위 정확도 (확률 층 · 과금) ═══════════════════════════


@dataclass(frozen=True)
class JudgeFixture:
    """고정 "claim 집합 + 근거 집합 + 기대 판정" 1건.

    단위가 claim 1개가 아니라 **claim 집합**인 이유는 판정이 시도당 1회 배치 호출이기
    때문이다 — 모순 감지가 claim·근거 교차 시야를 요구해서 쪼갤 수 없다. 측정 단위는
    호출 단위와 같아야 한다.
    """

    id: str
    category: str
    note: str
    evidences: tuple[Evidence, ...]
    claims: tuple[Claim, ...]
    expected_verdict: Verdict
    expected_reasons: tuple[RejectReason, ...]
    #: `claims` 와 같은 순서의 claim 별 기대 판정.
    expected_claim_verdicts: tuple[Verdict, ...]
    #: 기대 모순 근거쌍 — 파일에 적힌 순서 그대로.
    expected_contradiction_pairs: tuple[tuple[str, str], ...]

    @property
    def draft(self) -> Draft:
        """판정 입력이 되는 초안. L1 을 통과한 형태여야 한다(L2 는 L1 통과분만 본다)."""
        return Draft(claims=self.claims)

    @property
    def expected_contradictions(self) -> frozenset[frozenset[str]]:
        """대조용 — 모순은 **순서 없는 쌍**이다(a,b 와 b,a 는 같은 모순이다)."""
        return frozenset(frozenset(pair) for pair in self.expected_contradiction_pairs)

    @property
    def is_violation(self) -> bool:
        """기각되어야 하는 픽스처인가 — 검출률의 분모."""
        return self.expected_verdict is Verdict.REJECT


@dataclass(frozen=True)
class L2Outcome:
    """판정 픽스처 1건의 측정 결과.

    `error` 가 있으면 **판정하지 못한 것**이다 — 통과로도 기각으로도 세지 않고 검출률·
    오탐률의 분모에서 빠진다. 실패를 0 으로 접으면 리포트가 거짓말을 한다.
    """

    fixture_id: str
    category: str
    expected_verdict: Verdict
    actual_verdict: Verdict | None
    expected_reasons: tuple[RejectReason, ...]
    actual_reasons: tuple[RejectReason, ...]
    claim_total: int
    claim_verdict_matched: int
    contradiction_expected: int
    contradiction_matched: int
    contradiction_extra: int
    input_tokens: int
    output_tokens: int
    error: str | None
    #: 캐시 계열 토큰 — **판정 입력 토큰과 뭉뚱그리지 않는다.** 캐싱 켜짐 조건에서
    #: `input_tokens` 는 캐시 적중분을 뺀 비캐시 입력이고, write 와 read 는 단가가 다르다.
    #: 캐시를 재지 않은 실행에서는 0 이 아니라 `None`(해당 없음/미측정)이다.
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    #: 판정 호출에 흐른 벽시계(ms) — **형식 재시도와 전송 재시도를 포함한 합산**이다.
    #: 판정하지 못한 픽스처도 시간은 썼으므로 실패 경로에서도 값이 실린다. 값을 채우지
    #: 않고 조립한 결과에서는 0 이 아니라 `None`(미측정)이고, 집계 분모에서 빠진다.
    elapsed_ms: float | None = None

    @property
    def judged(self) -> bool:
        return self.error is None

    @property
    def verdict_matched(self) -> bool:
        return self.expected_verdict is self.actual_verdict

    @property
    def reasons_matched(self) -> bool:
        """사유 목록이 **순서까지** 같은가. 순서는 계약(`L2_REJECT_REASONS`)이 정한다."""
        return self.judged and self.expected_reasons == self.actual_reasons


@dataclass(frozen=True)
class JudgeAccuracy:
    """측정 3 의 집계. **확률 층이고 과금된다** — 재실행하면 값이 달라진다."""

    total: int
    error_total: int
    violation_total: int
    violation_detected: int
    clean_total: int
    clean_false_positive: int
    reason_set_exact: int
    claim_total: int
    claim_verdict_matched: int
    contradiction_expected_total: int
    contradiction_matched_total: int
    contradiction_extra_total: int
    input_tokens_total: int
    output_tokens_total: int
    breakdown: tuple[ReasonBreakdown, ...]
    outcomes: tuple[L2Outcome, ...]
    #: 캐시 write/read 합계. **한 건도 보고되지 않았으면 0 이 아니라 `None`(미측정)** 이다 —
    #: "캐시가 0 토큰 적중했다"와 "캐시를 잰 적이 없다"는 다른 상태다.
    cache_creation_tokens_total: int | None = None
    cache_read_tokens_total: int | None = None
    #: 판정 호출 지연의 세트 집계. 지금까지 측정 3 은 지연을 **아예 기록하지 않았다**.
    #: 미측정 픽스처는 분모에서 빠지고, 한 건도 재지 않았으면 `measured_cases` 가 0 이다.
    judge_latency: SpanAggregate | None = None

    @property
    def judged_total(self) -> int:
        """실제로 판정이 나온 픽스처 수 — 비율의 분모는 전부 이 안에서 나온다."""
        return self.total - self.error_total

    @property
    def detection_rate(self) -> float | None:
        """L2 검출률 — 기각되어야 할 픽스처 중 실제로 기각된 비율."""
        if self.violation_total == 0:
            return None
        return self.violation_detected / self.violation_total

    @property
    def false_positive_rate(self) -> float | None:
        """L2 오탐률 — 통과해야 할 픽스처 중 기각된 비율."""
        if self.clean_total == 0:
            return None
        return self.clean_false_positive / self.clean_total

    @property
    def reason_set_exact_rate(self) -> float | None:
        if self.judged_total == 0:
            return None
        return self.reason_set_exact / self.judged_total

    @property
    def claim_verdict_match_rate(self) -> float | None:
        """claim 단위 판정 일치율 — 픽스처 단위 판정보다 촘촘한 보조 지표."""
        if self.claim_total == 0:
            return None
        return self.claim_verdict_matched / self.claim_total

    @property
    def contradiction_recall(self) -> float | None:
        if self.contradiction_expected_total == 0:
            return None
        return self.contradiction_matched_total / self.contradiction_expected_total

    @property
    def tokens_per_fixture(self) -> float | None:
        if self.total == 0:
            return None
        return (self.input_tokens_total + self.output_tokens_total) / self.total

    @property
    def cache_measured(self) -> bool:
        """캐시 계열을 **잰** 실행인가. 안 잰 실행의 표기는 0 이 아니라 "미측정" 이다."""
        return (
            self.cache_creation_tokens_total is not None or self.cache_read_tokens_total is not None
        )


def _claims_from_json(rows: Sequence[Any], *, fixture_id: str) -> tuple[Claim, ...]:
    claims = tuple(
        Claim(
            text=str(row["text"]),
            citation_ids=tuple(str(value) for value in row["citation_ids"]),
        )
        for row in rows
    )
    if not claims:
        raise ValueError(f"{fixture_id}: claim 이 하나도 없다")
    return claims


def _expected_judgment_from_json(
    raw: Mapping[str, Any], *, fixture_id: str, claims: Sequence[Claim], evidence_ids: set[str]
) -> tuple[Verdict, tuple[RejectReason, ...], tuple[Verdict, ...], tuple[tuple[str, str], ...]]:
    """기대 판정 라벨을 읽고 **자기 정합성**을 검사한다 — 라벨이 어긋나면 지표가 거짓말을 한다.

    검사는 판정 모듈의 fail-closed 파싱(`judge._parse_judge_result`)과 같은 규칙이다:
    사유는 L2 2종뿐, verdict ⇔ 사유, `unsupported_claim` ⇔ reject claim 존재,
    `contradictory_evidence` 면 모순쌍 기록 필수, 모순 ID 는 수집 근거 안에서만 유효.
    """
    verdict = Verdict(raw["verdict"])
    reasons: list[RejectReason] = []
    for value in raw["reject_reasons"]:
        reason = RejectReason(value)
        if reason not in L2_REJECT_REASONS:
            raise ValueError(f"{fixture_id}: L2 사유 2종 밖의 값이다: {value}")
        reasons.append(reason)
    ordered = tuple(reason for reason in L2_REJECT_REASONS if reason in set(reasons))
    if tuple(reasons) != ordered:
        raise ValueError(f"{fixture_id}: reject_reasons 는 계약 순서여야 한다 {ordered!r}")
    if (verdict is Verdict.REJECT) != bool(ordered):
        raise ValueError(f"{fixture_id}: reject 는 사유가 1개 이상, pass 는 사유가 없어야 한다")

    judgments = list(raw["claim_judgments"])
    if [str(item["claim_text"]) for item in judgments] != [claim.text for claim in claims]:
        raise ValueError(f"{fixture_id}: claim_judgments 가 claim 전부와 순서까지 대응해야 한다")
    claim_verdicts = tuple(Verdict(item["verdict"]) for item in judgments)
    any_rejected = any(item is Verdict.REJECT for item in claim_verdicts)
    if (RejectReason.UNSUPPORTED_CLAIM in ordered) != any_rejected:
        raise ValueError(
            f"{fixture_id}: unsupported_claim 은 reject 인 claim 이 있을 때만, 있으면 반드시 붙는다"
        )

    pairs: list[tuple[str, str]] = []
    for item in raw.get("contradictions", ()):
        id_a, id_b = str(item["evidence_id_a"]), str(item["evidence_id_b"])
        if id_a == id_b:
            raise ValueError(f"{fixture_id}: 같은 근거끼리의 모순 쌍이다: {id_a}")
        for evidence_id in (id_a, id_b):
            if evidence_id not in evidence_ids:
                raise ValueError(f"{fixture_id}: 모순 쌍의 ID 가 근거 목록에 없다: {evidence_id}")
        pairs.append((id_a, id_b))
    if RejectReason.CONTRADICTORY_EVIDENCE in ordered and not pairs:
        raise ValueError(f"{fixture_id}: contradictory_evidence 인데 모순 근거쌍 기록이 없다")
    return verdict, ordered, claim_verdicts, tuple(pairs)


def load_judge_fixtures(path: Path = DEFAULT_JUDGE_FIXTURES_PATH) -> tuple[JudgeFixture, ...]:
    """판정 픽스처 셋을 읽는다. 라벨이 ground truth 이므로 어긋난 줄은 오류로 세운다."""
    fixtures: list[JudgeFixture] = []
    seen: set[str] = set()
    for row in _read_jsonl(path):
        fixture_id = str(row["id"])
        if fixture_id in seen:
            raise ValueError(f"판정 픽스처 ID 가 중복된다: {fixture_id}")
        seen.add(fixture_id)
        evidences = tuple(_evidence_from_json(item) for item in row["evidences"])
        if not evidences:
            raise ValueError(f"{fixture_id}: 근거가 하나도 없다")
        evidence_ids = {item.id for item in evidences}
        claims = _claims_from_json(row["claims"], fixture_id=fixture_id)
        # 판정 계약은 claim 을 **텍스트로 식별한다**(`ClaimJudgment.claim_text`). 한 픽스처
        # 안에 같은 텍스트가 두 번 있으면 판정 대조가 두 건을 한 판정으로 채점하는데
        # `claim_total` 은 2 로 센다 — 라벨 쪽에서 막는다.
        if len({claim.text for claim in claims}) != len(claims):
            raise ValueError(f"{fixture_id}: 한 픽스처 안의 claim 텍스트는 유일해야 한다")
        for claim in claims:
            # L1 을 통과한 초안만 L2 입력이 된다 — 인용 없음·미해석 인용은 L1 이 잡는 결함이다.
            if not claim.citation_ids:
                raise ValueError(f"{fixture_id}: 인용이 없는 claim 은 L1 이 먼저 기각한다")
            unknown = sorted(set(claim.citation_ids) - evidence_ids)
            if unknown:
                raise ValueError(f"{fixture_id}: 근거 목록에 없는 인용 ID 다: {unknown!r}")
        verdict, reasons, claim_verdicts, pairs = _expected_judgment_from_json(
            row["expected"], fixture_id=fixture_id, claims=claims, evidence_ids=evidence_ids
        )
        fixtures.append(
            JudgeFixture(
                id=fixture_id,
                category=str(row["category"]),
                note=str(row.get("note", "")),
                evidences=evidences,
                claims=claims,
                expected_verdict=verdict,
                expected_reasons=reasons,
                expected_claim_verdicts=claim_verdicts,
                expected_contradiction_pairs=pairs,
            )
        )
    if not fixtures:
        raise ValueError(f"판정 픽스처가 하나도 없다: {path}")
    return tuple(fixtures)


def evaluate_judge_fixture(*, fixture: JudgeFixture, judge: Judging) -> L2Outcome:
    """판정 픽스처 1건을 **실제 판정기**에 흘려 라벨과 대조한다.

    판정 실패(형식 불일치 소진·전송 오류)는 삼키지 않고 `error` 로 **기록**한다:
    11건 중 1건의 실패로 과금된 나머지 측정이 통째로 사라지면 안 되고, 그렇다고 실패를
    "통과"로 접으면 오탐률이 거짓이 된다. 자격 증명 부재(`MissingCredentialsError`)는
    업무 판정이 아니라 설정 오류이므로 여기서 잡지 않고 그대로 올라간다.
    """
    expected_pairs = fixture.expected_contradictions
    try:
        outcome = judge.judge(draft=fixture.draft, evidence=fixture.evidences)
    except (LLMFormatError, LLMCallError) as exc:
        return L2Outcome(
            fixture_id=fixture.id,
            category=fixture.category,
            expected_verdict=fixture.expected_verdict,
            actual_verdict=None,
            expected_reasons=fixture.expected_reasons,
            actual_reasons=(),
            claim_total=len(fixture.claims),
            claim_verdict_matched=0,
            contradiction_expected=len(expected_pairs),
            contradiction_matched=0,
            contradiction_extra=0,
            input_tokens=exc.input_tokens,
            output_tokens=exc.output_tokens,
            error=f"{type(exc).__name__}: {exc}",
            # 실행됐으나 실패한 호출의 캐시 write 도 실비용이다 — 0 으로 접지 않는다.
            cache_creation_input_tokens=exc.cache_creation_input_tokens,
            cache_read_input_tokens=exc.cache_read_input_tokens,
            # 판정하지 못한 픽스처도 시간은 썼다 — 토큰과 같은 규칙이다.
            elapsed_ms=exc.elapsed_ms,
        )

    result = outcome.result
    # 판정 계약이 claim 을 텍스트로 식별하므로 대조도 텍스트로 접는다 — 한 픽스처 안
    # claim 텍스트가 유일하다는 것은 로더(`load_judge_fixtures`)가 보장한다.
    verdict_by_text = {item.claim_text: item.verdict for item in result.claim_judgments}
    matched_claims = sum(
        1
        for claim, expected in zip(fixture.claims, fixture.expected_claim_verdicts, strict=True)
        if verdict_by_text.get(claim.text) is expected
    )
    actual_pairs = frozenset(
        frozenset((item.evidence_id_a, item.evidence_id_b)) for item in result.contradictions
    )
    return L2Outcome(
        fixture_id=fixture.id,
        category=fixture.category,
        expected_verdict=fixture.expected_verdict,
        actual_verdict=result.verdict,
        expected_reasons=fixture.expected_reasons,
        actual_reasons=result.reject_reasons,
        claim_total=len(fixture.claims),
        claim_verdict_matched=matched_claims,
        contradiction_expected=len(expected_pairs),
        contradiction_matched=len(expected_pairs & actual_pairs),
        contradiction_extra=len(actual_pairs - expected_pairs),
        input_tokens=outcome.input_tokens,
        output_tokens=outcome.output_tokens,
        error=None,
        cache_creation_input_tokens=outcome.cache_creation_input_tokens,
        cache_read_input_tokens=outcome.cache_read_input_tokens,
        elapsed_ms=outcome.elapsed_ms,
    )


def measure_judge_accuracy(
    *,
    fixtures: Sequence[JudgeFixture],
    judge: Judging,
    on_outcome: Callable[[L2Outcome], None] | None = None,
) -> JudgeAccuracy:
    """측정 3 — 판정 픽스처 전체를 **끝까지** 흘린다. 실제 판정 모델을 부르므로 과금된다."""
    outcomes: list[L2Outcome] = []
    for fixture in fixtures:
        outcome = evaluate_judge_fixture(fixture=fixture, judge=judge)
        outcomes.append(outcome)
        if on_outcome is not None:
            on_outcome(outcome)

    judged = [outcome for outcome in outcomes if outcome.judged]
    violations = [outcome for outcome in judged if outcome.expected_verdict is Verdict.REJECT]
    cleans = [outcome for outcome in judged if outcome.expected_verdict is Verdict.PASS]
    breakdown = tuple(
        ReasonBreakdown(
            reason=reason,
            expected_count=sum(1 for o in judged if reason in o.expected_reasons),
            detected_count=sum(
                1 for o in judged if reason in o.expected_reasons and reason in o.actual_reasons
            ),
            spurious_count=sum(
                1 for o in judged if reason not in o.expected_reasons and reason in o.actual_reasons
            ),
        )
        for reason in L2_REJECT_REASONS
    )
    return JudgeAccuracy(
        total=len(outcomes),
        error_total=sum(1 for outcome in outcomes if not outcome.judged),
        violation_total=len(violations),
        violation_detected=sum(1 for o in violations if o.actual_verdict is Verdict.REJECT),
        clean_total=len(cleans),
        clean_false_positive=sum(1 for o in cleans if o.actual_verdict is Verdict.REJECT),
        reason_set_exact=sum(1 for o in judged if o.reasons_matched),
        claim_total=sum(outcome.claim_total for outcome in judged),
        claim_verdict_matched=sum(outcome.claim_verdict_matched for outcome in judged),
        contradiction_expected_total=sum(outcome.contradiction_expected for outcome in judged),
        contradiction_matched_total=sum(outcome.contradiction_matched for outcome in judged),
        contradiction_extra_total=sum(outcome.contradiction_extra for outcome in judged),
        # 실패한 호출이 쓴 토큰도 실비용이므로 그대로 집계한다(판정 모듈의 규칙과 같다).
        input_tokens_total=sum(outcome.input_tokens for outcome in outcomes),
        output_tokens_total=sum(outcome.output_tokens for outcome in outcomes),
        breakdown=breakdown,
        outcomes=tuple(outcomes),
        # 캐시 계열은 `sum` 이 아니라 누적기로 접는다 — 미측정을 0 으로 만들지 않는다.
        cache_creation_tokens_total=reduce(
            accumulate_optional_tokens,
            (outcome.cache_creation_input_tokens for outcome in outcomes),
            cast(int | None, None),
        ),
        cache_read_tokens_total=reduce(
            accumulate_optional_tokens,
            (outcome.cache_read_input_tokens for outcome in outcomes),
            cast(int | None, None),
        ),
        # 판정 실패한 픽스처의 경과도 분모에 든다 — 그 호출도 판정 층이 쓴 시간이다.
        judge_latency=_span_aggregate(L2_JUDGE_STAGE, [outcome.elapsed_ms for outcome in outcomes]),
    )


# ══ 조건 지문 — 코드가 결정하는 항목 ════════════════════════════════════════
#
# **여기 있는 값들은 실행 인자도 설정도 시각도 보지 않는다.** 같은 코드면 언제 어디서
# 돌려도 같은 값이어야 한다 — 그렇지 않으면 다음 실행이 자기 자신과 "대조 불가"가 되어
# 지문이 스스로를 무력화한다. 그래서 집합은 반드시 정렬해서 담는다(파이썬 문자열 해시는
# 프로세스마다 다른 시드를 받아 `frozenset` 순회 순서가 실행마다 달라진다).
#
# **범위를 어떻게 갈랐나**: 소스 파일을 통째로 해시하지 않는다 — 주석 한 줄이 계보를
# 끊는다. 대신 규칙을 이루는 **선언**(패턴 표·허용 목록·enum)과, 선언에 없고 코드에만
# 있는 규칙의 **동작**(고정 탐침에 대한 산출)을 읽는다. 그래서 문면을 고쳐도 값이
# 그대로이고, 규칙을 고치면 값이 움직인다.


#: **판정 호출이 보내는 사고 과정(thinking) 설정.** 지문의 `judge_effort` 와 **다른 축이다**
#: — effort 는 "얼마나"이고 이쪽은 "켜는가"다. 지금 판정 래퍼는 thinking 설정을 **보내지
#: 않는다**: 미전송은 '끔'이 아니라 계열 기본을 따른다는 뜻이고, 그 기본은 계열마다 다르다
#: (adaptive thinking 이 켜짐인 계열도, 기능이 아예 없는 계열도 있다). 그래서 "기본값"
#: 한 단어로는 무엇으로 돌았는지 복원되지 않는다.
#:
#: 값이 코드 사실과 갈리지 않게 **구조 검사가 판정 요청 조립에 `thinking` 키가 없음을
#: 못박는다**(`tests/test_condition_fingerprint.py`). 보내기 시작하면 그 검사가 먼저 죽어
#: 이 값을 고치지 않고 지나갈 수 없다.
JUDGE_THINKING: Final = "미전송(계열 기본)"

#: 접기 규칙의 **동작**을 드러내는 고정 탐침. 눈으로 구분되지 않는 문자는 이스케이프로
#: 적는다 — 리터럴로 쓰면 무엇이 들어 있는지 읽을 수 없고, 편집 중에 조용히 ASCII 로
#: 바뀌어도 아무도 모른다(게이트 모듈이 같은 이유로 같은 규율을 쓴다).
#:
#: * U+FF10~U+FF19 — 전각 숫자(NFKC 가 반각으로 접는다)
#: * U+200B — 폭 없는 서식 문자(`Cf` 범주, 공통 접기가 지운다)
#: * U+2013 · U+2212 · U+30FB — 구분자 변종(번호 계열만 하이픈으로 접는다)
#: * U+0301 — 숫자 **자리 사이**의 결합 표식(`Mn` 범주, 번호 계열만 무시한다)
#: * 뒤쪽 평문 — 정규화(전화·숫자·이메일)가 실제로 무엇을 하는지 드러낸다
#:
#: **패턴 다섯이 전부 한 번씩은 매치돼야 한다.** 매치가 없는 패턴은 정규화 결과가 빈
#: 문자열이라, 그 패턴에 붙은 정규화 함수를 갈아 끼워도 지문이 움직이지 않는다.
#: **전화 정규화의 한국 국가번호 두 갈래도 각각 한 번씩 탄다**(`+82…` 와 `0082…`) —
#: 정규화는 입력의 **선두**를 보므로, 한 표기만 넣으면 다른 갈래가 죽은 채로 남는다.
_PII_RULE_PROBE: Final = (
    "\uff10\uff11\uff10\u200b-9999\u03018888 "
    "010\u20139999\u22128888 010\u30fb9999\u30fb8888 "
    "+82-10-1234-5678 0082 2 123 4567 1588-1234 900101-1234567  A_B@Example.COM "
)

#: 조회 가드 탐침이 쓰는 주문번호. **선검사를 통과한 형식**이어야 한다(아니면 호출 측
#: 오류로 죽는다). 값 자체는 판정에 쓰이지 않고 범위 한정 문면에만 들어간다.
_SQL_GUARD_PROBE_ORDER_NO: Final = "ORD-20260201-0001"

#: 조회 가드의 **유래 승인 규칙**을 가르는 고정 탐침. 이 규칙은 허용 목록이 아니라 코드에
#: 있어서(스코프를 타고 내려가는 증명) 선언만 읽으면 지문이 안 움직인다.
#:
#: 승인 쪽 넷(직접 컬럼 · 임시 테이블 경유 · 파생 테이블 경유 · 개인정보 모양이 아닌 빈칸
#: 채우기)과 불승인 쪽 넷(개인정보 모양 고정값 · `nullif` 의 비교 상대 자리 · 이어 붙이기 ·
#: 계산 컬럼), 그리고 거부 쪽 셋(목록 밖 캐스트 대상 타입 · 목록 밖 함수 · 주문 범위 미한정)을
#: 함께 든다. **양쪽을 다 담아야** 규칙이 어느 방향으로 움직여도 값이 따라간다.
_SQL_GUARD_PROBES: Final[tuple[str, ...]] = (
    f"SELECT customer_phone FROM orders WHERE order_no = '{_SQL_GUARD_PROBE_ORDER_NO}'",
    "WITH scoped AS ("
    f"SELECT order_no, customer_phone FROM orders WHERE order_no = '{_SQL_GUARD_PROBE_ORDER_NO}'"
    ") SELECT customer_phone FROM scoped",
    "SELECT d.customer_phone FROM ("
    f"SELECT order_no, customer_phone FROM orders WHERE order_no = '{_SQL_GUARD_PROBE_ORDER_NO}'"
    ") d",
    "SELECT coalesce(customer_phone, '미정') AS phone FROM orders "
    f"WHERE order_no = '{_SQL_GUARD_PROBE_ORDER_NO}'",
    "SELECT coalesce(customer_phone, '010-0000-0000') AS phone FROM orders "
    f"WHERE order_no = '{_SQL_GUARD_PROBE_ORDER_NO}'",
    "SELECT nullif('010-1234-5678', customer_phone) AS phone FROM orders "
    f"WHERE order_no = '{_SQL_GUARD_PROBE_ORDER_NO}'",
    "SELECT concat(customer_name, customer_phone) AS joined FROM orders "
    f"WHERE order_no = '{_SQL_GUARD_PROBE_ORDER_NO}'",
    "SELECT upper(customer_email) AS shouted FROM orders "
    f"WHERE order_no = '{_SQL_GUARD_PROBE_ORDER_NO}'",
    "SELECT cast(quantity AS text) AS qty FROM orders "
    f"WHERE order_no = '{_SQL_GUARD_PROBE_ORDER_NO}'",
    "SELECT cast('orders' AS regclass) AS catalog FROM orders "
    f"WHERE order_no = '{_SQL_GUARD_PROBE_ORDER_NO}'",
    f"SELECT pg_sleep(1) AS napped FROM orders WHERE order_no = '{_SQL_GUARD_PROBE_ORDER_NO}'",
    "SELECT customer_phone FROM orders",
)

#: 조회 가드 탐침의 결과 행 수 상한. 판정에 영향을 주지 않는 고정값이다.
_SQL_GUARD_PROBE_MAX_ROWS: Final = 50

#: 기권 게이트의 **미정의 경계**를 가르는 고정 점수열. 처분만 담으면 "측정 2건 미만" ·
#: "1위 코사인 0 이하" 두 경계가 움직여도 지문이 그대로다.
_ABSTENTION_PROBES: Final[tuple[tuple[float, ...], ...]] = (
    (),
    (0.9,),
    (0.0, 0.0),
    (-0.2, 0.4),
    (0.0, 0.5),
    (0.9, 0.3),
    (0.9, 0.9),
)

#: 검색 정렬의 tie-break 를 드러내는 고정 동점 후보. **정본 순서와 반대로 넣는다** —
#: 입력 순서대로 나오는 구현과 정렬하는 구현이 같은 답을 내면 탐침이 아무것도 못 가른다.
_RETRIEVAL_TIE_PROBE: Final[tuple[str, ...]] = ("b", "a", "c")

#: 정책 검색 SQL 의 정렬 절을 뽑는 패턴. **파일 전체가 아니라 이 절 하나만** 읽는다 —
#: 함수 안 주석이 바뀌었다고 계보가 끊기면 안 된다.
_ORDER_BY_PATTERN: Final = re.compile(r"ORDER BY\s+(.*)", re.DOTALL)

#: `ORDER BY` 뒤에서 절을 끝내는 표기. 여기까지만 정렬 키로 읽는다.
_ORDER_BY_TERMINATORS: Final[tuple[str, ...]] = ("LIMIT", "OFFSET", "FETCH")


def order_by_clause(source: str) -> str:
    """SQL 문면에서 `ORDER BY` 절만 뽑아 공백을 접는다. 없으면 `"정렬 없음"`.

    **없을 때 옛 값을 남기지 않는다** — 정렬 절이 사라진 것은 조용한 무변경이 아니라
    행동 계약이 바뀐 것이고, 지문은 그것을 값으로 드러내야 한다.
    """
    match = _ORDER_BY_PATTERN.search(source)
    if match is None:
        return "정렬 없음"
    clause = match.group(1)
    for terminator in _ORDER_BY_TERMINATORS:
        clause = clause.partition(terminator)[0]
    return " ".join(clause.split())


def _policy_search_sql() -> str:
    """정책 검색 함수가 **실제로 실행하는** SQL 문면.

    소스를 문자열로 훑지 않고 AST 로 본다 — 주석은 애초에 없고 docstring 은 제외할 수 있다.
    설명 문장에 `ORDER BY` 가 스치기만 해도 지문이 흔들리면, 문서를 고친 것이 계보를 끊는다.
    """
    module = ast.parse(textwrap.dedent(inspect.getsource(policy_index.search_policy_chunks)))
    function = module.body[0]
    if not isinstance(function, ast.FunctionDef):  # pragma: no cover - 방어
        return ""
    body = function.body[1:] if ast.get_docstring(function) is not None else function.body
    return "\n".join(
        node.value
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


def sql_guard_probe_outcomes() -> tuple[str, ...]:
    """조회 가드 탐침의 판정 결과. `ok|<PII 허용 출력명>` 또는 `reject|<규칙 코드>`.

    **거부 사유 문면은 담지 않는다** — 안내 문구를 다듬었다고 계보가 끊기면 안 된다.
    담는 것은 규칙 코드와 허용 출처 목록, 즉 **판정 자체**다.
    """
    outcomes: list[str] = []
    for probe in _SQL_GUARD_PROBES:
        try:
            validated = sql_guard.validate_sql(
                probe,
                order_no=_SQL_GUARD_PROBE_ORDER_NO,
                max_rows=_SQL_GUARD_PROBE_MAX_ROWS,
            )
        except SqlGuardRejection as rejection:
            outcomes.append(f"reject|{rejection.rule.value}")
        else:
            outcomes.append("ok|" + ",".join(validated.pii_safe_output_columns))
    return tuple(outcomes)


def _normalized_probe_matches(pattern: PiiPattern) -> str:
    """탐침에서 이 패턴이 뽑은 값들을 **매치별로** 정규화한 결과.

    **정규화는 실행 경로와 같은 자리에서 부른다** — L1 은 `pattern.regex.findall(fold(text))`
    가 뽑은 매치 하나하나에 정규화를 건다(`gate._normalized_matches`). 지문이 탐침 문자열
    **전체**에 정규화를 걸면 그 자리와 입력이 달라져, **입력의 선두를 보는 분기가 한 번도
    실행되지 않는다**: 전화 정규화의 한국 국가번호 두 갈래(`0082`·`82`)가 정확히 그랬다.
    두 갈래를 통째로 지워도 지문이 움직이지 않았고, 그 삭제는 근거의 `+82-10-1234-5678` 과
    초안의 `010-1234-5678` 을 다른 값으로 만들어 **정상 에코를 오기각시키는** 실제 동작
    변경이다.

    `findall` 을 쓰는 것도 같은 이유다 — 정규식에 캡처 그룹이 생기면 뽑히는 값이 달라지고
    실행 경로도 함께 달라진다. 지문이 그 변화를 따라가려면 같은 함수를 써야 한다.
    """
    return ",".join(
        str(pattern.normalize(match))
        for match in pattern.regex.findall(pattern.fold(_PII_RULE_PROBE))
    )


def _draft_rule_version() -> str:
    """초안 판정 규칙의 판 — **접기 규칙과 패턴 집합**의 내용 지문.

    읽는 것은 넷이다: 기각 사유의 고정 순서 · 번호 계열이 구분자로 받는 표기 변종 집합 ·
    두 접기(공통·번호)가 고정 탐침에 대해 내는 결과 · 패턴마다의 (이름, 정규식, 플래그,
    접기 결과, **매치별** 정규화 결과). 접기와 정규화는 **이름이 아니라 동작**으로 담는다 —
    함수 이름만 담으면 본문이 바뀌어도 지문이 안 움직인다.
    """
    parts = [
        "reasons=" + ",".join(reason.value for reason in REASON_ORDER),
        "separators=" + ",".join(f"U+{ord(ch):04X}" for ch in sorted(NUMERIC_SEPARATOR_VARIANTS)),
        "fold_common=" + fold_for_detection(_PII_RULE_PROBE),
        "fold_numeric=" + fold_numeric_for_detection(_PII_RULE_PROBE),
    ]
    parts.extend(
        f"pattern={pattern.name}|regex={pattern.regex.pattern}|flags={pattern.regex.flags}"
        f"|fold={pattern.fold(_PII_RULE_PROBE)}"
        f"|normalize={_normalized_probe_matches(pattern)}"
        for pattern in DEFAULT_PII_PATTERNS
    )
    return text_digest("\n".join(parts), prefix="draftrules-")


def _sql_guard_version() -> str:
    """조회 가드의 판 — **허용 함수 · 허용 캐스트 타입 · PII 유래 승인 규칙**의 내용 지문.

    앞의 둘은 생성 안내가 그대로 싣는 목록 문면(`describe_allowed_functions()`)에서 읽고,
    셋째는 코드에만 있어서 고정 탐침의 판정 결과로 읽는다.
    """
    parts = [sql_guard.describe_allowed_functions(), *sql_guard_probe_outcomes()]
    return text_digest("\n".join(parts), prefix="sqlguard-")


def _abstention_undefined_policy() -> str:
    """기권 게이트의 **미정의 처리 방식** 지문 — 사유별 처분과 그 경계.

    두 사유는 처분이 반대라 하나로 접지 않는다(2건 미만은 열어 두고, 1위 코사인 0 이하는
    기권). **경계까지 담는 이유**: 처분만 담으면 `<= 0` 을 `< 0` 으로 바꾸는 변경이 지문에
    아무 흔적을 남기지 않는다.
    """
    parts = [
        f"{reason.value}={'기권' if reason.abstains else '비기권'}"
        for reason in retrieval_strategies.AbstentionUndefined
    ]
    for scores in _ABSTENTION_PROBES:
        reason = retrieval_strategies.undefined_statistic_reason(scores)
        rendered = "정의됨" if reason is None else reason.value
        parts.append(f"probe({','.join(f'{score:g}' for score in scores)})={rendered}")
    return text_digest("\n".join(parts), prefix="abstain-")


def _retrieval_order_rule() -> str:
    """검색 순서 규칙 — **tie-break 유무**를 두 자리에서 함께 읽는다.

    DB 가 상위 `top_k` 를 자를 때의 정렬(`policy_index.search_policy_chunks`)과, 원문·재작성
    합집합을 접는 파이썬 병합(`retrieval_strategies.merge_rewritten_rankings`)이다. **두
    자리가 갈리면 재작성 켜짐/꺼짐이 다른 순서를 낸다** — 그래서 한 칸에 함께 싣는다.
    병합 쪽은 동점 후보를 정본 순서와 반대로 넣어 결과 순서를 그대로 적는다.
    """
    db_clause = order_by_clause(_policy_search_sql())
    merged = retrieval_strategies.merge_rewritten_rankings(
        original=[
            retrieval_strategies.VectorHit(rank=rank, evidence_id=evidence_id, similarity=0.5)
            for rank, evidence_id in enumerate(_RETRIEVAL_TIE_PROBE, start=1)
        ],
        rewritten=(),
    )
    return f"db=[{db_clause}] · merge=[{'>'.join(hit.evidence_id for hit in merged)}]"


def code_condition_fingerprint() -> dict[str, str]:
    """**코드가 결정하는** 조건 지문 항목. 실행 인자·설정·시각을 보지 않는다.

    실행 인자에 묶인 항목(골든셋·판정 픽스처·L1 채점표의 판 등)은 진입점이 명시 지문으로
    싣고, 명시가 이긴다 — 여기 값과 이름이 겹치면 진입점 쪽이 남는다.
    """
    return {
        "judge_thinking": JUDGE_THINKING,
        "draft_rule_version": _draft_rule_version(),
        "sql_guard_version": _sql_guard_version(),
        "retrieval_order": _retrieval_order_rule(),
        "abstention_undefined_policy": _abstention_undefined_policy(),
    }


# ══ 실행 조건 · 리포트 ══════════════════════════════════════════════════════


@dataclass(frozen=True)
class RunConditions:
    """무엇으로 돌렸는지. **대역으로 낸 수치를 실제 수치처럼 보고하지 않기 위한 기록이다.**"""

    started_at: str
    generation: str
    embedding: str
    #: 판정(L2) 모델 — 실제 모델인지 대역인지, 꺼져 있는지가 여기서 갈린다.
    #: L2 꺼짐 기준선과 켜짐 실측은 산출물 모양이 같으므로, 구분은 이 항목이 들고 있다.
    judge: str
    similarity_threshold: float
    top_k: int
    #: 검색 구성 요약 (scripts/AGENTS.md 불변식 15). **필수 필드**다 — 같은 지표를 다른 검색
    #: 설정으로 잰 산출물이 이름 계열은 하나이므로(리포트 이름을 늘리지 않기로 한 판단),
    #: 어떤 구성으로 잰 값인지는 실행 조건이 들고 있어야 한다. `embedding` 문자열이 모델을
    #: 담지만 차원·전략 조합은 거기 없다.
    embedding_dimensions: int
    #: 실행 경로에 켜진 검색 전략 조합. 예: `vector` · `vector+rewrite`.
    retrieval_strategy: str
    l1_fixture_count: int
    golden_case_count: int
    #: 판정 픽스처 수. **로드하지 못했으면 `None`** — 미실행을 0 으로 채우지 않는다.
    judge_fixture_count: int | None
    l1_fixtures_path: str
    golden_set_path: str
    judge_fixtures_path: str
    api_key_present: bool
    #: 판정 키(ANTHROPIC) 존재 여부 — 값이 아니라 **존재 여부만** 남긴다.
    judge_api_key_present: bool
    #: L2 판정 스위치. 지연·토큰·일치율을 기존 실측과 비교할 때 반드시 함께 읽어야 한다.
    l2_enabled: bool
    #: 측정 2 를 **실제 모델로 도는 실측 실행**으로 시작했는가. 대역이면 False 이고 리포트가
    #: 그렇게 적는다. 측정 2 가 도중에 중단돼 미실행으로 강등돼도 이 값은 True 로 남는다 —
    #: 과금은 이미 일어났고 리포트 이름(라이브 계열)도 이 값으로 확정된 뒤이기 때문이다.
    #: 그때 측정 2 절은 "미실행 + 중단 사유"라 수치가 실측으로 읽힐 자리가 없다.
    measurement2_is_real: bool
    #: 측정 3 수치가 **실제 판정 모델**로 낸 값인가. 대역(`testing.StubJudge` 등)으로 낸
    #: 값이면 False 이고, 그때 리포트의 `billed`·`deterministic` 과 확률층·과금 문구가
    #: 전부 뒤집힌다 — 실행 조건의 판정 모델 문자열 한 줄만으로 구분하면 리포트가 스스로
    #: "과금된 실측"이라고 거짓 신고할 수 있다.
    measurement3_is_real: bool
    #: **과금 실행인가.** 리포트 이름의 자격과 회귀 가드 실행 여부가 이 값을 본다.
    #: 측정 2 실측 여부와 갈릴 수 있다 — 측정 3 만 사는 실행도 과금이다.
    billed: bool = False
    #: 측정 범위. 풀셋인지 부분(예: 측정 3 단독)인지가 대조 가능성을 가른다.
    measurement_scope: str = "full"
    #: **조건 지문** — 두 실측이 애초에 대조 가능한지를 가르는 열린 맵이다
    #: (`regression_guard.FINGERPRINT_FIELDS` 가 하한이고 상한은 아니다). 새 검색·판정 축을
    #: 붙이는 작업은 **여기 값을 추가**하면 되고 이 모듈을 다시 열지 않아도 된다.
    #: 비어 있는 항목은 0 이 아니라 **미상**으로 읽힌다.
    condition_fingerprint: Mapping[str, str] = field(default_factory=dict)
    #: 이번 실행이 "의도적으로 바꾼 축"으로 **선언한** 지문 항목. 선언된 차이는 대조를
    #: 진행하며 차이 목록을 병기하고, 선언되지 않은 불일치만 "대조 불가"다.
    declared_experiment_fields: tuple[str, ...] = ()
    #: **도중에 중단된 측정의 이름들.** 중단 표시가 사람이 읽는 설명 문자열에만 붙으면
    #: 기계가 읽는 조건은 완주한 실행과 **같아진다** — 그러면 절반만 돈 실행의 케이스
    #: 결과가 완주한 실측과 한 세트로 묶인다. 선검사에 걸려 **처음부터 안 돈** 측정은
    #: 여기 들지 않는다(그건 `measurement_scope` 가 든다).
    aborted_measurements: tuple[str, ...] = ()

    @property
    def run_completion(self) -> str:
        """완주했는가 — 지문의 `run_completion` 항목이 되는 문면."""
        if not self.aborted_measurements:
            return "중단 없음"
        return "중단: " + "·".join(self.aborted_measurements)

    def fingerprint(self) -> ConditionFingerprint:
        """대조 가능성을 결정하는 지문. 파생값은 명시 지문에 덮인다(명시가 이긴다).

        **코드가 결정하는 항목은 여기서 합친다** — 진입점이 실행 인자로 만드는 명시 지문과
        같은 맵에 들어가고, 이름이 겹치면 명시가 이긴다. 옛 산출물을 읽을 때는 이 경로를
        타지 않으므로(`regression_guard.fingerprint_from_conditions` 가 JSON 을 그대로
        읽는다) **지금 코드의 값이 옛 산출물에 소급해 찍히지 않는다.** 그게 핵심이다:
        옛 산출물의 새 항목은 "미상"이지 이번 코드의 값이 아니다.
        """
        return fingerprint_from_conditions(
            {
                "similarity_threshold": self.similarity_threshold,
                "top_k": self.top_k,
                "embedding_dimensions": self.embedding_dimensions,
                "retrieval_strategy": self.retrieval_strategy,
                "generation": self.generation,
                "judge": self.judge,
                "measurement_scope": self.measurement_scope,
                "condition_fingerprint": {
                    **code_condition_fingerprint(),
                    **dict(self.condition_fingerprint),
                    # **중단 사실은 마지막에 쓴다 — 명시 지문이 덮을 수 없다.** 다른 항목은
                    # "명시가 이긴다"가 맞지만 이건 실행이 실제로 어떻게 끝났는가라서,
                    # 진입점이 같은 키를 내는 순간 중단이 조용히 완주로 덮인다.
                    "run_completion": self.run_completion,
                },
                "declared_experiment_fields": list(self.declared_experiment_fields),
            }
        )


@dataclass(frozen=True)
class MetricTarget:
    """지표 목표치 1개 — 이름·경계·비교 방향. 판정은 `met` 이 한다."""

    key: str
    label: str
    #: 경계값(비율). `at_most` 면 이하가 달성, 아니면 이상이 달성.
    bound: float
    at_most: bool = False
    #: 확률 층 지표인가. 대역 실행의 값으로는 판정하지 않는다 — 대역 수치를 실제 수치처럼
    #: "달성"으로 찍으면 리포트가 거짓말을 한다(scripts/AGENTS.md 불변식 6과 같은 규칙).
    probabilistic: bool = False
    #: **하한 경보선인가.** 달성 목표가 아니라 "그 아래면 무언가 크게 부서졌다"는 경보선이다
    #: (결정 0006 재확정). 미달을 "미달"이 아니라 "경보"로 적는다 — 사이클 성패 판정은
    #: 케이스 단위(회귀 가드)가 맡고 합산은 병기일 뿐이라는 것이 리포트 문면에 드러나야 한다.
    alert_only: bool = False

    def met(self, value: float | None) -> bool | None:
        """달성 여부. **측정하지 않았으면 `None`** — 미측정을 미달로 적지 않는다."""
        if value is None:
            return None
        return value <= self.bound if self.at_most else value >= self.bound

    def describe(self) -> str:
        """경계만 적는다 — 표에서 이름은 옆 칸이 이미 들고 있다."""
        return f"{'≤' if self.at_most else '≥'} {self.bound * 100:g}%"


#: 지표 목표치 — 2026-08-05 확정
#: (`docs/tracking/decisions/0006-지표-목표치를-실측-뒤에-확정한다.md`).
#:
#: L1 두 지표는 **결정론 층**이라 100%/0% 를 목표로 박을 수 있다: 픽스처가 고정이고
#: 게이트가 LLM 을 부르지 않으므로 달성은 재현되며, 미달은 곧 회귀다.
#: 일치율 75% 는 사이클 1 실측(71.1%) 위에 "닿지 않는 목표"로 둔 값이었다. **그 전제는
#: 끝났다** — 결정 0008 의 라벨 재정렬 뒤로는 **L2 를 꺼도 86.7%** 라 이 경계가 층 기여를
#: 판별하지 못한다. 2026-08-14 재확정으로 **값은 유지하되 성격이 하한 경보선으로 바뀌었고,
#: 회귀 판정은 케이스 단위(비악화 가드 + 케이스별 귀인)가 맡는다** — 결정 0006 의 "측정 2
#: 목표치 재확정" 절이 소유자다. 그 가드는 이제 상설이다: `regression_guard` 가 이전 실측을
#: 기준선으로 들고(리포트가 리포트를 읽는다) 케이스 단위로 판정하며, 여기 남은 75% 는
#: `alert_only` 로 성격이 박힌 **하한 경보선**이다.
#: 측정 3(L2 판정 단위 정확도)은 **여기 없다**: 목표치를 두지 않기로 했으므로 경계가 없다.
TARGETS: Final[tuple[MetricTarget, ...]] = (
    MetricTarget(key="detection_rate", label="측정 1 구조적 오류 검출률", bound=1.0),
    MetricTarget(
        key="false_positive_rate",
        label="측정 1 정상 초안 오탐률",
        bound=0.0,
        at_most=True,
    ),
    MetricTarget(
        key="match_rate",
        label="측정 2 허용 결과 집합 대비 일치율",
        bound=0.75,
        probabilistic=True,
        alert_only=True,
    ),
)


@dataclass(frozen=True)
class TargetAssessment:
    """목표치 1개의 판정 결과 — 마크다운 표·JSON·콘솔 요약이 같은 원본을 쓴다."""

    target: MetricTarget
    value: float | None
    #: 판정하지 않았으면(미측정·대역) `None`.
    met: bool | None
    #: 사람용 판정 문구: 달성 | 미달 | 미측정 | 대역 — 판정 없음.
    verdict: str


@dataclass(frozen=True)
class FailureAttributionCase:
    """기각·인계 1건을 검색 정답 라벨과 조인한 판정 근거."""

    case_id: str
    classification: str
    relevant_evidence_ids: tuple[str, ...]
    adopted_evidence_ids: tuple[str, ...]
    missing_relevant_evidence_ids: tuple[str, ...]
    rejected_at_least_once: bool
    escalated: bool
    #: 채택 근거 전체가 0건. SQL 조회 근거도 함께 센다.
    ended_with_zero_evidence: bool
    #: 정책 조항 근거가 0건 — "검색 0건"의 정확한 의미. SQL 근거는 검색 결과가 아니다.
    ended_with_zero_policy_evidence: bool
    l2_caught_with_evidence: bool
    #: 빈 정답 케이스에서만 bool. 다른 두 분류에서는 해당 없음.
    normal_behavior: bool | None
    normal_behavior_path: str | None
    anomaly_reason: str | None
    #: **정책 검색이 실제로 돌았는가.** 의도 분류가 `order` 로 라우팅해 검색이 아예 안 돈
    #: 케이스와, 검색이 돌았는데 걸러진 케이스는 다른 실패다 — 전자는 컷과 무관하다.
    #: 의도 해석이 무너졌으면 `None`(미상)이고, **미상을 False 로 접지 않는다.**
    policy_retrieval_ran: bool | None = None
    #: 위 판정의 근거 한 줄. 산출물만 보고 역추론하지 않아도 되게 남긴다.
    policy_retrieval_note: str | None = None
    #: 최종 상태와 라벨 일치 여부. `generation_issue` 가 종결 실패처럼 읽히는 것을 막는다 —
    #: 1차 기각 뒤 재생성으로 회복해 `answered`/일치로 끝난 케이스가 실제로 그렇다.
    final_status: str | None = None
    matched: bool = False


@dataclass(frozen=True)
class FailureAttribution:
    """검색 실패와 생성 문제 분해. 카운트는 산출된 때만 존재한다."""

    labels_path: str
    generation_issue_count: int
    #: 정답 조항을 하나도 채택하지 못한 기각·인계.
    retrieval_failure_count: int
    #: 정답 조항 중 일부만 채택한 기각·인계. 검색 실패 합계에 포함된다.
    partial_retrieval_failure_count: int
    #: 정답 조항이 빠진 채 답변이 확정된 케이스 — 게이트를 통과한 근거 부족.
    answered_without_relevant_evidence_count: int
    expected_no_answer_count: int
    expected_no_answer_anomaly_count: int
    cases: tuple[FailureAttributionCase, ...]

    @property
    def retrieval_failure_total(self) -> int:
        """전부 누락 + 일부 누락. 헤드라인이 검색 실패를 과소 보고하지 않게 한다."""
        return self.retrieval_failure_count + self.partial_retrieval_failure_count


@dataclass(frozen=True)
class UnavailableBreakdown:
    """분해를 산출하지 못한 사유. 0건 집계와 구분한다."""

    reason: str


@dataclass(frozen=True)
class EvaluationReport:
    """리포트 1건 — 사람이 읽는 마크다운과 기계가 읽는 JSON 의 공통 원본."""

    conditions: RunConditions
    gate_accuracy: GateAccuracy
    pipeline: PipelineAgreement | SkippedMeasurement
    judge_accuracy: JudgeAccuracy | SkippedMeasurement
    failure_attribution: FailureAttribution | UnavailableBreakdown
    #: 회귀 가드 두 줄 보고. 돌리지 않았으면 **미산출 + 사유**이지 통과가 아니다.
    regression_guard: RegressionGuard | GuardUnavailable = field(
        default_factory=lambda: GuardUnavailable(
            reason="호출자가 회귀 가드를 넘기지 않았다 — 대조하지 않았다는 뜻이다"
        )
    )

    def measured(self) -> dict[str, float | None]:
        """목표치 대조에 쓰는 실측값. 측정 2 미실행이면 일치율은 `None`.

        **측정 3 지표는 여기 들어오지 않는다** — 목표치를 두지 않으므로 대조할 경계가 없다.
        수치는 측정 3 절이 관측값으로 그대로 싣는다.
        """
        return {
            "detection_rate": self.gate_accuracy.detection_rate,
            "false_positive_rate": self.gate_accuracy.false_positive_rate,
            "match_rate": (
                None if isinstance(self.pipeline, SkippedMeasurement) else self.pipeline.match_rate
            ),
        }


def assess_targets(report: EvaluationReport) -> tuple[TargetAssessment, ...]:
    """목표치 전부를 판정한다. **미측정·대역은 미달도 달성도 아니다.**"""
    measured = report.measured()
    assessments: list[TargetAssessment] = []
    for target in TARGETS:
        value = measured[target.key]
        if value is None:
            assessments.append(
                TargetAssessment(target=target, value=None, met=None, verdict="미측정")
            )
            continue
        if target.probabilistic and not report.conditions.measurement2_is_real:
            # 대역 값은 존재하지만 판정에 쓰지 않는다 — 값 자체는 배관 검증용으로만 남긴다.
            assessments.append(
                TargetAssessment(target=target, value=value, met=None, verdict="대역 — 판정 없음")
            )
            continue
        met = target.met(value)
        # 하한 경보선의 미달은 사이클 판정이 아니라 **경보**다 — 판정은 케이스 단위가
        # 맡는다(결정 0006 재확정). 문면을 "미달"로 두면 합산이 다시 판정처럼 읽힌다.
        verdict = "달성" if met else "경보" if target.alert_only else "미달"
        assessments.append(TargetAssessment(target=target, value=value, met=met, verdict=verdict))
    return tuple(assessments)


def build_report(
    *,
    conditions: RunConditions,
    gate_accuracy: GateAccuracy,
    pipeline: PipelineAgreement | SkippedMeasurement,
    judge_accuracy: JudgeAccuracy | SkippedMeasurement,
    retrieval_labels_path: Path | None = None,
    regression_guard: RegressionGuard | GuardUnavailable | None = None,
) -> EvaluationReport:
    """리포트를 조립한다. **측정 3 도 명시해야 한다** — 기본값으로 비워 두면 미실행이
    조용히 "사유 없는 미실행"으로 찍힌다.

    회귀 가드는 리포트 JSON 을 입력으로 쓰므로 조립 뒤에 붙이는 것이 자연스럽다 —
    `attach_regression_guard` 가 그 순서를 들고 있다. 여기서 넘기지 않으면 가드 절은
    **미산출 + 사유**로 남는다."""
    failure_attribution = _build_failure_attribution(
        pipeline=pipeline,
        retrieval_labels_path=retrieval_labels_path,
    )
    report = EvaluationReport(
        conditions=conditions,
        gate_accuracy=gate_accuracy,
        pipeline=pipeline,
        judge_accuracy=judge_accuracy,
        failure_attribution=failure_attribution,
    )
    return (
        report if regression_guard is None else replace(report, regression_guard=regression_guard)
    )


def attach_regression_guard(
    report: EvaluationReport,
    *,
    stem: str,
    reports_dir: Path,
    promoted_reference_path: Path = DEFAULT_PROMOTED_BASELINE_PATH,
) -> EvaluationReport:
    """조립된 리포트에 회귀 가드 두 줄을 붙인다.

    **승격은 여기서 일어나지 않는다** — 이 경로는 승격 참조 파일을 읽기만 하고, 쓰는 코드는
    저장소 어디에도 없다(구조 테스트가 검사한다). 가드 산출이 실패해도 리포트를 죽이지
    않는다: 이미 과금된 측정 산출물을 잃지 않는 것이 우선이고, 실패는 **미산출 + 사유**로
    남는다(`scripts/AGENTS.md` 불변식 5·13 과 같은 규칙).
    """
    try:
        summary = run_summary_from_payload(report_to_json(report), stem=stem, source=stem)
        guard: RegressionGuard | GuardUnavailable = build_regression_guard(
            current=summary,
            reports_dir=reports_dir,
            promoted_reference_path=promoted_reference_path,
        )
    except Exception as error:  # 가드 실패가 이미 과금된 산출물을 죽이면 안 된다
        guard = GuardUnavailable(
            reason=f"회귀 가드 산출이 실패했다({type(error).__name__}): {error}"
        )
    return replace(report, regression_guard=guard)


def _build_failure_attribution(
    *,
    pipeline: PipelineAgreement | SkippedMeasurement,
    retrieval_labels_path: Path | None,
) -> FailureAttribution | UnavailableBreakdown:
    # 지연 import: retrieval_labels 는 골든셋 검증을 위해 evaluation 로더를 쓴다.
    # 보고서 조립 시점에만 반대 방향을 열어 모듈 순환 import 를 피한다.
    from reply_gate.retrieval_labels import (
        DEFAULT_RETRIEVAL_LABELS_PATH,
        load_retrieval_labels,
    )

    # 측정 2 미실행이 더 근본적인 사유다. 라벨 로드를 먼저 시도하면 둘 다 성립할 때
    # 미산출 사유가 라벨 실패로 덮여 진짜 원인이 사라진다.
    if isinstance(pipeline, SkippedMeasurement):
        return UnavailableBreakdown(reason=f"측정 2 미실행: {pipeline.reason}")

    path = DEFAULT_RETRIEVAL_LABELS_PATH if retrieval_labels_path is None else retrieval_labels_path
    try:
        labels = load_retrieval_labels(path)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return UnavailableBreakdown(reason=f"검색 정답 라벨 로드 실패({type(exc).__name__}): {exc}")

    relevant_by_id = {label.id: tuple(sorted(label.relevant_evidence_ids)) for label in labels}
    # 후보는 기각·인계만이 아니다. 정답 조항을 못 찾았는데도 답변이 확정된 케이스가 이
    # 제품에서 가장 위험한 실패이고, 그것이 분해표에서 빠지면 검색 실패가 과소 집계된다.
    candidates = tuple(pipeline.outcomes)
    missing_labels = sorted(
        outcome.case_id for outcome in candidates if outcome.case_id not in relevant_by_id
    )
    if missing_labels:
        return UnavailableBreakdown(
            reason="검색 정답 라벨에 없는 케이스: " + ", ".join(missing_labels)
        )

    attributed: list[FailureAttributionCase] = []
    for outcome in candidates:
        relevant = relevant_by_id[outcome.case_id]
        adopted = outcome.adopted_evidence_ids
        adopted_set = set(adopted)
        policy_adopted = tuple(
            evidence_id for evidence_id in adopted if is_policy_evidence_id(evidence_id)
        )
        is_failure = outcome.rejected_at_least_once or outcome.status is InquiryStatus.ESCALATED
        policy_retrieval_ran, policy_retrieval_note = _policy_retrieval_state(outcome)
        if relevant and not is_failure and set(relevant) <= adopted_set:
            # 정답 조항을 전부 채택하고 정상 답변한 케이스는 분해 대상이 아니다.
            continue
        l2_rejected = bool(set(outcome.reject_reasons).intersection(L2_REJECT_REASONS))
        normal_behavior: bool | None = None
        normal_behavior_path: str | None = None
        anomaly_reason: str | None = None
        if not relevant:
            classification = "expected_no_answer"
            zero_evidence_normal = (
                outcome.status is InquiryStatus.ESCALATED
                and outcome.escalation_reason is EscalationReason.NO_EVIDENCE
                and not adopted
                and not outcome.attempt_verdicts
            )
            l2_rejection_normal = (
                outcome.status is InquiryStatus.ESCALATED
                and outcome.escalation_reason is EscalationReason.REJECTED_TWICE
                and bool(adopted)
                and l2_rejected
            )
            # 주문 단계 사전 인계는 계약상 **정상 종결**이다 — business-rules 의
            # "구조적 사유가 이긴다"가 정책 근거를 모았든 아니든 인계를 시키기 때문이다.
            # 기대 경로 목록에 없으면 매 실행 비정상으로 찍힌다(findings 20번 ②).
            order_stage_normal = (
                outcome.status is InquiryStatus.ESCALATED
                and outcome.escalation_reason in ORDER_STAGE_ESCALATIONS
            )
            normal_behavior = zero_evidence_normal or l2_rejection_normal or order_stage_normal
            if order_stage_normal:
                normal_behavior_path = "order_stage_pre_handoff"
            elif zero_evidence_normal:
                normal_behavior_path = "retrieval_zero_evidence"
            elif l2_rejection_normal:
                normal_behavior_path = "l2_rejected_with_evidence"
            elif not adopted:
                anomaly_reason = "검색 0건이지만 no_evidence·시도 0건 종료가 아님"
            elif not l2_rejected:
                anomaly_reason = "근거를 채택했지만 L2 기각 사유 없음"
            else:
                anomaly_reason = "L2 기각 사유가 있지만 rejected_twice 인계가 아님"
        elif not is_failure:
            # 답변이 확정됐는데 정답 조항이 빠졌다 — 게이트를 통과한 근거 부족이다.
            classification = "answered_without_relevant_evidence"
        elif set(relevant) <= adopted_set:
            classification = "generation_issue"
        elif adopted_set.intersection(relevant):
            # 정답 조항 중 일부만 찾았다. 필요한 조항이 빠진 채 생성한 것이므로 생성 문제로
            # 세면 검색 실패가 과소 집계된다.
            classification = "partial_retrieval_failure"
        else:
            classification = "retrieval_failure"

        attributed.append(
            FailureAttributionCase(
                case_id=outcome.case_id,
                classification=classification,
                relevant_evidence_ids=relevant,
                adopted_evidence_ids=adopted,
                missing_relevant_evidence_ids=tuple(
                    evidence_id for evidence_id in relevant if evidence_id not in adopted_set
                ),
                rejected_at_least_once=outcome.rejected_at_least_once,
                escalated=outcome.status is InquiryStatus.ESCALATED,
                ended_with_zero_evidence=not adopted,
                ended_with_zero_policy_evidence=not policy_adopted,
                l2_caught_with_evidence=bool(adopted) and l2_rejected,
                normal_behavior=normal_behavior,
                normal_behavior_path=normal_behavior_path,
                anomaly_reason=anomaly_reason,
                policy_retrieval_ran=policy_retrieval_ran,
                policy_retrieval_note=policy_retrieval_note,
                final_status=None if outcome.status is None else outcome.status.value,
                matched=outcome.matched,
            )
        )

    return FailureAttribution(
        labels_path=display_path(path),
        generation_issue_count=sum(
            item.classification == "generation_issue" for item in attributed
        ),
        retrieval_failure_count=sum(
            item.classification == "retrieval_failure" for item in attributed
        ),
        partial_retrieval_failure_count=sum(
            item.classification == "partial_retrieval_failure" for item in attributed
        ),
        answered_without_relevant_evidence_count=sum(
            item.classification == "answered_without_relevant_evidence" for item in attributed
        ),
        expected_no_answer_count=sum(
            item.classification == "expected_no_answer" and item.normal_behavior is True
            for item in attributed
        ),
        expected_no_answer_anomaly_count=sum(
            item.classification == "expected_no_answer" and item.normal_behavior is False
            for item in attributed
        ),
        cases=tuple(attributed),
    )


#: 주문 단계에서 초안 전에 인계되는 **구조적** 사유. business-rules 의 "구조적 사유가
#: 이긴다"가 정하는 목록 그대로다 — 이 종결은 계약상 정상이지 anomaly 가 아니다.
ORDER_STAGE_ESCALATIONS: Final[frozenset[EscalationReason]] = frozenset(
    {
        EscalationReason.MISSING_ORDER_REF,
        EscalationReason.ORDER_NOT_FOUND,
        EscalationReason.SQL_FAILED,
    }
)


def _policy_retrieval_state(outcome: GoldenOutcome) -> tuple[bool | None, str | None]:
    """정책 검색이 실제로 돌았는가 — 의도 라우팅이 답을 들고 있다.

    **미상(`None`)을 False 로 접지 않는다.** 의도 해석이 무너진 문의는 "검색이 안 돌았다"가
    아니라 "돌았는지 알 수 없다"이고, 옛 산출물처럼 의도 필드가 없는 경우도 같다.
    """
    if outcome.intent is None:
        return None, "의도 라우팅 기록 없음 — 정책 검색 실행 여부 미상"
    if outcome.intent == IntentSource.ORDER.value:
        return False, "의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다"
    if outcome.intent in (IntentSource.POLICY.value, IntentSource.BOTH.value):
        return True, f"의도 분류가 `{outcome.intent}` 라 정책 검색이 실행됐다"
    return None, f"알 수 없는 의도 라벨: {outcome.intent}"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _pct(value: float | None) -> str:
    return "측정 불가" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None) -> str:
    return "측정 불가" if value is None else f"{value:.1f}"


def _int(value: int | None) -> str:
    return "측정 불가" if value is None else str(value)


def _count(value: int | None) -> str:
    """건수 — 세지 못했으면(로드 실패) **미실행**이라고 적는다. 0 으로 채우지 않는다."""
    return "미실행" if value is None else f"{value}건"


def _ms(value: float | None) -> str:
    """구간 시간 한 칸 — **미측정은 0 이 아니라 "미측정"** 이라고 적는다.

    사람이 읽는 줄과 리포트 JSON 이 **같은 값을 적어야 한다**는 것이 계약이다
    (커밋된 리포트에서 두 표면이 갈린 전례가 있다). 두 표면이 같은 원본에서 나오고,
    이 함수가 그 원본을 문자열로 옮기는 유일한 자리다.
    """
    return "미측정" if value is None else f"{value:.1f}"


def _render_stage_durations(
    aggregates: Sequence[SpanAggregate], conditions: RunConditions
) -> list[str]:
    """단계별 지연 표 — **미측정 케이스는 각 구간의 분모에서 빠진다.**

    구간 아홉을 밖으로 나가는 호출 여섯(의도 분류·질의 재작성·질의 임베딩·조회문 생성·
    초안 생성·L2 판정)과 코드만 도는 셋(벡터 검색·조회 실행·게이트 판정)으로 가른 것은
    "어느 단계가 몇 초인지"를 그 경계로 말할 수 있게 하기 위해서다.
    """
    lines = [
        "### 단계별 지연 (구간 아홉)",
        "",
        "밖으로 나가는 호출 여섯(의도 분류 · 질의 재작성 · 질의 임베딩 · 조회문 생성 ·",
        "초안 생성 · L2 판정)과 코드만 도는 셋(벡터 검색 · 조회 실행 · 게이트 판정)이다.",
        "한 구간의 시간은 그 구간의 **총 벽시계**이고 재시도·형식 실패·예외로 죽은 호출을",
        "포함한다. **미측정은 0 이 아니다** — 돌지 않은 구간은 분모에서 빠진다(0 을 섞어",
        "평균 내면 그 구간이 실제보다 빨라 보인다). 재생성이 돈 문의는 초안·게이트·판정",
        "구간이 시도별로 쌓이고, 시도별 값은 리포트 JSON 의 `attempt_durations` 에 있다.",
        "",
    ]
    if not conditions.measurement2_is_real:
        lines.extend(
            [
                "> **대역 실행에서는 밖으로 나가는 호출 여섯이 0 에 수렴한다** — 대역은",
                "> 프로세스 밖으로 나가지 않으므로 그 구간이 실제로 0 인 것이지 재지 않은",
                "> 것이 아니다(미측정은 `미측정` 으로 적힌다). 코드만 도는 셋(벡터 검색 ·",
                "> 조회 실행 · 게이트 판정)은 대역 실행에서도 실제 값이다.",
                "",
            ]
        )
    lines.extend(
        [
            "| 구간 | 측정 케이스 | 미측정 | 합계 ms | 평균 ms | p50 ms | p95 ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| `{item.span}` | {item.measured_cases} | {item.unmeasured_cases} | "
        f"{_ms(item.total_ms)} | {_ms(item.mean_ms)} | {_ms(item.p50_ms)} | {_ms(item.p95_ms)} |"
        for item in aggregates
    )
    lines.append("")
    return lines


_LIMITS: Final = """\
## 한계 (과장하지 않는다)

- **L1 은 패턴형 PII 만 본다.** 전화번호·이메일·주민등록번호처럼 정규식으로 잡히는 값만
  검사한다. 이름·주소 등 비패턴형 개인정보는 정규식으로 잡을 수 없어 **L1 의 검사 대상이
  아니다**(L2 도 근거-주장 정합만 보므로 대상이 아니다). 검출률 수치를 "개인정보 전반"으로
  읽으면 안 된다.
- **L1 은 내용의 진위를 보지 않는다.** citation 존재·무결성·스키마·PII 만 검사한다.
  근거를 인용했지만 내용이 근거와 어긋나는 답변은 L1 이 아니라 **L2 의미 검증**이 잡는다 —
  **L2 를 끈 실행에는 그 층이 통째로 없다**(실행 조건의 "L2 판정" 항목을 함께 읽어야 한다).
- **측정 2·3 은 확률 층이다.** 초안 생성과 판정이 비결정론이므로 재실행하면 값이 달라지고
  실제 모델 실행은 과금된다. 측정 1 만 100% 재현된다.
- **측정 2 의 일치율에는 초안 전 인계 경로가 포함된다** — 근거 0건·주문번호 없음·주문
  없음으로 끝난 건도 분모에 들어가므로 **L1 판정만의 지표가 아니다**.
- **측정 3 에는 목표치가 없다.** 무목표 관측이므로 이 수치에는 달성·미달 판정이 붙지 않는다.

## 이월 (다음 사이클)

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
        f"- L2 판정: {'켜짐' if conditions.l2_enabled else '꺼짐'} / 판정 모델: {conditions.judge}",
        f"- 검색 전략: {conditions.retrieval_strategy}"
        f" · 임베딩 {conditions.embedding_dimensions}차원",
        f"- 유사도 임계값: {conditions.similarity_threshold} / top k: {conditions.top_k}",
        f"- L1 픽스처: {conditions.l1_fixture_count}건 (`{conditions.l1_fixtures_path}`)",
        f"- 골든셋: {conditions.golden_case_count}건 (`{conditions.golden_set_path}`)",
        f"- 판정 픽스처: {_count(conditions.judge_fixture_count)} "
        f"(`{conditions.judge_fixtures_path}`)",
        f"- OPENAI_API_KEY 설정 여부: {'설정됨' if conditions.api_key_present else '없음'}",
        f"- ANTHROPIC_API_KEY 설정 여부: "
        f"{'설정됨' if conditions.judge_api_key_present else '없음'}",
        f"- 과금 실행: {'예' if conditions.billed else '아니오'}"
        f" · 측정 범위: {conditions.measurement_scope}",
        "",
    ]
    lines.extend(_render_fingerprint(conditions))
    lines.extend(_render_targets(report))
    lines.extend(_render_measurement_one(report.gate_accuracy))
    lines.extend(_render_measurement_two(report.pipeline, conditions))
    lines.extend(_render_failure_attribution(report.failure_attribution))
    lines.extend(_render_measurement_three(report.judge_accuracy, conditions))
    lines.extend(render_guard_section(report.regression_guard))
    lines.append(_LIMITS)
    return "\n".join(lines)


def _render_fingerprint(conditions: RunConditions) -> list[str]:
    """조건 지문 — **대조 가능성을 결정하는** 항목만 따로 세운다.

    실행 조건 전체를 눈으로 훑어 비교하지 않게 한다. 값이 없는 항목은 0 이 아니라
    **미상**으로 적히고, 이번 실행이 의도적으로 바꾼 축은 선언 목록이 들고 있다.
    """
    fingerprint = conditions.fingerprint()
    declared = set(fingerprint.declared)
    lines = [
        "### 조건 지문 (대조 가능성)",
        "",
        "선언 없이 달라진 조건끼리는 대조하지 않는다. **선언된 실험 변인**은 대조를 진행하고",
        "차이 목록을 병기한다.",
        "",
        "| 항목 | 값 | 선언 |",
        "| --- | --- | :---: |",
    ]
    lines.extend(
        f"| `{name}` | {fingerprint.describe(name)} | "
        f"{'**선언된 실험 변인**' if name in declared else ''} |"
        for name in fingerprint.values
    )
    lines.append("")
    if declared:
        lines.extend(
            [
                "선언된 실험 변인: " + ", ".join(f"`{name}`" for name in fingerprint.declared),
                "",
            ]
        )
    return lines


def _render_targets(report: EvaluationReport) -> list[str]:
    """목표치 대비 표. **미측정·대역은 미달이 아니다** — 그렇게 적으면 리포트가 거짓말을 한다."""
    lines = [
        "## 목표치 대비",
        "",
        "2026-08-05 확정(`docs/tracking/decisions/0006-지표-목표치를-실측-뒤에-확정한다.md`).",
        "미측정 지표와 대역으로 낸 확률 층 수치는 판정하지 않는다.",
        "",
        "**합산 일치율은 달성 목표가 아니라 하한 경보선이다**(결정 0006 재확정) — 그 아래로",
        "내려가면 무언가 크게 부서졌다는 신호일 뿐이고, **사이클 성패 판정은 케이스 단위**",
        "(회귀 가드의 비악화 판정 + 케이스별 귀인)가 맡는다. 합산은 병기다.",
        "",
        "| 지표 | 성격 | 경계 | 실측 | 판정 |",
        "| --- | --- | --- | ---: | :---: |",
    ]
    for item in assess_targets(report):
        value_cell = "미측정" if item.value is None else _pct(item.value)
        verdict_cell = "**미달**" if item.verdict == "미달" else item.verdict
        if item.verdict == "경보":
            verdict_cell = "**경보**"
        role = "하한 경보선" if item.target.alert_only else "달성 목표"
        lines.append(
            f"| {item.target.label} | {role} | {item.target.describe()} | "
            f"{value_cell} | {verdict_cell} |"
        )
    lines.append("")
    return lines


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
                "> 어휘 임베딩 · **판정 대역**)으로 하네스 배관만 검증한 실행이다. 일치율·기각",
                "> 재현율·지연·토큰 값을 제품 지표로 인용하면 안 된다 — **판정 계열 토큰도",
                "> 대역이 만든 휴리스틱 값이라 합산까지 대역이다.** 실행 조건 절의 생성 LLM·",
                "> 임베딩·판정 모델 항목이 무엇으로 돌았는지를 함께 읽어야 한다.",
                "",
            ]
        )

    lines.extend(
        [
            f"- 골든셋 {pipeline.total}건 처리",
            f"- **허용 결과 집합 대비 일치율: {_pct(pipeline.match_rate)}** "
            f"({pipeline.matched}/{pipeline.total}) "
            "— **초안 전 인계 경로 포함이며 L1 판정만의 지표가 아니다.**",
            f"- **미끼 문의(reject_bait)의 기각 재현율: {_pct(pipeline.bait_reject_recall)}** "
            f"({pipeline.bait_reject_reproduced}/{pipeline.bait_total}) "
            "— 목표 없는 관측값이다(결정 0006·0008)."
            + (
                f" **미측정 {pipeline.bait_unmeasured}건은 분모 밖이다**"
                " — L2 판정 호출이 실패해 게이트가 돌지 못했다."
                if pipeline.bait_unmeasured
                else ""
            ),
            f"- 정상 PII 에코 감시 케이스: {pipeline.forbidden_watch_total}건 중 "
            f"금지 사유 발화 {pipeline.forbidden_violations}건",
            f"- 지연 p50: {_int(pipeline.latency_p50_ms)} ms / "
            f"p95: {_int(pipeline.latency_p95_ms)} ms "
            "(파이프라인 `run` 의 벽시계 시간 — 처리 기록 저장은 포함하지 않는다)",
            f"- 검색 단계 폴백: {pipeline.retrieval_fallback_total}건"
            + (
                " — 재작성을 얻지 못해 원문 질의로 검색한 문의다. **인계가 아니다.** "
                "이 수가 크면 재작성 층이 사실상 꺼진 실행이므로 검색 지표를 그렇게 읽어야 한다."
                if pipeline.retrieval_fallback_total
                else " (전건 재작성 성공 — 검색 구성이 실행 조건 그대로 돌았다)"
            ),
            *_abstention_undefined_lines(pipeline),
            "",
            "### 문의 1건당 토큰 (생성·임베딩·판정·검색 구분)",
            "",
            "provider 와 단가가 다른 계열을 합산하면 건당 비용 지표가 무너진다 — 네 계열은",
            "끝까지 분리해서 센다. L2 미실행이면 판정 계열은, 재작성을 쓰지 않았으면 검색",
            "계열은 0 이다.",
            "",
            "| 계열 | 합계 | 건당 |",
            "| --- | ---: | ---: |",
            *_token_rows(pipeline, conditions),
            "",
            *_generation_cache_token_lines(pipeline),
            "",
            *_render_stage_durations(pipeline.stage_durations, conditions),
            "### 종결 분포",
            "",
            f"- 최종 상태: {_counts(pipeline.status_counts)}",
            f"- 인계 사유: {_counts(pipeline.escalation_counts)}",
            "",
            "### 케이스별 채택 근거",
            "",
        ]
    )
    lines.extend(
        f"- `{outcome.case_id}`: "
        + (", ".join(f"`{evidence_id}`" for evidence_id in outcome.adopted_evidence_ids) or "없음")
        for outcome in pipeline.outcomes
    )
    lines.append("")

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


def _abstention_undefined_lines(pipeline: PipelineAgreement) -> list[str]:
    """기권 게이트 통계량이 미정의였던 분기 — **사유마다 처분이 다르므로 갈라 적는다.**

    사유가 한 건도 없었으면 그 사실을 적는다. 줄 자체를 빼면 "재지 않았다"와 "없었다"가
    산출물에서 같아진다(미실행을 0 으로 채우지 않는다는 규칙의 같은 자리다).
    """
    counts = pipeline.abstention_undefined_counts
    if not counts:
        return [
            "- 기권 게이트 통계량 미정의: 0건 (전건에서 통계량이 정의됐거나 게이트가 돌지 않았다)"
        ]
    rendered = " · ".join(
        f"`{reason}` {hits}건({'기권' if AbstentionUndefined(reason).abstains else '비기권'})"
        for reason, hits in counts.items()
    )
    return [
        f"- 기권 게이트 통계량 미정의: {rendered}"
        " — **사유마다 처분이 다르다.** 처분만 보면 두 분기가 구분되지 않는다."
    ]


def _token_rows(pipeline: PipelineAgreement, conditions: RunConditions) -> list[str]:
    """토큰 표의 본문 — 계열 넷(생성·임베딩·판정·검색)을 끝까지 분리해서 적는다.

    대역 실행에서는 **판정·검색 행에 `(대역)` 을 붙인다**: `--stub-llm` 은 판정자와 생성
    클라이언트를 모두 대역으로 갈아 끼우므로 이 행들의 값은 대역의 휴리스틱 산출이고
    합산에도 그대로 들어간다.

    **검색 계열을 생성 소계에 넣지 않는다.** 초안을 만들지도 않은 문의가 초안 생성 토큰을
    쓴 것으로 찍히면 성공 판정 ②("시도 0건 + 판정 토큰 0")를 표에서 읽을 수 없다.
    """
    # 대역 실행이면 판정 계열도 대역이다 — 표만 보고 실제 판정 비용으로 읽으면 안 된다.
    judge_mark = "" if conditions.measurement2_is_real else " (대역)"

    def per_inquiry(total: int) -> float | None:
        return None if pipeline.total == 0 else total / pipeline.total

    generation_total = pipeline.input_tokens_total + pipeline.output_tokens_total
    judge_total = pipeline.judge_input_tokens_total + pipeline.judge_output_tokens_total
    retrieval_total = pipeline.retrieval_input_tokens_total + pipeline.retrieval_output_tokens_total
    rows: list[tuple[str, int, float | None]] = [
        ("생성 입력", pipeline.input_tokens_total, per_inquiry(pipeline.input_tokens_total)),
        ("생성 출력", pipeline.output_tokens_total, per_inquiry(pipeline.output_tokens_total)),
        ("생성 소계", generation_total, pipeline.generation_tokens_per_inquiry),
        ("임베딩", pipeline.embedding_tokens_total, pipeline.embedding_tokens_per_inquiry),
        (
            f"판정 입력{judge_mark}",
            pipeline.judge_input_tokens_total,
            per_inquiry(pipeline.judge_input_tokens_total),
        ),
        (
            f"판정 출력{judge_mark}",
            pipeline.judge_output_tokens_total,
            per_inquiry(pipeline.judge_output_tokens_total),
        ),
        (f"판정 소계{judge_mark}", judge_total, pipeline.judge_tokens_per_inquiry),
        (
            f"검색 입력{judge_mark}",
            pipeline.retrieval_input_tokens_total,
            per_inquiry(pipeline.retrieval_input_tokens_total),
        ),
        (
            f"검색 출력{judge_mark}",
            pipeline.retrieval_output_tokens_total,
            per_inquiry(pipeline.retrieval_output_tokens_total),
        ),
        (f"검색 소계{judge_mark}", retrieval_total, pipeline.retrieval_tokens_per_inquiry),
    ]
    lines = [f"| {label} | {total} | {_num(value)} |" for label, total, value in rows]
    lines.append(
        f"| **합산** | {_grand_total(pipeline)} | **{_num(pipeline.total_tokens_per_inquiry)}** |"
    )
    return lines


def _render_failure_attribution(
    attribution: FailureAttribution | UnavailableBreakdown,
) -> list[str]:
    lines = ["### 검색 실패 / 생성 문제 분해", ""]
    if isinstance(attribution, UnavailableBreakdown):
        lines.extend(
            [
                f"**미산출 (사유: {attribution.reason})**",
                "",
                "미산출을 0건·빈 집계·성공으로 대체하지 않는다.",
                "",
            ]
        )
        return lines

    lines.extend(
        [
            f"- 검색 정답 라벨: `{attribution.labels_path}`",
            f"- 생성 문제: **{attribution.generation_issue_count}건** — "
            "정답 조항을 **전부** 채택했지만 기각·인계",
            f"- 검색 실패 합계: **{attribution.retrieval_failure_total}건** "
            f"(전부 누락 {attribution.retrieval_failure_count}건 · "
            f"일부 누락 {attribution.partial_retrieval_failure_count}건)",
            f"- 근거 없이 답변 확정: **{attribution.answered_without_relevant_evidence_count}건** "
            "— 정답 조항이 빠진 채 게이트를 통과했다",
            f"- 빈 정답 정상 인계: **{attribution.expected_no_answer_count}건** — "
            "앞의 분류에 포함하지 않음",
            f"- 빈 정답 비정상 종결: **{attribution.expected_no_answer_anomaly_count}건** — "
            "정상 인계 집계에서 제외",
            "",
            "#### 케이스별 판정 근거",
            "",
        ]
    )
    labels = {
        "generation_issue": "생성 문제",
        "retrieval_failure": "검색 실패(전부 누락)",
        "partial_retrieval_failure": "검색 실패(일부 누락)",
        "answered_without_relevant_evidence": "근거 없이 답변 확정",
        "expected_no_answer": "빈 정답 정상 인계",
    }
    normal_paths = {
        "retrieval_zero_evidence": "검색 0건 종료",
        "l2_rejected_with_evidence": "근거 채택 후 L2 검출",
        "order_stage_pre_handoff": "주문 단계 사전 인계 — 구조적 사유가 이긴다(계약상 정상)",
    }
    for item in attribution.cases:
        relevant = ", ".join(f"`{value}`" for value in item.relevant_evidence_ids) or "없음"
        adopted = ", ".join(f"`{value}`" for value in item.adopted_evidence_ids) or "없음"
        route = ""
        if item.classification == "expected_no_answer":
            if item.normal_behavior is False:
                route = f" / 비정상: {item.anomaly_reason}"
            elif item.normal_behavior_path in normal_paths:
                route = f" / {normal_paths[item.normal_behavior_path]}"
        # 정책 검색이 아예 안 돈 케이스는 컷과 무관한 실패다 — 이름으로 갈라 적는다.
        if item.policy_retrieval_ran is False:
            route += " / **정책 검색 미실행**"
            if item.policy_retrieval_note:
                route += f"({item.policy_retrieval_note})"
        elif item.policy_retrieval_ran is None:
            route += " / 정책 검색 실행 여부 미상"
        # 1차 기각 뒤 재생성으로 회복한 케이스가 종결 실패처럼 읽히지 않게 병기한다.
        if item.classification == "generation_issue":
            route += (
                f" / 최종 {item.final_status or '미상'}"
                f"·{'라벨 일치' if item.matched else '라벨 불일치'}"
            )
        label = (
            "빈 정답 비정상 종결"
            if item.classification == "expected_no_answer" and item.normal_behavior is False
            else labels[item.classification]
        )
        lines.append(
            f"- `{item.case_id}`: **{label}** — 정답 근거 {relevant} / 채택 근거 {adopted}{route}"
        )
    lines.append("")
    return lines


def _generation_cache_token_lines(pipeline: PipelineAgreement) -> list[str]:
    """생성·검색 계열의 프롬프트 캐시 — **계열마다 한 줄, 입력 칸과 분리해서** 적는다.

    두 계열은 같은 클라이언트를 지나므로 한 줄로 합치면 어느 계열의 캐시였는지 되짚을 수
    없다. 그리고 이 줄이 없으면 달러 환산이 전부 정가를 곱해, 리포트의 비용이 **실제
    청구액이 아니라 그 이상일 수 없는 값**(상한)이 된다.

    **위 표의 '생성 입력'·'검색 입력'에서 이 값을 빼지 않았다.** OpenAI 응답에서 캐시
    적중분은 입력 토큰 **안에** 들어 있다(Anthropic 판정 계열은 반대로 제외돼 있다).
    빼면 옛 산출물과 정의가 갈리고, 합산에 더하면 같은 토큰을 두 번 센다.

    **재지 않은 계열은 0 이 아니라 "미측정"** 이다(`scripts/AGENTS.md` 불변식 5).
    """

    def line(label: str, measured: bool, creation: int | None, read: int | None) -> str:
        if not measured:
            return (
                f"- {label} 계열 프롬프트 캐시: **미측정** "
                "(캐시 계열 토큰을 보고하지 않은 실행 — 0 이 아니다)"
            )
        return (
            f"- {label} 계열 프롬프트 캐시({label} 입력 토큰과 별도 칸): "
            f"write {_int_or_unmeasured(creation)} / read {_int_or_unmeasured(read)} "
            f"— 위 '{label} 입력'은 **캐시 적중분을 포함한 총 입력**이다(빼지도 더하지도 않는다)"
        )

    return [
        line(
            "생성",
            pipeline.generation_cache_measured,
            pipeline.generation_cache_creation_total,
            pipeline.generation_cache_read_total,
        ),
        line(
            "검색",
            pipeline.retrieval_cache_measured,
            pipeline.retrieval_cache_creation_total,
            pipeline.retrieval_cache_read_total,
        ),
    ]


def _int_or_unmeasured(value: int | None) -> str:
    """한쪽 칸만 보고된 계열의 나머지 칸 — 0 으로 채우지 않는다."""
    return "미측정" if value is None else str(value)


def _grand_total(pipeline: PipelineAgreement) -> int:
    """네 계열 합산 — 한 계열이 빠지면 건당 비용이 실제보다 작아진다."""
    return (
        pipeline.input_tokens_total
        + pipeline.output_tokens_total
        + pipeline.embedding_tokens_total
        + pipeline.judge_input_tokens_total
        + pipeline.judge_output_tokens_total
        + pipeline.retrieval_input_tokens_total
        + pipeline.retrieval_output_tokens_total
    )


def _judge_latency_lines(accuracy: JudgeAccuracy) -> list[str]:
    """판정 호출 지연 — 픽스처 측정도 이제 지연을 기록한다.

    **판정하지 못한 픽스처의 경과도 분모에 든다**(그 호출도 판정 층이 쓴 시간이다).
    한 건도 재지 않은 실행은 0 이 아니라 "미측정"이다.
    """
    latency = accuracy.judge_latency
    if latency is None or latency.measured_cases == 0:
        return ["- 판정 호출 지연: **미측정** (경과를 기록하지 않은 실행 — 0 이 아니다)"]
    return [
        f"- 판정 호출 지연: 평균 {_ms(latency.mean_ms)} ms / "
        f"p50 {_ms(latency.p50_ms)} ms / p95 {_ms(latency.p95_ms)} ms "
        f"(측정 {latency.measured_cases}건 / 미측정 {latency.unmeasured_cases}건 — "
        "재시도와 판정 실패 호출을 포함한 총 벽시계다)"
    ]


def _cache_token_lines(accuracy: JudgeAccuracy) -> list[str]:
    """판정 프롬프트 캐시 계열 — **판정 입력 토큰과 분리해서** 적는다.

    캐싱 켜짐 조건에서 `input_tokens` 는 캐시 적중분을 **제외한** 비캐시 입력이다. 세 값을
    뭉뚱그려 "입력 토큰이 줄었다"고 적으면 그 자체가 은폐다 — write 는 일반 입력보다 비싸고
    read 는 싸므로 단가가 다른 값을 한 칸에 넣을 수 없다.

    **재지 않은 실행은 0 이 아니라 "미측정"** 이다(`scripts/AGENTS.md` 불변식 5).
    달러 환산은 싣지 않는다 — 저장소에 기준일·출처가 붙은 단가 표가 아직 없다.
    """
    if not accuracy.cache_measured:
        return [
            "- 판정 프롬프트 캐시: **미측정** (캐시 계열 토큰을 보고하지 않은 실행 — 0 이 아니다)"
        ]
    return [
        f"- 판정 프롬프트 캐시(판정 입력 토큰과 별도 칸): "
        f"write {accuracy.cache_creation_tokens_total} / "
        f"read {accuracy.cache_read_tokens_total} "
        f"— 위 '판정 토큰 입력'은 **캐시 적중분을 제외한 비캐시 입력**이다"
    ]


def _render_measurement_three(
    accuracy: JudgeAccuracy | SkippedMeasurement, conditions: RunConditions
) -> list[str]:
    lines = ["## 측정 3 — L2 판정 단위 정확도 (확률 층)", ""]
    if isinstance(accuracy, SkippedMeasurement):
        lines.extend(
            [
                f"**미실행 (사유: {accuracy.reason})**",
                "",
                "수치를 0 이나 빈 값으로 채우지 않는다 — 미실행은 미실행으로 남긴다.",
                "",
            ]
        )
        return lines

    if not conditions.measurement3_is_real:
        lines.extend(
            [
                "> **경고 — 아래 수치는 실제 판정 모델 수치가 아니다.** 결정론 판정 대역으로",
                "> 배관만 검증한 실행이라 **과금되지 않았고 재실행해도 같은 값이 나온다**.",
                "> 검출률·오탐률·사유 일치·판정 토큰을 판정 모델의 성능으로 인용하면 안 된다.",
                "",
            ]
        )

    failed_note = f" / 판정 실패 {accuracy.error_total} — 분모 제외" if accuracy.error_total else ""
    nature_label = (
        "실제 판정 모델(과금)" if conditions.measurement3_is_real else "대역(비과금·결정론)"
    )
    nature = (
        [
            "고정 claim 집합을 **판정기에 직접** 흘려 기대 판정과 대조했다(무엇으로 판정했는지는",
            "아래 판정 모델 항목이 들고 있다). 측정 1 과 같은 모양의 수치지만 이 실행은",
            "**확률 층이고 과금된다** — 재실행하면 값이 달라진다.",
        ]
        if conditions.measurement3_is_real
        else [
            "고정 claim 집합을 **판정기에 직접** 흘려 기대 판정과 대조했다(무엇으로 판정했는지는",
            "아래 판정 모델 항목이 들고 있다). 이 실행은 **결정론 대역**으로 돌았다 —",
            "**과금되지 않았고 재실행해도 같은 값**이며, 판정 모델의 정확도가 아니다.",
        ]
    )
    lines.extend(
        [
            *nature,
            "목표치: **없음** (무목표 관측 — 미측정을 미달로도 달성으로도 적지 않는",
            "규칙과 같은 이유로, 경계가 없는 지표에 판정을 붙이지 않는다).",
            "",
            f"- 판정 모델: {conditions.judge}",
            f"- 실측 여부: {nature_label}",
            f"- 픽스처 총수: **{accuracy.total}건** "
            f"(기각 기대 {accuracy.violation_total} / 통과 기대 {accuracy.clean_total}"
            f"{failed_note})",
            f"- **L2 검출률: {_pct(accuracy.detection_rate)}** "
            f"({accuracy.violation_detected}/{accuracy.violation_total})",
            f"- **L2 오탐률: {_pct(accuracy.false_positive_rate)}** "
            f"({accuracy.clean_false_positive}/{accuracy.clean_total})",
            f"- 사유 목록까지 정확히 일치: {_pct(accuracy.reason_set_exact_rate)} "
            f"({accuracy.reason_set_exact}/{accuracy.judged_total})",
            f"- claim 단위 판정 일치: {_pct(accuracy.claim_verdict_match_rate)} "
            f"({accuracy.claim_verdict_matched}/{accuracy.claim_total})",
            f"- 모순 근거쌍: 기대 {accuracy.contradiction_expected_total}건 중 "
            f"{accuracy.contradiction_matched_total}건 검출 "
            f"(기대 밖 검출 {accuracy.contradiction_extra_total}건)",
            f"- 판정 토큰: 입력 {accuracy.input_tokens_total} / "
            f"출력 {accuracy.output_tokens_total} "
            f"(픽스처당 {_num(accuracy.tokens_per_fixture)})",
            *_judge_latency_lines(accuracy),
            *_cache_token_lines(accuracy),
            "",
            "### 사유 2종별 내역",
            "",
            "| 사유 | 기대 픽스처 | 검출 | 검출률 | 오발화(기대하지 않은 발화) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
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
            f"기대 {outcome.expected_verdict.value}{_reason_list(outcome.expected_reasons)} → "
            + (
                f"판정 실패({outcome.error})"
                if outcome.actual_verdict is None
                else f"실제 {outcome.actual_verdict.value}{_reason_list(outcome.actual_reasons)}"
            )
            for outcome in failures
        )
    else:
        lines.append("모든 픽스처가 기대 판정·기대 사유 목록과 일치했다.")
    lines.append("")
    return lines


def _counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "없음"
    return ", ".join(f"{key} {value}건" for key, value in sorted(counts.items()))


def report_to_json(report: EvaluationReport) -> dict[str, Any]:
    """기계가 읽는 형식. 마크다운과 같은 원본에서 나온다."""
    conditions = report.conditions
    accuracy = report.gate_accuracy
    payload: dict[str, Any] = {
        "targets": {
            "decided_at": "2026-08-05",
            "decision": "docs/tracking/decisions/0006-지표-목표치를-실측-뒤에-확정한다.md",
            "metrics": [
                {
                    "key": item.target.key,
                    "label": item.target.label,
                    "bound": item.target.bound,
                    "direction": "at_most" if item.target.at_most else "at_least",
                    "measured": item.value,
                    # 미측정·대역이면 null — 판정하지 않은 것을 미달로도 달성으로도 적지 않는다.
                    "met": item.met,
                    "verdict": item.verdict,
                }
                for item in assess_targets(report)
            ],
        },
        "conditions": {
            "started_at": conditions.started_at,
            "generation": conditions.generation,
            "embedding": conditions.embedding,
            "embedding_dimensions": conditions.embedding_dimensions,
            "judge": conditions.judge,
            "l2_enabled": conditions.l2_enabled,
            "retrieval_strategy": conditions.retrieval_strategy,
            "similarity_threshold": conditions.similarity_threshold,
            "top_k": conditions.top_k,
            "l1_fixture_count": conditions.l1_fixture_count,
            "golden_case_count": conditions.golden_case_count,
            "judge_fixture_count": conditions.judge_fixture_count,
            "l1_fixtures_path": conditions.l1_fixtures_path,
            "golden_set_path": conditions.golden_set_path,
            "judge_fixtures_path": conditions.judge_fixtures_path,
            "api_key_present": conditions.api_key_present,
            "judge_api_key_present": conditions.judge_api_key_present,
            "measurement2_is_real": conditions.measurement2_is_real,
            "measurement3_is_real": conditions.measurement3_is_real,
            "billed": conditions.billed,
            "measurement_scope": conditions.measurement_scope,
            # 대조 가능성을 결정하는 지문. 값이 없는 항목은 null(미상)이고 0 이 아니다.
            "condition_fingerprint": dict(conditions.fingerprint().values),
            "declared_experiment_fields": list(conditions.declared_experiment_fields),
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
            "pii": "L1 은 패턴형 PII 만 검사한다. 이름·주소 등 비패턴형은 어느 층의 대상도 아니다.",
            "content": "L1 은 내용의 진위를 검사하지 않는다 — 그 층은 L2 이고, 끄면 없다.",
            "measurement_2": "확률 층이라 재실행하면 값이 달라진다. 일치율에 초안 전 인계 포함.",
            "measurement_3": "확률 층이고 과금된다. 목표치를 두지 않아 달성 판정이 없다.",
        },
        "deferred": [
            "L1 필터링에 의한 L2 호출 감소율",
            "RAG 검색 품질 단계별 개선표",
            "비패턴형 개인정보(이름·주소) 검출",
        ],
    }
    payload["measurement_2_pipeline_agreement"] = _measurement_two_json(report.pipeline)
    payload["failure_attribution"] = _failure_attribution_json(report.failure_attribution)
    payload["measurement_3_l2_judge_accuracy"] = _measurement_three_json(
        report.judge_accuracy, conditions
    )
    payload["regression_guard"] = guard_to_json(report.regression_guard)
    return payload


def _failure_attribution_json(
    attribution: FailureAttribution | UnavailableBreakdown,
) -> dict[str, Any]:
    if isinstance(attribution, UnavailableBreakdown):
        return {"computed": False, "reason": attribution.reason}
    return {
        "computed": True,
        "labels_path": attribution.labels_path,
        "generation_issue_count": attribution.generation_issue_count,
        "retrieval_failure_count": attribution.retrieval_failure_count,
        "partial_retrieval_failure_count": attribution.partial_retrieval_failure_count,
        "retrieval_failure_total": attribution.retrieval_failure_total,
        "answered_without_relevant_evidence_count": (
            attribution.answered_without_relevant_evidence_count
        ),
        "expected_no_answer_count": attribution.expected_no_answer_count,
        "expected_no_answer_anomaly_count": attribution.expected_no_answer_anomaly_count,
        "cases": [
            {
                "case_id": item.case_id,
                "classification": item.classification,
                "relevant_evidence_ids": list(item.relevant_evidence_ids),
                "adopted_evidence_ids": list(item.adopted_evidence_ids),
                "missing_relevant_evidence_ids": list(item.missing_relevant_evidence_ids),
                "rejected_at_least_once": item.rejected_at_least_once,
                "escalated": item.escalated,
                "ended_with_zero_evidence": item.ended_with_zero_evidence,
                "ended_with_zero_policy_evidence": item.ended_with_zero_policy_evidence,
                "l2_caught_with_evidence": item.l2_caught_with_evidence,
                "normal_behavior": item.normal_behavior,
                "normal_behavior_path": item.normal_behavior_path,
                "anomaly_reason": item.anomaly_reason,
                "policy_retrieval_ran": item.policy_retrieval_ran,
                "policy_retrieval_note": item.policy_retrieval_note,
                "final_status": item.final_status,
                "matched": item.matched,
            }
            for item in attribution.cases
        ],
    }


def _span_aggregate_json(item: SpanAggregate) -> dict[str, Any]:
    """구간 집계 한 줄. **미측정은 0 이 아니라 `null`** 이고 분모는 측정 케이스 수다."""
    return {
        "span": item.span,
        "measured_cases": item.measured_cases,
        "unmeasured_cases": item.unmeasured_cases,
        "total_ms": item.total_ms,
        "mean_ms": item.mean_ms,
        "p50_ms": item.p50_ms,
        "p95_ms": item.p95_ms,
    }


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
        #: 판정이 돌지 못해 분모 밖으로 뺀 미끼 건수. 0 으로 채우지 않는다.
        "bait_unmeasured": pipeline.bait_unmeasured,
        "forbidden_watch_total": pipeline.forbidden_watch_total,
        "forbidden_violations": pipeline.forbidden_violations,
        "latency_p50_ms": pipeline.latency_p50_ms,
        "latency_p95_ms": pipeline.latency_p95_ms,
        #: 검색 단계가 폴백한 문의 수. 인계가 아니라 "재작성 없이 원문으로 돌았다"이다.
        "retrieval_fallback_total": pipeline.retrieval_fallback_total,
        #: 기권 게이트 통계량이 미정의였던 문의 수 — **사유별**이다. 두 사유의 처분이
        #: 반대라 한 칸으로 접으면 어느 분기였는지 되짚을 수 없다. 사유가 한 번도 없었으면
        #: 빈 맵이고, 없는 사유를 0 으로 채우지 않는다.
        "abstention_undefined_counts": dict(pipeline.abstention_undefined_counts),
        #: 구간 아홉의 세트 집계. **미측정 케이스는 그 구간의 분모에서 빠지고**,
        #: 한 건도 재지 않은 구간의 값은 0 이 아니라 `null` 이다. 사람이 읽는 줄
        #: ("단계별 지연" 표)과 **같은 값**이며 같은 원본에서 나온다.
        "stage_durations": [_span_aggregate_json(item) for item in pipeline.stage_durations],
        "tokens": {
            "generation_input_total": pipeline.input_tokens_total,
            "generation_output_total": pipeline.output_tokens_total,
            "embedding_total": pipeline.embedding_tokens_total,
            "judge_input_total": pipeline.judge_input_tokens_total,
            "judge_output_total": pipeline.judge_output_tokens_total,
            "retrieval_input_total": pipeline.retrieval_input_tokens_total,
            "retrieval_output_total": pipeline.retrieval_output_tokens_total,
            #: 캐시 계열은 **계열마다 한 쌍**이고 입력 칸과 분리 표기한다. 위 두 입력 칸은
            #: 캐시 적중분을 **포함한** 총 입력이므로(OpenAI 는 적중분이 입력 토큰 안에 있다)
            #: 이 값을 합산에 다시 더하지 않는다. 재지 않은 실행은 0 이 아니라 `null` 이다.
            "generation_cache_creation_total": pipeline.generation_cache_creation_total,
            "generation_cache_read_total": pipeline.generation_cache_read_total,
            "retrieval_cache_creation_total": pipeline.retrieval_cache_creation_total,
            "retrieval_cache_read_total": pipeline.retrieval_cache_read_total,
            "generation_per_inquiry": pipeline.generation_tokens_per_inquiry,
            "embedding_per_inquiry": pipeline.embedding_tokens_per_inquiry,
            "judge_per_inquiry": pipeline.judge_tokens_per_inquiry,
            "retrieval_per_inquiry": pipeline.retrieval_tokens_per_inquiry,
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
                #: 판정 호출이 무너져 게이트가 돌지 못했는가. `attempt_verdicts` 만으로는
                #: "돌았고 통과"와 구분되지 않는다(docs/contracts.md "층별 판정 키" ③).
                "gate_never_ran": outcome.gate_never_ran,
                "reject_reasons": [reason.value for reason in outcome.reject_reasons],
                "adopted_evidence_ids": list(outcome.adopted_evidence_ids),
                "latency_ms": outcome.latency_ms,
                "input_tokens": outcome.input_tokens,
                "output_tokens": outcome.output_tokens,
                "embedding_tokens": outcome.embedding_tokens,
                "judge_input_tokens": outcome.judge_input_tokens,
                "judge_output_tokens": outcome.judge_output_tokens,
                "retrieval_input_tokens": outcome.retrieval_input_tokens,
                "retrieval_output_tokens": outcome.retrieval_output_tokens,
                #: 케이스별 캐시 칸도 계열별이다. 미측정은 0 이 아니라 `null` 이다.
                "generation_cache_creation_tokens": outcome.generation_cache_creation_tokens,
                "generation_cache_read_tokens": outcome.generation_cache_read_tokens,
                "retrieval_cache_creation_tokens": outcome.retrieval_cache_creation_tokens,
                "retrieval_cache_read_tokens": outcome.retrieval_cache_read_tokens,
                "retrieval_fallback_reason": outcome.retrieval_fallback_reason,
                #: 기권 게이트 통계량이 미정의였던 사유. `null` 은 게이트 꺼짐 · 정책 검색
                #: 미실행 · 통계량 정의됨 셋을 함께 덮는다(처분이 아니라 **사유**다).
                "abstention_undefined_reason": outcome.abstention_undefined_reason,
                # 정책 검색이 돌았는지를 산출물에서 바로 읽게 한다 — 역추론 금지.
                "intent": outcome.intent,
                #: 케이스별 구간 시간(ms). 키는 아홉이 항상 있고 미측정은 `null` 이다.
                "stage_durations": outcome.stage_durations.as_mapping(),
                #: 시도별 구간. 재생성이 돌면 2건이고, 합계는 위 `stage_durations` 다.
                "attempt_durations": [
                    {
                        "draft_ms": item.draft_ms,
                        "gate_ms": item.gate_ms,
                        "l2_judge_ms": item.l2_judge_ms,
                    }
                    for item in outcome.attempt_durations
                ],
                "matched": outcome.matched,
                "mismatches": list(outcome.mismatches),
                "error": outcome.error,
            }
            for outcome in pipeline.outcomes
        ],
    }


def _measurement_three_json(
    accuracy: JudgeAccuracy | SkippedMeasurement, conditions: RunConditions
) -> dict[str, Any]:
    if isinstance(accuracy, SkippedMeasurement):
        return {"executed": False, "skip_reason": accuracy.reason}
    # 대역으로 만든 수치를 "과금된 실측"이라고 적으면 리포트가 스스로 거짓 신고를 한다 —
    # 세 플래그가 전부 실측 여부를 따른다(측정 2 의 `measurement2_is_real` 과 같은 처리).
    is_real = conditions.measurement3_is_real
    return {
        "executed": True,
        "is_real": is_real,
        "deterministic": not is_real,
        "billed": is_real,
        # 목표치를 두지 않기로 했다 — 경계가 없으므로 달성 여부도 없다(`null`).
        "target": None,
        "target_note": "없음 — 무목표 관측 (결정 0006)",
        "total": accuracy.total,
        "judged_total": accuracy.judged_total,
        "error_total": accuracy.error_total,
        "violation_total": accuracy.violation_total,
        "violation_detected": accuracy.violation_detected,
        "detection_rate": accuracy.detection_rate,
        "clean_total": accuracy.clean_total,
        "clean_false_positive": accuracy.clean_false_positive,
        "false_positive_rate": accuracy.false_positive_rate,
        "reason_set_exact": accuracy.reason_set_exact,
        "reason_set_exact_rate": accuracy.reason_set_exact_rate,
        "claim_total": accuracy.claim_total,
        "claim_verdict_matched": accuracy.claim_verdict_matched,
        "claim_verdict_match_rate": accuracy.claim_verdict_match_rate,
        "contradiction_expected_total": accuracy.contradiction_expected_total,
        "contradiction_matched_total": accuracy.contradiction_matched_total,
        "contradiction_extra_total": accuracy.contradiction_extra_total,
        "contradiction_recall": accuracy.contradiction_recall,
        "tokens": {
            "input_total": accuracy.input_tokens_total,
            "output_total": accuracy.output_tokens_total,
            "per_fixture": accuracy.tokens_per_fixture,
            #: 캐시 계열은 **판정 토큰 정의를 왜곡하지 않게 분리 표기**한다. 켜짐 조건의
            #: `input_total` 은 캐시 적중분을 제외한 비캐시 입력이고, write/read 는 단가가
            #: 서로 다르다. 재지 않은 실행은 0 이 아니라 `null`(미측정)이다.
            "cache_creation_total": accuracy.cache_creation_tokens_total,
            "cache_read_total": accuracy.cache_read_tokens_total,
        },
        #: 판정 호출 지연의 세트 집계 — 사람이 읽는 줄과 같은 원본이다. 판정에 실패한
        #: 픽스처의 경과도 분모에 들고, 한 건도 재지 않았으면 `measured_cases` 가 0 이다.
        "judge_latency": (
            None if accuracy.judge_latency is None else _span_aggregate_json(accuracy.judge_latency)
        ),
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
                "actual_verdict": (
                    None if outcome.actual_verdict is None else outcome.actual_verdict.value
                ),
                "expected_reasons": [reason.value for reason in outcome.expected_reasons],
                "actual_reasons": [reason.value for reason in outcome.actual_reasons],
                "claim_total": outcome.claim_total,
                "claim_verdict_matched": outcome.claim_verdict_matched,
                "contradiction_expected": outcome.contradiction_expected,
                "contradiction_matched": outcome.contradiction_matched,
                "contradiction_extra": outcome.contradiction_extra,
                "input_tokens": outcome.input_tokens,
                "output_tokens": outcome.output_tokens,
                "cache_creation_input_tokens": outcome.cache_creation_input_tokens,
                "cache_read_input_tokens": outcome.cache_read_input_tokens,
                #: 판정 호출에 흐른 벽시계(ms). 판정하지 못한 픽스처도 시간은 썼다.
                "elapsed_ms": outcome.elapsed_ms,
                "matched": outcome.reasons_matched,
                "error": outcome.error,
            }
            for outcome in accuracy.outcomes
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
        if stage == QUERY_REWRITE_STAGE:
            return {"rewritten": self._rewrite(user)}
        raise AssertionError(f"대역이 모르는 단계다: {stage!r}")

    @staticmethod
    def _rewrite(user: str) -> str:
        """**재작성하지 않는다** — 문의 원문을 그대로 돌려준다.

        이 대역이 하는 유일한 정직한 산출이다. 재작성의 값어치는 구어체를 문서체 어휘로
        옮기는 의미 변환인데, 대역은 어휘 규칙밖에 없어 그것을 흉내내면 실제 모델이 내지
        않는 질의로 검색 배관을 재게 된다 — `StubJudge` 가 모순 감지를 흉내내지 않는 것과
        같은 이유다. 원문과 같은 문자열은 픽스처 계약이 명시적으로 허용하는 산출이고,
        그때 수집기는 검색을 한 번만 돈다(합집합 경로는 단위 테스트가 덮는다).

        **호출 자체는 실제로 나간다** — 토큰이 검색 계열에 집계되고 단계가 기록에 남으므로,
        대역 완주가 "재작성 배선이 실제로 걸려 있는가"까지 확인한다.
        """
        return user.removeprefix("[문의]\n")

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
    """`llm.JsonCompletion` 과 같은 모양의 대역 산출 (llm 모듈을 import 하지 않기 위해).

    `transport_attempts` 는 대역이라 항상 1 이다 — 전송이 없으니 전송 수도 1회분으로 센다.
    모양을 맞추지 않으면 형식 루프가 이 값을 읽는 자리에서 대역만 터진다(실제로 그랬다).
    """

    data: Any
    input_tokens: int
    output_tokens: int
    transport_attempts: int = 1
    #: 대역은 **밖으로 나가지 않으므로 이 값이 0.0 이다** — "재지 않았다"가 아니라
    #: "밖으로 나간 시간이 0"이라는 측정값이다. 여기서 실제 시계를 재면 대역 산출이
    #: 비결정론이 되어 `--stub-llm` 실행이 재현되지 않는다(대역 결정론은 이 저장소의
    #: 계약이고 `tests/test_testing_doubles.py` 가 지킨다).
    elapsed_ms: float = 0.0
    #: 대역은 캐시를 재지 않는다 — 0 이 아니라 **미측정**이다. 모양을 맞추지 않으면 캐시
    #: 칸을 읽는 자리에서 대역만 터진다(경과 칸이 이미 겪은 사고다).
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
