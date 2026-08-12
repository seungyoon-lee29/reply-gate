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
    Claim,
    Draft,
    EscalationReason,
    Evidence,
    EvidenceSource,
    InquiryStatus,
    RejectReason,
    Verdict,
    is_policy_evidence_id,
)
from reply_gate.gate import DEFAULT_PII_PATTERNS, REASON_ORDER, evaluate_draft
from reply_gate.judge import L2_REJECT_REASONS
from reply_gate.llm import LLMCallError, LLMFormatError
from reply_gate.pipeline import (
    Judging,
    ProcessedInquiry,
    ReceiptError,
    accept_inquiry,
    new_inquiry_id,
)

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
    "StubGenerationClient",
    "TargetAssessment",
    "assess_targets",
    "build_report",
    "display_path",
    "load_golden_set",
    "load_judge_fixtures",
    "load_l1_fixtures",
    "measure_gate_accuracy",
    "measure_judge_accuracy",
    "measure_pipeline_agreement",
    "render_markdown",
    "report_to_json",
    "resolve_report_stem",
    "write_report",
]

_ROOT: Final = Path(__file__).resolve().parents[2]

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
    measurement2_is_real: bool,
    l2_enabled: bool,
    out_dir: Path,
) -> str:
    """리포트 이름을 확정한다 — 라이브 실측 산출물을 잃지 않게 막는 유일한 자리다.

    규칙은 두 겹이고 **둘 다 양방향**이다.

    **(1) 라이브 이름 ⇔ 실측**

    * 실측이 아닌 실행(측정 2 미실행·대역)은 라이브 이름을 쓸 수 없다. `--live` 를 줬어도
      키·DB 문제로 측정 2 가 미실행이면 비실측이므로 기본 이름은 `evaluation` 으로 떨어진다.
    * 실측(과금) 실행은 라이브 이름만 쓸 수 있다 — 실측 결과가 gitignore 되는 이름으로
      새어 나가 다음 기본 실행에 덮이는 것을 막는다.

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
    del live_requested  # 판정 근거는 "실측인가"이지 "실측을 요청했는가"가 아니다.
    series = LIVE_L2_REPORT_STEM if l2_enabled else LIVE_REPORT_STEM

    if requested is None:
        if not measurement2_is_real:
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

    if is_live_name and not measurement2_is_real:
        raise ReportStemError(
            f"거부: `{stem}` 은 라이브 실측 리포트 이름인데 이 실행은 실측이 아니다"
            f"(측정 2 미실행 또는 대역). 라이브 산출물을 덮어쓰면 문서가 인용하는 수치의 "
            f"근거가 사라진다. 다른 --report-stem 을 쓰거나 기본값(`{DEFAULT_REPORT_STEM}`)을 쓰라."
        )
    if measurement2_is_real and not stem.startswith(LIVE_REPORT_STEM):
        raise ReportStemError(
            f"거부: 실측(과금) 실행이 `{stem}` 에 쓰면 산출물이 gitignore 되어 다음 기본 "
            f"실행에 덮인다 — 실제로 한 번 그렇게 잃었다. `{series}` 로 시작하는 이름을 쓰라."
        )
    if measurement2_is_real and l2_enabled and not stem.startswith(LIVE_L2_REPORT_STEM):
        raise ReportStemError(
            f"거부: L2 켜짐 실측이 `{stem}` 에 쓰면 L2 꺼짐 기준선 계열과 섞인다 — 산출물만 "
            f"보고는 어느 쪽이 L2 를 포함해 잰 값인지 알 수 없게 된다. "
            f"`{LIVE_L2_REPORT_STEM}` 로 시작하는 이름을 쓰라 (예: "
            f"--report-stem {_next_free_live_stem(out_dir, prefix=LIVE_L2_REPORT_STEM)})."
        )
    if measurement2_is_real and not l2_enabled and is_l2_name:
        raise ReportStemError(
            f"거부: `{stem}` 은 L2 켜짐 실측 계열 이름인데 이 실행은 L2 가 꺼져 있다. "
            f"꺼짐 기준선은 `{LIVE_REPORT_STEM}` 계열에만 쓴다 — 계열이 뒤섞이면 두 실측을 "
            f"대조하는 근거가 사라진다."
        )
    if measurement2_is_real:
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

    @property
    def rejected_at_least_once(self) -> bool:
        """시도 중 **최소 1건**이 기각인가 — **층 무관**이다.

        종합 verdict 를 보므로 L1 기각도 L2 기각도 같은 자격으로 들어온다. 기각 재현율의
        정의가 층에 묶이면 L2 도입이 지표 정의를 바꾸게 되고, 도입 전후 비교가 깨진다.
        """
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
    #: 판정 계열은 생성·임베딩과 분리해서 센다. L2 켜짐 실측에서 이 계열을 빠뜨리면
    #: 건당 비용 지표가 판정 비용을 통째로 누락한다.
    judge_input_tokens_total: int
    judge_output_tokens_total: int
    status_counts: Mapping[str, int]
    escalation_counts: Mapping[str, int]
    outcomes: tuple[GoldenOutcome, ...]

    @property
    def match_rate(self) -> float | None:
        return None if self.total == 0 else self.matched / self.total

    @property
    def bait_reject_recall(self) -> float | None:
        """미끼(`reject_bait`) 문의의 기각 재현율 — 목표 없는 관측값(결정 0006·0008).

        분자의 정의는 **시도 중 최소 1건 기각, 층 무관**이다(L2 도입으로 바뀌지 않는다).
        분모는 라벨(`expect_reject`)이 아니라 범주다 — 라벨을 판정 규칙에 정렬하면서
        (결정 0008) 기각 요구는 해제했지만 관측은 유지하기 때문이다.
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
    def total_tokens_per_inquiry(self) -> float | None:
        """세 계열 합산 — 판정 계열을 빠뜨리면 L2 켜짐 실측의 건당 비용이 거짓이 된다."""
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
        if case.category == BAIT_CATEGORY
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
        judge_input_tokens_total=sum(outcome.judge_input_tokens for outcome in outcomes),
        judge_output_tokens_total=sum(outcome.judge_output_tokens for outcome in outcomes),
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
    )


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
#: 일치율은 확률 층이라 반복 실행마다 흔들린다 — 3회 실측 71.1%(70.0~73.3%) 위에
#: 75% 를 둔 것은 **지금 닿지 않는 목표**이고, 그 간극을 이번 사이클의 L2·임계값이 겨냥한다.
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
        assessments.append(
            TargetAssessment(target=target, value=value, met=met, verdict="달성" if met else "미달")
        )
    return tuple(assessments)


def build_report(
    *,
    conditions: RunConditions,
    gate_accuracy: GateAccuracy,
    pipeline: PipelineAgreement | SkippedMeasurement,
    judge_accuracy: JudgeAccuracy | SkippedMeasurement,
    retrieval_labels_path: Path | None = None,
) -> EvaluationReport:
    """리포트를 조립한다. **측정 3 도 명시해야 한다** — 기본값으로 비워 두면 미실행이
    조용히 "사유 없는 미실행"으로 찍힌다."""
    failure_attribution = _build_failure_attribution(
        pipeline=pipeline,
        retrieval_labels_path=retrieval_labels_path,
    )
    return EvaluationReport(
        conditions=conditions,
        gate_accuracy=gate_accuracy,
        pipeline=pipeline,
        judge_accuracy=judge_accuracy,
        failure_attribution=failure_attribution,
    )


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
            normal_behavior = zero_evidence_normal or l2_rejection_normal
            if zero_evidence_normal:
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
        f"- 유사도 임계값: {conditions.similarity_threshold} / top k: {conditions.top_k}",
        f"- L1 픽스처: {conditions.l1_fixture_count}건 (`{conditions.l1_fixtures_path}`)",
        f"- 골든셋: {conditions.golden_case_count}건 (`{conditions.golden_set_path}`)",
        f"- 판정 픽스처: {_count(conditions.judge_fixture_count)} "
        f"(`{conditions.judge_fixtures_path}`)",
        f"- OPENAI_API_KEY 설정 여부: {'설정됨' if conditions.api_key_present else '없음'}",
        f"- ANTHROPIC_API_KEY 설정 여부: "
        f"{'설정됨' if conditions.judge_api_key_present else '없음'}",
        "",
    ]
    lines.extend(_render_targets(report))
    lines.extend(_render_measurement_one(report.gate_accuracy))
    lines.extend(_render_measurement_two(report.pipeline, conditions))
    lines.extend(_render_failure_attribution(report.failure_attribution))
    lines.extend(_render_measurement_three(report.judge_accuracy, conditions))
    lines.append(_LIMITS)
    return "\n".join(lines)


def _render_targets(report: EvaluationReport) -> list[str]:
    """목표치 대비 표. **미측정·대역은 미달이 아니다** — 그렇게 적으면 리포트가 거짓말을 한다."""
    lines = [
        "## 목표치 대비",
        "",
        "2026-08-05 확정(`docs/tracking/decisions/0006-지표-목표치를-실측-뒤에-확정한다.md`).",
        "미측정 지표와 대역으로 낸 확률 층 수치는 판정하지 않는다.",
        "",
        "| 지표 | 목표 | 실측 | 판정 |",
        "| --- | --- | ---: | :---: |",
    ]
    for item in assess_targets(report):
        value_cell = "미측정" if item.value is None else _pct(item.value)
        verdict_cell = "**미달**" if item.verdict == "미달" else item.verdict
        lines.append(
            f"| {item.target.label} | {item.target.describe()} | {value_cell} | {verdict_cell} |"
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
            "— 목표 없는 관측값이다(결정 0006·0008).",
            f"- 정상 PII 에코 감시 케이스: {pipeline.forbidden_watch_total}건 중 "
            f"금지 사유 발화 {pipeline.forbidden_violations}건",
            f"- 지연 p50: {_int(pipeline.latency_p50_ms)} ms / "
            f"p95: {_int(pipeline.latency_p95_ms)} ms "
            "(파이프라인 `run` 의 벽시계 시간 — 처리 기록 저장은 포함하지 않는다)",
            "",
            "### 문의 1건당 토큰 (생성·임베딩·판정 구분)",
            "",
            "provider 와 단가가 다른 계열을 합산하면 건당 비용 지표가 무너진다 — 세 계열은",
            "끝까지 분리해서 센다. L2 미실행이면 판정 계열은 0 이다.",
            "",
            "| 계열 | 합계 | 건당 |",
            "| --- | ---: | ---: |",
            *_token_rows(pipeline, conditions),
            "",
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


def _token_rows(pipeline: PipelineAgreement, conditions: RunConditions) -> list[str]:
    """토큰 표의 본문 — 계열 셋(생성·임베딩·판정)을 끝까지 분리해서 적는다.

    대역 실행에서는 **판정 행에 `(대역)` 을 붙인다**: `--stub-llm` 은 판정자까지 대역으로
    갈아 끼우므로 이 행의 값은 대역의 휴리스틱 산출이고 합산에도 그대로 들어간다.
    """
    # 대역 실행이면 판정 계열도 대역이다 — 표만 보고 실제 판정 비용으로 읽으면 안 된다.
    judge_mark = "" if conditions.measurement2_is_real else " (대역)"

    def per_inquiry(total: int) -> float | None:
        return None if pipeline.total == 0 else total / pipeline.total

    generation_total = pipeline.input_tokens_total + pipeline.output_tokens_total
    judge_total = pipeline.judge_input_tokens_total + pipeline.judge_output_tokens_total
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
    for item in attribution.cases:
        relevant = ", ".join(f"`{value}`" for value in item.relevant_evidence_ids) or "없음"
        adopted = ", ".join(f"`{value}`" for value in item.adopted_evidence_ids) or "없음"
        route = ""
        if item.classification == "expected_no_answer":
            if item.normal_behavior is False:
                route = f" / 비정상: {item.anomaly_reason}"
            elif item.normal_behavior_path == "retrieval_zero_evidence":
                route = " / 검색 0건 종료"
            elif item.normal_behavior_path == "l2_rejected_with_evidence":
                route = " / 근거 채택 후 L2 검출"
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


def _grand_total(pipeline: PipelineAgreement) -> int:
    """세 계열 합산 — 판정 계열이 빠지면 L2 켜짐 실측의 건당 비용이 실제보다 작아진다."""
    return (
        pipeline.input_tokens_total
        + pipeline.output_tokens_total
        + pipeline.embedding_tokens_total
        + pipeline.judge_input_tokens_total
        + pipeline.judge_output_tokens_total
    )


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
            "judge": conditions.judge,
            "l2_enabled": conditions.l2_enabled,
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
            }
            for item in attribution.cases
        ],
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
        "forbidden_watch_total": pipeline.forbidden_watch_total,
        "forbidden_violations": pipeline.forbidden_violations,
        "latency_p50_ms": pipeline.latency_p50_ms,
        "latency_p95_ms": pipeline.latency_p95_ms,
        "tokens": {
            "generation_input_total": pipeline.input_tokens_total,
            "generation_output_total": pipeline.output_tokens_total,
            "embedding_total": pipeline.embedding_tokens_total,
            "judge_input_total": pipeline.judge_input_tokens_total,
            "judge_output_total": pipeline.judge_output_tokens_total,
            "generation_per_inquiry": pipeline.generation_tokens_per_inquiry,
            "embedding_per_inquiry": pipeline.embedding_tokens_per_inquiry,
            "judge_per_inquiry": pipeline.judge_tokens_per_inquiry,
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
                "adopted_evidence_ids": list(outcome.adopted_evidence_ids),
                "latency_ms": outcome.latency_ms,
                "input_tokens": outcome.input_tokens,
                "output_tokens": outcome.output_tokens,
                "embedding_tokens": outcome.embedding_tokens,
                "judge_input_tokens": outcome.judge_input_tokens,
                "judge_output_tokens": outcome.judge_output_tokens,
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
        },
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
