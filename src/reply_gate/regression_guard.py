"""회귀 가드 — 이중 기준선 대조·조건 지문·비악화 판정.

**이 모듈이 존재하는 이유는 합산 지표가 못 잡은 회귀 하나다.** 커밋된 라이브 산출물이
그것을 그대로 보여준다: `reports/evaluation-live-l2-1` 은 G18 을
`reject_reasons=["contradictory_evidence"]` 로 끝냈고 `-l2-4`·`-l2-6` 은
`reject_reasons=[]` 에 채택 근거가 `["policy:refund:2-6"]` 단독인데 **6회 전부
`matched=True`** 다. 합산 일치율도, 기존 비악화 대조도, 픽스처 기반 측정 3 도 그 소멸을
보지 못했다(`docs/tracking/findings.md` 18번). 그래서 **가드의 술어를 `matched` 하나로
되돌리면 안 된다** — 근거 부분 손실 검사가 같은 자격의 판정 조건이다.

구조는 셋이다.

* **조건 지문**(`ConditionFingerprint`) — 두 실측이 애초에 대조 가능한지를 가른다.
  불일치의 처리는 **선언 여부로 갈린다**: 실행이 "이번에 의도적으로 바꾼 축"으로 선언한
  항목이면 대조를 **진행**하고 차이 목록을 병기하고, 선언 없이 달라졌으면 그 줄은
  **"대조 불가 + 어긋난 항목"** 이다. 0 이나 "통과"로 채우지 않는다.
* **이중 기준선**(`RegressionGuard`) — **승격 기준선**이 구속하고 **직전 라이브**가 경보한다.
  두 줄이 상반되면 승격이 이긴다. 승격은 **사람이 참조 파일을 바꾸는 것**뿐이고 이 모듈에는
  참조 파일을 쓰는 경로가 없다(구조 테스트가 검사한다).
* **비악화 판정**(`compare_run_sets`) — 케이스 단위 두 겹. ① 기준선에서 3회 모두 일치한
  케이스가 새 실측 3회 중 2회 이상 일치. ② 기준선이 채택했던 **정답 근거 ID** 가 새 실측의
  케이스별 다수결에서 빠지면 미달. 감시 모집단은 "기준선 3/3" 으로 한정하지 않는다.

입력은 리포트 JSON **그대로**다 — 이 모듈은 `evaluation` 을 import 하지 않는다. 커밋된
산출물을 사후 편집하지 않기 때문에 **옛 산출물에는 새 필드가 없고**, 그때는 없다고 적을 뿐
0 으로도 통과로도 채우지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

__all__ = [
    "DEFAULT_PROMOTED_BASELINE_PATH",
    "FINGERPRINT_FIELDS",
    "LIVE_SERIES_PREFIX",
    "PAIRED_FINGERPRINT_FIELDS",
    "RUN_SET_SIZE",
    "BaselineLine",
    "BaselineNotRegistered",
    "CaseObservation",
    "ConditionFingerprint",
    "EvidenceLoss",
    "FieldDifference",
    "FingerprintComparison",
    "GuardUnavailable",
    "MatchFinding",
    "PromotedBaseline",
    "RegressionGuard",
    "RunSet",
    "RunSummary",
    "build_regression_guard",
    "content_digest",
    "discover_recent_live_set",
    "guard_to_json",
    "load_promoted_baseline",
    "load_run_summary",
    "render_guard_section",
    "run_summary_from_payload",
]

_ROOT: Final = Path(__file__).resolve().parents[2]

#: 승격 기준선 참조 파일. **사람만 바꾼다** — 이 파일을 바꾸는 것이 곧 승격이고 재등재다.
DEFAULT_PROMOTED_BASELINE_PATH: Final = _ROOT / "data" / "promoted_baseline.json"

#: 한 세트의 실측 횟수. 판정 규칙("3회 중 2회 이상")의 분모다.
RUN_SET_SIZE: Final = 3

#: 라이브 리포트 이름 접두 — 가드가 자동 탐색하는 대상.
LIVE_SERIES_PREFIX: Final = "evaluation-live"

#: 대조 가능성을 결정하는 지문 항목. **이 목록은 하한이지 상한이 아니다** —
#: 실행 조건에 실린 키는 목록 밖이어도 같은 규칙으로 대조된다. 그래서 새 축(예: 기권 게이트
#: 통계량·τ)을 붙이는 작업은 **값만 추가**하면 되고 이 모듈을 다시 열지 않아도 된다.
FINGERPRINT_FIELDS: Final[tuple[str, ...]] = (
    "label_version",
    "acceptance_cut",
    "abstention_gate_statistic",
    "abstention_tau",
    "query_rewrite",
    "embedding_model",
    "embedding_dimensions",
    "top_k",
    "generation_model",
    "judge_model",
    "judge_effort",
    "judge_prompt_version",
    "judge_fixture_version",
    "judge_prompt_caching",
    "measurement_scope",
)

#: **짝으로 읽는 지문 항목.** τ 는 임베딩 모델을 넘어 이전되지 않았다(손계산) — 임베딩
#: 모델이 다른 두 실측은 τ 값이 같아도 같은 조건이 아니다. 그래서 τ 의 비교값에 모델을
#: 함께 싣는다. 짝의 한쪽이 선언된 실험 변인이면 그 차이는 선언된 것으로 읽는다.
PAIRED_FINGERPRINT_FIELDS: Final[Mapping[str, str]] = {"abstention_tau": "embedding_model"}

#: 미상 표기. 0 이나 "통과"로 채우지 않는다는 규칙의 문자열 형태다.
UNKNOWN: Final = "미상"


def content_digest(path: Path, *, prefix: str = "") -> str | None:
    """파일 내용의 짧은 지문. 읽지 못하면 `None`(미상)이다 — 0 으로 채우지 않는다.

    프롬프트·픽스처·라벨처럼 **바뀌면 대조가 깨지는 입력**의 버전을 손으로 적지 않고
    내용에서 끌어낸다. 손으로 적는 버전 문자열은 갱신을 잊는 순간 조용한 드리프트가 된다.
    """
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    return f"{prefix}{hashlib.sha256(payload).hexdigest()[:12]}"


def text_digest(text: str, *, prefix: str = "") -> str:
    """문자열의 짧은 지문 — 프롬프트 상수처럼 파일이 아닌 입력에 쓴다."""
    return f"{prefix}{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"


# ══ 조건 지문 ═══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FieldDifference:
    """지문 항목 1개의 차이 — 어긋난 항목을 이름으로 찍기 위한 자료형."""

    field: str
    baseline: str
    candidate: str

    def describe(self) -> str:
        return f"{self.field}: 기준선 `{self.baseline}` → 이번 `{self.candidate}`"

    def describe_as(self, *, left: str, right: str) -> str:
        """대조 문맥이 "기준선 대 이번"이 아닐 때 쓴다 — 등재 정합성 검사가 그렇다."""
        return f"{self.field}: {left} `{self.candidate}` / {right} `{self.baseline}`"


@dataclass(frozen=True)
class FingerprintComparison:
    """지문 대조 결과. **선언 여부가 대조 가능성을 가른다.**"""

    #: 이번 실행이 "의도적으로 바꾼 축"으로 선언한 차이. 대조는 진행하고 목록을 병기한다.
    declared_differences: tuple[FieldDifference, ...] = ()
    #: 선언 없이 달라진 항목. 하나라도 있으면 그 줄은 대조 불가다.
    undeclared_differences: tuple[FieldDifference, ...] = ()
    #: 한쪽에 값이 없는 항목. 같다고도 다르다고도 적지 않고 **미상**으로 남긴다.
    unknown_fields: tuple[str, ...] = ()

    @property
    def comparable(self) -> bool:
        """선언되지 않은 불일치가 없을 때만 대조 가능하다."""
        return not self.undeclared_differences


@dataclass(frozen=True)
class ConditionFingerprint:
    """실행 조건 중 **대조 가능성을 결정하는** 부분.

    값은 전부 문자열이거나 `None`(미상)이다. 필수 항목(`FINGERPRINT_FIELDS`)은 값이 없어도
    키가 존재하고 `None` 을 든다 — 항목이 통째로 사라져 "같다"로 읽히는 것을 막는다.
    """

    values: Mapping[str, str | None]
    #: 이번 실행이 선언한 실험 변인 이름들.
    declared: tuple[str, ...] = ()

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Any] | None,
        *,
        declared: Sequence[str] = (),
    ) -> ConditionFingerprint:
        normalized: dict[str, str | None] = dict.fromkeys(FINGERPRINT_FIELDS)
        for key, value in (values or {}).items():
            normalized[str(key)] = None if value is None else _as_text(value)
        return cls(values=normalized, declared=tuple(declared))

    def effective(self, name: str) -> tuple[str | None, ...]:
        """비교에 쓰는 값. 짝이 있는 항목은 짝의 값을 함께 싣는다."""
        partner = PAIRED_FINGERPRINT_FIELDS.get(name)
        if partner is None:
            return (self.values.get(name),)
        return (self.values.get(name), self.values.get(partner))

    def describe(self, name: str) -> str:
        value = self.values.get(name)
        return UNKNOWN if value is None else value

    def compare(self, baseline: ConditionFingerprint) -> FingerprintComparison:
        """기준선과 대조한다. `self` 가 이번 실행이고 선언 목록도 이번 실행의 것이다."""
        declared: list[FieldDifference] = []
        undeclared: list[FieldDifference] = []
        unknown: list[str] = []
        for name in _ordered_keys(self.values, baseline.values):
            mine = self.values.get(name)
            theirs = baseline.values.get(name)
            partner = PAIRED_FINGERPRINT_FIELDS.get(name)
            if mine is None or theirs is None:
                unknown.append(name)
                continue
            if partner is not None and (
                self.values.get(partner) is None or baseline.values.get(partner) is None
            ):
                # 짝의 한쪽이 미상이면 τ 만 비교해도 같은 조건이라고 말할 수 없다.
                unknown.append(name)
                continue
            if self.effective(name) == baseline.effective(name):
                continue
            difference = FieldDifference(field=name, baseline=theirs, candidate=mine)
            is_declared = name in self.declared or (
                partner is not None and partner in self.declared
            )
            (declared if is_declared else undeclared).append(difference)
        return FingerprintComparison(
            declared_differences=tuple(declared),
            undeclared_differences=tuple(undeclared),
            unknown_fields=tuple(unknown),
        )


def _ordered_keys(*mappings: Mapping[str, Any]) -> tuple[str, ...]:
    """필수 항목을 먼저, 그 뒤에 추가된 항목을 발견 순서로. 출력 순서를 고정한다."""
    seen: list[str] = [name for name in FINGERPRINT_FIELDS]
    for mapping in mappings:
        for name in mapping:
            if name not in seen:
                seen.append(name)
    return tuple(seen)


def _as_text(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        # 0.30 과 0.3 이 다른 조건으로 읽히지 않게 정규화한다.
        return f"{value:g}"
    return str(value)


# ══ 실측 1회 요약 — 리포트 JSON 을 그대로 읽는다 ════════════════════════════


@dataclass(frozen=True)
class CaseObservation:
    """케이스 1건의 실측 1회 관측.

    `relevant_evidence_ids` 가 `None` 이면 **귀인 절에 실리지 않은 케이스**다 — 귀인은
    정답 조항을 전부 채택하고 정상 답변한 케이스를 싣지 않으므로, 부재는 곧 "전부 채택"이다.
    `evidence_unknown` 은 다르다: 귀인 절 자체가 미산출이라 **채택 여부를 알 수 없다**는 뜻이고,
    그때는 손실로도 보존으로도 세지 않는다.
    """

    case_id: str
    matched: bool
    relevant_evidence_ids: frozenset[str] | None
    missing_relevant_evidence_ids: frozenset[str] | None
    evidence_unknown: bool

    def accepted_relevant(self, relevant_reference: frozenset[str]) -> frozenset[str] | None:
        """이 실측에서 채택된 **정답** 근거. 알 수 없으면 `None`."""
        if self.evidence_unknown:
            return None
        if self.relevant_evidence_ids is None:
            # 귀인 절에 없다 = 정답 근거를 전부 채택하고 정상 답변했다.
            return relevant_reference
        missing = self.missing_relevant_evidence_ids
        if missing is None:
            return None
        return frozenset(self.relevant_evidence_ids) - missing


@dataclass(frozen=True)
class RunSummary:
    """리포트 1건에서 가드가 쓰는 부분만 추린 요약."""

    stem: str
    source: str
    started_at: str
    billed: bool
    l2_enabled: bool
    fingerprint: ConditionFingerprint
    cases: Mapping[str, CaseObservation]
    attribution_computed: bool
    attribution_reason: str | None
    detection_rate: float | None
    false_positive_rate: float | None
    match_rate: float | None
    measurement3_executed: bool
    measurement3_detection_rate: float | None


class RunSummaryError(ValueError):
    """리포트 JSON 을 요약으로 읽지 못했다 — 조용히 빈 요약으로 대체하지 않는다."""


def run_summary_from_payload(payload: Mapping[str, Any], *, stem: str, source: str) -> RunSummary:
    """리포트 JSON 1건을 요약으로 읽는다. **옛 산출물의 없는 필드는 미상이다.**"""
    if not isinstance(payload, Mapping):  # pragma: no cover - 방어
        raise RunSummaryError(f"{source}: 리포트 JSON 이 객체가 아니다")
    conditions = _mapping(payload.get("conditions"))
    measurement2 = _mapping(payload.get("measurement_2_pipeline_agreement"))
    attribution = _mapping(payload.get("failure_attribution"))
    measurement1 = _mapping(payload.get("measurement_1_l1_gate_accuracy"))
    measurement3 = _mapping(payload.get("measurement_3_l2_judge_accuracy"))

    attribution_computed = bool(attribution.get("computed"))
    attributed: dict[str, Mapping[str, Any]] = {}
    if attribution_computed:
        for row in attribution.get("cases") or ():
            item = _mapping(row)
            case_id = item.get("case_id")
            if isinstance(case_id, str):
                attributed[case_id] = item

    cases: dict[str, CaseObservation] = {}
    for row in measurement2.get("outcomes") or ():
        outcome = _mapping(row)
        case_id = outcome.get("case_id")
        if not isinstance(case_id, str):
            continue
        entry = attributed.get(case_id)
        relevant: frozenset[str] | None = None
        missing: frozenset[str] | None = None
        if entry is not None:
            relevant = frozenset(_str_list(entry.get("relevant_evidence_ids")))
            raw_missing = entry.get("missing_relevant_evidence_ids")
            missing = None if raw_missing is None else frozenset(_str_list(raw_missing))
        cases[case_id] = CaseObservation(
            case_id=case_id,
            matched=bool(outcome.get("matched")),
            relevant_evidence_ids=relevant,
            missing_relevant_evidence_ids=missing,
            evidence_unknown=not attribution_computed,
        )

    return RunSummary(
        stem=stem,
        source=source,
        started_at=str(conditions.get("started_at") or ""),
        # `billed` 는 새 필드다. 옛 산출물은 측정 2 실측 여부가 그 자리를 대신한다 —
        # 그 시절에는 그 둘이 같은 값이었다.
        billed=bool(conditions.get("billed", conditions.get("measurement2_is_real", False))),
        l2_enabled=bool(conditions.get("l2_enabled", False)),
        fingerprint=fingerprint_from_conditions(conditions),
        cases=cases,
        attribution_computed=attribution_computed,
        attribution_reason=_optional_text(attribution.get("reason")),
        detection_rate=_optional_float(measurement1.get("detection_rate")),
        false_positive_rate=_optional_float(measurement1.get("false_positive_rate")),
        match_rate=_optional_float(measurement2.get("match_rate")),
        measurement3_executed=bool(measurement3.get("executed")),
        measurement3_detection_rate=_optional_float(measurement3.get("detection_rate")),
    )


def load_run_summary(path: Path) -> RunSummary:
    """리포트 JSON 파일을 요약으로 읽는다."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunSummaryError(f"{path.name}: 리포트 JSON 을 읽지 못했다: {exc}") from exc
    return run_summary_from_payload(payload, stem=path.stem, source=_display(path))


def fingerprint_from_conditions(conditions: Mapping[str, Any]) -> ConditionFingerprint:
    """실행 조건에서 지문을 끌어낸다.

    **새 필드(`condition_fingerprint`)가 없는 옛 산출물도 읽힌다** — 그 시절에도 있던
    항목(컷·`top_k`·차원·검색 전략)은 파생으로 채우고 나머지는 미상으로 남는다. 파생값은
    명시 지문이 있으면 그것에 덮인다(명시가 항상 이긴다).
    """
    derived: dict[str, Any] = {
        "acceptance_cut": conditions.get("similarity_threshold"),
        "top_k": conditions.get("top_k"),
        "embedding_dimensions": conditions.get("embedding_dimensions"),
        "query_rewrite": _rewrite_flag(conditions.get("retrieval_strategy")),
        # **모델은 사람이 읽는 설명이 아니라 벌거벗은 id 로 비교한다.** 옛 산출물의 설명
        # 문자열(예: "OpenAI `gpt-5.6-terra` (effort=기본값)")을 그대로 쓰면 새 실행의 명시
        # 지문(`gpt-5.6-terra`)과 매번 어긋나, **아무것도 바뀌지 않았는데** 커밋된 기준선마다
        # "대조 불가"가 뜬다. 지문 규칙이 대조를 죽이면 지문이 스스로를 무력화한 것이다.
        "generation_model": _bare_model_id(conditions.get("generation")),
        "generation_effort": _effort(conditions.get("generation")),
        "embedding_model": _bare_model_id(conditions.get("embedding")),
        "judge_model": _bare_model_id(conditions.get("judge")),
        "judge_effort": _effort(conditions.get("judge")),
        "measurement_scope": conditions.get("measurement_scope"),
    }
    derived = {key: value for key, value in derived.items() if value is not None}
    explicit = conditions.get("condition_fingerprint")
    if isinstance(explicit, Mapping):
        derived.update({str(key): value for key, value in explicit.items()})
    declared = tuple(_str_list(conditions.get("declared_experiment_fields")))
    return ConditionFingerprint.from_values(derived, declared=declared)


#: 실행 조건의 모델 설명은 모델 id 를 백틱으로 감싼다 — 그 한 조각만 지문에 쓴다.
_BACKTICKED: Final = re.compile(r"`([^`]+)`")
#: `(effort=...)` 꼬리. 판정 호출의 thinking/effort 설정이 여기 실려 있다.
_EFFORT: Final = re.compile(r"effort=([^)]*)")


def _bare_model_id(label: Any) -> str | None:
    """모델 설명에서 id 만 꺼낸다. 꺼낼 것이 없으면 `None`(미상)이다 — 추측하지 않는다."""
    if not isinstance(label, str):
        return None
    match = _BACKTICKED.search(label)
    return match.group(1) if match else None


def _effort(label: Any) -> str | None:
    """모델 설명에서 effort 설정만 꺼낸다. 없으면 `None`(미상)."""
    if not isinstance(label, str):
        return None
    match = _EFFORT.search(label)
    return match.group(1).strip() if match else None


def _rewrite_flag(strategy: Any) -> str | None:
    if not isinstance(strategy, str) or not strategy:
        return None
    return "on" if "rewrite" in strategy else "off"


# ══ 실측 세트 ═══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RunSet:
    """같은 조건으로 반복한 실측 묶음. 판정 단위는 개별 실행이 아니라 이 세트다."""

    label: str
    runs: tuple[RunSummary, ...]

    @property
    def stems(self) -> tuple[str, ...]:
        return tuple(run.stem for run in self.runs)

    @property
    def fingerprint(self) -> ConditionFingerprint:
        return self.runs[0].fingerprint if self.runs else ConditionFingerprint.from_values({})

    def case_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for run in self.runs:
            for case_id in run.cases:
                if case_id not in seen:
                    seen.append(case_id)
        return tuple(sorted(seen))


def _collect_live_runs(
    reports_dir: Path, *, l2_enabled: bool, exclude: Iterable[str] = ()
) -> list[RunSummary]:
    """`reports/` 의 라이브 산출물을 최신 순으로 읽는다. 읽지 못한 파일은 건너뛴다."""
    skipped = set(exclude)
    runs: list[RunSummary] = []
    if not reports_dir.is_dir():
        return runs
    for path in sorted(reports_dir.glob(f"{LIVE_SERIES_PREFIX}*.json")):
        if path.stem in skipped:
            continue
        try:
            summary = load_run_summary(path)
        except RunSummaryError:
            # 읽히지 않는 산출물은 대조에서 빠질 뿐이다 — 리포트 생성을 죽이지 않는다.
            continue
        if summary.billed and summary.l2_enabled == l2_enabled:
            runs.append(summary)
    runs.sort(key=lambda run: (run.started_at, run.stem), reverse=True)
    return runs


def assemble_candidate_set(
    *,
    current: RunSummary,
    reports_dir: Path,
    size: int = RUN_SET_SIZE,
    exclude: Iterable[str] = (),
) -> RunSet:
    """이번 실측 세트를 모은다 — 현재 실행 + **지문이 같은** 직전 실행들.

    승격 기준선으로 등재된 산출물은 제외한다. 그것을 이번 세트에 섞으면 기준선이 자기
    자신과 대조되어 판정이 자기추인이 된다.
    """
    skipped = {current.stem, *exclude}
    runs = [current]
    for summary in _collect_live_runs(reports_dir, l2_enabled=current.l2_enabled, exclude=skipped):
        if len(runs) >= size:
            break
        comparison = summary.fingerprint.compare(current.fingerprint)
        # 지문이 **똑같은** 실행만 한 세트다. 선언된 차이도 여기서는 다른 조건이다 —
        # 선언은 기준선과의 대조를 진행시키는 장치이지 세트를 섞는 장치가 아니다.
        if comparison.comparable and not comparison.declared_differences:
            runs.append(summary)
    runs.sort(key=lambda run: (run.started_at, run.stem))
    return RunSet(label="이번 실측", runs=tuple(runs))


def discover_recent_live_set(
    *,
    reports_dir: Path,
    l2_enabled: bool,
    size: int = RUN_SET_SIZE,
    exclude: Iterable[str] = (),
) -> RunSet | None:
    """직전 라이브 세트를 자동 탐색한다 — 가장 최근 실행과 **지문이 같은** 것들."""
    runs = _collect_live_runs(reports_dir, l2_enabled=l2_enabled, exclude=exclude)
    if not runs:
        return None
    head = runs[0]
    grouped = [head]
    for summary in runs[1:]:
        if len(grouped) >= size:
            break
        comparison = summary.fingerprint.compare(head.fingerprint)
        if comparison.comparable and not comparison.declared_differences:
            grouped.append(summary)
    grouped.sort(key=lambda run: (run.started_at, run.stem))
    return RunSet(label="직전 라이브", runs=tuple(grouped))


# ══ 승격 기준선 참조 ════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PromotedBaseline:
    """사람이 명시적으로 승격한 기준선. **이 값을 만드는 코드 경로는 없다.**"""

    path: str
    promoted_at: str
    promoted_by: str
    reason: str
    #: 재등재 이력 — 이 승격이 대신한 직전 승격의 리포트 스템들.
    supersedes: tuple[str, ...]
    report_stems: tuple[str, ...]
    fingerprint: ConditionFingerprint

    @property
    def repromotion(self) -> bool:
        return bool(self.supersedes)


@dataclass(frozen=True)
class BaselineNotRegistered:
    """승격 기준선이 등재되지 않았다 — **0 이나 "통과"로 채우지 않는다.**"""

    path: str
    reason: str


def load_promoted_baseline(
    path: Path = DEFAULT_PROMOTED_BASELINE_PATH,
) -> PromotedBaseline | BaselineNotRegistered:
    """승격 참조 파일을 읽는다. 읽기 전용이다 — 이 모듈은 이 파일을 쓰지 않는다."""
    display = _display(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return BaselineNotRegistered(path=display, reason=f"승격 참조 파일이 없다: {display}")
    except (OSError, ValueError) as exc:
        return BaselineNotRegistered(
            path=display, reason=f"승격 참조 파일을 읽지 못했다({type(exc).__name__}): {exc}"
        )
    if not isinstance(raw, Mapping):
        return BaselineNotRegistered(path=display, reason="승격 참조 파일이 객체가 아니다")

    promotion = raw.get("promotion")
    stems = tuple(_str_list(raw.get("report_stems")))
    if not isinstance(promotion, Mapping) or not stems:
        return BaselineNotRegistered(
            path=display,
            reason=(
                "승격 기준선이 등재되지 않았다 — `promotion` 과 `report_stems` 가 채워져야 "
                "한다. 승격은 사람이 이 파일을 채우는 것뿐이고 하네스가 대신 채우지 않는다."
            ),
        )
    missing = [key for key in ("promoted_at", "promoted_by", "reason") if not promotion.get(key)]
    if missing:
        return BaselineNotRegistered(
            path=display,
            reason=(
                "승격 기록이 불완전하다 — 빠진 항목: " + ", ".join(missing) + ". "
                "누가 언제 무엇을 근거로 승격했는지가 없으면 구속 판정의 출처가 사라진다."
            ),
        )
    return PromotedBaseline(
        path=display,
        promoted_at=str(promotion["promoted_at"]),
        promoted_by=str(promotion["promoted_by"]),
        reason=str(promotion["reason"]),
        supersedes=tuple(_str_list(promotion.get("supersedes"))),
        report_stems=stems,
        fingerprint=ConditionFingerprint.from_values(raw.get("fingerprint")),
    )


# ══ 비악화 판정 ═════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MatchFinding:
    """케이스 1건의 일치 횟수 변화."""

    case_id: str
    baseline_matched: int
    baseline_runs: int
    candidate_matched: int
    candidate_runs: int

    def describe(self) -> str:
        return (
            f"`{self.case_id}` — 기준선 {self.baseline_matched}/{self.baseline_runs} → "
            f"이번 {self.candidate_matched}/{self.candidate_runs}"
        )


@dataclass(frozen=True)
class EvidenceLoss:
    """기준선이 채택했던 정답 근거가 새 실측의 다수결에서 빠졌다."""

    case_id: str
    dropped_evidence_ids: tuple[str, ...]
    baseline_accepted_ids: tuple[str, ...]
    candidate_accepted_ids: tuple[str, ...]

    def describe(self) -> str:
        dropped = ", ".join(f"`{value}`" for value in self.dropped_evidence_ids)
        return f"`{self.case_id}` — 빠진 정답 근거: {dropped}"


@dataclass(frozen=True)
class BaselineLine:
    """기준선 한 줄. 역할이 **구속**인지 **경보**인지가 판정 발동을 가른다."""

    label: str
    role: str
    verdict: str
    verdict_reason: str
    baseline_stems: tuple[str, ...] = ()
    baseline_source: str = ""
    fingerprint: FingerprintComparison = field(default_factory=FingerprintComparison)
    match_shortfalls: tuple[MatchFinding, ...] = ()
    match_collapses: tuple[MatchFinding, ...] = ()
    match_decreases: tuple[MatchFinding, ...] = ()
    evidence_losses: tuple[EvidenceLoss, ...] = ()
    measurement1_changes: tuple[str, ...] = ()
    unknown_notes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


#: 판정 문면. 리포트·JSON·테스트가 같은 문자열을 쓴다.
VERDICT_PASS: Final = "통과"
VERDICT_FAIL: Final = "미달"
VERDICT_HELD: Final = "보류"
VERDICT_INCOMPARABLE: Final = "대조 불가"
VERDICT_NOT_REGISTERED: Final = "기준선 미등재"
VERDICT_NO_BASELINE: Final = "직전 라이브 없음"

ROLE_BINDING: Final = "구속"
ROLE_ALERT: Final = "경보"


def _majority(count: int) -> int:
    """다수결 하한. 3회면 2회, 2회면 2회, 1회면 1회다."""
    return count // 2 + 1


def compare_run_sets(
    *,
    baseline: RunSet,
    candidate: RunSet,
    label: str,
    role: str,
    baseline_source: str = "",
    expected_run_count: int = RUN_SET_SIZE,
) -> BaselineLine:
    """비악화 판정 두 겹 + 지문 대조 + 측정 1 무변경. **측정 3 은 대상이 아니다.**"""
    comparison = candidate.fingerprint.compare(baseline.fingerprint)
    notes: list[str] = []
    if comparison.declared_differences:
        notes.append(
            "선언된 실험 변인이 있어 대조를 진행한다 — 차이: "
            + " · ".join(item.describe() for item in comparison.declared_differences)
        )
    notes.append(_measurement3_note(baseline=baseline, candidate=candidate))
    notes.append("합산 일치율은 하한 경보선이고 사이클 판정은 케이스 단위가 맡는다.")

    if not comparison.comparable:
        return BaselineLine(
            label=label,
            role=role,
            verdict=VERDICT_INCOMPARABLE,
            verdict_reason=(
                "선언되지 않은 조건 불일치 — 어긋난 항목: "
                + " · ".join(item.describe() for item in comparison.undeclared_differences)
            ),
            baseline_stems=baseline.stems,
            baseline_source=baseline_source,
            fingerprint=comparison,
            notes=tuple(notes),
        )

    shortfalls: list[MatchFinding] = []
    collapses: list[MatchFinding] = []
    decreases: list[MatchFinding] = []
    losses: list[EvidenceLoss] = []
    unknown: list[str] = []
    unobserved: list[str] = []

    baseline_runs = len(baseline.runs)
    candidate_runs = len(candidate.runs)
    #: 두 세트에 **공통으로** 실린 케이스 수. 0 이면 케이스 단위 판정이 돌지 않은 것이다.
    compared_cases = 0

    # 근거 부분 손실 검사의 입력은 귀인 절이다. 커밋된 옛 산출물에는 그 절이 없고,
    # 없는 것을 "손실 0건"으로 적으면 이 사이클을 촉발한 회귀가 그대로 통과한다.
    # **없다고 적을 뿐 0 으로도 통과로도 채우지 않는다.**
    baseline_evidence_known = any(run.attribution_computed for run in baseline.runs)
    candidate_evidence_known = any(run.attribution_computed for run in candidate.runs)
    if not baseline_evidence_known:
        unknown.append(
            "기준선 산출물에 귀인 절이 없다(미산출) — 근거 부분 손실을 판정하지 않는다. "
            "새 필드는 새 실행부터이므로 기준선을 다시 등재하기 전까지 이 겹은 비어 있다."
        )
    if not candidate_evidence_known:
        unknown.append("이번 실측에 귀인 절이 없다(미산출) — 근거 부분 손실을 판정하지 않는다.")
    for case_id in sorted(set(baseline.case_ids()) | set(candidate.case_ids())):
        baseline_seen = [run.cases[case_id] for run in baseline.runs if case_id in run.cases]
        candidate_seen = [run.cases[case_id] for run in candidate.runs if case_id in run.cases]
        if not baseline_seen:
            unknown.append(f"`{case_id}` 은 기준선 산출물에 없다 — 대조하지 않는다")
            continue
        if not candidate_seen:
            unknown.append(f"`{case_id}` 이 이번 실측에 없다 — 대조하지 않는다")
            continue

        compared_cases += 1
        baseline_matched = sum(1 for item in baseline_seen if item.matched)
        candidate_matched = sum(1 for item in candidate_seen if item.matched)
        finding = MatchFinding(
            case_id=case_id,
            baseline_matched=baseline_matched,
            baseline_runs=len(baseline_seen),
            candidate_matched=candidate_matched,
            candidate_runs=len(candidate_seen),
        )
        # ① 기준선 전회 일치 → 새 실측 다수결 일치.
        if baseline_matched == len(baseline_seen) and candidate_matched < _majority(
            len(candidate_seen)
        ):
            shortfalls.append(finding)
        # ③ 감시 모집단은 "기준선 3/3" 으로 한정하지 않는다 — 붕괴는 미달, 감소는 경보.
        elif baseline_matched > 0 and candidate_matched == 0:
            collapses.append(finding)
        elif candidate_matched < baseline_matched:
            decreases.append(finding)

        if baseline_evidence_known and candidate_evidence_known:
            check = _evidence_loss(
                case_id=case_id, baseline=baseline_seen, candidate=candidate_seen
            )
            if check.loss is not None:
                losses.append(check.loss)
            if check.note is not None:
                unknown.append(check.note)
            if check.unobserved:
                unobserved.append(case_id)

    if unobserved:
        unknown.append(
            "정답 근거 ID 를 어느 실측의 귀인 절에서도 관측하지 못한 케이스 — 양쪽 모두 정답 "
            "근거를 **전부 채택**한 것으로 읽히지만(귀인 절은 그런 케이스를 싣지 않는다) "
            "ID 를 이름으로 확인하지는 못했다: " + ", ".join(f"`{case}`" for case in unobserved)
        )

    measurement1_changes = tuple(
        change
        for change in (
            _no_change(
                "측정 1 구조적 오류 검출률",
                (run.detection_rate for run in baseline.runs),
                (run.detection_rate for run in candidate.runs),
            ),
            _no_change(
                "측정 1 정상 초안 오탐률",
                (run.false_positive_rate for run in baseline.runs),
                (run.false_positive_rate for run in candidate.runs),
            ),
        )
        if change is not None
    )

    failed = bool(shortfalls or collapses or losses or measurement1_changes)
    # **돌지 않은 겹이 있으면 통과가 아니다.** 판정의 정의인 검사가 실행되지 않았는데
    # 헤드라인이 "통과"라고 적히면, 그것이 바로 하드 게이트가 금지하는 "미실행을 통과로
    # 채우기"다. 사유가 아래 미판정 줄에만 남아 있는 것으로는 부족하다 — 판정 문면이 말해야 한다.
    held: list[str] = []
    if candidate_runs < expected_run_count:
        held.append(f"실측 {candidate_runs}/{expected_run_count} — 세트가 아직 안 찼다")
    if not baseline_evidence_known:
        held.append("기준선에 귀인 절이 없어 **근거 부분 손실 검사가 돌지 않았다**")
    if not candidate_evidence_known:
        held.append("이번 실측에 귀인 절이 없어 **근거 부분 손실 검사가 돌지 않았다**")
    if not compared_cases:
        held.append("두 세트에 공통으로 실린 케이스가 없어 **케이스 단위 판정이 돌지 않았다**")

    if failed:
        # 관측된 회귀는 보류로 덮지 않는다 — 미달이 더 강한 정보다.
        verdict = VERDICT_FAIL
        reason = "케이스 단위 비악화 판정 미달 — 아래 케이스를 이름으로 찍는다."
    elif held:
        verdict = VERDICT_HELD
        reason = "판정 보류 — " + " · ".join(held) + ". 아래 항목은 잠정 관측이다."
    else:
        verdict = VERDICT_PASS
        reason = (
            f"기준선 {baseline_runs}회 대비 케이스 단위 비악화 통과 "
            f"(케이스 {compared_cases}건 · 일치 하한 {_majority(candidate_runs)}/"
            f"{candidate_runs} · 근거 부분 손실 없음)."
        )

    return BaselineLine(
        label=label,
        role=role,
        verdict=verdict,
        verdict_reason=reason,
        baseline_stems=baseline.stems,
        baseline_source=baseline_source,
        fingerprint=comparison,
        match_shortfalls=tuple(shortfalls),
        match_collapses=tuple(collapses),
        match_decreases=tuple(decreases),
        evidence_losses=tuple(losses),
        measurement1_changes=measurement1_changes,
        unknown_notes=tuple(unknown),
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class _EvidenceCheck:
    """근거 부분 손실 검사 1건의 결과. 셋은 서로 다른 상태다 — 섞으면 무지가 통과가 된다."""

    loss: EvidenceLoss | None = None
    note: str | None = None
    #: 정답 근거 ID 를 **양쪽 어디서도 관측하지 못했다.** 손실이 없다는 판정 자체는
    #: 성립하지만(양쪽 모두 전부 채택) ID 를 이름으로 확인하지는 못했다 — 한 줄로 묶어 적는다.
    unobserved: bool = False


#: 판정 층에 속하는 지문 항목. 측정 3 의 변화를 무엇으로 설명할 수 있는지가 여기서 갈린다.
JUDGING_FINGERPRINT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "judge_model",
        "judge_effort",
        "judge_prompt_version",
        "judge_fixture_version",
        "judge_prompt_caching",
    }
)


def _measurement3_note(*, baseline: RunSet, candidate: RunSet) -> str:
    """측정 3 에 대해 **이 실행에서 실제로 성립하는 것**을 적는다.

    규칙 자체는 고정이다 — 측정 3 은 무변경 검사 대상이 아니다. 판정 층 개선의 성공이
    무관한 축의 원복을 발동시키면 안 되기 때문이다. 하지만 그 뒤에 붙는 문장까지 "선언된
    실험 변인이라" 로 고정하면, 아무것도 선언하지 않은 실행의 리포트가 거짓을 적는다.
    """
    before = _common(run.measurement3_detection_rate for run in baseline.runs)
    after = _common(run.measurement3_detection_rate for run in candidate.runs)
    rule = (
        "측정 3 은 무변경 검사 대상이 아니다 — 판정 층 개선의 성공이 무관한 축의 원복을 "
        "발동시키면 안 되기 때문이다."
    )
    values = f"기준선 {_rate(before)} / 이번 {_rate(after)}"
    if before is None or after is None:
        state = "한쪽이 미상이거나 세트 안에서 흔들려 대조하지 않았다"
    elif before == after:
        state = "두 세트의 값이 같다"
    else:
        declared = tuple(
            name for name in candidate.fingerprint.declared if name in JUDGING_FINGERPRINT_FIELDS
        )
        state = (
            "값이 달라졌고 이 실행이 판정 층 변경을 선언했다("
            + ", ".join(f"`{name}`" for name in declared)
            + ")"
            if declared
            else "값이 달라졌는데 이 실행은 판정 층 변경을 선언하지 않았다 — 관측으로만 "
            "남긴다(가드는 이것으로 미달을 내지 않는다)"
        )
    return f"{rule} {values} — {state}."


def _evidence_loss(
    *,
    case_id: str,
    baseline: Sequence[CaseObservation],
    candidate: Sequence[CaseObservation],
) -> _EvidenceCheck:
    """근거 부분 손실 — **`matched` 가 못 잡는 회귀를 잡는 자리다.**

    판정 단위는 케이스별 다수결이다: 기준선 다수결에서 채택됐던 정답 근거 ID 가 새 실측
    다수결에서 빠지면 미달이고, 빠진 ID 를 이름으로 찍는다.
    """
    baseline_reference = _relevant_reference(baseline)
    candidate_reference = _relevant_reference(candidate)
    observed = [item for item in (baseline_reference, candidate_reference) if item is not None]
    if not observed:
        # 어느 실측도 이 케이스를 귀인 절에 싣지 않았다 = 양쪽 모두 정답 근거를 전부
        # 채택했다는 뜻이지만, **ID 를 하나도 관측하지 못했으므로 이름으로 말할 수 없다.**
        # 아무 말도 하지 않으면 그 무지가 "손실 없음"으로 읽힌다 — 관측 한계로 남긴다.
        return _EvidenceCheck(unobserved=True)
    if len(observed) == 2 and observed[0] != observed[1]:
        return _EvidenceCheck(
            note=(
                f"`{case_id}` 의 정답 라벨이 기준선과 이번 실측에서 다르다 — 근거 손실 검사를 "
                "건너뛴다(라벨 변경은 회귀가 아니라 조건 변경이다)"
            )
        )
    # **한쪽만 없는 것은 무지가 아니라 정보다.** 귀인 절은 "정답 조항을 전부 채택하고 정상
    # 답변한" 케이스를 싣지 않으므로, 부재는 곧 **전부 채택**이다. 그래서 없는 쪽의 라벨을
    # 있는 쪽에서 빌려 온다 — 이 보정이 없으면 **기준선이 3회 모두 멀쩡했던 케이스**가
    # 통째로 감시 밖으로 나간다. 그것이 이 층이 가장 지켜야 할 모집단인데도 그렇다.
    reference = observed[0]

    baseline_accepted = _accepted_majority(baseline, reference)
    candidate_accepted = _accepted_majority(candidate, reference)
    if baseline_accepted is None:
        return _EvidenceCheck(
            note=(
                f"`{case_id}` 의 채택 근거가 기준선 산출물에 없다 — 근거 부분 손실을 판정하지 "
                "않는다(0 으로 채우지 않는다)"
            )
        )
    if candidate_accepted is None:
        return _EvidenceCheck(
            note=f"`{case_id}` 의 채택 근거가 이번 실측에 없다 — 근거 부분 손실 미판정"
        )
    dropped = tuple(sorted(baseline_accepted - candidate_accepted))
    if not dropped:
        return _EvidenceCheck()
    return _EvidenceCheck(
        loss=EvidenceLoss(
            case_id=case_id,
            dropped_evidence_ids=dropped,
            baseline_accepted_ids=tuple(sorted(baseline_accepted)),
            candidate_accepted_ids=tuple(sorted(candidate_accepted)),
        )
    )


def _relevant_reference(observations: Sequence[CaseObservation]) -> frozenset[str] | None:
    """이 세트가 **관측한** 정답 근거 집합. 어느 실측도 싣지 않았으면 `None`.

    `None` 은 "정답 근거가 없다"가 아니라 "이 세트의 산출물만으로는 ID 를 알 수 없다"다 —
    귀인 절이 정상 케이스를 싣지 않기 때문이다. 호출자가 반대편 세트에서 빌려 온다.
    """
    seen = [
        item.relevant_evidence_ids
        for item in observations
        if item.relevant_evidence_ids is not None
    ]
    if not seen:
        return None
    return frozenset().union(*seen)


def _accepted_majority(
    observations: Sequence[CaseObservation], reference: frozenset[str]
) -> frozenset[str] | None:
    """3회 중 2회 이상 채택된 정답 근거. 채택 여부를 알 수 없는 세트면 `None`."""
    accepted_per_run = [item.accepted_relevant(reference) for item in observations]
    known = [item for item in accepted_per_run if item is not None]
    if not known:
        return None
    threshold = _majority(len(known))
    return frozenset(
        evidence_id
        for evidence_id in reference
        if sum(1 for item in known if evidence_id in item) >= threshold
    )


def _no_change(
    label: str, baseline: Iterable[float | None], candidate: Iterable[float | None]
) -> str | None:
    """무변경 검사 1건. 어느 쪽이든 미상이면 변경으로 적지 않는다."""
    before = _common(baseline)
    after = _common(candidate)
    if before is None or after is None or before == after:
        return None
    return f"{label}: 기준선 {_rate(before)} → 이번 {_rate(after)}"


def _common(values: Iterable[float | None]) -> float | None:
    """세트 전체가 같은 값일 때만 그 값. 흔들리거나 미상이면 `None`."""
    collected = list(values)
    if not collected or any(value is None for value in collected):
        return None
    unique = {value for value in collected}
    return collected[0] if len(unique) == 1 else None


def _rate(value: float | None) -> str:
    return UNKNOWN if value is None else f"{value * 100:.1f}%"


# ══ 두 줄 보고 ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RegressionGuard:
    """이중 기준선 두 줄 보고. **판정은 구속 줄이 가진다.**"""

    candidate_stems: tuple[str, ...]
    candidate_run_count: int
    expected_run_count: int
    binding: BaselineLine
    alert: BaselineLine
    promotion: PromotedBaseline | BaselineNotRegistered

    @property
    def verdict(self) -> str:
        return self.binding.verdict

    @property
    def disagreement(self) -> bool:
        """두 줄이 **상반되는가**. 상반돼도 판정은 승격 기준선이 이긴다.

        한쪽이 대조 자체를 못 한 경우(미등재·대조 불가·직전 없음)는 상반이 아니다 —
        비교되지 않은 것을 "의견이 갈렸다"고 적으면 사용자가 읽을 신호가 흐려진다.
        """
        compared = {VERDICT_PASS, VERDICT_FAIL, VERDICT_HELD}
        if self.binding.verdict not in compared or self.alert.verdict not in compared:
            return False
        return self.binding.verdict != self.alert.verdict


@dataclass(frozen=True)
class GuardUnavailable:
    """가드를 돌리지 않았다는 **명시적 기록**. 통과로 대체하지 않는다."""

    reason: str


def build_regression_guard(
    *,
    current: RunSummary,
    reports_dir: Path,
    promoted_reference_path: Path = DEFAULT_PROMOTED_BASELINE_PATH,
    run_set_size: int = RUN_SET_SIZE,
) -> RegressionGuard | GuardUnavailable:
    """이번 실행의 두 줄 보고를 만든다."""
    if not current.billed:
        return GuardUnavailable(
            reason="과금 실행이 아니다 — 대역 수치를 기준선과 대조하면 대조 자체가 거짓이 된다."
        )

    promotion = load_promoted_baseline(promoted_reference_path)
    promoted_stems = promotion.report_stems if isinstance(promotion, PromotedBaseline) else ()
    candidate = assemble_candidate_set(
        current=current, reports_dir=reports_dir, size=run_set_size, exclude=promoted_stems
    )

    binding = _binding_line(
        promotion=promotion,
        candidate=candidate,
        reports_dir=reports_dir,
        run_set_size=run_set_size,
    )
    recent = discover_recent_live_set(
        reports_dir=reports_dir,
        l2_enabled=current.l2_enabled,
        size=run_set_size,
        exclude=candidate.stems,
    )
    if recent is None or not recent.runs:
        alert = BaselineLine(
            label="직전 라이브",
            role=ROLE_ALERT,
            verdict=VERDICT_NO_BASELINE,
            verdict_reason="`reports/` 에서 대조할 직전 라이브 세트를 찾지 못했다.",
        )
    else:
        alert = compare_run_sets(
            baseline=recent,
            candidate=candidate,
            label="직전 라이브",
            role=ROLE_ALERT,
            baseline_source="reports/ 자동 탐색",
            expected_run_count=run_set_size,
        )
    return RegressionGuard(
        candidate_stems=candidate.stems,
        candidate_run_count=len(candidate.runs),
        expected_run_count=run_set_size,
        binding=binding,
        alert=alert,
        promotion=promotion,
    )


def _promotion_drift(*, promotion: PromotedBaseline, runs: Sequence[RunSummary]) -> tuple[str, ...]:
    """등재된 지문·스템이 실제 산출물과 어긋나는 지점. 어긋나지 않으면 빈 튜플.

    두 가지를 본다: ① 등재된 스템들이 서로 같은 조건인가(한 세트인가), ② 참조 파일에 적힌
    조건 지문이 그 산출물의 지문과 같은가. 참조에 적지 않은 항목은 미상이라 어긋남이 아니다
    — 사람이 다 적지 않아도 되지만, **적은 것은 맞아야 한다.**
    """
    findings: list[str] = []
    head = runs[0]
    for other in runs[1:]:
        conflicts = _fingerprint_conflicts(other.fingerprint, head.fingerprint)
        if conflicts:
            findings.append(
                "등재된 산출물끼리 조건이 다르다: "
                + " · ".join(
                    item.describe_as(left=f"`{other.stem}`", right=f"`{head.stem}`")
                    for item in conflicts
                )
            )
    registered = _fingerprint_conflicts(promotion.fingerprint, head.fingerprint)
    if registered:
        findings.append(
            "참조 파일의 조건 지문이 산출물과 다르다: "
            + " · ".join(
                item.describe_as(left="참조", right=f"산출물 `{head.stem}`") for item in registered
            )
        )
    return tuple(findings)


def _fingerprint_conflicts(
    left: ConditionFingerprint, right: ConditionFingerprint
) -> tuple[FieldDifference, ...]:
    """두 지문의 **모든** 값 차이. 선언 여부와 무관하다 — 등재 정합성 검사에는 선언이 없다."""
    comparison = ConditionFingerprint(values=left.values).compare(
        ConditionFingerprint(values=right.values)
    )
    return comparison.undeclared_differences


def _binding_line(
    *,
    promotion: PromotedBaseline | BaselineNotRegistered,
    candidate: RunSet,
    reports_dir: Path,
    run_set_size: int,
) -> BaselineLine:
    if isinstance(promotion, BaselineNotRegistered):
        return BaselineLine(
            label="승격 기준선",
            role=ROLE_BINDING,
            verdict=VERDICT_NOT_REGISTERED,
            verdict_reason=promotion.reason,
            baseline_source=promotion.path,
        )
    runs: list[RunSummary] = []
    missing: list[str] = []
    for stem in promotion.report_stems:
        path = reports_dir / f"{stem}.json"
        try:
            runs.append(load_run_summary(path))
        except RunSummaryError as exc:
            missing.append(f"{stem}: {exc}")
    if missing or not runs:
        return BaselineLine(
            label="승격 기준선",
            role=ROLE_BINDING,
            verdict=VERDICT_INCOMPARABLE,
            verdict_reason=(
                "승격 기준선 산출물을 읽지 못했다 — " + " · ".join(missing or ["등재된 스템 없음"])
            ),
            baseline_stems=promotion.report_stems,
            baseline_source=promotion.path,
        )
    runs.sort(key=lambda run: (run.started_at, run.stem))
    # **등재가 자기가 가리키는 산출물과 어긋나면 그 등재는 기준선을 설명하지 못한다.**
    # 참조 파일은 "승격 대상 리포트 스템 + 조건 지문"이라 둘이 한 쌍이어야 하고, 등재된
    # 스템들끼리도 같은 조건이어야 한다. 어긋난 채로 구속 판정을 내면 무엇을 기준으로
    # 통과·미달을 말한 것인지 산출물만 보고는 되짚을 수 없다.
    drift = _promotion_drift(promotion=promotion, runs=runs)
    if drift:
        return BaselineLine(
            label="승격 기준선",
            role=ROLE_BINDING,
            verdict=VERDICT_INCOMPARABLE,
            verdict_reason=(
                "승격 등재가 자기가 가리키는 산출물과 어긋난다 — " + " · ".join(drift) + ". "
                "**재등재**가 필요하다(승격과 같은 자격 — 사람이 참조 파일을 바꾼다)."
            ),
            baseline_stems=promotion.report_stems,
            baseline_source=promotion.path,
        )
    return compare_run_sets(
        baseline=RunSet(label="승격 기준선", runs=tuple(runs)),
        candidate=candidate,
        label="승격 기준선",
        role=ROLE_BINDING,
        baseline_source=promotion.path,
        expected_run_count=run_set_size,
    )


# ══ 표기 — 마크다운과 JSON 은 같은 원본에서 나온다 ═══════════════════════════


def render_guard_section(guard: RegressionGuard | GuardUnavailable) -> list[str]:
    """사람이 읽는 회귀 가드 절."""
    lines = ["## 회귀 가드 — 이중 기준선 두 줄 보고", ""]
    if isinstance(guard, GuardUnavailable):
        lines.extend(
            [f"**미산출 (사유: {guard.reason})**", "", "미산출을 통과로 대체하지 않는다.", ""]
        )
        return lines

    lines.extend(
        [
            f"- 이번 실측 세트: {guard.candidate_run_count}/{guard.expected_run_count}회 "
            + (", ".join(f"`{stem}`" for stem in guard.candidate_stems) or "없음"),
            f"- **판정: {guard.verdict}** — 승격 기준선(구속) 줄이 결정한다. "
            "직전 라이브 줄은 경보이고 판정을 발동시키지 않는다.",
        ]
    )
    if guard.disagreement:
        lines.append(
            f"- **두 줄이 상반된다** (구속 {guard.binding.verdict} / 경보 {guard.alert.verdict}) "
            "— 승격 기준선이 이기고 직전 라이브의 악화는 사용자 판단으로 넘긴다."
        )
    promotion = guard.promotion
    if isinstance(promotion, PromotedBaseline):
        mark = "재등재" if promotion.repromotion else "승격"
        lines.append(
            f"- {mark}: {promotion.promoted_at} · {promotion.promoted_by} — {promotion.reason} "
            f"(참조 `{promotion.path}`)"
        )
        if promotion.repromotion:
            lines.append(
                "  - 재등재로 대신한 직전 승격: "
                + ", ".join(f"`{stem}`" for stem in promotion.supersedes)
            )
    else:
        lines.append(f"- 승격 참조: **기준선 미등재** — {promotion.reason}")
    lines.append("")
    lines.extend(_render_line(guard.binding))
    lines.extend(_render_line(guard.alert))
    return lines


def _render_line(line: BaselineLine) -> list[str]:
    lines = [
        f"### {line.label} ({line.role}) — **{line.verdict}**",
        "",
        f"- 사유: {line.verdict_reason}",
    ]
    if line.baseline_stems:
        lines.append("- 기준선 산출물: " + ", ".join(f"`{stem}`" for stem in line.baseline_stems))
    if line.baseline_source:
        lines.append(f"- 기준선 출처: `{line.baseline_source}`")
    comparison = line.fingerprint
    if comparison.undeclared_differences:
        lines.append(
            "- **대조 불가 — 선언 없이 어긋난 항목**: "
            + " · ".join(item.describe() for item in comparison.undeclared_differences)
        )
    if comparison.declared_differences:
        lines.append(
            "- 선언된 실험 변인(대조 진행): "
            + " · ".join(item.describe() for item in comparison.declared_differences)
        )
    if comparison.unknown_fields:
        lines.append(
            "- 지문 미상 항목(기준선 또는 이번 실측에 값이 없다 — 같다고도 다르다고도 적지 "
            "않는다): " + ", ".join(f"`{name}`" for name in comparison.unknown_fields)
        )
    for title, findings in (
        ("일치 미달(기준선 전회 일치 → 다수결 미달)", line.match_shortfalls),
        ("일치 붕괴(기준선 일치 있음 → 이번 0회)", line.match_collapses),
        ("일치 감소(경보)", line.match_decreases),
    ):
        if findings:
            lines.append(f"- **{title}**: " + " · ".join(item.describe() for item in findings))
    if line.evidence_losses:
        lines.append(
            "- **근거 부분 손실** — 기준선이 채택했던 정답 근거가 빠졌다: "
            + " · ".join(item.describe() for item in line.evidence_losses)
        )
    if line.measurement1_changes:
        lines.append("- **측정 1 변경**: " + " · ".join(line.measurement1_changes))
    for note in line.unknown_notes:
        lines.append(f"- 미판정: {note}")
    for note in line.notes:
        lines.append(f"- 참고: {note}")
    lines.append("")
    return lines


def guard_to_json(guard: RegressionGuard | GuardUnavailable) -> dict[str, Any]:
    """기계가 읽는 형식. 마크다운과 같은 원본에서 나온다."""
    if isinstance(guard, GuardUnavailable):
        return {"computed": False, "reason": guard.reason}
    promotion = guard.promotion
    return {
        "computed": True,
        "verdict": guard.verdict,
        "verdict_owner": "승격 기준선(구속)",
        "disagreement": guard.disagreement,
        "candidate_stems": list(guard.candidate_stems),
        "candidate_run_count": guard.candidate_run_count,
        "expected_run_count": guard.expected_run_count,
        "promotion": (
            {
                "registered": True,
                "path": promotion.path,
                "promoted_at": promotion.promoted_at,
                "promoted_by": promotion.promoted_by,
                "reason": promotion.reason,
                "repromotion": promotion.repromotion,
                "supersedes": list(promotion.supersedes),
                "report_stems": list(promotion.report_stems),
            }
            if isinstance(promotion, PromotedBaseline)
            else {"registered": False, "path": promotion.path, "reason": promotion.reason}
        ),
        "binding": _line_to_json(guard.binding),
        "alert": _line_to_json(guard.alert),
    }


def _line_to_json(line: BaselineLine) -> dict[str, Any]:
    comparison = line.fingerprint
    return {
        "label": line.label,
        "role": line.role,
        "verdict": line.verdict,
        "verdict_reason": line.verdict_reason,
        "baseline_stems": list(line.baseline_stems),
        "baseline_source": line.baseline_source,
        "fingerprint": {
            "comparable": comparison.comparable,
            "declared_differences": [
                {"field": item.field, "baseline": item.baseline, "candidate": item.candidate}
                for item in comparison.declared_differences
            ],
            "undeclared_differences": [
                {"field": item.field, "baseline": item.baseline, "candidate": item.candidate}
                for item in comparison.undeclared_differences
            ],
            "unknown_fields": list(comparison.unknown_fields),
        },
        "match_shortfalls": [_match_to_json(item) for item in line.match_shortfalls],
        "match_collapses": [_match_to_json(item) for item in line.match_collapses],
        "match_decreases": [_match_to_json(item) for item in line.match_decreases],
        "evidence_losses": [
            {
                "case_id": item.case_id,
                "dropped_evidence_ids": list(item.dropped_evidence_ids),
                "baseline_accepted_ids": list(item.baseline_accepted_ids),
                "candidate_accepted_ids": list(item.candidate_accepted_ids),
            }
            for item in line.evidence_losses
        ],
        "measurement1_changes": list(line.measurement1_changes),
        "unknown_notes": list(line.unknown_notes),
        "notes": list(line.notes),
    }


def _match_to_json(item: MatchFinding) -> dict[str, Any]:
    return {
        "case_id": item.case_id,
        "baseline_matched": item.baseline_matched,
        "baseline_runs": item.baseline_runs,
        "candidate_matched": item.candidate_matched,
        "candidate_runs": item.candidate_runs,
    }


# ══ 작은 도우미 ═════════════════════════════════════════════════════════════


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [str(item) for item in value]


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _display(path: Path) -> str:
    """저장소 안이면 상대 경로로 적는다 — 커밋되는 산출물에 로컬 절대 경로를 남기지 않는다."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_ROOT))
    except ValueError:
        return str(resolved)
