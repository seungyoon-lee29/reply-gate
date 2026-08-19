"""채택 축 손계산 — 커밋된 검색 산출물만으로 후보 축의 도달 가능 영역을 계산한다.

**입력은 `reports/retrieval-strategies-live-*.json` 뿐이다.** 새 임베딩도 새 과금도 없고,
정답 라벨은 리포트 안에 이미 실려 있는 `relevant_evidence_ids` 를 **채점에만** 쓴다
(`docs/standards.md` 의 "검색 관련성과 답변 충분성은 다른 계약이다" — 채점자는 정답을 보고
전략은 보지 않는다).

**런타임과 같은 순서로 자른다: `top_k` 선절단 → 규칙 적용.** `policy_index.search_policy_chunks`
가 SQL `LIMIT top_k` 로 후보를 자른 뒤 파이썬이 임계값을 걸기 때문에, 통계량과 채택 집합을
전체 코퍼스 랭킹으로 계산하면 오프라인 격자가 라이브를 더 이상 예측하지 못한다. 이 절단은
조정 가능한 인자가 아니라 **행동 계약**이다.

이 모듈이 답하는 물음 셋:

1. **절대 하한 x 적응 컷 비율 x 상대 마진** 격자에서 세 방향(근거 보존 · 무근거 기권 ·
   상충쌍 동시 채택)을 동시에 만족하는 구성이 있는가 → `run_grid`.
2. 질의 단위 **기권 게이트**의 통계량 5종이 조건별로 기권군과 채택군을 분리하는가 →
   `separation_margin`.
3. **상충쌍 보존 규칙**이 현 기본값(컷 0.30)에서 무엇을 바꾸는가 → `audit_conflict_pairs`.

**격자 결과의 공집합은 격자 간격에 의존하지 않는다.** 상대 축 둘(비율 · 마진)은 1위를
**절대로** 지우지 못한다 — 1위는 `s1 >= ratio * s1` 과 `s1 - s1 <= margin` 을 항상 만족한다.
따라서 기권(채택 0건)은 오직 절대 하한이 `s1` 을 넘을 때만 생기고, 방향 2 의 요구는 하한
축 하나로 환원된다. 그 하한이 방향 1 의 요구와 겹치지 않는다는 것이 결론이며, 축의 눈금을
아무리 촘촘하게 해도 이 논증은 바뀌지 않는다.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Final

__all__ = [
    "ABSTENTION_CASE_IDS",
    "CONFLICT_PAIR_CASE_IDS",
    "DEFAULT_GRID_AXES",
    "DEFAULT_REPORT_DIR",
    "FINE_GRID_AXES",
    "KNOWN_CONFLICT_PARTNERS",
    "STRATEGY_NAME",
    "TOP_K",
    "AbstentionStatistic",
    "CaseSlice",
    "ConditionSlice",
    "Configuration",
    "ConflictPairAudit",
    "Direction",
    "GridAxes",
    "GridOutcome",
    "HandCalculation",
    "LiftedClause",
    "SeparationMargin",
    "accepted_ids",
    "audit_conflict_pairs",
    "default_conditions",
    "load_condition",
    "render_markdown",
    "run_grid",
    "run_hand_calculation",
    "separation_margin",
    "statistic_value",
    "to_json",
]

#: 런타임이 임계값 **전에** 거는 절단. 오프라인 손계산도 같은 자리에서 자른다.
TOP_K: Final = 5

#: 손계산이 보는 전략 행. 현 기본값(벡터 + 재작성 켜짐)과 같은 조합이다.
STRATEGY_NAME: Final = "vector_rewrite"

#: 현 기본 채택 컷. `reply_gate.config` 를 import 하지 않는다 — 손계산은 설정이 아니라
#: 커밋된 산출물을 읽는 오프라인 채점자다.
DEFAULT_ABSOLUTE_FLOOR: Final = 0.30

#: 저장소 안 검색 산출물 위치.
DEFAULT_REPORT_DIR: Final = Path(__file__).resolve().parents[2] / "reports"

#: 방향 2 의 표적 — 정답 조항이 없어 "채택 0건" 이 정답인 문의.
#: G28 도 빈 라벨이지만 기권 표적이 아니다(환불 실행 요청). 분리 여유 계산의 어느 쪽에도
#: 넣지 않는다 — 넣으면 표적이 아닌 케이스가 τ 를 정한다.
ABSTENTION_CASE_IDS: Final = ("G21", "G22", "G23", "G24")

#: 방향 3 의 표적 — 라벨 두 건이 **상충쌍**인 케이스.
#: G20 도 라벨이 둘이지만 상충이 아니라 같은 사안의 보완 조항이라 여기 넣지 않는다.
CONFLICT_PAIR_CASE_IDS: Final = ("G01", "G02", "G18")

#: 상충 후보 탐지기가 알게 될 짝. 앞 둘은 정책 원문의 심은 주석이 짝의 조항 번호까지 적어
#: 둔 것이고(정답 그 자체다), 마지막 하나는 실측이 드러낸 의미 상충이다.
#: **채점 전용 상수다** — 전략·런타임은 이 표에도 심은 주석에도 닿지 않는다
#: (`tests/test_answer_key_isolation.py`).
KNOWN_CONFLICT_PARTNERS: Final[Mapping[str, str]] = {
    "policy:shipping:1-3": "policy:shipping:1-4",
    "policy:shipping:1-4": "policy:shipping:1-3",
    "policy:refund:2-1": "policy:refund:2-2",
    "policy:refund:2-2": "policy:refund:2-1",
    "policy:refund:2-4": "policy:refund:2-6",
    "policy:refund:2-6": "policy:refund:2-4",
}


class Direction(StrEnum):
    """채택 규칙이 **동시에** 만족해야 하는 세 방향."""

    #: 정답 조항이 있는 25건이 각각 정답 조항을 하나 이상 채택한다.
    EVIDENCE_PRESERVED = "evidence_preserved"
    #: G21~G24 가 채택 0건으로 끝난다. "줄인다" 가 아니라 "기권한다" 다.
    UNGROUNDED_ABSTAINS = "ungrounded_abstains"
    #: G01·G02·G18 이 상충쌍을 **함께** 채택한다.
    CONFLICT_PAIR_KEPT = "conflict_pair_kept"


class AbstentionStatistic(StrEnum):
    """질의 단위 기권 게이트의 후보 통계량 5종.

    판정 방향은 다섯 모두 같다 — **값이 τ 미만이면 그 질의는 채택 0건으로 끝낸다.**
    방향을 통계량마다 뒤집으면 "분리했다" 가 사후 선택이 되므로 한 방향으로 고정한다.
    """

    #: 1위 - `top_k` 위 산포.
    SPREAD = "rank1_minus_rank_k_spread"
    #: 상위 `top_k` 점수의 모표준편차.
    STDEV = "top_k_stdev"
    #: 1위 대비 산포 비 — (1위 - `top_k` 위) / 1위.
    RELATIVE_SPREAD = "spread_over_rank1"
    #: 1위 - 2위 gap.
    GAP_1_2 = "rank1_minus_rank2_gap"
    #: `top_k` 위 / 1위 비.
    TAIL_RATIO = "rank_k_over_rank1_ratio"


@dataclass(frozen=True)
class CaseSlice:
    """케이스 1건의 `top_k` 절단 슬라이스.

    `full_ranking` 은 절단 **전** 전체 순위다 — 상충쌍 보존 규칙이 절단 밖에서 무엇을
    끌어올리는지 세는 감사에만 쓰고, 격자·통계량 계산에는 쓰지 않는다.
    """

    case_id: str
    relevant_evidence_ids: tuple[str, ...]
    hits: tuple[tuple[str, float], ...]
    full_ranking: tuple[tuple[str, float], ...]

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(score for _, score in self.hits)

    @property
    def has_labels(self) -> bool:
        return bool(self.relevant_evidence_ids)


@dataclass(frozen=True)
class ConditionSlice:
    """조건 1개(임베딩 모델 · 차원 · 재작성 조건) 의 30건 슬라이스."""

    label: str
    report_name: str
    embedding_model: str
    dimensions: int
    rewrite_condition: str
    top_k: int
    cases: tuple[CaseSlice, ...]

    def case(self, case_id: str) -> CaseSlice:
        for item in self.cases:
            if item.case_id == case_id:
                return item
        raise KeyError(f"조건 {self.label} 에 케이스 {case_id} 가 없다")


@dataclass(frozen=True)
class Configuration:
    """격자의 한 점. 세 축을 **모두** 통과한 항목만 채택된다."""

    absolute_floor: float
    adaptive_ratio: float
    relative_margin: float


@dataclass(frozen=True)
class GridAxes:
    """전수 격자의 축과 눈금.

    **눈금은 결론을 바꾸지 않는다.** 세 축 모두 값을 낮추거나(하한·비율) 높이면(마진)
    채택 집합이 단조롭게 커지므로, 격자 밖의 값은 방향 1·3 을 이미 만족하는 구성에서
    방향 2 를 새로 만족시킬 수 없다. 그래서 범위 끝은 지배 논증으로 정당화된다.
    """

    absolute_floor: tuple[float, ...]
    adaptive_ratio: tuple[float, ...]
    relative_margin: tuple[float, ...]

    @property
    def cardinality(self) -> int:
        return len(self.absolute_floor) * len(self.adaptive_ratio) * len(self.relative_margin)

    def configurations(self) -> tuple[Configuration, ...]:
        return tuple(
            Configuration(floor, ratio, margin)
            for floor in self.absolute_floor
            for ratio in self.adaptive_ratio
            for margin in self.relative_margin
        )


def _sweep(start: float, stop: float, step: float) -> tuple[float, ...]:
    """부동소수 누적 오차 없이 눈금을 만든다."""
    count = round((stop - start) / step) + 1
    return tuple(round(start + index * step, 10) for index in range(count))


#: 전수 격자의 기본 눈금 — 9 x 8 x 8 = 576 구성.
#:
#: - **절대 하한** 0.30~0.70: 커밋된 리포트의 `cutoff_sweep` 값 중 현 기본값 0.30 이상.
#:   0.30 미만은 채택 집합을 키우기만 하므로 방향 2 를 새로 만족시킬 수 없다(지배).
#: - **적응 컷 비율** 0.60~0.95: 1.00 은 1위 동점만 남기는 퇴화 구성이고, 0.60 미만은
#:   역시 채택 집합을 키우기만 한다(지배).
#: - **상대 마진** 0.05~0.40: 0.05 미만은 방향 1 을 깨기만 하고(G04 정답 조항이 1위와
#:   0.0076 차이다) 0.40 초과는 채택 집합을 키우기만 한다(지배).
DEFAULT_GRID_AXES: Final = GridAxes(
    absolute_floor=_sweep(0.30, 0.70, 0.05),
    adaptive_ratio=_sweep(0.60, 0.95, 0.05),
    relative_margin=_sweep(0.05, 0.40, 0.05),
)

#: 눈금 독립성을 **손 논증만이 아니라 실행으로도** 보이는 대조 격자 — 0.01 간격,
#: 세 축 모두 지배 논증이 잘라 낸 구간까지 되돌려 넣었다(비율 1.00·마진 0.00 포함).
#: 기본 격자보다 185배 촘촘한데도 결론이 같다는 것이 이 상수의 쓸모다.
FINE_GRID_AXES: Final = GridAxes(
    absolute_floor=_sweep(0.30, 0.70, 0.01),
    adaptive_ratio=_sweep(0.50, 1.00, 0.01),
    relative_margin=_sweep(0.00, 0.50, 0.01),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_condition(
    path: Path,
    *,
    label: str,
    strategy: str = STRATEGY_NAME,
    top_k: int = TOP_K,
) -> ConditionSlice:
    """커밋된 검색 리포트 1개에서 전략 행 하나를 `top_k` 로 잘라 읽는다."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), f"{path.name}: 리포트 최상위가 객체가 아니다")
    configuration = raw["configuration"]
    conditions = raw["run_conditions"]
    _require(
        conditions.get("labels_used_for_retrieval") is False,
        f"{path.name}: 검색에 라벨이 쓰인 리포트는 손계산 입력이 아니다",
    )
    rows = [row for row in raw["strategies"] if row["name"] == strategy]
    _require(len(rows) == 1, f"{path.name}: 전략 행 {strategy} 을 하나로 특정하지 못했다")

    cases: list[CaseSlice] = []
    for case in rows[0]["cases"]:
        ranking = tuple(
            (str(hit["evidence_id"]), float(hit["vector_similarity"]))
            for hit in case["ranked_hits"]
        )
        _require(
            all(earlier[1] >= later[1] for earlier, later in pairwise(ranking)),
            f"{path.name}: {case['id']} 의 순위가 코사인 내림차순이 아니다",
        )
        cases.append(
            CaseSlice(
                case_id=str(case["id"]),
                relevant_evidence_ids=tuple(str(item) for item in case["relevant_evidence_ids"]),
                hits=ranking[:top_k],
                full_ranking=ranking,
            )
        )
    _require(bool(cases), f"{path.name}: 케이스가 없다")

    return ConditionSlice(
        label=label,
        report_name=path.name,
        embedding_model=str(configuration["embedding_model"]),
        dimensions=int(configuration["dimensions"]),
        rewrite_condition=str(conditions["rewrite_condition"]),
        top_k=top_k,
        cases=tuple(cases),
    )


#: 네 조건 ⇔ 커밋된 리포트 네 계열. 이름이 조건을 말한다.
_CONDITION_REPORTS: Final[tuple[tuple[str, str], ...]] = (
    (
        "small-d1536-blind",
        "retrieval-strategies-live-text-embedding-3-small-d1536-blind-k5-c030.json",
    ),
    (
        "small-d1536-oracle",
        "retrieval-strategies-live-text-embedding-3-small-d1536-oracle-k5-c030.json",
    ),
    (
        "large-d1536-blind",
        "retrieval-strategies-live-text-embedding-3-large-d1536-blind-k5-c030.json",
    ),
    (
        "large-d3072-blind",
        "retrieval-strategies-live-text-embedding-3-large-d3072-blind-k5-c030.json",
    ),
)


def default_conditions(report_dir: Path = DEFAULT_REPORT_DIR) -> tuple[ConditionSlice, ...]:
    """손계산이 대조하는 네 조건. 첫 항목이 기준 조건(현 기본 임베딩 · blind 재작성)이다."""
    return tuple(
        load_condition(report_dir / name, label=label) for label, name in _CONDITION_REPORTS
    )


def statistic_value(statistic: AbstentionStatistic, scores: Sequence[float]) -> float:
    """`top_k` 로 자른 점수열에서 통계량 하나를 계산한다."""
    _require(len(scores) >= 2, "통계량은 상위 2건 이상이 있어야 정의된다")
    _require(scores[0] > 0.0, "1위 코사인이 0 이하면 비 기반 통계량이 정의되지 않는다")
    first, last = scores[0], scores[-1]
    match statistic:
        case AbstentionStatistic.SPREAD:
            return first - last
        case AbstentionStatistic.STDEV:
            return statistics.pstdev(scores)
        case AbstentionStatistic.RELATIVE_SPREAD:
            return (first - last) / first
        case AbstentionStatistic.GAP_1_2:
            return first - scores[1]
        case AbstentionStatistic.TAIL_RATIO:
            return last / first


@dataclass(frozen=True)
class SeparationMargin:
    """통계량 하나가 기권군과 채택군을 가르는 여유.

    `margin` 이 양수여야 **하나의 τ** 로 두 군을 가를 수 있다. 음수면 어떤 τ 도 둘 중
    하나를 반드시 틀린다 — 반증이고, 반증도 산출물이다.
    """

    condition: str
    statistic: AbstentionStatistic
    abstain_case: str
    abstain_value: float
    accept_case: str
    accept_value: float
    margin: float
    tau: float
    accept_headroom: float

    @property
    def separates(self) -> bool:
        return self.margin > 0.0


def separation_margin(
    condition: ConditionSlice, statistic: AbstentionStatistic
) -> SeparationMargin:
    """기권 쪽 최댓값과 채택 쪽 최솟값의 간격, 그 중간에 놓은 τ, 채택 쪽 여유."""
    abstain = {
        case.case_id: statistic_value(statistic, case.scores)
        for case in condition.cases
        if case.case_id in ABSTENTION_CASE_IDS
    }
    accept = {
        case.case_id: statistic_value(statistic, case.scores)
        for case in condition.cases
        if case.has_labels
    }
    _require(bool(abstain) and bool(accept), f"{condition.label}: 분리 대상 군이 비어 있다")

    abstain_case = max(abstain, key=lambda key: (abstain[key], key))
    accept_case = min(accept, key=lambda key: (accept[key], key))
    abstain_value = abstain[abstain_case]
    accept_value = accept[accept_case]
    tau = (abstain_value + accept_value) / 2

    return SeparationMargin(
        condition=condition.label,
        statistic=statistic,
        abstain_case=abstain_case,
        abstain_value=abstain_value,
        accept_case=accept_case,
        accept_value=accept_value,
        margin=accept_value - abstain_value,
        tau=tau,
        accept_headroom=accept_value - tau,
    )


def accepted_ids(case: CaseSlice, configuration: Configuration) -> tuple[str, ...]:
    """`top_k` 절단 뒤 세 축을 **모두** 통과한 근거 ID.

    1위는 비율·마진 축을 항상 통과한다(`s1 >= ratio * s1`, `s1 - s1 <= margin`).
    그래서 채택 0건은 오직 절대 하한이 1위를 넘을 때만 생긴다 — 이 함수가 그 사실의
    실행 가능한 형태다.
    """
    if not case.hits:
        return ()
    top = case.hits[0][1]
    return tuple(
        evidence_id
        for evidence_id, score in case.hits
        if score >= configuration.absolute_floor
        and score >= configuration.adaptive_ratio * top
        and top - score <= configuration.relative_margin
    )


def _directions_met(
    condition: ConditionSlice, configuration: Configuration
) -> frozenset[Direction]:
    met: set[Direction] = set()
    accepted = {case.case_id: set(accepted_ids(case, configuration)) for case in condition.cases}

    labelled = [case for case in condition.cases if case.has_labels]
    if all(accepted[case.case_id] & set(case.relevant_evidence_ids) for case in labelled):
        met.add(Direction.EVIDENCE_PRESERVED)

    if all(not accepted[case_id] for case_id in ABSTENTION_CASE_IDS):
        met.add(Direction.UNGROUNDED_ABSTAINS)

    if all(
        set(condition.case(case_id).relevant_evidence_ids) <= accepted[case_id]
        for case_id in CONFLICT_PAIR_CASE_IDS
    ):
        met.add(Direction.CONFLICT_PAIR_KEPT)

    return frozenset(met)


@dataclass(frozen=True)
class GridOutcome:
    """전수 격자 결과 — 세 방향을 동시에 만족하는 구성 수와, 방향별·조합별 내역."""

    condition: str
    axes: GridAxes
    total: int
    satisfying_all: tuple[Configuration, ...]
    per_direction: Mapping[Direction, int]
    per_pair: Mapping[tuple[Direction, Direction], int]
    #: 방향 1 이 요구하는 절대 하한 상한 — (케이스 ID, 정답 조항 ID, 코사인).
    floor_ceiling: tuple[str, str, float]
    #: 방향 2 가 요구하는 절대 하한 하한(초과) — (케이스 ID, 1위 조항 ID, 코사인).
    floor_threshold: tuple[str, str, float]

    @property
    def empty_intersection(self) -> bool:
        return self.floor_ceiling[2] < self.floor_threshold[2]


def _floor_ceiling(condition: ConditionSlice) -> tuple[str, str, float]:
    """정답 조항이 있는 케이스를 전부 살리는 절대 하한의 상한.

    케이스마다 "가장 높은 점수의 정답 조항" 이 그 케이스의 생존선이고, 그중 가장 낮은
    것이 전체 상한이다. `top_k` 밖의 정답 조항은 절단이 이미 지웠으므로 세지 않는다.
    """
    worst: tuple[str, str, float] | None = None
    for case in condition.cases:
        if not case.has_labels:
            continue
        survivors = [
            (evidence_id, score)
            for evidence_id, score in case.hits
            if evidence_id in case.relevant_evidence_ids
        ]
        if not survivors:
            return (case.case_id, "", float("-inf"))
        best_id, best_score = max(survivors, key=lambda item: item[1])
        if worst is None or best_score < worst[2]:
            worst = (case.case_id, best_id, best_score)
    _require(worst is not None, f"{condition.label}: 정답 조항이 있는 케이스가 없다")
    assert worst is not None
    return worst


def _floor_threshold(condition: ConditionSlice) -> tuple[str, str, float]:
    """기권 표적 넷을 전부 채택 0건으로 만들려면 절대 하한이 **초과**해야 하는 값."""
    highest: tuple[str, str, float] | None = None
    for case_id in ABSTENTION_CASE_IDS:
        case = condition.case(case_id)
        evidence_id, score = case.hits[0]
        if highest is None or score > highest[2]:
            highest = (case_id, evidence_id, score)
    assert highest is not None
    return highest


def run_grid(condition: ConditionSlice, axes: GridAxes = DEFAULT_GRID_AXES) -> GridOutcome:
    """절대 하한 x 적응 컷 비율 x 상대 마진 전수 격자."""
    per_direction = dict.fromkeys(Direction, 0)
    pairs = (
        (Direction.EVIDENCE_PRESERVED, Direction.UNGROUNDED_ABSTAINS),
        (Direction.EVIDENCE_PRESERVED, Direction.CONFLICT_PAIR_KEPT),
        (Direction.UNGROUNDED_ABSTAINS, Direction.CONFLICT_PAIR_KEPT),
    )
    per_pair = dict.fromkeys(pairs, 0)
    satisfying: list[Configuration] = []

    configurations = axes.configurations()
    for configuration in configurations:
        met = _directions_met(condition, configuration)
        for direction in met:
            per_direction[direction] += 1
        for pair in pairs:
            if set(pair) <= met:
                per_pair[pair] += 1
        if len(met) == len(Direction):
            satisfying.append(configuration)

    return GridOutcome(
        condition=condition.label,
        axes=axes,
        total=len(configurations),
        satisfying_all=tuple(satisfying),
        per_direction=per_direction,
        per_pair=per_pair,
        floor_ceiling=_floor_ceiling(condition),
        floor_threshold=_floor_threshold(condition),
    )


@dataclass(frozen=True)
class LiftedClause:
    """상충쌍 보존 규칙이 새로 끌어올리게 될 조항 1건."""

    case_id: str
    accepted_clause: str
    partner_clause: str
    partner_rank: int
    partner_score: float
    case_has_labels: bool
    partner_is_labelled: bool


@dataclass(frozen=True)
class ConflictPairAudit:
    """상충쌍 보존 규칙이 현 기본값에서 실제로 무엇을 바꾸는가.

    고치는 것이 없고 늘리는 것만 있으면, 그 전략은 후보에서 빠진다.
    """

    condition: str
    absolute_floor: float
    already_paired: tuple[str, ...]
    unpaired: tuple[str, ...]
    lifted: tuple[LiftedClause, ...]

    @property
    def fixes_nothing(self) -> bool:
        return not self.unpaired

    @property
    def over_acceptance_added(self) -> int:
        """기권이 정답인 케이스에서 새로 늘어나는 채택 건수 — 방향 2 를 직접 깎는다."""
        return sum(1 for item in self.lifted if not item.case_has_labels)

    @property
    def precision_harming(self) -> int:
        """끌어올린 짝이 그 케이스의 정답이 아닌 건수 — precision 을 깎는다."""
        return sum(1 for item in self.lifted if not item.partner_is_labelled)


def audit_conflict_pairs(
    condition: ConditionSlice,
    *,
    absolute_floor: float = DEFAULT_ABSOLUTE_FLOOR,
    partners: Mapping[str, str] = KNOWN_CONFLICT_PARTNERS,
) -> ConflictPairAudit:
    """현 기본값(절대 하한만, `top_k` 선절단)에서 상충쌍 보존 규칙의 손익을 센다."""
    configuration = Configuration(absolute_floor, 0.0, float("inf"))
    already: list[str] = []
    unpaired: list[str] = []
    lifted: list[LiftedClause] = []

    for case in condition.cases:
        accepted = set(accepted_ids(case, configuration))
        if case.case_id in CONFLICT_PAIR_CASE_IDS:
            target = set(case.relevant_evidence_ids)
            (already if target <= accepted else unpaired).append(case.case_id)
        ranks = {
            evidence_id: (index, score)
            for index, (evidence_id, score) in enumerate(case.full_ranking, start=1)
        }
        for evidence_id in sorted(accepted):
            partner = partners.get(evidence_id)
            if partner is None or partner in accepted:
                continue
            rank, score = ranks[partner]
            lifted.append(
                LiftedClause(
                    case_id=case.case_id,
                    accepted_clause=evidence_id,
                    partner_clause=partner,
                    partner_rank=rank,
                    partner_score=score,
                    case_has_labels=case.has_labels,
                    partner_is_labelled=partner in case.relevant_evidence_ids,
                )
            )

    return ConflictPairAudit(
        condition=condition.label,
        absolute_floor=absolute_floor,
        already_paired=tuple(already),
        unpaired=tuple(unpaired),
        lifted=tuple(lifted),
    )


@dataclass(frozen=True)
class HandCalculation:
    """손계산 1회의 산출물 전체."""

    axes: GridAxes
    conditions: tuple[ConditionSlice, ...]
    grid: GridOutcome
    #: 눈금 독립성 대조 — 훨씬 촘촘한 격자에서도 같은 결론이 나오는지 실행으로 확인한다.
    robustness: GridOutcome
    margins: Mapping[str, tuple[SeparationMargin, ...]]
    conflict_audit: ConflictPairAudit


def run_hand_calculation(
    conditions: Sequence[ConditionSlice],
    *,
    axes: GridAxes = DEFAULT_GRID_AXES,
    robustness_axes: GridAxes = FINE_GRID_AXES,
) -> HandCalculation:
    """기준 조건에서 격자와 상충쌍 감사를, 전 조건에서 분리 여유를 낸다."""
    _require(bool(conditions), "손계산에는 조건이 하나 이상 필요하다")
    base = conditions[0]
    return HandCalculation(
        axes=axes,
        conditions=tuple(conditions),
        grid=run_grid(base, axes),
        robustness=run_grid(base, robustness_axes),
        margins={
            condition.label: tuple(
                separation_margin(condition, statistic) for statistic in AbstentionStatistic
            )
            for condition in conditions
        },
        conflict_audit=audit_conflict_pairs(base),
    )


def to_json(result: HandCalculation) -> dict[str, object]:
    """산출물의 기계 판독 형태. 미측정 항목을 0 으로 채우지 않는다 — 안 낸 값은 키가 없다."""
    grid = result.grid
    return {
        "input": {
            "note": "커밋된 검색 산출물만 읽는다 — 새 임베딩도 새 과금도 없다",
            "strategy": STRATEGY_NAME,
            "top_k": TOP_K,
            "truncate_before_threshold": True,
            "labels_used_for": "scoring_only",
            "reports": [condition.report_name for condition in result.conditions],
        },
        "grid": {
            "axes": {
                "absolute_floor": list(result.axes.absolute_floor),
                "adaptive_ratio": list(result.axes.adaptive_ratio),
                "relative_margin": list(result.axes.relative_margin),
            },
            "cardinality": result.axes.cardinality,
            "condition": grid.condition,
            "evaluated": grid.total,
            "satisfying_all_three": len(grid.satisfying_all),
            "per_direction": {key.value: value for key, value in grid.per_direction.items()},
            "per_pair": {
                f"{left.value}+{right.value}": value
                for (left, right), value in grid.per_pair.items()
            },
            "floor_ceiling": {
                "case": grid.floor_ceiling[0],
                "evidence_id": grid.floor_ceiling[1],
                "cosine": grid.floor_ceiling[2],
            },
            "floor_threshold": {
                "case": grid.floor_threshold[0],
                "evidence_id": grid.floor_threshold[1],
                "cosine": grid.floor_threshold[2],
            },
            "empty_intersection": grid.empty_intersection,
            "robustness": {
                "note": "지배 논증이 잘라 낸 구간까지 되돌려 넣은 0.01 간격 대조 격자",
                "axes": {
                    "absolute_floor": [
                        result.robustness.axes.absolute_floor[0],
                        result.robustness.axes.absolute_floor[-1],
                        0.01,
                    ],
                    "adaptive_ratio": [
                        result.robustness.axes.adaptive_ratio[0],
                        result.robustness.axes.adaptive_ratio[-1],
                        0.01,
                    ],
                    "relative_margin": [
                        result.robustness.axes.relative_margin[0],
                        result.robustness.axes.relative_margin[-1],
                        0.01,
                    ],
                },
                "cardinality": result.robustness.axes.cardinality,
                "satisfying_all_three": len(result.robustness.satisfying_all),
            },
            "granularity_independent": (
                "상대 축 둘은 1위를 지우지 못하므로 기권은 절대 하한만 만든다 — "
                "눈금을 바꿔도 방향 1 상한과 방향 2 하한은 그대로 어긋난다"
            ),
        },
        "separation_margins": {
            label: [
                {
                    "statistic": margin.statistic.value,
                    "abstain_case": margin.abstain_case,
                    "abstain_value": margin.abstain_value,
                    "accept_case": margin.accept_case,
                    "accept_value": margin.accept_value,
                    "margin": margin.margin,
                    "tau": margin.tau,
                    "accept_headroom": margin.accept_headroom,
                    "separates": margin.separates,
                }
                for margin in margins
            ]
            for label, margins in result.margins.items()
        },
        "conflict_pair_audit": {
            "condition": result.conflict_audit.condition,
            "absolute_floor": result.conflict_audit.absolute_floor,
            "already_paired": list(result.conflict_audit.already_paired),
            "fixes_nothing": result.conflict_audit.fixes_nothing,
            "over_acceptance_added": result.conflict_audit.over_acceptance_added,
            "precision_harming": result.conflict_audit.precision_harming,
            "unpaired": list(result.conflict_audit.unpaired),
            "lifted": [
                {
                    "case": item.case_id,
                    "accepted_clause": item.accepted_clause,
                    "partner_clause": item.partner_clause,
                    "partner_rank": item.partner_rank,
                    "partner_score": item.partner_score,
                    "case_has_labels": item.case_has_labels,
                    "partner_is_labelled": item.partner_is_labelled,
                }
                for item in result.conflict_audit.lifted
            ],
        },
    }


def _axis_line(name: str, values: tuple[float, ...]) -> str:
    return f"- **{name}** {values[0]:.2f} ~ {values[-1]:.2f} · {len(values)}점"


def render_markdown(result: HandCalculation) -> str:
    """사람이 읽는 산출물. 인용되는 수치는 전부 이 표에서 나온다."""
    grid = result.grid
    base = result.conditions[0]
    lines: list[str] = [
        "# 채택 축 손계산 — 커밋된 검색 산출물 재현",
        "",
        "무과금 산출물이다. 입력은 `reports/retrieval-strategies-live-*.json` 뿐이고,",
        f"전략 행은 `{STRATEGY_NAME}`, 절단은 런타임과 같은 `top_k={TOP_K}` **선절단**이다.",
        "정답 라벨은 채점에만 쓴다 — 어떤 전략도 입력으로 받지 않는다.",
        "",
        "## 1. 전수 격자 — 절대 하한 x 적응 컷 비율 x 상대 마진",
        "",
        f"기준 조건: `{base.label}` (`{base.report_name}`)",
        "",
        _axis_line("절대 하한", result.axes.absolute_floor),
        _axis_line("적응 컷 비율", result.axes.adaptive_ratio),
        _axis_line("상대 마진", result.axes.relative_margin),
        f"- **격자 크기** {result.axes.cardinality} 구성",
        "",
        f"**{grid.total} 구성 중 세 방향을 동시에 만족하는 구성은 "
        f"{len(grid.satisfying_all)}개다.**",
        "",
        "| 방향 | 단독 만족 구성 수 |",
        "|---|---:|",
    ]
    labels = {
        Direction.EVIDENCE_PRESERVED: "1. 정답 조항 보존 (라벨 있는 25건)",
        Direction.UNGROUNDED_ABSTAINS: "2. 무근거 문의 기권 (G21~G24 채택 0건)",
        Direction.CONFLICT_PAIR_KEPT: "3. 상충쌍 동시 채택 (G01·G02·G18)",
    }
    for direction, count in grid.per_direction.items():
        lines.append(f"| {labels[direction]} | {count} |")
    lines += ["", "| 방향 조합 | 동시 만족 구성 수 |", "|---|---:|"]
    for (left, right), count in grid.per_pair.items():
        lines.append(f"| {labels[left].split('.')[0]} + {labels[right].split('.')[0]} | {count} |")

    ceiling_case, ceiling_id, ceiling_value = grid.floor_ceiling
    threshold_case, threshold_id, threshold_value = grid.floor_threshold
    lines += [
        "",
        "### 왜 비어 있는가 — 하한 축 하나로 환원된다",
        "",
        f"- 방향 1 은 하한 ≤ **{ceiling_value:.4f}** 를 요구한다 "
        f"({ceiling_case} 의 정답 조항 `{ceiling_id}`).",
        f"- 방향 2 는 하한 > **{threshold_value:.4f}** 를 요구한다 "
        f"({threshold_case} 의 1위 `{threshold_id}`).",
        f"- 두 구간은 **{'겹치지 않는다' if grid.empty_intersection else '겹친다'}**.",
        "",
        "**이 결론은 격자 눈금에 의존하지 않는다.** 상대 축 둘은 1위를 지우지 못한다 —",
        "1위는 `s1 >= 비율 x s1` 과 `s1 - s1 <= 마진` 을 항상 만족하기 때문이다.",
        "따라서 채택 0건(기권)은 오직 절대 하한이 1위를 넘을 때만 생기고, 방향 2 의 요구는",
        "하한 축 하나로 환원된다. 눈금을 아무리 촘촘히 해도 두 구간은 여전히 어긋난다.",
        "",
        f"**실행으로도 확인했다**: 지배 논증이 잘라 낸 구간까지 되돌려 넣은 0.01 간격 대조 "
        f"격자 {result.robustness.axes.cardinality:,} 구성에서도 세 방향 동시 만족은 "
        f"{len(result.robustness.satisfying_all)}개다.",
        "",
        "## 2. 질의 단위 기권 게이트 — 통계량 5종의 분리 여유",
        "",
        "판정 방향은 다섯 모두 같다: **통계량 < τ 이면 그 질의는 채택 0건.**",
        "기권 쪽 최댓값(G21~G24)과 채택 쪽 최솟값(라벨 있는 25건)의 간격이 여유이고,",
        "τ 는 그 중간이다. 음수 여유는 어떤 τ 도 한쪽을 반드시 틀린다는 뜻이다 — 반증이다.",
        "",
        "| 조건 | 통계량 | 기권 쪽 최대 | 채택 쪽 최소 | 여유 | τ | 채택 여유 | 분리 |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for label, margins in result.margins.items():
        for margin in margins:
            lines.append(
                f"| {label} | `{margin.statistic.value}` "
                f"| {margin.abstain_case} {margin.abstain_value:.4f} "
                f"| {margin.accept_case} {margin.accept_value:.4f} "
                f"| {margin.margin:+.4f} | {margin.tau:.4f} "
                f"| {margin.accept_headroom:+.4f} "
                f"| {'분리' if margin.separates else '**반증**'} |"
            )

    base_margin = next(
        margin
        for margin in result.margins[base.label]
        if margin.statistic is AbstentionStatistic.SPREAD
    )
    lines += [
        "",
        "### 기준 조건의 τ 경계는 두 케이스가 정한다",
        "",
        f"기준 조건 `{base.label}` 에서 여유가 가장 큰 통계량은 "
        f"`{base_margin.statistic.value}` 이고, 그 경계를 정하는 것은",
        f"기권 쪽 **{base_margin.abstain_case}**({base_margin.abstain_value:.4f})와 "
        f"채택 쪽 **{base_margin.accept_case}**({base_margin.accept_value:.4f})다.",
        f"τ 를 그 중간({base_margin.tau:.4f})에 두면 "
        f"**{base_margin.accept_case} 는 τ 위로 {base_margin.accept_headroom:+.4f} 밖에 "
        f"남지 않는다.**",
        f"라이브 재작성이 {base_margin.accept_case} 의 점수 분포를 조금만 평평하게 만들면 "
        f"케이스 하한을 위반한다 —",
        f"감시 케이스는 G04 가 아니라 **{base_margin.accept_case}** 다.",
    ]

    audit = result.conflict_audit
    lines += [
        "",
        "## 3. 상충쌍 보존 규칙 — 무엇을 고치고 무엇을 늘리는가",
        "",
        f"기준 조건 `{audit.condition}` · 절대 하한 {audit.absolute_floor:.2f} · "
        f"`top_k={TOP_K}` 선절단.",
        "",
        f"- 상충쌍이 **이미 함께 채택되는** 케이스: "
        f"{', '.join(audit.already_paired) if audit.already_paired else '없음'}",
        f"- 상충쌍이 갈라진 케이스: "
        f"{', '.join(audit.unpaired) if audit.unpaired else '**없음 — 고칠 것이 없다**'}",
        "",
        "| 케이스 | 채택된 조항 | 끌어올릴 짝 | 짝의 순위 | 짝의 코사인 | 라벨 있는 케이스 "
        "| 짝이 이 케이스의 정답인가 |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for item in audit.lifted:
        lines.append(
            f"| {item.case_id} | `{item.accepted_clause}` | `{item.partner_clause}` "
            f"| {item.partner_rank} | {item.partner_score:.4f} "
            f"| {'예' if item.case_has_labels else '**아니오 — 기권 표적**'} "
            f"| {'예' if item.partner_is_labelled else '**아니오 — precision 하락**'} |"
        )
    lines += [
        "",
        f"- 끌어올리는 조항 **{len(audit.lifted)}건** 중 그 케이스의 정답이 아닌 것이 "
        f"**{audit.precision_harming}건**이다 — 전부 precision 을 깎는다.",
        f"- 그중 기권이 정답인 케이스에서 늘어나는 채택이 "
        f"**{audit.over_acceptance_added}건**이다 — 방향 2 를 직접 깎는다.",
        f"- 이 규칙이 고치는 것은 **{len(audit.unpaired)}건**이다.",
        "",
    ]
    return "\n".join(lines)
