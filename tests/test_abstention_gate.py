"""질의 단위 기권 게이트와 그 격자의 채점 규칙.

게이트는 **결정론 후처리**다 — 점수는 그대로 두고 질의 단위로 채택 집합을 비운다. 그래서
이 테스트가 확인하는 것은 둘이다.

1. **게이트 자체**(`retrieval_strategies`)는 점수만 본다. 라벨도 케이스 정체성도 LLM 도
   입력이 아니다. 구조 경계는 `tests/test_answer_key_isolation.py` 가 지키고, 여기서는
   같은 점수열이면 언제나 같은 판정이 나온다는 행동을 못박는다.
2. **채점자**(`retrieval_eval`)는 라벨을 본다. 케이스 하한 → 상충쌍 보존 → 기권 → 비악화
   순서로 구성을 자르고, 위반 구성이 **실제로 탈락으로 찍히는지**를 음성 대조와 함께 본다.

손계산(`reply_gate.adoption_axis`)이 이미 낸 수치가 있으므로, 커밋된 검색 산출물로 격자가
그 수치를 그대로 재현하는지도 여기서 확인한다 — 무과금이다.
"""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from reply_gate.adoption_axis import (
    ABSTENTION_CASE_IDS,
    CONFLICT_PAIR_CASE_IDS,
)
from reply_gate.adoption_axis import (
    AbstentionStatistic as HandCalcStatistic,
)
from reply_gate.adoption_axis import (
    statistic_value as handcalc_statistic_value,
)
from reply_gate.retrieval_eval import (
    DEFAULT_ABSTENTION_TAU_AXES,
    AbstentionGrid,
    AbstentionGridPoint,
    AbstentionGridUnmeasured,
    AdoptionConstraint,
    AdoptionVerdict,
    RankedHit,
    StrategyRetrieval,
    StrategyRetrievedCase,
    run_abstention_grid,
    run_retrieval_comparison,
)
from reply_gate.retrieval_labels import RetrievalLabel
from reply_gate.retrieval_strategies import (
    AbstentionGate,
    AbstentionStatistic,
    RetrievalStage,
    StrategyDefinition,
    abstention_statistic,
    apply_abstention_gate,
    truncate_for_gate,
)

_ROOT = Path(__file__).resolve().parents[1]
_BASE_REPORT = (
    _ROOT / "reports" / "retrieval-strategies-live-text-embedding-3-small-d1536-blind-k5-c030.json"
)
#: 결정 기록이 소수 4자리로 인용한다.
_QUOTED = 5e-5
_LIVE_CUTOFF = 0.30
_TOP_K = 5

_VECTOR_REWRITE = StrategyDefinition(
    "vector_rewrite", (RetrievalStage.VECTOR, RetrievalStage.REWRITE)
)


def _case(case_id: str, scores: Sequence[float]) -> StrategyRetrievedCase:
    hits = tuple(
        RankedHit(rank=index, evidence_id=f"policy:x:{index}", similarity=score)
        for index, score in enumerate(scores, start=1)
    )
    return StrategyRetrievedCase(case_id=case_id, ranked_hits=hits, accept_candidates=hits)


def _label(case_id: str, *evidence_ids: str) -> RetrievalLabel:
    return RetrievalLabel(id=case_id, relevant_evidence_ids=frozenset(evidence_ids), note="")


#: 제약을 건드리지 않는 채움 케이스. 격자는 표적 케이스가 **전부** 있어야 돌므로(없으면
#: 미실행 + 사유), 합성 픽스처는 관심 없는 표적을 중립값으로 채운다.
_FILLER_ABSTAIN = (0.10, 0.09, 0.08, 0.07, 0.06)
_FILLER_PAIR = (0.90, 0.89, 0.88, 0.87, 0.86)


def _grid(
    cases: Sequence[StrategyRetrievedCase],
    labels: Sequence[RetrievalLabel],
    *,
    cutoff: float,
) -> AbstentionGrid | AbstentionGridUnmeasured:
    """관심 케이스만 주고 나머지 표적은 중립값으로 채워 격자를 돌린다."""
    supplied = {case.case_id for case in cases}
    padded = list(cases)
    padded_labels = list(labels)
    for case_id in ABSTENTION_CASE_IDS:
        if case_id not in supplied:
            padded.append(_case(case_id, _FILLER_ABSTAIN))
            padded_labels.append(_label(case_id))
    for case_id in CONFLICT_PAIR_CASE_IDS:
        if case_id not in supplied:
            padded.append(_case(case_id, _FILLER_PAIR))
            padded_labels.append(_label(case_id, "policy:x:1", "policy:x:2"))
    retrieval = StrategyRetrieval(
        strategy=_VECTOR_REWRITE, accept_limit=_TOP_K, cases=tuple(padded)
    )
    return run_abstention_grid(retrieval, tuple(padded_labels), cutoff=cutoff)


def _measured(
    cases: Sequence[StrategyRetrievedCase],
    labels: Sequence[RetrievalLabel],
    *,
    cutoff: float,
) -> AbstentionGrid:
    result = _grid(cases, labels, cutoff=cutoff)
    assert isinstance(result, AbstentionGrid)
    return result


def _live_slice() -> tuple[StrategyRetrieval, tuple[RetrievalLabel, ...]]:
    """커밋된 라이브 산출물의 `vector_rewrite` 행을 격자 입력 형태로 되살린다.

    `accept_candidates` 를 전체 순위로 두고 `accept_limit=5` 로 자르는 것이 런타임과 같다 —
    `search_policy_chunks` 가 SQL `LIMIT top_k` 로 자른 뒤 파이썬이 임계값을 건다.
    """
    raw = json.loads(_BASE_REPORT.read_text(encoding="utf-8"))
    row = next(item for item in raw["strategies"] if item["name"] == "vector_rewrite")
    cases: list[StrategyRetrievedCase] = []
    labels: list[RetrievalLabel] = []
    for case in row["cases"]:
        hits = tuple(
            RankedHit(
                rank=int(hit["rank"]),
                evidence_id=str(hit["evidence_id"]),
                similarity=hit["vector_similarity"],
            )
            for hit in case["ranked_hits"]
        )
        cases.append(
            StrategyRetrievedCase(case_id=str(case["id"]), ranked_hits=hits, accept_candidates=hits)
        )
        labels.append(
            _label(str(case["id"]), *(str(item) for item in case["relevant_evidence_ids"]))
        )
    retrieval = StrategyRetrieval(strategy=_VECTOR_REWRITE, accept_limit=_TOP_K, cases=tuple(cases))
    return retrieval, tuple(labels)


@pytest.fixture(scope="module")
def live_grid() -> AbstentionGrid:
    retrieval, labels = _live_slice()
    result = run_abstention_grid(retrieval, labels, cutoff=_LIVE_CUTOFF)
    assert isinstance(result, AbstentionGrid)
    return result


def _point(grid: AbstentionGrid, statistic: AbstentionStatistic, tau: float) -> AbstentionGridPoint:
    return next(
        point
        for point in grid.points
        if point.gate is not None
        and point.gate.statistic is statistic
        and point.gate.tau == pytest.approx(tau)
    )


# ---------------------------------------------------------------------------
# 계약 A — 결정론 후처리. 점수는 그대로 두고 질의 단위로 채택 집합만 비운다.
# ---------------------------------------------------------------------------


def test_게이트는_점수를_건드리지_않고_질의_단위로_채택_집합만_비운다() -> None:
    case = _case("G15", (0.90, 0.89, 0.88, 0.87, 0.86))

    grid = _measured((case,), (_label("G15", "policy:x:1"),), cutoff=0.30)
    fired = _point(grid, AbstentionStatistic.SPREAD, 0.10)
    open_gate = _point(grid, AbstentionStatistic.SPREAD, 0.00)

    rows_fired = {row.case_id: row for row in fired.cases}
    rows_open = {row.case_id: row for row in open_gate.cases}
    # 채택 0건이 되는 것은 질의 전체다 — 항목별로 일부만 남지 않는다.
    assert rows_fired["G15"].accepted_count == 0
    assert rows_fired["G15"].gate_fired is True
    assert rows_open["G15"].accepted_count == 5
    # 순위와 점수는 그대로다. 게이트는 채택 집합만 만진다.
    assert case.ranked_hits[0].similarity == 0.90


def test_절대_하한은_그대로_남고_게이트가_그_위에_얹힌다() -> None:
    """항목 채택은 여전히 절대 하한이 자른다(결정 0009) — 게이트는 질의 단위 층이다."""
    grid = _measured(
        (_case("G15", (0.90, 0.20, 0.10, 0.05, 0.01)),),
        (_label("G15", "policy:x:1"),),
        cutoff=0.30,
    )

    open_gate = _point(grid, AbstentionStatistic.SPREAD, 0.00)
    rows = {row.case_id: row for row in open_gate.cases}

    assert rows["G15"].accepted_count == 1
    assert rows["G15"].accepted_evidence_ids == ("policy:x:1",)


def test_게이트_판정은_같은_점수열에_대해_항상_같다() -> None:
    scores = (0.51, 0.48, 0.47, 0.46, 0.45)
    gate = AbstentionGate(statistic=AbstentionStatistic.SPREAD, tau=0.10)

    assert apply_abstention_gate(gate, scores) == apply_abstention_gate(gate, scores)
    assert apply_abstention_gate(gate, scores).abstains is True


def test_임계값과_같은_값은_기권시키지_않는다() -> None:
    """`τ 미만이면 기권` 이다 — 등호 방향이 뒤집히면 경계 케이스의 판정이 뒤집힌다.

    등호 자리를 정확히 보려면 이진수로 정확한 값이어야 한다(0.5·0.25). 그렇지 않은 값은
    부동소수 오차가 판정을 먼저 정한다 — τ 를 눈금 위에 두는 것이 그래서 중요하다.
    """
    gate = AbstentionGate(statistic=AbstentionStatistic.SPREAD, tau=0.25)

    exact = apply_abstention_gate(gate, (0.5, 0.4, 0.35, 0.3, 0.25))
    below = apply_abstention_gate(gate, (0.5, 0.4, 0.35, 0.3, 0.28))

    assert exact.value == 0.25
    assert exact.abstains is False
    assert below.abstains is True


# ---------------------------------------------------------------------------
# 계약 B — 통계량 5종을 전부 격자에 넣는다. 반증된 둘도 산출물이다.
# ---------------------------------------------------------------------------


def test_격자는_통계량_다섯을_전부_축으로_돈다(live_grid: AbstentionGrid) -> None:
    swept = {point.gate.statistic for point in live_grid.points if point.gate is not None}

    assert swept == set(AbstentionStatistic)
    assert len(AbstentionStatistic) == 5
    assert set(DEFAULT_ABSTENTION_TAU_AXES) == set(AbstentionStatistic)


def test_반증된_통계량_둘도_격자에_남아_결과를_남긴다(live_grid: AbstentionGrid) -> None:
    """손계산이 분리하지 못한 둘을 격자에서 빼지 않는다 — 반증도 산출물이다."""
    separates = {item.statistic for item in live_grid.separations if item.separates}

    assert separates == {
        AbstentionStatistic.SPREAD,
        AbstentionStatistic.STDEV,
        AbstentionStatistic.RELATIVE_SPREAD,
    }
    refuted = set(AbstentionStatistic) - separates
    assert refuted == {AbstentionStatistic.GAP_1_2, AbstentionStatistic.TAIL_RATIO}
    for statistic in refuted:
        points = [
            point
            for point in live_grid.points
            if point.gate is not None and point.gate.statistic is statistic
        ]
        # 축이 통째로 돌았고, 그 축의 어떤 τ 도 채택 후보가 되지 못한다.
        assert points, f"{statistic} 축이 격자에서 빠졌다"
        assert all(point.verdict is AdoptionVerdict.ELIMINATED for point in points)


def test_반증된_통계량의_여유가_손계산_인용값과_같다(live_grid: AbstentionGrid) -> None:
    by_statistic = {item.statistic: item for item in live_grid.separations}

    assert by_statistic[AbstentionStatistic.GAP_1_2].margin == pytest.approx(-0.0324, abs=_QUOTED)
    assert by_statistic[AbstentionStatistic.TAIL_RATIO].margin == pytest.approx(
        -0.4677, abs=_QUOTED
    )


# ---------------------------------------------------------------------------
# 계약 C — 통계량은 상위 `top_k` 슬라이스만 본다(결정 0012 의 선절단 계약).
# ---------------------------------------------------------------------------


def test_통계량은_top_k_선절단_슬라이스만_본다() -> None:
    similarities = [0.90, 0.88, 0.86, 0.84, 0.82, 0.10, 0.05]

    cut = truncate_for_gate(similarities, top_k=5)

    assert cut == (0.90, 0.88, 0.86, 0.84, 0.82)
    assert abstention_statistic(AbstentionStatistic.SPREAD, cut) == pytest.approx(0.08)
    # 전체 순위로 계산하면 다른 수가 나온다 — 그래서 이 절단이 행동 계약이다.
    assert abstention_statistic(AbstentionStatistic.SPREAD, tuple(similarities)) == pytest.approx(
        0.85
    )


def test_격자도_top_k_밖의_꼬리를_통계량에_넣지_않는다() -> None:
    """`accept_candidates` 가 전체 순위여도 게이트는 `accept_limit` 까지만 본다."""
    grid = _measured(
        (_case("G15", (0.90, 0.88, 0.86, 0.84, 0.82, 0.01)),),
        (_label("G15", "policy:x:1"),),
        cutoff=0.30,
    )

    point = _point(grid, AbstentionStatistic.SPREAD, 0.00)
    rows = {row.case_id: row for row in point.cases}

    assert rows["G15"].statistic_value == pytest.approx(0.08)


def test_측정되지_않은_후보는_통계량에서_빠지고_0_으로_채우지_않는다() -> None:
    """어휘 다리에만 있던 조항은 코사인이 **미측정**이다 — 0.0 과 같은 값이 아니다."""
    assert truncate_for_gate([0.9, None, 0.7], top_k=3) == (0.9, 0.7)

    single = apply_abstention_gate(
        AbstentionGate(statistic=AbstentionStatistic.SPREAD, tau=0.05), (0.9,)
    )

    assert single.value is None
    assert single.undefined_reason is not None
    # 미정의를 0 으로 읽으면 모든 τ 에서 기권한다 — 게이트는 발동하지 않고 사유를 남긴다.
    assert single.abstains is False


def test_통계량이_미정의면_케이스_행에도_사유가_남는다() -> None:
    grid = _measured(
        (_case("G15", (0.90,)),),
        (_label("G15", "policy:x:1"),),
        cutoff=0.30,
    )

    point = _point(grid, AbstentionStatistic.SPREAD, 0.10)
    rows = {row.case_id: row for row in point.cases}

    assert rows["G15"].statistic_value is None
    assert rows["G15"].statistic_undefined_reason is not None
    assert rows["G15"].margin_to_tau is None
    assert rows["G15"].gate_fired is False


# ---------------------------------------------------------------------------
# 계약 D — τ 는 스윕 축이고, 경계가 보일 만큼 촘촘하다.
# ---------------------------------------------------------------------------


def test_tau_축은_관측_범위를_덮고_경계_양쪽에_점을_둔다(live_grid: AbstentionGrid) -> None:
    axis = DEFAULT_ABSTENTION_TAU_AXES[AbstentionStatistic.SPREAD]
    separation = next(
        item for item in live_grid.separations if item.statistic is AbstentionStatistic.SPREAD
    )

    assert axis[0] == 0.0
    assert axis[-1] > separation.accept_value
    # 경계 양쪽에 점이 있어야 "이긴 τ" 만이 아니라 **경계** 가 보인다.
    assert any(tau <= separation.abstain_value for tau in axis)
    assert any(separation.abstain_value < tau <= separation.accept_value for tau in axis)
    assert any(tau > separation.accept_value for tau in axis)


def test_tau_0_은_게이트_꺼짐과_같은_구성이다(live_grid: AbstentionGrid) -> None:
    """스윕 안에 대조군이 들어 있다 — 게이트가 무엇을 바꿨는지 같은 표에서 읽힌다."""
    zero = _point(live_grid, AbstentionStatistic.SPREAD, 0.0)

    assert zero.aggregate == live_grid.baseline.aggregate
    assert all(not row.gate_fired for row in zero.cases)


# ---------------------------------------------------------------------------
# 계약 E — 제약 셋 + 비악화, 그 순서.
# ---------------------------------------------------------------------------


def test_정답_조항을_못_채택하면_오답을_채택했어도_즉시_탈락한다() -> None:
    """ "근거 0건 금지" 가 아니라 **정답 조항 채택** 이다 — G04 의 1위는 오답이다."""
    cases = (_case("G04", (0.3647, 0.3571, 0.35, 0.34, 0.33)),)
    labels = (_label("G04", "policy:x:2"),)

    # 컷 0.36 은 오답 1위만 남긴다 — 근거는 1건이지만 정답은 0건이다.
    wrong_only = _measured(cases, labels, cutoff=0.36)
    both = _measured(cases, labels, cutoff=0.30)

    wrong_row = next(row for row in wrong_only.baseline.cases if row.case_id == "G04")
    assert wrong_row.accepted_count == 1
    assert wrong_row.correct_clause_accepted is False
    assert AdoptionConstraint.CASE_FLOOR in wrong_only.baseline.failed_constraints
    assert "G04" in wrong_only.baseline.case_floor_violations
    # 음성 대조 — 정답 조항이 채택되면 같은 케이스가 하한을 통과한다.
    both_row = next(row for row in both.baseline.cases if row.case_id == "G04")
    assert both_row.correct_clause_accepted is True
    assert AdoptionConstraint.CASE_FLOOR not in both.baseline.failed_constraints


def test_케이스_하한은_라벨이_빈_케이스에는_걸리지_않는다() -> None:
    """G21~G24·G28 은 기권이 정답이다 — 하한 대상으로 세면 방향 1 과 2 가 서로를 부순다."""
    grid = _measured((_case("G28", (0.50, 0.49, 0.48, 0.47, 0.46)),), (_label("G28"),), cutoff=0.30)

    rows = {row.case_id: row for row in grid.baseline.cases}

    assert rows["G28"].correct_clause_accepted is None
    assert rows["G28"].labelled is False
    assert "G28" not in grid.case_floor_case_ids
    assert "G21" not in grid.case_floor_case_ids
    assert "G01" in grid.case_floor_case_ids


def test_라이브_슬라이스의_케이스_하한_대상은_25건이다(live_grid: AbstentionGrid) -> None:
    assert len(live_grid.case_floor_case_ids) == 25
    assert len(live_grid.baseline.cases) == 30
    assert set(ABSTENTION_CASE_IDS).isdisjoint(live_grid.case_floor_case_ids)


def test_상충쌍이_갈라지면_표기되고_탈락한다() -> None:
    cases = (_case("G18", (0.50, 0.29, 0.28, 0.27, 0.26)),)
    labels = (_label("G18", "policy:x:1", "policy:x:2"),)

    split = _measured(cases, labels, cutoff=0.30)
    together = _measured(cases, labels, cutoff=0.25)

    rows = {row.case_id: row for row in split.baseline.cases}
    assert rows["G18"].conflict_pair_kept is False
    assert AdoptionConstraint.CONFLICT_PAIR in split.baseline.failed_constraints
    # 케이스 하한은 통과한다 — 정답 조항 하나는 채택됐다. 하한으로는 못 잡는 **부분 손실**이다.
    assert AdoptionConstraint.CASE_FLOOR not in split.baseline.failed_constraints
    # 음성 대조.
    together_rows = {row.case_id: row for row in together.baseline.cases}
    assert together_rows["G18"].conflict_pair_kept is True
    assert AdoptionConstraint.CONFLICT_PAIR not in together.baseline.failed_constraints


def test_상충쌍_제약은_G18_G01_G02_셋을_본다(live_grid: AbstentionGrid) -> None:
    """사이클 3 문서가 G02 를 빠뜨렸다 — 셋이어야 한다."""
    assert set(CONFLICT_PAIR_CASE_IDS) == {"G01", "G02", "G18"}
    assert set(live_grid.baseline.conflict_pair_kept) == {"G01", "G02", "G18"}
    assert all(live_grid.baseline.conflict_pair_kept.values())


def test_과채택을_줄이기만_하고_기권하지_못한_구성은_탈락한다(
    live_grid: AbstentionGrid,
) -> None:
    """줄인 구성이 오프라인을 통과하면 라이브 표적(기권 ≥10/12)에서 0/12 로 떨어진다."""
    reduced = _point(live_grid, AbstentionStatistic.SPREAD, 0.05)
    abstained = _point(live_grid, AbstentionStatistic.SPREAD, 0.06)
    baseline_total = sum(live_grid.baseline.abstention_accepted_counts.values())

    # 넷 중 셋이 기권했다 — 과채택은 분명히 **줄었다**.
    reduced_total = sum(reduced.abstention_accepted_counts.values())
    assert 0 < reduced_total < baseline_total
    # 그런데도 탈락이고, 탈락 사유는 기권 표적 하나다.
    assert reduced.verdict is AdoptionVerdict.ELIMINATED
    assert reduced.failed_constraints == (AdoptionConstraint.ABSTENTION_TARGET,)
    # 음성 대조 — 넷 전부 채택 0건이면 통과한다.
    assert sum(abstained.abstention_accepted_counts.values()) == 0
    assert abstained.verdict is AdoptionVerdict.ADOPTABLE


def test_비악화는_기준선_대비로_판정하고_recall_은_등호가_기본선이다(
    live_grid: AbstentionGrid,
) -> None:
    baseline = live_grid.baseline
    winner = _point(live_grid, AbstentionStatistic.SPREAD, 0.06)

    assert baseline.aggregate.accepted_recall == pytest.approx(1.0)
    assert winner.aggregate.accepted_recall == pytest.approx(1.0)
    # 게이트는 순위를 건드리지 않는다 — r@1 은 정의상 기준선과 같다.
    assert winner.aggregate.recall_at_1 == baseline.aggregate.recall_at_1
    assert winner.degraded_metrics == ()
    assert AdoptionConstraint.NON_DEGRADATION not in winner.failed_constraints


def test_정답_조항을_잃은_구성은_하한과_비악화_둘_다_위반하고_하한이_먼저다(
    live_grid: AbstentionGrid,
) -> None:
    """감시 케이스 G15 가 τ 위로 +0.0073 밖에 안 남는다 — 한 눈금 위에서 잘린다."""
    too_high = _point(live_grid, AbstentionStatistic.SPREAD, 0.07)

    assert too_high.verdict is AdoptionVerdict.ELIMINATED
    assert "G15" in too_high.case_floor_violations
    assert "accepted_recall" in too_high.degraded_metrics
    assert AdoptionConstraint.NON_DEGRADATION in too_high.failed_constraints
    # 우선순위 — 케이스 하한이 먼저다. macro 수치가 무엇이든 즉시 탈락이다.
    assert too_high.eliminated_by is AdoptionConstraint.CASE_FLOOR


def test_탈락_사유는_제약_우선순위_순서로_기록된다() -> None:
    assert tuple(AdoptionConstraint) == (
        AdoptionConstraint.CASE_FLOOR,
        AdoptionConstraint.CONFLICT_PAIR,
        AdoptionConstraint.ABSTENTION_TARGET,
        AdoptionConstraint.NON_DEGRADATION,
    )


def test_precision_만_오른_구성은_구제되지_않는다(live_grid: AbstentionGrid) -> None:
    """기권한 라벨 케이스는 precision 분모에서 **빠져** macro precision 을 올린다."""
    too_high = _point(live_grid, AbstentionStatistic.SPREAD, 0.07)
    baseline_precision = live_grid.baseline.aggregate.accepted_precision
    assert baseline_precision is not None
    precision = too_high.aggregate.accepted_precision
    assert precision is not None

    assert precision > baseline_precision
    assert too_high.verdict is AdoptionVerdict.ELIMINATED


# ---------------------------------------------------------------------------
# 계약 F — 채점자는 라벨을 보고, 전략은 못 본다.
# ---------------------------------------------------------------------------


def test_게이트는_라벨도_케이스_정체성도_입력으로_받지_않는다() -> None:
    parameters = set(inspect.signature(apply_abstention_gate).parameters)

    assert parameters == {"gate", "scores"}
    assert "labels" not in set(inspect.signature(abstention_statistic).parameters)
    assert "case_id" not in set(inspect.signature(truncate_for_gate).parameters)


def test_같은_점수열이면_어느_케이스든_같은_판정이다() -> None:
    gate = AbstentionGate(statistic=AbstentionStatistic.STDEV, tau=0.02)
    scores = (0.48, 0.47, 0.465, 0.46, 0.455)

    grid = _measured(
        (_case("G21", scores), _case("G15", scores)),
        (_label("G21"), _label("G15", "policy:x:1")),
        cutoff=0.30,
    )
    point = _point(grid, AbstentionStatistic.STDEV, 0.02)
    rows = {row.case_id: row for row in point.cases}

    assert rows["G21"].statistic_value == rows["G15"].statistic_value
    assert rows["G21"].gate_fired == rows["G15"].gate_fired
    assert rows["G21"].gate_fired is apply_abstention_gate(gate, scores).abstains


# ---------------------------------------------------------------------------
# 계약 G — 리포트는 구성별 macro 와 케이스별 행, 그리고 경계까지의 여유를 싣는다.
# ---------------------------------------------------------------------------


def test_구성마다_케이스별_행과_경계_여유가_남는다(live_grid: AbstentionGrid) -> None:
    winner = _point(live_grid, AbstentionStatistic.SPREAD, 0.06)

    assert len(winner.cases) == 30
    row = next(item for item in winner.cases if item.case_id == "G15")
    assert row.correct_clause_accepted is True
    assert row.statistic_value == pytest.approx(0.0668, abs=_QUOTED)
    assert row.margin_to_tau == pytest.approx(0.0068, abs=_QUOTED)
    assert set(winner.abstention_accepted_counts) == set(ABSTENTION_CASE_IDS)
    assert set(winner.conflict_pair_kept) == set(CONFLICT_PAIR_CASE_IDS)


def test_경계_케이스는_이름으로_찍힌다(live_grid: AbstentionGrid) -> None:
    """τ 가 얼마나 아슬아슬한지가 라이브 이전 가능성의 유일한 사전 신호다."""
    winner = _point(live_grid, AbstentionStatistic.SPREAD, 0.06)
    assert winner.accept_boundary is not None
    assert winner.abstain_boundary is not None

    assert winner.accept_boundary.case_id == "G15"
    assert winner.accept_boundary.margin == pytest.approx(0.0068, abs=_QUOTED)
    assert winner.abstain_boundary.case_id == "G23"
    assert winner.abstain_boundary.margin == pytest.approx(-0.0079, abs=_QUOTED)


def test_구성_식별자가_축을_전부_들고_있다(live_grid: AbstentionGrid) -> None:
    winner = _point(live_grid, AbstentionStatistic.SPREAD, 0.06)

    assert (
        winner.configuration_id
        == "vector_rewrite/cut=0.300/stat=rank1_minus_rank_k_spread/tau=0.0600"
    )
    assert live_grid.baseline.configuration_id == "vector_rewrite/cut=0.300/gate=off"
    # 구분자에 `|` 를 쓰면 Markdown 표 칸이 쪼개진다 — 구성 이름 하나가 표를 망가뜨린다.
    assert "|" not in winner.configuration_id


# ---------------------------------------------------------------------------
# 손계산 재현 — 커밋된 산출물로 결정 0012 의 수치가 그대로 나온다(무과금).
# ---------------------------------------------------------------------------


def test_격자가_손계산의_분리_여유와_tau_를_그대로_낸다(live_grid: AbstentionGrid) -> None:
    spread = next(
        item for item in live_grid.separations if item.statistic is AbstentionStatistic.SPREAD
    )

    assert spread.abstain_case == "G23"
    assert spread.abstain_value == pytest.approx(0.0521, abs=_QUOTED)
    assert spread.accept_case == "G15"
    assert spread.accept_value == pytest.approx(0.0668, abs=_QUOTED)
    assert spread.margin == pytest.approx(0.0147, abs=_QUOTED)
    assert spread.tau == pytest.approx(0.0595, abs=_QUOTED)
    assert spread.accept_headroom == pytest.approx(0.0073, abs=_QUOTED)


def test_통계량이_손계산_모듈과_같은_수를_낸다() -> None:
    """게이트는 런타임 쪽, 손계산은 채점자 쪽에 있다 — 두 구현이 갈라지면 배선이 어긋난다."""
    retrieval, _ = _live_slice()
    pairs = {
        AbstentionStatistic.SPREAD: HandCalcStatistic.SPREAD,
        AbstentionStatistic.STDEV: HandCalcStatistic.STDEV,
        AbstentionStatistic.RELATIVE_SPREAD: HandCalcStatistic.RELATIVE_SPREAD,
        AbstentionStatistic.GAP_1_2: HandCalcStatistic.GAP_1_2,
        AbstentionStatistic.TAIL_RATIO: HandCalcStatistic.TAIL_RATIO,
    }

    assert {item.value for item in AbstentionStatistic} == {
        item.value for item in HandCalcStatistic
    }
    for case in retrieval.cases:
        scores = truncate_for_gate([hit.similarity for hit in case.accept_candidates], top_k=_TOP_K)
        for statistic, handcalc in pairs.items():
            assert abstention_statistic(statistic, scores) == pytest.approx(
                handcalc_statistic_value(handcalc, scores), abs=1e-12
            )


def test_격자가_고르는_채택_후보가_손계산_결론과_같다(live_grid: AbstentionGrid) -> None:
    """셋은 분리하고 둘은 못 한다 — 유료 실측 전에 커밋된 산출물이 이미 답한 부분이다."""
    adoptable = {
        (point.gate.statistic, point.gate.tau)
        for point in live_grid.adoptable
        if point.gate is not None
    }

    assert adoptable == {
        (AbstentionStatistic.SPREAD, 0.06),
        (AbstentionStatistic.STDEV, 0.02),
        (AbstentionStatistic.STDEV, 0.025),
        (AbstentionStatistic.RELATIVE_SPREAD, 0.12),
    }


def test_격자_크기가_선언된_축과_일치한다(live_grid: AbstentionGrid) -> None:
    expected = sum(len(axis) for axis in DEFAULT_ABSTENTION_TAU_AXES.values())

    assert len(live_grid.points) == expected == 165


# ---------------------------------------------------------------------------
# 미실행은 0 이 아니라 사유다.
# ---------------------------------------------------------------------------


def test_표적_케이스가_없으면_격자는_미실행_사유를_남긴다() -> None:
    retrieval = StrategyRetrieval(
        strategy=_VECTOR_REWRITE,
        accept_limit=_TOP_K,
        cases=(_case("A", (0.9, 0.8, 0.7, 0.6, 0.5)),),
    )

    result = run_abstention_grid(retrieval, (_label("A", "policy:x:1"),), cutoff=0.30)

    assert isinstance(result, AbstentionGridUnmeasured)
    assert "G21" in result.reason


def test_기권_표적에_정답_라벨이_있으면_대조_불가다() -> None:
    """방향 1 과 2 가 충돌하는 입력은 0 으로 채우지 않고 사유로 남긴다."""
    result = _grid(
        (_case("G21", (0.9, 0.8, 0.7, 0.6, 0.5)),),
        (_label("G21", "policy:x:1"),),
        cutoff=0.30,
    )

    assert isinstance(result, AbstentionGridUnmeasured)
    assert "G21" in result.reason


def test_대역_임베딩으로_통계량_5종_x_tau_스윕이_완주한다(tmp_path: Path) -> None:
    paths = run_retrieval_comparison(
        live=False,
        output_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
        strategies=(
            StrategyDefinition("vector", (RetrievalStage.VECTOR,)),
            StrategyDefinition("vector_rewrite", (RetrievalStage.VECTOR, RetrievalStage.REWRITE)),
        ),
    )
    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    markdown = paths.markdown.read_text(encoding="utf-8")

    grid = payload["abstention_grid"]
    assert grid["measured"] is True
    assert grid["strategy"] == "vector_rewrite"
    assert grid["labels_used_for"] == "scoring_only"
    assert grid["truncate_before_threshold"] is True
    assert len(grid["configurations"]) == 165
    assert {row["statistic"] for row in grid["configurations"]} == {
        item.value for item in AbstentionStatistic
    }
    for row in grid["configurations"]:
        assert len(row["cases"]) == 30
        assert row["verdict"] in {item.value for item in AdoptionVerdict}
        assert math.isfinite(row["tau"])
    assert "## 기권 게이트 격자" in markdown
    assert "케이스 하한" in markdown
    _assert_table_columns_align(markdown, "### 구성별 결과")


def _assert_table_columns_align(markdown: str, heading: str) -> None:
    """표의 칸 수가 헤더와 같은지 본다 — 구성 이름 안의 `|` 하나가 표를 통째로 망가뜨린다."""
    section = markdown.split(heading, 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("|")]
    assert rows, f"{heading} 아래에 표가 없다"
    expected = rows[0].count("|")
    misaligned = [row for row in rows if row.count("|") != expected]
    assert not misaligned, misaligned[:2]


def test_격자를_끄면_리포트에_미실행_사유가_남는다(tmp_path: Path) -> None:
    """무과금 격자라도 끌 수 있어야 하고, 끈 실행은 0 이 아니라 사유를 남긴다."""
    paths = run_retrieval_comparison(
        live=False,
        output_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
        strategies=(StrategyDefinition("vector", (RetrievalStage.VECTOR,)),),
        abstention_grid=False,
    )
    payload = json.loads(paths.json.read_text(encoding="utf-8"))

    grid = payload["abstention_grid"]
    assert grid["measured"] is False
    assert "미실행" in grid["reason"]
    assert "configurations" not in grid
    assert grid["reason"] in paths.markdown.read_text(encoding="utf-8")
