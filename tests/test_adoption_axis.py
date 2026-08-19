"""채택 축 손계산이 커밋된 산출물에서 문서가 인용하는 수치를 그대로 낸다.

수치를 인용만 하고 재현 경로를 두지 않으면 다음 사람이 검증할 수 없다. 이 테스트가
그 경로다 — 입력은 저장소에 커밋된 `reports/retrieval-strategies-live-*.json` 뿐이고
과금은 없다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reply_gate.adoption_axis import (
    ABSTENTION_CASE_IDS,
    DEFAULT_GRID_AXES,
    DEFAULT_REPORT_DIR,
    FINE_GRID_AXES,
    TOP_K,
    AbstentionStatistic,
    ConditionSlice,
    Configuration,
    Direction,
    accepted_ids,
    audit_conflict_pairs,
    default_conditions,
    load_condition,
    render_markdown,
    run_grid,
    run_hand_calculation,
    separation_margin,
    statistic_value,
    to_json,
)

#: 문서가 소수 4자리로 인용하므로 그 자리까지 맞는지 본다.
_QUOTED = 5e-5


@pytest.fixture(scope="module")
def conditions() -> tuple[ConditionSlice, ...]:
    return default_conditions()


@pytest.fixture(scope="module")
def base(conditions: tuple[ConditionSlice, ...]) -> ConditionSlice:
    return conditions[0]


def test_기준_조건은_현_기본값_조합의_blind_실측이다(base: ConditionSlice) -> None:
    assert base.embedding_model == "text-embedding-3-small"
    assert base.dimensions == 1536
    assert base.rewrite_condition == "blind"
    assert base.top_k == TOP_K
    assert len(base.cases) == 30
    # 라벨이 빈 케이스는 다섯이고 그중 넷이 기권 표적이다.
    empty = {case.case_id for case in base.cases if not case.has_labels}
    assert empty == {"G21", "G22", "G23", "G24", "G28"}
    assert set(ABSTENTION_CASE_IDS) < empty


def test_전수_격자는_576_구성이고_세_방향_동시_만족은_0개다(base: ConditionSlice) -> None:
    outcome = run_grid(base, DEFAULT_GRID_AXES)

    assert DEFAULT_GRID_AXES.cardinality == 576
    assert outcome.total == 576
    assert outcome.satisfying_all == ()
    assert outcome.per_pair[(Direction.EVIDENCE_PRESERVED, Direction.UNGROUNDED_ABSTAINS)] == 0


def test_방향_1_상한과_방향_2_하한이_공집합이다(base: ConditionSlice) -> None:
    outcome = run_grid(base, DEFAULT_GRID_AXES)

    assert outcome.floor_ceiling[0] == "G04"
    assert outcome.floor_ceiling[1] == "policy:support:4-6"
    assert outcome.floor_ceiling[2] == pytest.approx(0.3571, abs=_QUOTED)
    assert outcome.floor_threshold[0] == "G21"
    assert outcome.floor_threshold[1] == "policy:shipping:1-6"
    assert outcome.floor_threshold[2] == pytest.approx(0.4812, abs=_QUOTED)
    assert outcome.empty_intersection


def test_상대_축은_1위를_지우지_못한다_기권은_절대_하한만_만든다(base: ConditionSlice) -> None:
    """공집합이 격자 눈금이 아니라 축의 성질에서 나온다는 것을 실행으로 못박는다."""
    for case in base.cases:
        top_id, top_score = case.hits[0]
        # 비율 1.0 · 마진 0.0 — 상대 축을 최대로 조여도 1위는 남는다.
        strict = Configuration(absolute_floor=0.0, adaptive_ratio=1.0, relative_margin=0.0)
        assert accepted_ids(case, strict)[0] == top_id
        # 절대 하한이 1위를 넘어야만 채택 0건이 된다.
        assert accepted_ids(case, Configuration(top_score + 1e-9, 0.0, float("inf"))) == ()


def test_훨씬_촘촘한_격자에서도_결론이_같다(base: ConditionSlice) -> None:
    fine = run_grid(base, FINE_GRID_AXES)

    assert fine.axes.cardinality > 100_000
    assert fine.satisfying_all == ()


def test_통계량_다섯_중_셋이_분리하고_둘이_반증된다(base: ConditionSlice) -> None:
    margins = {statistic: separation_margin(base, statistic) for statistic in AbstentionStatistic}

    separated = {statistic for statistic, margin in margins.items() if margin.separates}
    assert separated == {
        AbstentionStatistic.SPREAD,
        AbstentionStatistic.STDEV,
        AbstentionStatistic.RELATIVE_SPREAD,
    }
    refuted = set(AbstentionStatistic) - separated
    assert refuted == {AbstentionStatistic.GAP_1_2, AbstentionStatistic.TAIL_RATIO}
    # 반증도 산출물이다 — 값이 남아 있어야 한다.
    assert margins[AbstentionStatistic.GAP_1_2].margin == pytest.approx(-0.0324, abs=_QUOTED)
    assert margins[AbstentionStatistic.TAIL_RATIO].margin == pytest.approx(-0.4677, abs=_QUOTED)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("small-d1536-blind", 0.0147),
        ("small-d1536-oracle", 0.0534),
        ("large-d1536-blind", -0.0114),
        ("large-d3072-blind", -0.0160),
    ],
)
def test_조건별_분리_여유가_문서_인용값과_같다(
    conditions: tuple[ConditionSlice, ...], label: str, expected: float
) -> None:
    condition = next(item for item in conditions if item.label == label)

    margin = separation_margin(condition, AbstentionStatistic.SPREAD)

    assert margin.margin == pytest.approx(expected, abs=_QUOTED)
    # 전 조건 공통 τ 는 없다 — large 계열은 어떤 τ 로도 갈리지 않는다.
    assert margin.separates is (expected > 0)


def test_기준_조건의_경계는_G23_과_G15_가_정하고_G15_여유는_0_0073_이다(
    base: ConditionSlice,
) -> None:
    margin = separation_margin(base, AbstentionStatistic.SPREAD)

    assert margin.abstain_case == "G23"
    assert margin.accept_case == "G15"
    assert margin.abstain_value == pytest.approx(0.0521, abs=_QUOTED)
    assert margin.accept_value == pytest.approx(0.0668, abs=_QUOTED)
    assert margin.accept_headroom == pytest.approx(0.0073, abs=_QUOTED)


def test_top_k_선절단은_행동_계약이라_전체_랭킹으로_계산하면_수가_달라진다(
    base: ConditionSlice,
) -> None:
    """런타임이 임계값 전에 `LIMIT top_k` 로 자르므로 통계량도 그 슬라이스에서 나온다.

    전체 코퍼스 랭킹으로 계산해도 되는 값이면 이 계약이 조정 가능한 인자로 읽힌다.
    실제로는 값이 갈리고, 갈린다는 것을 여기서 못박는다.
    """
    g15 = base.case("G15")
    g23 = base.case("G23")

    truncated = statistic_value(AbstentionStatistic.SPREAD, g15.scores) - statistic_value(
        AbstentionStatistic.SPREAD, g23.scores
    )
    full = statistic_value(
        AbstentionStatistic.SPREAD, [score for _, score in g15.full_ranking]
    ) - statistic_value(AbstentionStatistic.SPREAD, [score for _, score in g23.full_ranking])

    assert len(g15.hits) == TOP_K < len(g15.full_ranking)
    assert truncated == pytest.approx(0.0147, abs=_QUOTED)
    assert full != pytest.approx(truncated, abs=1e-3)


def test_컷_030_은_상충쌍_셋을_이미_함께_채택한다(base: ConditionSlice) -> None:
    audit = audit_conflict_pairs(base, absolute_floor=0.30)

    assert set(audit.already_paired) == {"G01", "G02", "G18"}
    assert audit.unpaired == ()
    assert audit.fixes_nothing


def test_상충쌍_보존은_G22_에서_과채택을_늘린다(base: ConditionSlice) -> None:
    audit = audit_conflict_pairs(base, absolute_floor=0.30)

    lifted = {(item.case_id, item.partner_clause): item for item in audit.lifted}
    g22 = lifted[("G22", "policy:shipping:1-3")]
    assert g22.accepted_clause == "policy:shipping:1-4"
    # `top_k=5` 선절단이 지금 이 조항을 자른다 — 컷 0.30 은 이미 넘는다.
    assert g22.partner_rank == 16
    assert g22.partner_score == pytest.approx(0.3274, abs=_QUOTED)
    assert g22.partner_score > 0.30
    assert not g22.case_has_labels

    assert audit.over_acceptance_added == 1
    assert audit.precision_harming == len(audit.lifted) == 10


def test_반복_실측에서도_같은_수치가_나온다() -> None:
    """같은 조건의 두 번째 라이브 리포트가 손계산을 재현한다 — 산출물 우연이 아니다."""
    repeat = load_condition(
        DEFAULT_REPORT_DIR
        / "retrieval-strategies-live-text-embedding-3-small-d1536-blind-k5-c030-2.json",
        label="small-d1536-blind-2",
    )

    assert separation_margin(repeat, AbstentionStatistic.SPREAD).margin == pytest.approx(
        0.0147, abs=_QUOTED
    )
    assert run_grid(repeat, DEFAULT_GRID_AXES).satisfying_all == ()


def test_라벨을_검색에_쓴_리포트는_손계산_입력이_아니다(tmp_path: Path) -> None:
    """채점자만 정답을 본다 — 검색이 정답을 본 산출물은 이 계산의 입력이 될 수 없다."""
    source = json.loads(
        (
            DEFAULT_REPORT_DIR
            / "retrieval-strategies-live-text-embedding-3-small-d1536-blind-k5-c030.json"
        ).read_text(encoding="utf-8")
    )
    source["run_conditions"]["labels_used_for_retrieval"] = True
    path = tmp_path / "tainted.json"
    path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="검색에 라벨이 쓰인 리포트"):
        load_condition(path, label="tainted")


def test_산출물이_인용되는_수치를_전부_들고_있다(conditions: tuple[ConditionSlice, ...]) -> None:
    result = run_hand_calculation(conditions)
    payload = to_json(result)
    markdown = render_markdown(result)

    grid = payload["grid"]
    assert isinstance(grid, dict)
    assert grid["cardinality"] == 576
    assert grid["satisfying_all_three"] == 0
    assert grid["empty_intersection"] is True
    payload_input = payload["input"]
    assert isinstance(payload_input, dict)
    assert payload_input["top_k"] == TOP_K
    assert payload_input["labels_used_for"] == "scoring_only"
    assert payload_input["truncate_before_threshold"] is True

    for quoted in ("576", "0.3571", "0.4812", "+0.0147", "+0.0534", "-0.0114", "-0.0160"):
        assert quoted in markdown
    assert "G15" in markdown and "G23" in markdown
