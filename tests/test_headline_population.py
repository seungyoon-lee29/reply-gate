"""헤드라인 수치의 **모집단이 문서에 고정돼 있고, 그 규칙으로 재집계한 값이 문서와 같다.**

**왜 이 파일이 있는가.** 미끼 기각 재현 `12/95` · `G06` 계보 `18/19` · `G30` 계보 `4/7` 은
오래 사람이 커밋된 리포트에서 손으로 재집계한 값이었고 **그 수를 다시 계산하는 코드가
0줄**이었다([미해결 36](../docs/tracking/findings.md)).

**순서를 지킨다.** 모집단의 정본은 `docs/tracking/status.md` 의 표이고 재집계기는 그것을
**읽는다**. 반대로 세우면 *"구현이 스스로 모집단을 정한다"* 가 되고, 그것이 36 번이 미뤄진
이유였다. 그래서 이 파일의 음성 대조는 두 방향을 다 잰다 — 문서를 바꾸면 재집계가 따라
갈리는가(정본이 맞는가), 그리고 모르는 어휘를 만나면 조용히 넘어가지 않는가(fail-closed).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.recount_headline import RULES_DOC, RULES_HEADING, main, parse_rules, recount

#: 문서가 고정한 모집단 크기. 규칙이 조용히 넓어지면 여기서 먼저 걸린다.
_모집단_크기 = {"미끼 기각 재현": 19, "G06 계보": 19, "G30 계보": 7}


def _문서_사본(tmp_path: Path, *, 치환: tuple[str, str] | None = None) -> Path:
    """정본 문서를 복사해 한 곳만 바꾼 사본. 실제 문서는 건드리지 않는다."""
    본문 = RULES_DOC.read_text(encoding="utf-8")
    if 치환 is not None:
        옛것, 새것 = 치환
        assert 옛것 in 본문, f"음성 대조가 겨냥한 문면이 문서에 없다: {옛것!r}"
        본문 = 본문.replace(옛것, 새것, 1)
    사본 = tmp_path / "status.md"
    사본.write_text(본문, encoding="utf-8")
    return 사본


# ── 구속 ────────────────────────────────────────────────────────────────────


def test_문서가_고정한_규칙으로_재집계한_값이_문서와_같다() -> None:
    """**이 파일의 구속력 있는 검사다.** 실패하면 고칠 것은 문서가 아니라 인용이다."""
    어긋난_것 = [
        f"{항목.rule.name}: 재집계 {항목.numerator}/{항목.denominator} "
        f"≠ 문서 {항목.rule.documented[0]}/{항목.rule.documented[1]}"
        for 항목 in recount()
        if not 항목.matches
    ]

    assert 어긋난_것 == [], "헤드라인 재집계가 문서와 갈렸다:\n" + "\n".join(어긋난_것)


def test_규칙_표가_비어_있지_않고_모집단_크기가_고정돼_있다() -> None:
    """표를 못 읽으면 위 대조가 조용히 초록이 된다 — 재집계기의 첫 실패 모드다."""
    항목들 = recount()

    assert {항목.rule.name for 항목 in 항목들} == set(_모집단_크기)
    assert {항목.rule.name: len(항목.stems) for 항목 in 항목들} == _모집단_크기
    # 분모가 회차 수와 같은 규칙과 케이스-실행 수인 규칙이 함께 있다는 것이 이 표의 요점이다.
    assert {항목.rule.documented[1] for 항목 in 항목들} == {95, 19, 7}


def test_측정_2_를_돌리지_않은_회차는_모집단에_들지_않는다() -> None:
    """`measurement_scope` 로는 안 갈린다 — `full` 은 10회뿐이고 아홉은 그 필드 이전이다."""
    미끼 = next(항목 for 항목 in recount() if 항목.rule.name == "미끼 기각 재현")

    assert "evaluation-live-l2-13" not in 미끼.stems, "측정 2 를 제외한 회차가 분모에 들었다"
    assert {"evaluation-live-1", "evaluation-live-l2-31"} <= set(미끼.stems)


# ── 음성 대조 — 정본이 실제로 정본인가 ──────────────────────────────────────


def test_문서의_재집계_칸을_바꾸면_어긋남으로_잡힌다(tmp_path: Path) -> None:
    """**문서가 정본이라는 주장의 내용이다** — 문서를 고치면 판정이 따라 움직인다."""
    사본 = _문서_사본(tmp_path, 치환=("| **12/95** |", "| **11/95** |"))

    항목들 = recount(doc=사본)
    미끼 = next(항목 for 항목 in 항목들 if 항목.rule.name == "미끼 기각 재현")

    assert not 미끼.matches
    assert (미끼.numerator, 미끼.denominator) == (12, 95), "재집계는 리포트에서 나온다"


def test_술어를_지우면_모집단이_넓어지고_판정이_갈린다(tmp_path: Path) -> None:
    """`G30` 계보의 판별자는 기권 게이트 배선 하나다 — 빼면 7회가 19회로 넓어진다."""
    사본 = _문서_사본(
        tmp_path,
        치환=(
            "`measurement_2_executed` · `fingerprint:abstention_gate_statistic="
            "rank1_minus_rank_k_spread`",
            "`measurement_2_executed`",
        ),
    )

    g30 = next(항목 for 항목 in recount(doc=사본) if 항목.rule.name == "G30 계보")

    assert len(g30.stems) == 19
    assert not g30.matches


def test_절_머리말이_사라지면_규칙이_0건이고_그것이_통과가_아니다(tmp_path: Path) -> None:
    """빈 대조는 통과가 아니다 — 규칙 0건이면 진입점이 `1` 을 돌려준다."""
    사본 = _문서_사본(tmp_path, 치환=(RULES_HEADING, "### 지워진 머리말"))

    assert parse_rules(사본) == ()
    assert recount(doc=사본) == ()
    assert main([], doc=사본) == 1


def test_모르는_어휘는_조용히_넘어가지_않는다(tmp_path: Path) -> None:
    """**fail-closed** — 술어를 모르면 참으로 넘기지 않는다. 넘기면 모집단이 조용히 넓어진다."""
    사본 = _문서_사본(tmp_path, 치환=("`measurement_2_executed` |", "`아무거나_참` |"))

    with pytest.raises(ValueError, match="모르는 모집단 술어"):
        recount(doc=사본)


# ── 진입점 — 종료 코드를 실제로 문다 ────────────────────────────────────────


def test_진입점은_전부_일치하면_0_어긋나면_1_이다(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`main` 을 실제로 부른다 — 종료 코드를 테스트 쪽에서 다시 계산하지 않는다."""
    assert main([]) == 0
    assert "일치" in capsys.readouterr().out

    사본 = _문서_사본(tmp_path, 치환=("| **4/7** |", "| **7/7** |"))

    assert main([], doc=사본) == 1
    assert "어긋남" in capsys.readouterr().out
