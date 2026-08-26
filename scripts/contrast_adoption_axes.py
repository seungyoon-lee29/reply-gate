"""채택 축 두 변경의 **전/후**를 커밋된 검색 산출물 전수로 센다 (무과금·오프라인).

```bash
uv run python -m scripts.contrast_adoption_axes
```

**왜 있나.** 두 변경 다 "지금 값이 틀렸다"가 아니라 "보장이 없다"를 닫는다 — 미정의 사유를
갈라 처분하는 것과, 검색 정렬에 결정론 tie-break 를 넣는 것. 그래서 유료 실측 앞에 무과금
대조를 두고 **무엇이 어느 방향으로 몇 건 바뀌는지**를 먼저 센다. 결과는 둘 중 하나다:
**조건 보존(변화 0건)** 이거나 **변화 N건**이고, 후자면 그 수가 라이브 관측의 사전 예측이 된다.

**모집단 — 전수다.** 커밋된 `reports/retrieval-strategies-live-*.json` 전부와 전략 4종과
골든셋 케이스의 곱. 자유 모집단이 있는 축이라 프로브가 아니라 전수로 셀 수 있다.

**왜 전체 코퍼스 점수열인가.** 각 리포트의 `ranked_hits` 는 잘리지 않은 조항 26개 전체의
코사인을 담는다. 상위 `top_k` 만 담은 산출물로는 **절단선의 동점을 볼 수 없다** — 절단선
바로 아래가 무엇이었는지가 사라지기 때문이다. 그래서 대조는 전체 순위를 읽고 `top_k` 절단을
런타임과 **같은 자리에서** 다시 건다.

**두 축의 모집단이 다르다.**

- **미정의 축**은 전략 4종 전부를 센다. 게이트는 그 전략이 올린 상위 `top_k` 슬라이스를
  보므로, 융합·리랭크 단의 슬라이스도 게이트 입력 자격이 있다.
  - **분기 ①** 측정된 상위 점수가 2건 미만 → 게이트는 열린 채로 남는다(무변경).
  - **분기 ②** 1위 코사인이 0 이하 → 이번 변경으로 **기권으로 바뀐다.**
- **정렬 축**은 **코사인 내림차순으로 적힌 행만** 센다. tie-break 가 고치는 것은 코사인
  정렬의 동점이고, 융합·리랭크 단의 순서는 코사인이 아니라 RRF·리랭크가 정한다 — 그 행을
  코사인으로 다시 세우면 대조가 엉뚱한 것을 잰다. 대상 여부는 손으로 정하지 않고 **행마다
  실제로 코사인 내림차순인지 확인해서** 가른다.

동점이 0건이면 tie-break 도입은 어떤 순위도 바꿀 수 없다 — 관측이 아니라 논증이다. 그래서
관측된 **최소 여유**(1위 코사인의 최솟값 · 절단선 간격의 최솟값)를 함께 찍어, 0건이 "간격이
넓어서"인지 아니면 아슬아슬했는지를 숫자로 남긴다.

산출물은 표준 출력뿐이다 — 파일을 남기지 않는다. 재현이 무과금이라 재실행이 곧 근거다.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

from reply_gate.retrieval_strategies import (
    AbstentionUndefined,
    truncate_for_gate,
    undefined_statistic_reason,
)

_ROOT: Final = Path(__file__).resolve().parents[1]
_REPORT_DIR: Final = _ROOT / "reports"
_REPORT_GLOB: Final = "retrieval-strategies-live-*.json"

#: 제품 기본값. 리포트가 `configuration.top_k` 를 담고 있으면 그쪽이 이긴다.
_DEFAULT_TOP_K: Final = 5


@dataclass(frozen=True)
class ScoreRow:
    """대조 단위 1건 — 질의 하나의 전체 코퍼스 점수열. 어디서 왔는지를 함께 든다."""

    origin: str
    #: `(근거 ID, 코사인)` 을 산출물에 적힌 순서 그대로. 측정되지 않은 후보는 `None` 이다.
    ranking: tuple[tuple[str, float | None], ...]
    top_k: int

    @property
    def measured_slice(self) -> tuple[float, ...]:
        """게이트가 보는 슬라이스 — `top_k` 선절단 뒤의 측정된 코사인만."""
        return truncate_for_gate([score for _, score in self.ranking], top_k=self.top_k)

    @property
    def cosine_ordered(self) -> bool:
        """적힌 순서가 코사인 내림차순인가 — 정렬 축의 대상 여부를 이것으로 가른다."""
        scores = [score for _, score in self.ranking]
        if any(score is None for score in scores):
            return False
        return all(
            earlier >= later
            for earlier, later in pairwise(score for score in scores if score is not None)
        )

    @property
    def tied_pairs(self) -> int:
        """코사인이 같은 이웃 쌍의 수. 0 이면 tie-break 가 바꿀 수 있는 것이 없다."""
        scores = [score for _, score in self.ranking]
        return sum(
            1 for earlier, later in pairwise(scores) if earlier is not None and earlier == later
        )

    @property
    def tie_at_cut(self) -> bool:
        """`top_k` 절단선을 코사인 동점이 가로지르는가."""
        if len(self.ranking) <= self.top_k:
            return False
        last_in, first_out = self.ranking[self.top_k - 1][1], self.ranking[self.top_k][1]
        return last_in is not None and last_in == first_out

    @property
    def cut_headroom(self) -> float | None:
        """절단선을 사이에 둔 두 코사인의 간격. 절단선이 없으면 `None`."""
        if len(self.ranking) <= self.top_k:
            return None
        last_in, first_out = self.ranking[self.top_k - 1][1], self.ranking[self.top_k][1]
        if last_in is None or first_out is None:
            return None
        return last_in - first_out

    def survivors(self, *, tie_break: bool) -> tuple[str, ...]:
        """상위 `top_k` 의 근거 ID. `tie_break` 면 유사도 → 근거 ID 전순서로 자른다.

        `tie_break=False` 는 **수정 전**이다 — 산출물에 적힌 순서를 그대로 쓴다. 파이썬에서
        다시 정렬하면 그 순서가 지워져 대조가 무의미해진다.
        """
        ordered = self.ranking
        if tie_break:
            ordered = tuple(sorted(ordered, key=_tie_break_key))
        return tuple(evidence_id for evidence_id, _ in ordered[: self.top_k])


def _tie_break_key(item: tuple[str, float | None]) -> tuple[float, str]:
    """유사도 내림차순 → 근거 ID 오름차순. 미측정은 맨 뒤로 보낸다."""
    evidence_id, score = item
    return (-score if score is not None else float("inf"), evidence_id)


@dataclass(frozen=True)
class Contrast:
    """모집단 하나의 대조 결과. **전수이므로 프로브 단서를 달지 않는다.**"""

    population: str
    rows: int
    undefined: dict[AbstentionUndefined, tuple[str, ...]]
    #: 정렬 축의 대상 — 적힌 순서가 실제로 코사인 내림차순인 행.
    ordered_rows: int
    ties_anywhere: tuple[str, ...]
    ties_at_cut: tuple[str, ...]
    reordered: tuple[str, ...]
    min_rank1: float | None
    min_cut_headroom: float | None

    @property
    def headline(self) -> str:
        insufficient = len(self.undefined[AbstentionUndefined.INSUFFICIENT_SCORES])
        nonpositive = len(self.undefined[AbstentionUndefined.NONPOSITIVE_RANK1])
        return (
            f"{self.population}\n"
            f"    미정의 축 (점수열 {self.rows}건): "
            f"① 측정 2건 미만 {insufficient}건 · ② 1위 코사인 0 이하 {nonpositive}건\n"
            f"    정렬 축 (코사인 순서 {self.ordered_rows}건): "
            f"동점 {len(self.ties_anywhere)}건 · 절단선 동점 {len(self.ties_at_cut)}건 · "
            f"순위 변화 {len(self.reordered)}건"
        )

    @property
    def condition_preserved(self) -> bool:
        """모든 축이 0 건인가 — 그러면 이 변경은 커밋된 조건을 보존한다."""
        return not (
            self.undefined[AbstentionUndefined.INSUFFICIENT_SCORES]
            or self.undefined[AbstentionUndefined.NONPOSITIVE_RANK1]
            or self.ties_anywhere
            or self.ties_at_cut
            or self.reordered
        )


def contrast(population: str, rows: Sequence[ScoreRow]) -> Contrast:
    undefined: dict[AbstentionUndefined, list[str]] = {reason: [] for reason in AbstentionUndefined}
    ties_anywhere: list[str] = []
    ties_at_cut: list[str] = []
    reordered: list[str] = []
    rank1_values: list[float] = []
    headrooms: list[float] = []
    ordered_rows = 0
    for row in rows:
        reason = undefined_statistic_reason(row.measured_slice)
        if reason is not None:
            undefined[reason].append(row.origin)
        if row.measured_slice:
            rank1_values.append(row.measured_slice[0])
        if not row.cosine_ordered:
            continue
        ordered_rows += 1
        if row.tied_pairs:
            ties_anywhere.append(row.origin)
        if row.tie_at_cut:
            ties_at_cut.append(row.origin)
        if row.survivors(tie_break=False) != row.survivors(tie_break=True):
            reordered.append(row.origin)
        headroom = row.cut_headroom
        if headroom is not None:
            headrooms.append(headroom)
    return Contrast(
        population=population,
        rows=len(rows),
        undefined={reason: tuple(origins) for reason, origins in undefined.items()},
        ordered_rows=ordered_rows,
        ties_anywhere=tuple(ties_anywhere),
        ties_at_cut=tuple(ties_at_cut),
        reordered=tuple(reordered),
        min_rank1=min(rank1_values) if rank1_values else None,
        min_cut_headroom=min(headrooms) if headrooms else None,
    )


# ── 모집단 ──────────────────────────────────────────────────────────────────


def report_rows(path: Path) -> list[ScoreRow]:
    """리포트 1개의 전략마다 케이스 전부를 점수열로 편다."""
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    top_k = int(raw.get("configuration", {}).get("top_k", _DEFAULT_TOP_K))
    rows: list[ScoreRow] = []
    for strategy in raw["strategies"]:
        for case in strategy["cases"]:
            ranking = tuple(
                (
                    str(hit["evidence_id"]),
                    None if hit["vector_similarity"] is None else float(hit["vector_similarity"]),
                )
                for hit in case["ranked_hits"]
            )
            rows.append(
                ScoreRow(
                    origin=f"{path.name}:{strategy['name']}:{case['id']}",
                    ranking=ranking,
                    top_k=top_k,
                )
            )
    return rows


def committed_reports() -> list[Path]:
    return sorted(_REPORT_DIR.glob(_REPORT_GLOB))


def _report(contrasts: Iterable[Contrast]) -> None:
    for item in contrasts:
        print(item.headline)
        for reason, origins in item.undefined.items():
            for origin in origins:
                print(f"        미정의({reason.value}): {origin}")
        for origin in item.ties_anywhere:
            print(f"        동점: {origin}")
        for origin in item.ties_at_cut:
            print(f"        절단선 동점: {origin}")
        for origin in item.reordered:
            print(f"        순위 변화: {origin}")
        rank1 = "없음" if item.min_rank1 is None else f"{item.min_rank1:.4f}"
        headroom = "없음" if item.min_cut_headroom is None else f"{item.min_cut_headroom:.4f}"
        print(f"        최소 1위 코사인 {rank1} · 최소 절단선 간격 {headroom}")


def main() -> None:
    print("채택 축 두 변경의 전/후 대조 — 무과금·오프라인, 산출물 없음\n")
    reports = committed_reports()
    if not reports:
        raise SystemExit(f"대조 입력이 없다: {_REPORT_DIR}/{_REPORT_GLOB}")

    _report(contrast(path.name, report_rows(path)) for path in reports)

    total = contrast("합계", [row for path in reports for row in report_rows(path)])
    print(f"\n{total.headline}")
    print(
        "조건 보존 — 모든 축이 0건이다."
        if total.condition_preserved
        else "변화가 있다 — 위 목록이 라이브 관측의 사전 예측이다."
    )


if __name__ == "__main__":
    main()
