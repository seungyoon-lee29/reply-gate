"""기권 게이트 **제품 배선** — 결정 0014 가 고른 구성 하나가 실행 경로 위에 있는가.

`tests/test_abstention_gate.py` 는 게이트 원시연산과 **오프라인 격자의 채점 규칙**을 본다.
이 파일은 그 격자가 고른 구성 하나가 **런타임 채택 경로**에 그대로 앉았는지를 본다. 확인
대상은 넷이다.

1. **켜짐/꺼짐이 채택 집합을 의도대로 가른다.** 게이트가 발동하면 그 질의의 정책 근거가
   통째로 비고, 꺼지면 절대 하한만 남는다.
2. **절단 계약** — 통계량은 `search_policy_chunks` 가 `LIMIT top_k` 로 자른 뒤 **컷 전**
   슬라이스를 본다(`docs/tracking/decisions/0014`). 컷 뒤 슬라이스로 계산하면 G04 모양의
   케이스가 기권해 케이스 하한을 깬다. 그 음성 대조를 값으로 둔다.
3. **런타임과 오프라인 격자가 같은 수를 낸다.** 커밋된 검색 산출물 하나를 양쪽에 먹여
   케이스 30건의 채택 근거 ID 까지 대조한다 — 절단 계약이 존재하는 이유가 이것이다.
4. **조건 지문이 실제 값을 싣는다.** `미배선` 자리표시가 남으면 게이트 변경이 회귀 가드가
   볼 수 없는 조용한 드리프트가 된다.

τ 와 통계량은 **설정값이지 데이터가 아니다** — 적재물도 캐시도 만들지 않고, 임베딩 모델이
바뀌어도 따라가지 않는다(결정 0012 가 `-3-large` 계열에서 이 통계량의 반증을 기록했다).
"""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import psycopg
import pytest
from psycopg.rows import DictRow
from scripts import evaluate

from reply_gate.adoption_axis import AbstentionStatistic as HandCalcStatistic
from reply_gate.adoption_axis import statistic_value as handcalc_statistic_value
from reply_gate.config import Settings
from reply_gate.contracts import EvidenceSource, IntentSource
from reply_gate.evidence import (
    EvidenceCollector,
    adopt_policy_hits,
    merge_policy_rankings,
)
from reply_gate.llm import GenerationClient, JsonCompletion
from reply_gate.policy_index import (
    PolicySearchHit,
    index_policy_documents,
    load_policy_documents,
    search_policy_chunks,
)
from reply_gate.retrieval_eval import (
    AbstentionGrid,
    RankedHit,
    StrategyRetrieval,
    StrategyRetrievedCase,
    run_abstention_grid,
)
from reply_gate.retrieval_labels import RetrievalLabel
from reply_gate.retrieval_strategies import (
    AbstentionGate,
    AbstentionStatistic,
    RetrievalStage,
    StrategyDefinition,
    abstention_statistic,
    truncate_for_gate,
)
from reply_gate.testing import LexicalEmbeddingClient
from tests.conftest import declared_settings

_ROOT = Path(__file__).resolve().parents[1]
#: 결정 0014 격자의 입력. 커밋된 산출물이라 이 대조는 무과금이다.
_BASE_REPORT = (
    _ROOT / "reports" / "retrieval-strategies-live-text-embedding-3-small-d1536-blind-k5-c030.json"
)
#: 결정 기록이 소수 4자리로 인용한다.
_QUOTED = 5e-5
_TOP_K = 5
_CUT = 0.30
#: 결정 0014 채택 구성.
_TAU = 0.06
#: 그 구성에서 게이트가 발동하는 케이스 — 빈 라벨 5건 전부다(표적 넷 + G28).
_ABSTAINING = {"G21", "G22", "G23", "G24", "G28"}


def _hit(evidence_id: str, similarity: float) -> PolicySearchHit:
    return PolicySearchHit(
        evidence_id=evidence_id,
        document_slug="refund",
        document_title="환불 정책",
        article=evidence_id.rsplit(":", 1)[-1],
        article_title="조항",
        content="본문",
        similarity=similarity,
    )


def _ranked(scores: Sequence[float]) -> tuple[PolicySearchHit, ...]:
    """유사도 내림차순 순위 — DB 검색 결과의 모양이다."""
    return tuple(_hit(f"policy:x:{index}", score) for index, score in enumerate(scores, start=1))


def _adopt(
    scores: Sequence[float],
    *,
    gate: AbstentionGate | None,
    cut: float = _CUT,
    top_k: int = _TOP_K,
) -> tuple[str, ...]:
    adoption = adopt_policy_hits(
        candidates=_ranked(scores), top_k=top_k, similarity_threshold=cut, gate=gate
    )
    return tuple(hit.evidence_id for hit in adoption.hits)


def _gate(*, tau: float = _TAU) -> AbstentionGate:
    return AbstentionGate(statistic=AbstentionStatistic.SPREAD, tau=tau)


# ---------------------------------------------------------------------------
# 계약 A — 결정 0014 의 구성 하나가 설정 기본값이다.
# ---------------------------------------------------------------------------


def test_기본_설정이_결정_0014_가_고른_구성을_그대로_들고_있다() -> None:
    """격자 165 구성 중 채택 규칙 다섯을 순서대로 통과한 것은 이 하나뿐이다.

    **선언값을 읽는다** — 인자 없는 `Settings()` 는 개발자의 `.env` 까지 읽어서 이 검사가
    결정 기록 대신 로컬 환경을 재게 된다.
    """
    settings = declared_settings()

    assert settings.abstention_gate_enabled is True
    assert settings.abstention_gate_statistic is AbstentionStatistic.SPREAD
    assert settings.abstention_tau == pytest.approx(0.06)
    # 컷과 top_k 는 이 사이클에서 건드리지 않는다.
    assert settings.vector_similarity_threshold == pytest.approx(0.30)
    assert settings.vector_top_k == 5


def test_설정이_조립한_게이트가_통계량과_tau_를_그대로_싣는다() -> None:
    gate = declared_settings().abstention_gate()

    assert gate == AbstentionGate(statistic=AbstentionStatistic.SPREAD, tau=0.06)


def test_스위치를_끄면_게이트가_아예_조립되지_않는다() -> None:
    """끄기는 설정 한 줄이다 — 원복 범위가 게이트 하나로 남는다(결정 0011)."""
    assert declared_settings(abstention_gate_enabled=False).abstention_gate() is None


def test_통계량과_tau_는_설정값이라_다른_구성으로도_조립된다() -> None:
    """반증된 축으로 되돌리는 것도 설정 한 줄이어야 한다 — 코드 수정이 아니다."""
    settings = declared_settings(
        abstention_gate_statistic=AbstentionStatistic.STDEV, abstention_tau=0.02
    )

    assert settings.abstention_gate() == AbstentionGate(
        statistic=AbstentionStatistic.STDEV, tau=0.02
    )


def test_게이트는_환경_변수로_켜고_끌_수_있다(monkeypatch: pytest.MonkeyPatch) -> None:
    """설정값이지 데이터가 아니다 — 적재물도 캐시도 거치지 않고 환경에서 바로 온다."""
    monkeypatch.setenv("ABSTENTION_GATE_ENABLED", "false")
    assert Settings().abstention_gate() is None

    monkeypatch.setenv("ABSTENTION_GATE_ENABLED", "true")
    monkeypatch.setenv("ABSTENTION_GATE_STATISTIC", "spread_over_rank1")
    monkeypatch.setenv("ABSTENTION_TAU", "0.12")

    assert Settings().abstention_gate() == AbstentionGate(
        statistic=AbstentionStatistic.RELATIVE_SPREAD, tau=0.12
    )


# ---------------------------------------------------------------------------
# 계약 B — 켜짐/꺼짐이 채택 집합을 의도대로 가른다.
# ---------------------------------------------------------------------------


def test_게이트가_발동하면_그_질의의_채택_집합이_통째로_빈다() -> None:
    """1위가 컷을 훨씬 넘어도 지워진다 — 항목 규칙이 아니라 질의 단위 판정이다."""
    flat = (0.52, 0.51, 0.50, 0.49, 0.48)

    assert _adopt(flat, gate=_gate()) == ()


def test_같은_입력이_게이트가_꺼지면_절대_하한만_지난다() -> None:
    """켜짐/꺼짐의 차이가 곧 이 사이클이 바꾼 것 전부다."""
    flat = (0.52, 0.51, 0.50, 0.49, 0.48)

    assert len(_adopt(flat, gate=None)) == 5


def test_게이트가_발동하지_않으면_컷이_자른_것과_같다() -> None:
    """양성 대조 — 게이트가 늘 비우는 것이 아니어야 켜짐/꺼짐 대조가 의미를 갖는다."""
    spread = (0.52, 0.44, 0.36, 0.28, 0.20)

    assert _adopt(spread, gate=_gate()) == _adopt(spread, gate=None)
    assert _adopt(spread, gate=_gate()) == ("policy:x:1", "policy:x:2", "policy:x:3")


def test_통계량이_tau_와_같으면_기권시키지_않는다() -> None:
    """판정은 `< τ` 다. 경계값을 기권으로 넘기면 채택 쪽 여유가 한 눈금 사라진다."""
    exactly_tau = (0.50, 0.49, 0.48, 0.47, 0.44)

    assert abstention_statistic(AbstentionStatistic.SPREAD, exactly_tau) == pytest.approx(_TAU)
    assert len(_adopt(exactly_tau, gate=_gate())) == 5


# ---------------------------------------------------------------------------
# 계약 C — 절단 계약. 통계량은 **컷 전** 상위 `top_k` 슬라이스를 본다.
# ---------------------------------------------------------------------------


def test_게이트_입력은_컷_전의_상위_top_k_슬라이스다() -> None:
    """G04 모양 — 컷 위 2건, 아래 3건. 컷 뒤 슬라이스로 재면 기권해 하한을 깬다."""
    g04_shaped = (0.3647, 0.3571, 0.2601, 0.2400, 0.2300)

    # 음성 대조: 컷이 먼저 걸린 슬라이스의 통계량은 τ 아래다.
    above_cut = tuple(score for score in g04_shaped if score >= _CUT)
    assert abstention_statistic(AbstentionStatistic.SPREAD, above_cut) < _TAU

    # 런타임이 실제로 보는 것은 절단 슬라이스라 게이트가 발동하지 않는다.
    assert _adopt(g04_shaped, gate=_gate()) == ("policy:x:1", "policy:x:2")


def test_top_k_밖의_꼬리는_통계량에_들어가지_않는다() -> None:
    """`LIMIT top_k` 뒤가 입력이다 — 6위 이하가 산포를 늘리면 오프라인과 갈린다."""
    with_tail = (0.52, 0.51, 0.50, 0.49, 0.48, 0.01)

    assert _adopt(with_tail, gate=_gate(), top_k=5) == ()


def test_절단_슬라이스가_같으면_뒤에_무엇이_붙어도_판정이_같다() -> None:
    head = (0.52, 0.51, 0.50, 0.49, 0.48)

    assert _adopt(head, gate=_gate()) == _adopt((*head, 0.30, 0.29), gate=_gate())


# ---------------------------------------------------------------------------
# 계약 D — 미정의는 0 이 아니다. 0 으로 채우면 모든 양수 τ 에서 기권이 된다.
# ---------------------------------------------------------------------------


def test_측정된_후보가_두_건_미만이면_게이트는_열린_채_남는다() -> None:
    assert _adopt((0.9,), gate=_gate()) == ("policy:x:1",)


def test_후보가_아예_없으면_채택도_없고_게이트가_사유_없이_발동하지_않는다() -> None:
    adoption = adopt_policy_hits(
        candidates=(), top_k=_TOP_K, similarity_threshold=_CUT, gate=_gate()
    )

    assert adoption.hits == ()
    # 게이트가 아예 돌지 않았다 — 판정이 없는 것과 "판정했고 정의됐다"는 다른 상태다.
    assert adoption.verdict is None


def test_미정의_판정은_기권이_아니라_사유를_남긴다() -> None:
    """원시연산의 계약을 배선이 뒤집지 않는다 — 0 으로 채우면 여기서 기권이 된다."""
    from reply_gate.retrieval_strategies import apply_abstention_gate

    verdict = apply_abstention_gate(_gate(), truncate_for_gate([0.9], top_k=_TOP_K))

    assert verdict.abstains is False
    assert verdict.value is None
    assert verdict.undefined_reason


# ---------------------------------------------------------------------------
# 계약 E — 런타임과 오프라인 격자가 같은 수를 낸다(무과금 대조).
# ---------------------------------------------------------------------------


def _committed_cases() -> tuple[tuple[str, tuple[PolicySearchHit, ...], tuple[str, ...]], ...]:
    """커밋된 라이브 산출물의 `vector_rewrite` 행을 **런타임 후보 모양**으로 되살린다."""
    raw = json.loads(_BASE_REPORT.read_text(encoding="utf-8"))
    row = next(item for item in raw["strategies"] if item["name"] == "vector_rewrite")
    cases: list[tuple[str, tuple[PolicySearchHit, ...], tuple[str, ...]]] = []
    for case in row["cases"]:
        hits = tuple(
            _hit(str(item["evidence_id"]), float(item["vector_similarity"]))
            for item in case["ranked_hits"]
            if item["vector_similarity"] is not None
        )
        labels = tuple(str(item) for item in case["relevant_evidence_ids"])
        cases.append((str(case["id"]), hits, labels))
    return tuple(cases)


def _offline_grid() -> AbstentionGrid:
    """같은 입력을 채점자에게 먹인다 — 채점자는 라벨을 보고, 런타임은 못 본다."""
    cases: list[StrategyRetrievedCase] = []
    labels: list[RetrievalLabel] = []
    for case_id, hits, relevant in _committed_cases():
        ranked = tuple(
            RankedHit(rank=rank, evidence_id=hit.evidence_id, similarity=hit.similarity)
            for rank, hit in enumerate(hits, start=1)
        )
        cases.append(
            StrategyRetrievedCase(case_id=case_id, ranked_hits=ranked, accept_candidates=ranked)
        )
        labels.append(
            RetrievalLabel(id=case_id, relevant_evidence_ids=frozenset(relevant), note="")
        )
    retrieval = StrategyRetrieval(
        strategy=StrategyDefinition(
            "vector_rewrite", (RetrievalStage.VECTOR, RetrievalStage.REWRITE)
        ),
        accept_limit=_TOP_K,
        cases=tuple(cases),
    )
    grid = run_abstention_grid(retrieval, tuple(labels), cutoff=_CUT)
    assert isinstance(grid, AbstentionGrid)
    return grid


def _runtime_accepted(hits: Sequence[PolicySearchHit]) -> tuple[str, ...]:
    """런타임 정책 경로 그대로 — 합집합 → 상위 `top_k` → 게이트 → 컷."""
    candidates = merge_policy_rankings(original=hits, rewritten=(), top_k=_TOP_K)
    adoption = adopt_policy_hits(
        candidates=candidates,
        top_k=_TOP_K,
        similarity_threshold=_CUT,
        gate=declared_settings().abstention_gate(),
    )
    return tuple(hit.evidence_id for hit in adoption.hits)


def test_런타임_채택_집합이_오프라인_격자와_케이스마다_같다() -> None:
    """절단 계약이 존재하는 이유 — 같은 입력에 두 경로가 같은 답을 내야 한다."""
    grid = _offline_grid()
    point = next(
        item
        for item in grid.points
        if item.gate is not None
        and item.gate.statistic is AbstentionStatistic.SPREAD
        and item.gate.tau == pytest.approx(_TAU)
    )
    offline = {row.case_id: tuple(sorted(row.accepted_evidence_ids)) for row in point.cases}

    runtime = {
        case_id: tuple(sorted(_runtime_accepted(hits))) for case_id, hits, _ in _committed_cases()
    }

    assert runtime == offline
    # 양성 대조 — 빈 집합끼리 같은 것이 아니다.
    assert sum(len(ids) for ids in runtime.values()) > 0


def test_런타임이_기권시키는_케이스가_결정_0014_의_표와_같다() -> None:
    abstained = {case_id for case_id, hits, _ in _committed_cases() if not _runtime_accepted(hits)}

    assert abstained == _ABSTAINING


def test_경계_케이스_G15_와_G23_이_결정_0014_의_값을_그대로_낸다() -> None:
    """τ 를 정한 두 케이스. 여기가 움직이면 채택 구성이 바뀐 것이다."""
    values = {
        case_id: abstention_statistic(
            AbstentionStatistic.SPREAD,
            truncate_for_gate([hit.similarity for hit in hits], top_k=_TOP_K),
        )
        for case_id, hits, _ in _committed_cases()
    }

    assert values["G15"] == pytest.approx(0.0668, abs=_QUOTED)
    assert values["G23"] == pytest.approx(0.0521, abs=_QUOTED)
    assert values["G15"] - _TAU == pytest.approx(0.0068, abs=_QUOTED)
    assert values["G23"] - _TAU == pytest.approx(-0.0079, abs=_QUOTED)


def test_런타임_통계량이_채점자_손계산과_같은_수를_낸다() -> None:
    """구현이 둘이라 핀이 필요하다 — 셋째 벌을 만들지 않는 대신 이 대조를 둔다."""
    for _case_id, hits, _ in _committed_cases():
        scores = truncate_for_gate([hit.similarity for hit in hits], top_k=_TOP_K)
        assert abstention_statistic(AbstentionStatistic.SPREAD, scores) == pytest.approx(
            handcalc_statistic_value(HandCalcStatistic.SPREAD, scores), abs=1e-12
        )


# ---------------------------------------------------------------------------
# 계약 E-2 — 구현은 한 벌이다. 셋째 벌이 생기면 이 대조가 무의미해진다.
# ---------------------------------------------------------------------------


def test_런타임_채택_경로는_게이트_원시연산을_다시_구현하지_않는다() -> None:
    """통계량 계산이 `evidence.py` 안에 또 생기면 대조 테스트가 자기 자신을 대조한다.

    현재 구현은 둘이다 — 런타임 쪽(`retrieval_strategies`)과 채점자 쪽(`adoption_axis`).
    둘은 교차 테스트로 핀이 박혀 있고, **셋째 벌은 만들지 않는다.** 채택 경로가 원시연산을
    import 해서 쓰는지, 스스로 다시 계산하지 않는지를 AST 로 본다.
    """
    source = (_ROOT / "src" / "reply_gate" / "evidence.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="evidence.py")

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "reply_gate.retrieval_strategies":
            imported.update(alias.asname or alias.name for alias in node.names)

    assert {"apply_abstention_gate", "truncate_for_gate"} <= imported
    # 통계량 계산을 스스로 하지 않는다 — 원시연산이 쓰는 도구가 여기 나타나면 안 된다.
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "pstdev" not in called
    # `import os, statistics` 처럼 alias 가 여럿이면 첫 이름만 보는 검사는 뚫린다.
    assert "statistics" not in {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }


# ---------------------------------------------------------------------------
# 계약 F — 게이트를 끈 런타임은 배선 전 동작과 같다(원복 경로).
# ---------------------------------------------------------------------------


def test_게이트를_끄면_컷을_먼저_걸던_예전_순서와_채택_집합이_같다() -> None:
    """컷을 검색 안에서 걸던 순서와 합침 뒤에 거는 순서는 결과가 같다.

    컷 미만 후보는 컷 이상 후보보다 항상 아래에 정렬되므로, 합집합 상위 `top_k` 에
    들어가는 경우는 컷 이상 후보가 `top_k` 개보다 적을 때뿐이고 그때도 컷이 다시 지운다.
    """
    original = _ranked((0.90, 0.80, 0.70, 0.29, 0.28))
    rewritten = _ranked((0.85, 0.60, 0.31, 0.27, 0.26))

    old_order = merge_policy_rankings(
        original=tuple(hit for hit in original if hit.similarity >= _CUT),
        rewritten=tuple(hit for hit in rewritten if hit.similarity >= _CUT),
        top_k=_TOP_K,
    )
    new_order = adopt_policy_hits(
        candidates=merge_policy_rankings(original=original, rewritten=rewritten, top_k=_TOP_K),
        top_k=_TOP_K,
        similarity_threshold=_CUT,
        gate=None,
    ).hits

    assert [hit.evidence_id for hit in new_order] == [hit.evidence_id for hit in old_order]
    assert new_order  # 양성 대조


# ---------------------------------------------------------------------------
# 계약 G — 조건 지문이 실제 값을 싣는다. 자리표시가 남으면 조용한 드리프트다.
# ---------------------------------------------------------------------------


def _fingerprint(settings: Settings) -> dict[str, str]:
    args = argparse.Namespace(
        golden_set=_ROOT / "data" / "golden_set.jsonl",
        judge_fixtures=_ROOT / "data" / "judge_fixtures.jsonl",
        stub_llm=False,
    )
    return evaluate._condition_fingerprint(args=args, settings=settings, run_settings=settings)


def test_지문이_기권_게이트의_실제_통계량과_tau_를_싣는다() -> None:
    values = _fingerprint(declared_settings())

    assert values["abstention_gate_statistic"] == "rank1_minus_rank_k_spread"
    assert values["abstention_tau"] == "0.06"
    assert values["acceptance_cut"] == "0.3"


def test_지문에_미배선_자리표시가_남아_있지_않다() -> None:
    """음성 대조 — 자리표시가 남으면 게이트 변경이 가드에 보이지 않는다."""
    values = _fingerprint(declared_settings())

    assert "미배선" not in values["abstention_gate_statistic"]
    assert "미배선" not in values["abstention_tau"]


def test_tau_나_통계량을_바꾸면_지문_값이_따라_바뀐다() -> None:
    values = _fingerprint(
        declared_settings(abstention_gate_statistic=AbstentionStatistic.STDEV, abstention_tau=0.025)
    )

    assert values["abstention_gate_statistic"] == "top_k_stdev"
    assert values["abstention_tau"] == "0.025"


def test_게이트를_끄면_지문이_0_이_아니라_꺼짐으로_남는다() -> None:
    """끈 실행을 τ=0 으로 적으면 "모든 질의를 통과시킨 실행"과 구분되지 않는다."""
    values = _fingerprint(declared_settings(abstention_gate_enabled=False))

    assert values["abstention_gate_statistic"] == "꺼짐"
    assert values["abstention_tau"] == "꺼짐"


def test_지문_항목_이름이_회귀_가드가_보는_이름과_같다() -> None:
    """가드는 이름으로 짝을 찾는다 — 이름이 갈리면 대조가 조용히 빠진다."""
    from reply_gate.regression_guard import FINGERPRINT_FIELDS, PAIRED_FINGERPRINT_FIELDS

    values = _fingerprint(declared_settings())

    assert {"abstention_gate_statistic", "abstention_tau"} <= set(FINGERPRINT_FIELDS)
    assert {"abstention_gate_statistic", "abstention_tau"} <= values.keys()
    # τ 는 임베딩 모델과 짝으로 읽힌다 — 모델이 다르면 τ 가 같아도 대조 불가다.
    assert PAIRED_FINGERPRINT_FIELDS["abstention_tau"] == "embedding_model"
    assert values["embedding_model"] == declared_settings().embedding_model


# ---------------------------------------------------------------------------
# 계약 H — 수집기가 실제로 그 경로를 탄다(DB 통합).
# ---------------------------------------------------------------------------


class _PolicyIntentClient:
    """정책 의도 하나만 돌려주는 생성 대역. 재작성은 꺼서 부르지 않는다."""

    def complete_json(self, **kwargs: object) -> JsonCompletion:
        assert kwargs["stage"] == "intent"
        return JsonCompletion(data={"source": "policy"}, input_tokens=1, output_tokens=1)


def _policy_collector(*, gate_enabled: bool) -> EvidenceCollector:
    return EvidenceCollector(
        generation_client=cast(GenerationClient, _PolicyIntentClient()),
        embedding_client=LexicalEmbeddingClient(dimensions=1536),
        settings=Settings(
            vector_similarity_threshold=0.0,
            query_rewrite_enabled=False,
            abstention_gate_enabled=gate_enabled,
        ),
    )


@pytest.mark.db
def test_수집기가_평평한_분포의_문의에서_정책_근거를_한_건도_채택하지_않는다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """대역 임베딩이라 τ 는 이 조건의 값이 아니다 — 여기서 보는 것은 **배선**이다."""
    index_policy_documents(
        conn=app_conn,
        documents=load_policy_documents(),
        embedder=LexicalEmbeddingClient(dimensions=1536),
    )
    flat = "매장에 직접 방문해서 수령할 수 있나요?"

    gated = _policy_collector(gate_enabled=True).collect(
        inquiry_id="11111111-2222-3333-4444-555555555555",
        content=flat,
        order_no=None,
        app_conn=app_conn,
        readonly_conn=app_conn,
    )
    ungated = _policy_collector(gate_enabled=False).collect(
        inquiry_id="11111111-2222-3333-4444-555555555555",
        content=flat,
        order_no=None,
        app_conn=app_conn,
        readonly_conn=app_conn,
    )

    embedder = LexicalEmbeddingClient(dimensions=1536)
    vector = embedder.embed(stage="t", texts=[flat]).vectors[0]
    candidates = search_policy_chunks(
        conn=app_conn,
        query_vector=vector,
        top_k=5,
        embedding_model=embedder.model,
        embedding_dimensions=embedder.dimensions,
    )
    scores = truncate_for_gate([hit.similarity for hit in candidates], top_k=5)
    assert abstention_statistic(AbstentionStatistic.SPREAD, scores) < _TAU

    assert gated.intent is IntentSource.POLICY
    assert not [item for item in gated.evidence if item.source is EvidenceSource.POLICY]
    # 양성 대조 — 게이트를 끄면 같은 문의가 근거를 얻는다.
    assert [item for item in ungated.evidence if item.source is EvidenceSource.POLICY]
