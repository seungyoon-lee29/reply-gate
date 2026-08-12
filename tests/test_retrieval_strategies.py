"""정답 라벨과 분리된 재사용 가능 검색 전략의 공개 동작."""

from pathlib import Path
from typing import Any, cast

import pytest

from reply_gate.llm import GenerationClient, JsonCompletion, LLMCallError
from reply_gate.retrieval_strategies import (
    Bm25Hit,
    FusedHit,
    RetrievalStage,
    VectorHit,
    bm25_rank,
    default_strategy_ladder,
    llm_rerank,
    merge_rewritten_rankings,
    reciprocal_rank_fusion,
)


class _RecordingGenerationClient:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> JsonCompletion:
        self.calls.append(kwargs)
        outcome = self._outcomes[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return cast(JsonCompletion, outcome)


def _fused_candidates() -> tuple[FusedHit, ...]:
    return (
        FusedHit(1, "policy:a:1", 0.03, 1, 0.9, 2, 1.0),
        FusedHit(2, "policy:b:1", 0.02, 2, 0.8, 1, 2.0),
        FusedHit(3, "policy:c:1", 0.01, 3, 0.7, 3, 0.5),
    )


def test_default_strategy_ladder_is_four_cumulative_compositions() -> None:
    ladder = default_strategy_ladder()

    assert [strategy.name for strategy in ladder] == [
        "vector",
        "vector_rewrite",
        "vector_rewrite_hybrid",
        "vector_rewrite_hybrid_rerank",
    ]
    assert [strategy.stages for strategy in ladder] == [
        (RetrievalStage.VECTOR,),
        (RetrievalStage.VECTOR, RetrievalStage.REWRITE),
        (RetrievalStage.VECTOR, RetrievalStage.REWRITE, RetrievalStage.HYBRID),
        (
            RetrievalStage.VECTOR,
            RetrievalStage.REWRITE,
            RetrievalStage.HYBRID,
            RetrievalStage.RERANK,
        ),
    ]


def test_rewrite_union_keeps_original_as_fallback_and_uses_max_similarity() -> None:
    original = (
        VectorHit(rank=1, evidence_id="policy:original:1", similarity=0.81),
        VectorHit(rank=2, evidence_id="policy:shared:1", similarity=0.62),
    )
    rewritten = (
        VectorHit(rank=1, evidence_id="policy:rewritten:1", similarity=0.91),
        VectorHit(rank=2, evidence_id="policy:shared:1", similarity=0.77),
    )

    merged = merge_rewritten_rankings(original=original, rewritten=rewritten)

    assert [(hit.evidence_id, hit.similarity) for hit in merged] == [
        ("policy:rewritten:1", 0.91),
        ("policy:original:1", 0.81),
        ("policy:shared:1", 0.77),
    ]
    assert [hit.rank for hit in merged] == [1, 2, 3]


def test_bm25_uses_korean_character_bigrams_for_score_and_rank() -> None:
    ranked = bm25_rank(
        query="환불",
        documents=(
            ("policy:long:1", "환불가능"),
            ("policy:exact:1", "환불"),
            ("policy:other:1", "배송"),
        ),
    )

    assert [hit.evidence_id for hit in ranked] == [
        "policy:exact:1",
        "policy:long:1",
        "policy:other:1",
    ]
    assert ranked[0].score == pytest.approx(0.5619608610546839)
    assert ranked[1].score == pytest.approx(0.3541123234043214)
    assert ranked[2].score == 0.0


def test_rrf_combines_rankings_and_breaks_ties_by_evidence_id() -> None:
    vector = (
        VectorHit(rank=1, evidence_id="policy:a:1", similarity=0.9),
        VectorHit(rank=2, evidence_id="policy:b:1", similarity=0.8),
        VectorHit(rank=3, evidence_id="policy:c:1", similarity=0.7),
    )
    bm25 = (
        Bm25Hit(rank=1, evidence_id="policy:b:1", score=2.0),
        Bm25Hit(rank=2, evidence_id="policy:a:1", score=1.0),
        Bm25Hit(rank=3, evidence_id="policy:d:1", score=0.5),
    )

    fused = reciprocal_rank_fusion(vector=vector, bm25=bm25, rrf_k=60)

    assert [hit.evidence_id for hit in fused] == [
        "policy:a:1",
        "policy:b:1",
        "policy:c:1",
        "policy:d:1",
    ]
    assert fused[0].rrf_score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1].rrf_score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[2].rrf_score == pytest.approx(1 / 63)
    assert fused[3].rrf_score == pytest.approx(1 / 63)


def test_llm_rerank_uses_returned_order_without_sampling_parameters(tmp_path: Path) -> None:
    client = _RecordingGenerationClient(
        [
            JsonCompletion(
                data={"evidence_ids": ["policy:c:1", "policy:a:1", "policy:b:1"]},
                input_tokens=10,
                output_tokens=4,
            )
        ]
    )

    reranked = llm_rerank(
        query="환불 문의",
        candidates=_fused_candidates(),
        policy_texts={
            "policy:a:1": "A 조항",
            "policy:b:1": "B 조항",
            "policy:c:1": "C 조항",
        },
        client=cast(GenerationClient, client),
        model="gpt-cheap",
        cache_dir=tmp_path,
    )

    assert [hit.evidence_id for hit in reranked] == [
        "policy:c:1",
        "policy:a:1",
        "policy:b:1",
    ]
    assert [hit.rank for hit in reranked] == [1, 2, 3]
    assert "temperature" not in client.calls[0]
    assert "top_p" not in client.calls[0]
    assert "top_k" not in client.calls[0]


@pytest.mark.parametrize(
    "outcome",
    [
        LLMCallError(stage="retrieval_rerank", reason="transport_error", attempts=2),
        JsonCompletion(data={"evidence_ids": ["policy:a:1"]}, input_tokens=1, output_tokens=1),
    ],
)
def test_llm_rerank_failure_or_unparseable_output_keeps_previous_order(
    tmp_path: Path, outcome: object
) -> None:
    client = _RecordingGenerationClient([outcome])

    reranked = llm_rerank(
        query="환불 문의",
        candidates=_fused_candidates(),
        policy_texts={
            "policy:a:1": "A 조항",
            "policy:b:1": "B 조항",
            "policy:c:1": "C 조항",
        },
        client=cast(GenerationClient, client),
        model="gpt-cheap",
        cache_dir=tmp_path,
    )

    assert reranked == _fused_candidates()


def test_llm_rerank_reuses_cache_for_same_input_and_model(tmp_path: Path) -> None:
    client = _RecordingGenerationClient(
        [
            JsonCompletion(
                data={"evidence_ids": ["policy:b:1", "policy:c:1", "policy:a:1"]},
                input_tokens=10,
                output_tokens=4,
            )
        ]
    )
    policy_texts = {
        "policy:a:1": "A 조항",
        "policy:b:1": "B 조항",
        "policy:c:1": "C 조항",
    }

    first = llm_rerank(
        query="환불 문의",
        candidates=_fused_candidates(),
        policy_texts=policy_texts,
        client=cast(GenerationClient, client),
        model="gpt-cheap",
        cache_dir=tmp_path,
    )
    second = llm_rerank(
        query="환불 문의",
        candidates=_fused_candidates(),
        policy_texts=policy_texts,
        client=cast(GenerationClient, client),
        model="gpt-cheap",
        cache_dir=tmp_path,
    )

    assert second == first
    assert len(client.calls) == 1
