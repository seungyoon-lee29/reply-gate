"""정답 라벨과 분리된 재사용 가능 검색 전략의 공개 동작."""

import json
from pathlib import Path
from typing import Any, cast

import pytest

from reply_gate import retrieval_strategies
from reply_gate.llm import GenerationClient, JsonCompletion, LLMCallError
from reply_gate.retrieval_strategies import (
    Bm25Hit,
    FusedHit,
    RerankOutcome,
    RetrievalStage,
    VectorHit,
    bm25_rank,
    default_strategy_ladder,
    llm_rerank,
    merge_rewritten_bm25,
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

    outcome = llm_rerank(
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

    assert [hit.evidence_id for hit in outcome.hits] == [
        "policy:c:1",
        "policy:a:1",
        "policy:b:1",
    ]
    assert [hit.rank for hit in outcome.hits] == [1, 2, 3]
    assert not outcome.fell_back
    assert outcome.fallback_reason is None
    assert (outcome.input_tokens, outcome.output_tokens) == (10, 4)
    assert "temperature" not in client.calls[0]
    assert "top_p" not in client.calls[0]
    assert "top_k" not in client.calls[0]


def test_리랭크_입력에_재작성문이_함께_실리고_원문을_지우지_않는다(tmp_path: Path) -> None:
    client = _RecordingGenerationClient(
        [
            JsonCompletion(
                data={"evidence_ids": ["policy:a:1", "policy:b:1", "policy:c:1"]},
                input_tokens=1,
                output_tokens=1,
            )
        ]
    )

    llm_rerank(
        query="환불 언제 되나요",
        rewritten_query="환불 처리 기간",
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

    payload = json.loads(cast(str, client.calls[0]["user"]))
    assert payload["query"] == "환불 언제 되나요"
    assert payload["rewritten_query"] == "환불 처리 기간"


def test_리랭크_캐시는_지시문이_바뀌면_무효화된다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """지시문도 입력이다 — 프롬프트를 고친 재실행이 옛 순위를 조용히 재사용하면 안 된다."""
    policy_texts = {"policy:a:1": "A 조항", "policy:b:1": "B 조항", "policy:c:1": "C 조항"}

    def rerank_once(client: _RecordingGenerationClient) -> RerankOutcome:
        return llm_rerank(
            query="환불 문의",
            candidates=_fused_candidates(),
            policy_texts=policy_texts,
            client=cast(GenerationClient, client),
            model="gpt-cheap",
            cache_dir=tmp_path,
        )

    first_client = _RecordingGenerationClient(
        [
            JsonCompletion(
                data={"evidence_ids": ["policy:b:1", "policy:c:1", "policy:a:1"]},
                input_tokens=10,
                output_tokens=4,
            )
        ]
    )
    rerank_once(first_client)

    monkeypatch.setattr(retrieval_strategies, "_RERANK_SYSTEM", "다른 지시문")
    second_client = _RecordingGenerationClient(
        [
            JsonCompletion(
                data={"evidence_ids": ["policy:c:1", "policy:b:1", "policy:a:1"]},
                input_tokens=10,
                output_tokens=4,
            )
        ]
    )
    second = rerank_once(second_client)

    assert len(second_client.calls) == 1
    assert not second.served_from_cache
    assert [hit.evidence_id for hit in second.hits] == [
        "policy:c:1",
        "policy:b:1",
        "policy:a:1",
    ]


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

    result = llm_rerank(
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

    assert result.hits == _fused_candidates()
    # 폴백은 순위를 지키지만 조용히 지나가지 않는다 — 사유가 호출자에게 올라간다.
    assert result.fell_back
    assert result.fallback_reason is not None
    assert not result.served_from_cache


def test_리랭크_폴백은_캐시에_기록되지_않는다(tmp_path: Path) -> None:
    """실패한 순위를 캐시에 남기면 다음 실행이 그것을 성공으로 재사용한다."""
    failing = _RecordingGenerationClient(
        [LLMCallError(stage="retrieval_rerank", reason="transport_error", attempts=2)]
    )
    policy_texts = {"policy:a:1": "A 조항", "policy:b:1": "B 조항", "policy:c:1": "C 조항"}
    first = llm_rerank(
        query="환불 문의",
        candidates=_fused_candidates(),
        policy_texts=policy_texts,
        client=cast(GenerationClient, failing),
        model="gpt-cheap",
        cache_dir=tmp_path,
    )
    assert first.fell_back

    succeeding = _RecordingGenerationClient(
        [
            JsonCompletion(
                data={"evidence_ids": ["policy:c:1", "policy:b:1", "policy:a:1"]},
                input_tokens=2,
                output_tokens=2,
            )
        ]
    )
    second = llm_rerank(
        query="환불 문의",
        candidates=_fused_candidates(),
        policy_texts=policy_texts,
        client=cast(GenerationClient, succeeding),
        model="gpt-cheap",
        cache_dir=tmp_path,
    )

    assert len(succeeding.calls) == 1
    assert not second.fell_back
    assert [hit.evidence_id for hit in second.hits] == [
        "policy:c:1",
        "policy:b:1",
        "policy:a:1",
    ]


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

    assert second.hits == first.hits
    assert len(client.calls) == 1
    # 캐시 재사용은 과금이 아니다 — 토큰 0 과 재사용 표시가 함께 올라간다.
    assert second.served_from_cache
    assert (second.input_tokens, second.output_tokens) == (0, 0)
    assert not first.served_from_cache
    assert (first.input_tokens, first.output_tokens) == (10, 4)


def test_재작성_어휘_순위는_더_큰_점수로_합쳐진다() -> None:
    original = (
        Bm25Hit(rank=1, evidence_id="policy:a:1", score=3.0),
        Bm25Hit(rank=2, evidence_id="policy:b:1", score=1.0),
    )
    rewritten = (
        Bm25Hit(rank=1, evidence_id="policy:b:1", score=5.0),
        Bm25Hit(rank=2, evidence_id="policy:c:1", score=2.0),
    )

    merged = merge_rewritten_bm25(original=original, rewritten=rewritten)

    assert [(hit.rank, hit.evidence_id, hit.score) for hit in merged] == [
        (1, "policy:b:1", 5.0),
        (2, "policy:a:1", 3.0),
        (3, "policy:c:1", 2.0),
    ]
