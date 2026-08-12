"""정답 라벨을 모르는 순수 정책 검색 전략.

오프라인 비교와 후속 런타임 배선이 같은 구현을 재사용한다. 이 모듈은 정책 원본을
직접 읽지 않고, 호출자가 주입한 문의·조항·후보만 다룬다.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import cast

from reply_gate.llm import GenerationClient

__all__ = [
    "Bm25Hit",
    "FusedHit",
    "RetrievalStage",
    "StrategyDefinition",
    "VectorHit",
    "bm25_rank",
    "default_strategy_ladder",
    "llm_rerank",
    "merge_rewritten_rankings",
    "reciprocal_rank_fusion",
]

_BM25_K1 = 1.2
_BM25_B = 0.75
_SEARCHABLE_CHARACTERS = re.compile(r"[0-9A-Za-z가-힣]+")


class RetrievalStage(StrEnum):
    """앞 단계 결과에 누적 적용되는 검색 전략 단계."""

    VECTOR = "vector"
    REWRITE = "rewrite"
    HYBRID = "hybrid_bm25_rrf"
    RERANK = "llm_rerank"


_STAGE_ORDER = (
    RetrievalStage.VECTOR,
    RetrievalStage.REWRITE,
    RetrievalStage.HYBRID,
    RetrievalStage.RERANK,
)


@dataclass(frozen=True)
class StrategyDefinition:
    """누적 단계 조합으로 표현한 검색 전략."""

    name: str
    stages: tuple[RetrievalStage, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("전략 name은 비어 있지 않아야 한다")
        if not self.stages or self.stages != _STAGE_ORDER[: len(self.stages)]:
            raise ValueError("검색 전략 단계는 vector부터 누적되는 prefix여야 한다")


def default_strategy_ladder() -> tuple[StrategyDefinition, ...]:
    """오프라인 비교의 기본 네 단계 누적 사다리."""
    return (
        StrategyDefinition("vector", _STAGE_ORDER[:1]),
        StrategyDefinition("vector_rewrite", _STAGE_ORDER[:2]),
        StrategyDefinition("vector_rewrite_hybrid", _STAGE_ORDER[:3]),
        StrategyDefinition("vector_rewrite_hybrid_rerank", _STAGE_ORDER),
    )


@dataclass(frozen=True)
class VectorHit:
    """벡터 검색 한 건의 코사인 순위."""

    rank: int
    evidence_id: str
    similarity: float


@dataclass(frozen=True)
class Bm25Hit:
    """BM25 검색 한 건의 순위와 점수."""

    rank: int
    evidence_id: str
    score: float


@dataclass(frozen=True)
class FusedHit:
    """RRF로 결합한 순위와 각 검색기의 원 점수."""

    rank: int
    evidence_id: str
    rrf_score: float
    vector_rank: int | None
    vector_similarity: float | None
    bm25_rank: int | None
    bm25_score: float | None


def _character_ngrams(text: str, *, size: int) -> tuple[str, ...]:
    if size <= 0:
        raise ValueError("ngram_size는 1 이상이어야 한다")
    normalized = "".join(_SEARCHABLE_CHARACTERS.findall(text.casefold()))
    if not normalized:
        return ()
    if len(normalized) < size:
        return (normalized,)
    return tuple(normalized[index : index + size] for index in range(len(normalized) - size + 1))


def bm25_rank(
    *, query: str, documents: Sequence[tuple[str, str]], ngram_size: int = 2
) -> tuple[Bm25Hit, ...]:
    """주입된 전체 정책 조항을 문자 n-gram BM25로 순위화한다."""
    document_ids = [evidence_id for evidence_id, _ in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("정책 근거 ID가 중복된다")
    tokenized = [
        _character_ngrams(document_text, size=ngram_size) for _, document_text in documents
    ]
    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized) if tokenized else 0.0
    query_terms = tuple(dict.fromkeys(_character_ngrams(query, size=ngram_size)))
    document_frequency = {term: sum(term in tokens for tokens in tokenized) for term in query_terms}

    scored: list[tuple[str, float]] = []
    for (evidence_id, _), tokens in zip(documents, tokenized, strict=True):
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if frequency == 0:
                continue
            containing = document_frequency[term]
            inverse_frequency = math.log(
                1.0 + (len(documents) - containing + 0.5) / (containing + 0.5)
            )
            length_ratio = len(tokens) / average_length if average_length else 0.0
            denominator = frequency + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * length_ratio)
            score += inverse_frequency * (frequency * (_BM25_K1 + 1.0)) / denominator
        scored.append((evidence_id, score))

    ordered = sorted(scored, key=lambda item: (-item[1], item[0]))
    return tuple(
        Bm25Hit(rank=rank, evidence_id=evidence_id, score=score)
        for rank, (evidence_id, score) in enumerate(ordered, start=1)
    )


def reciprocal_rank_fusion(
    *, vector: Sequence[VectorHit], bm25: Sequence[Bm25Hit], rrf_k: int = 60
) -> tuple[FusedHit, ...]:
    """두 순위의 합집합을 reciprocal rank fusion으로 결합한다."""
    if rrf_k <= 0:
        raise ValueError("rrf_k는 1 이상이어야 한다")
    vector_by_id = {hit.evidence_id: hit for hit in vector}
    bm25_by_id = {hit.evidence_id: hit for hit in bm25}
    if len(vector_by_id) != len(vector) or len(bm25_by_id) != len(bm25):
        raise ValueError("각 순위 안의 정책 근거 ID는 중복될 수 없다")

    fused: list[tuple[str, float]] = []
    for evidence_id in vector_by_id.keys() | bm25_by_id.keys():
        vector_hit = vector_by_id.get(evidence_id)
        bm25_hit = bm25_by_id.get(evidence_id)
        score = 0.0
        if vector_hit is not None:
            score += 1.0 / (rrf_k + vector_hit.rank)
        if bm25_hit is not None:
            score += 1.0 / (rrf_k + bm25_hit.rank)
        fused.append((evidence_id, score))

    ordered = sorted(fused, key=lambda item: (-item[1], item[0]))
    return tuple(
        FusedHit(
            rank=rank,
            evidence_id=evidence_id,
            rrf_score=score,
            vector_rank=(
                None if vector_by_id.get(evidence_id) is None else vector_by_id[evidence_id].rank
            ),
            vector_similarity=(
                None
                if vector_by_id.get(evidence_id) is None
                else vector_by_id[evidence_id].similarity
            ),
            bm25_rank=(
                None if bm25_by_id.get(evidence_id) is None else bm25_by_id[evidence_id].rank
            ),
            bm25_score=(
                None if bm25_by_id.get(evidence_id) is None else bm25_by_id[evidence_id].score
            ),
        )
        for rank, (evidence_id, score) in enumerate(ordered, start=1)
    )


_RERANK_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["evidence_ids"],
    "additionalProperties": False,
}


def _rerank_input(
    *, query: str, candidates: Sequence[FusedHit], policy_texts: Mapping[str, str]
) -> dict[str, object]:
    missing = [hit.evidence_id for hit in candidates if hit.evidence_id not in policy_texts]
    if missing:
        raise ValueError(f"리랭크 후보 본문이 없다: {', '.join(missing)}")
    return {
        "query": query,
        "candidates": [
            {"evidence_id": hit.evidence_id, "text": policy_texts[hit.evidence_id]}
            for hit in candidates
        ],
    }


def _rerank_cache_path(*, cache_dir: Path, model: str, input_hash: str) -> Path:
    cache_key = json.dumps(
        {"model": model, "input_hash": input_hash},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return cache_dir / f"rerank-{hashlib.sha256(cache_key.encode('utf-8')).hexdigest()}.json"


def _validated_order(data: object, *, expected: tuple[str, ...]) -> tuple[str, ...] | None:
    if not isinstance(data, dict):
        return None
    payload = cast(dict[str, object], data)
    raw_ids = payload.get("evidence_ids")
    if not isinstance(raw_ids, list) or any(not isinstance(item, str) for item in raw_ids):
        return None
    evidence_ids = tuple(cast(list[str], raw_ids))
    if len(evidence_ids) != len(expected) or set(evidence_ids) != set(expected):
        return None
    return evidence_ids


def _read_rerank_cache(
    *, path: Path, model: str, input_hash: str, expected: tuple[str, ...]
) -> tuple[str, ...] | None:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    payload = cast(dict[str, object], raw)
    if payload.get("model") != model or payload.get("input_hash") != input_hash:
        return None
    return _validated_order({"evidence_ids": payload.get("evidence_ids")}, expected=expected)


def _write_rerank_cache(
    *, path: Path, model: str, input_hash: str, evidence_ids: Sequence[str]
) -> None:
    payload = {
        "model": model,
        "input_hash": input_hash,
        "evidence_ids": list(evidence_ids),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def llm_rerank(
    *,
    query: str,
    candidates: Sequence[FusedHit],
    policy_texts: Mapping[str, str],
    client: GenerationClient,
    model: str,
    cache_dir: Path,
) -> tuple[FusedHit, ...]:
    """후보를 OpenAI 생성 경계로 재정렬하고 실패하면 입력 순위를 유지한다."""
    if not model.strip():
        raise ValueError("리랭크 model은 비어 있지 않아야 한다")
    candidates_tuple = tuple(candidates)
    expected = tuple(hit.evidence_id for hit in candidates_tuple)
    if len(expected) != len(set(expected)):
        raise ValueError("리랭크 후보의 정책 근거 ID는 중복될 수 없다")
    if not candidates_tuple:
        return ()

    rerank_input = _rerank_input(
        query=query, candidates=candidates_tuple, policy_texts=policy_texts
    )
    canonical_input = json.dumps(
        rerank_input, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    input_hash = hashlib.sha256(canonical_input.encode("utf-8")).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _rerank_cache_path(cache_dir=cache_dir, model=model, input_hash=input_hash)
    evidence_ids = _read_rerank_cache(
        path=cache_path, model=model, input_hash=input_hash, expected=expected
    )
    if evidence_ids is None:
        try:
            completion = client.complete_json(
                stage="retrieval_rerank",
                system=(
                    "문의에 답하는 데 직접 관련된 정책 조항부터 정렬한다. "
                    "후보 evidence_id를 빠짐없이 정확히 한 번씩 반환한다."
                ),
                user=canonical_input,
                schema=_RERANK_SCHEMA,
                schema_name="retrieval_rerank",
                max_output_tokens=2000,
            )
        except Exception:
            return candidates_tuple
        evidence_ids = _validated_order(completion.data, expected=expected)
        if evidence_ids is None:
            return candidates_tuple
        _write_rerank_cache(
            path=cache_path,
            model=model,
            input_hash=input_hash,
            evidence_ids=evidence_ids,
        )

    by_id = {hit.evidence_id: hit for hit in candidates_tuple}
    return tuple(
        replace(by_id[evidence_id], rank=rank)
        for rank, evidence_id in enumerate(evidence_ids, start=1)
    )


def merge_rewritten_rankings(
    *, original: Sequence[VectorHit], rewritten: Sequence[VectorHit]
) -> tuple[VectorHit, ...]:
    """원문과 재작성문 결과를 합치고 같은 근거에는 더 큰 유사도를 쓴다."""
    max_similarity: dict[str, float] = {}
    for hit in (*original, *rewritten):
        max_similarity[hit.evidence_id] = max(
            hit.similarity, max_similarity.get(hit.evidence_id, float("-inf"))
        )
    ordered = sorted(max_similarity.items(), key=lambda item: (-item[1], item[0]))
    return tuple(
        VectorHit(rank=rank, evidence_id=evidence_id, similarity=similarity)
        for rank, (evidence_id, similarity) in enumerate(ordered, start=1)
    )
