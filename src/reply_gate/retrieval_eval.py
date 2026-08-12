"""DB를 거치지 않는 정책 검색 전략 사다리 비교와 채점."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from typing import Any, Final, cast

from reply_gate.config import get_settings
from reply_gate.evaluation import DEFAULT_GOLDEN_SET_PATH, GoldenCase, load_golden_set
from reply_gate.llm import (
    EmbeddingClient,
    GenerationClient,
    JsonCompletion,
    OpenAIEmbeddingClient,
    OpenAIGenerationClient,
)
from reply_gate.policy_index import (
    DEFAULT_POLICY_DIR,
    PolicyDocument,
    load_policy_documents,
)
from reply_gate.retrieval_labels import (
    DEFAULT_RETRIEVAL_LABELS_PATH,
    RetrievalLabel,
    load_retrieval_labels,
)
from reply_gate.retrieval_strategies import (
    FusedHit,
    RetrievalStage,
    StrategyDefinition,
    VectorHit,
    bm25_rank,
    default_strategy_ladder,
    llm_rerank,
    merge_rewritten_rankings,
    reciprocal_rank_fusion,
)
from reply_gate.testing import LexicalEmbeddingClient

__all__ = [
    "DEFAULT_EMBEDDING_CACHE_DIR",
    "DEFAULT_NGRAM_SIZE",
    "DEFAULT_RERANK_MODEL",
    "DEFAULT_RETRIEVAL_REPORT_DIR",
    "DEFAULT_RRF_CUTOFF",
    "DEFAULT_RRF_K",
    "AggregateMetrics",
    "CaseScore",
    "CutoffSweepPoint",
    "RankedHit",
    "ReportPaths",
    "RetrievalConfigurationError",
    "RetrievalEvalConfig",
    "RetrievalEvaluation",
    "RetrievalQuery",
    "RetrievalScore",
    "RetrievedCase",
    "StrategyComparison",
    "StrategyCutoffKind",
    "StrategyCutoffs",
    "StrategyEvaluation",
    "StrategyRetrievedCase",
    "evaluate_retrieval",
    "evaluate_strategy_ladder",
    "retrieve_cases",
    "retrieve_strategy_ladder",
    "run_retrieval_comparison",
    "score_retrieval",
    "score_strategy_ladder",
    "write_report",
    "write_strategy_report",
]

DEFAULT_CUTOFF_SWEEP: Final = tuple(round(0.10 + index * 0.05, 2) for index in range(13))
_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_RETRIEVAL_REPORT_DIR: Final = _ROOT / "reports"
DEFAULT_EMBEDDING_CACHE_DIR: Final = _ROOT / ".retrieval-cache"
STUB_EMBEDDING_MODEL: Final = "lexical-2gram-v1"
STUB_DEFAULT_CUTOFF: Final = 0.10
DEFAULT_RERANK_MODEL: Final = "gpt-5.6-luna"
DEFAULT_RRF_K: Final = 60
DEFAULT_RRF_CUTOFF: Final = 0.02
DEFAULT_NGRAM_SIZE: Final = 2


class RetrievalConfigurationError(ValueError):
    """외부 호출 전 발견한 실행 구성 오류."""


class _IdentityRerankClient:
    """대역 사다리 배관용 무과금 리랭커. 입력 후보 순서를 그대로 돌려준다."""

    def complete_json(self, **kwargs: Any) -> JsonCompletion:
        raw: object = json.loads(cast(str, kwargs["user"]))
        if not isinstance(raw, dict):
            raise ValueError("리랭크 대역 입력이 object가 아니다")
        candidates = cast(dict[str, object], raw).get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("리랭크 대역 후보가 array가 아니다")
        evidence_ids: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("리랭크 대역 후보가 object가 아니다")
            evidence_id = cast(dict[str, object], candidate).get("evidence_id")
            if not isinstance(evidence_id, str):
                raise ValueError("리랭크 대역 후보 ID가 문자열이 아니다")
            evidence_ids.append(evidence_id)
        return JsonCompletion(data={"evidence_ids": evidence_ids}, input_tokens=0, output_tokens=0)


@dataclass(frozen=True)
class RetrievalEvalConfig:
    """벡터 단독 비교의 재현 가능한 실행 조건."""

    model: str
    dimensions: int
    top_k: int = 5
    cutoff: float = 0.30
    cutoff_sweep: tuple[float, ...] = DEFAULT_CUTOFF_SWEEP
    is_stub: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model은 비어 있지 않아야 한다")
        if self.dimensions <= 0:
            raise ValueError("dimensions는 1 이상이어야 한다")
        if self.top_k <= 0:
            raise ValueError("top_k는 1 이상이어야 한다")
        if not 0.0 <= self.cutoff <= 1.0:
            raise ValueError("cutoff는 0.0 이상 1.0 이하여야 한다")
        if not self.cutoff_sweep:
            raise ValueError("cutoff_sweep은 비어 있을 수 없다")
        if any(not 0.0 <= cutoff <= 1.0 for cutoff in self.cutoff_sweep):
            raise ValueError("cutoff_sweep의 모든 값은 0.0 이상 1.0 이하여야 한다")


@dataclass(frozen=True)
class RetrievalQuery:
    """검색 전략에 전달되는 문의 입력. 정답 라벨을 포함하지 않는다."""

    case_id: str
    text: str


@dataclass(frozen=True)
class RankedHit:
    """문의 한 건에 대한 조항의 전체 순위와 코사인 유사도."""

    rank: int
    evidence_id: str
    similarity: float
    bm25_score: float | None = None
    rrf_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None


@dataclass(frozen=True)
class RetrievedCase:
    """라벨을 보지 않고 만든 문의 한 건의 전체 벡터 순위."""

    case_id: str
    ranked_hits: tuple[RankedHit, ...]


class StrategyCutoffKind(StrEnum):
    """전략 단계마다 의미가 다른 채택 축."""

    COSINE_SIMILARITY = "cosine_similarity"
    RRF_SCORE = "rrf_score"
    RANK_LIMIT = "rank_limit"


@dataclass(frozen=True)
class StrategyCutoffs:
    """서로 단위가 다른 검색 전략 컷."""

    cosine_similarity: float = 0.30
    rrf_score: float = 0.02
    rerank_top_n: int = 5

    def __post_init__(self) -> None:
        if not 0.0 <= self.cosine_similarity <= 1.0:
            raise ValueError("cosine_similarity는 0.0 이상 1.0 이하여야 한다")
        if self.rrf_score < 0.0:
            raise ValueError("rrf_score는 0.0 이상이어야 한다")
        if self.rerank_top_n <= 0:
            raise ValueError("rerank_top_n은 1 이상이어야 한다")


@dataclass(frozen=True)
class StrategyRetrievedCase:
    """라벨을 보지 않고 검색·컷 적용까지 끝낸 전략별 케이스."""

    case_id: str
    ranked_hits: tuple[RankedHit, ...]
    accepted_hits: tuple[RankedHit, ...]


@dataclass(frozen=True)
class StrategyRetrieval:
    """한 전략의 라벨 없는 검색 결과."""

    strategy: StrategyDefinition
    cutoff_kind: StrategyCutoffKind
    cutoff_value: float | int
    cases: tuple[StrategyRetrievedCase, ...]


@dataclass(frozen=True)
class StrategyCaseScore:
    """전략 검색 결과를 독립 라벨로 채점한 케이스."""

    case_id: str
    relevant_evidence_ids: frozenset[str]
    ranked_hits: tuple[RankedHit, ...]
    accepted_hits: tuple[RankedHit, ...]
    recall_at_1: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    accepted_precision: float | None
    accepted_recall: float | None


@dataclass(frozen=True)
class StrategyEvaluation:
    """한 누적 전략의 컷·케이스·집계."""

    strategy: StrategyDefinition
    cutoff_kind: StrategyCutoffKind
    cutoff_value: float | int
    cases: tuple[StrategyCaseScore, ...]
    aggregate: AggregateMetrics


@dataclass(frozen=True)
class StrategyComparison:
    """동일 입력에 대한 네 단계 누적 전략 비교."""

    embedding_config: RetrievalEvalConfig
    cutoffs: StrategyCutoffs
    rrf_k: int
    ngram_size: int
    rerank_model: str
    strategies: tuple[StrategyEvaluation, ...]


@dataclass(frozen=True)
class CaseScore:
    """검색 결과에 독립 라벨을 붙여 채점한 문의 한 건."""

    case_id: str
    relevant_evidence_ids: frozenset[str]
    ranked_hits: tuple[RankedHit, ...]
    accepted_hits: tuple[RankedHit, ...]
    recall_at_1: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    accepted_precision: float | None
    accepted_recall: float | None


@dataclass(frozen=True)
class AggregateMetrics:
    """문의 단위 macro 평균. 각 분모의 케이스 수를 함께 보존한다."""

    recall_at_1: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    accepted_precision: float | None
    accepted_recall: float | None
    precision_case_count: int
    recall_case_count: int


@dataclass(frozen=True)
class RetrievalScore:
    """지정 컷에서의 케이스별 결과와 집계."""

    cutoff: float
    top_k: int
    cases: tuple[CaseScore, ...]
    aggregate: AggregateMetrics


@dataclass(frozen=True)
class CutoffSweepPoint:
    """한 코사인 컷에서의 채택분 macro 품질."""

    cutoff: float
    accepted_precision: float | None
    accepted_recall: float | None
    macro_f1: float | None
    precision_case_count: int
    recall_case_count: int


@dataclass(frozen=True)
class RetrievalEvaluation:
    """한 벡터 구성의 기본 컷 결과, 컷 스윕, 최적점."""

    config: RetrievalEvalConfig
    score: RetrievalScore
    sweep: tuple[CutoffSweepPoint, ...]
    best_cutoff: float | None


@dataclass(frozen=True)
class ReportPaths:
    """같은 stem을 공유하는 Markdown/JSON 보고서 경로."""

    markdown: Path
    json: Path


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _recall_at(ranked_hits: Sequence[RankedHit], relevant: frozenset[str], k: int) -> float | None:
    if not relevant:
        return None
    found = {hit.evidence_id for hit in ranked_hits[:k]}
    return len(found & relevant) / len(relevant)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_path(*, cache_dir: Path, model: str, dimensions: int, text_hash: str) -> Path:
    key = json.dumps(
        {"model": model, "dimensions": dimensions, "text_hash": text_hash},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"


def _read_cached_vector(
    *, path: Path, model: str, dimensions: int, text_hash: str
) -> list[float] | None:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    payload = cast(dict[str, object], raw)
    vector = payload.get("vector")
    if (
        payload.get("model") != model
        or payload.get("dimensions") != dimensions
        or payload.get("text_hash") != text_hash
        or not isinstance(vector, list)
        or len(vector) != dimensions
        or any(not isinstance(value, int | float) for value in vector)
    ):
        return None
    return [float(value) for value in vector]


def _write_cached_vector(
    *,
    path: Path,
    model: str,
    dimensions: int,
    text_hash: str,
    vector: Sequence[float],
) -> None:
    payload = {
        "model": model,
        "dimensions": dimensions,
        "text_hash": text_hash,
        "vector": list(vector),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def _embed_with_cache(
    *,
    texts: Sequence[str],
    stage: str,
    embedder: EmbeddingClient,
    config: RetrievalEvalConfig,
    cache_dir: Path,
) -> list[list[float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors: list[list[float] | None] = [None] * len(texts)
    misses: dict[str, list[int]] = {}
    paths: dict[str, Path] = {}
    for index, target_text in enumerate(texts):
        text_hash = _text_hash(target_text)
        path = _cache_path(
            cache_dir=cache_dir,
            model=config.model,
            dimensions=config.dimensions,
            text_hash=text_hash,
        )
        cached = _read_cached_vector(
            path=path,
            model=config.model,
            dimensions=config.dimensions,
            text_hash=text_hash,
        )
        if cached is not None:
            vectors[index] = cached
            continue
        misses.setdefault(target_text, []).append(index)
        paths[target_text] = path

    if misses:
        missing_texts = list(misses)
        embedded = embedder.embed(stage=stage, texts=missing_texts)
        if len(embedded.vectors) != len(missing_texts):
            raise ValueError(
                "임베딩 개수가 대상 텍스트 수와 다르다: "
                f"{len(embedded.vectors)} != {len(missing_texts)}"
            )
        for target_text, vector in zip(missing_texts, embedded.vectors, strict=True):
            if len(vector) != config.dimensions:
                raise ValueError(
                    f"임베딩 차원이 구성과 다르다: {len(vector)} != {config.dimensions}"
                )
            text_hash = _text_hash(target_text)
            _write_cached_vector(
                path=paths[target_text],
                model=config.model,
                dimensions=config.dimensions,
                text_hash=text_hash,
                vector=vector,
            )
            for index in misses[target_text]:
                vectors[index] = list(vector)

    if any(vector is None for vector in vectors):
        raise RuntimeError("임베딩 캐시 조립이 완료되지 않았다")
    return cast(list[list[float]], vectors)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def retrieve_cases(
    *,
    queries: Sequence[RetrievalQuery],
    policy_texts: Sequence[tuple[str, str]],
    embedder: EmbeddingClient,
    config: RetrievalEvalConfig,
    cache_dir: Path,
) -> tuple[RetrievedCase, ...]:
    """정답 라벨 없이 모든 조항을 메모리 코사인 유사도로 순위화한다."""
    if embedder.dimensions != config.dimensions:
        raise ValueError(
            f"임베더 차원이 구성과 다르다: {embedder.dimensions} != {config.dimensions}"
        )
    policy_ids = [evidence_id for evidence_id, _ in policy_texts]
    if len(policy_ids) != len(set(policy_ids)):
        raise ValueError("정책 근거 ID가 중복된다")

    policy_vectors = _embed_with_cache(
        texts=[text for _, text in policy_texts],
        stage="retrieval_policy",
        embedder=embedder,
        config=config,
        cache_dir=cache_dir,
    )
    query_vectors = _embed_with_cache(
        texts=[query.text for query in queries],
        stage="retrieval_query",
        embedder=embedder,
        config=config,
        cache_dir=cache_dir,
    )

    retrieved: list[RetrievedCase] = []
    for query, query_vector in zip(queries, query_vectors, strict=True):
        similarities = sorted(
            (
                (evidence_id, _cosine_similarity(query_vector, policy_vector))
                for evidence_id, policy_vector in zip(policy_ids, policy_vectors, strict=True)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        retrieved.append(
            RetrievedCase(
                case_id=query.case_id,
                ranked_hits=tuple(
                    RankedHit(rank=rank, evidence_id=evidence_id, similarity=similarity)
                    for rank, (evidence_id, similarity) in enumerate(similarities, start=1)
                ),
            )
        )
    return tuple(retrieved)


def _ranked_vector_hits(hits: Sequence[VectorHit]) -> tuple[RankedHit, ...]:
    return tuple(
        RankedHit(
            rank=hit.rank,
            evidence_id=hit.evidence_id,
            similarity=hit.similarity,
            vector_rank=hit.rank,
        )
        for hit in hits
    )


def _ranked_fused_hits(hits: Sequence[FusedHit]) -> tuple[RankedHit, ...]:
    return tuple(
        RankedHit(
            rank=hit.rank,
            evidence_id=hit.evidence_id,
            similarity=hit.vector_similarity or 0.0,
            bm25_score=hit.bm25_score,
            rrf_score=hit.rrf_score,
            vector_rank=hit.vector_rank,
            bm25_rank=hit.bm25_rank,
        )
        for hit in hits
    )


def _validate_rewritten_queries(
    *,
    queries: Sequence[RetrievalQuery],
    rewritten_queries: Mapping[str, str] | None,
    strategies: Sequence[StrategyDefinition],
) -> dict[str, str]:
    if not any(RetrievalStage.REWRITE in strategy.stages for strategy in strategies):
        return {}
    if rewritten_queries is None:
        raise RetrievalConfigurationError(
            "재작성 전략에는 케이스별 rewritten_queries 호출자 주입이 필요하다"
        )
    query_id_list = [query.case_id for query in queries]
    query_ids = set(query_id_list)
    if len(query_ids) != len(query_id_list):
        raise RetrievalConfigurationError("문의 케이스 ID가 중복되어 재작성문과 1:1 대응할 수 없다")
    rewrite_ids = set(rewritten_queries)
    if query_ids != rewrite_ids:
        missing = ", ".join(sorted(query_ids - rewrite_ids)) or "없음"
        extra = ", ".join(sorted(rewrite_ids - query_ids)) or "없음"
        raise RetrievalConfigurationError(
            f"문의와 주입된 재작성 ID가 다르다(누락={missing}, 추가={extra})"
        )
    empty_ids = sorted(
        case_id
        for case_id, text in rewritten_queries.items()
        if not isinstance(text, str) or not text.strip()
    )
    if empty_ids:
        raise RetrievalConfigurationError(f"주입된 재작성문이 비어 있다: {', '.join(empty_ids)}")
    return dict(rewritten_queries)


def retrieve_strategy_ladder(
    *,
    queries: Sequence[RetrievalQuery],
    policy_texts: Sequence[tuple[str, str]],
    rewritten_queries: Mapping[str, str] | None,
    embedder: EmbeddingClient,
    embedding_config: RetrievalEvalConfig,
    cutoffs: StrategyCutoffs,
    reranker: GenerationClient | None,
    rerank_model: str,
    cache_dir: Path,
    rerank_cache_dir: Path,
    rrf_k: int = 60,
    ngram_size: int = 2,
    strategies: Sequence[StrategyDefinition] = default_strategy_ladder(),
) -> tuple[StrategyRetrieval, ...]:
    """정답 라벨 없이 누적 전략 검색과 각 단계의 채택 컷을 적용한다."""
    strategy_tuple = tuple(strategies)
    if not strategy_tuple:
        raise ValueError("검색 전략은 하나 이상이어야 한다")
    uses_rewrite = any(RetrievalStage.REWRITE in strategy.stages for strategy in strategy_tuple)
    uses_hybrid = any(RetrievalStage.HYBRID in strategy.stages for strategy in strategy_tuple)
    uses_rerank = any(RetrievalStage.RERANK in strategy.stages for strategy in strategy_tuple)
    injected_rewrites = _validate_rewritten_queries(
        queries=queries,
        rewritten_queries=rewritten_queries,
        strategies=strategy_tuple,
    )
    if uses_rerank:
        if reranker is None:
            raise ValueError("리랭크 전략에는 GenerationClient가 필요하다")
        if not rerank_model.strip():
            raise ValueError("rerank_model은 비어 있지 않아야 한다")

    original = retrieve_cases(
        queries=queries,
        policy_texts=policy_texts,
        embedder=embedder,
        config=embedding_config,
        cache_dir=cache_dir,
    )
    rewritten = (
        retrieve_cases(
            queries=tuple(
                RetrievalQuery(case_id=query.case_id, text=injected_rewrites[query.case_id])
                for query in queries
            ),
            policy_texts=policy_texts,
            embedder=embedder,
            config=embedding_config,
            cache_dir=cache_dir,
        )
        if uses_rewrite
        else ()
    )
    original_by_id = {case.case_id: case for case in original}
    rewritten_by_id = {case.case_id: case for case in rewritten}
    policy_by_id = dict(policy_texts)
    if len(policy_by_id) != len(policy_texts):
        raise ValueError("정책 근거 ID가 중복된다")

    cases_by_strategy: dict[str, list[StrategyRetrievedCase]] = {
        strategy.name: [] for strategy in strategy_tuple
    }
    for query in queries:
        original_vector = tuple(
            VectorHit(hit.rank, hit.evidence_id, hit.similarity)
            for hit in original_by_id[query.case_id].ranked_hits
        )
        rewritten_vector = (
            tuple(
                VectorHit(hit.rank, hit.evidence_id, hit.similarity)
                for hit in rewritten_by_id[query.case_id].ranked_hits
            )
            if uses_rewrite
            else ()
        )
        merged_vector = (
            merge_rewritten_rankings(original=original_vector, rewritten=rewritten_vector)
            if uses_rewrite
            else original_vector
        )
        fused = (
            reciprocal_rank_fusion(
                vector=merged_vector,
                bm25=bm25_rank(query=query.text, documents=policy_texts, ngram_size=ngram_size),
                rrf_k=rrf_k,
            )
            if uses_hybrid
            else ()
        )
        hybrid_candidates = (
            tuple(
                hit for hit in fused[: embedding_config.top_k] if hit.rrf_score >= cutoffs.rrf_score
            )
            if uses_hybrid
            else ()
        )
        reranked = (
            llm_rerank(
                query=query.text,
                candidates=hybrid_candidates,
                policy_texts=policy_by_id,
                client=reranker,
                model=rerank_model,
                cache_dir=rerank_cache_dir,
            )
            if uses_rerank and reranker is not None
            else ()
        )

        for strategy in strategy_tuple:
            if RetrievalStage.RERANK in strategy.stages:
                ranked = _ranked_fused_hits(reranked)
                accepted = ranked[: cutoffs.rerank_top_n]
            elif RetrievalStage.HYBRID in strategy.stages:
                ranked = _ranked_fused_hits(fused)
                accepted = _ranked_fused_hits(hybrid_candidates)
            elif RetrievalStage.REWRITE in strategy.stages:
                ranked = _ranked_vector_hits(merged_vector)
                accepted = tuple(
                    hit
                    for hit in ranked[: embedding_config.top_k]
                    if hit.similarity >= cutoffs.cosine_similarity
                )
            else:
                ranked = _ranked_vector_hits(original_vector)
                accepted = tuple(
                    hit
                    for hit in ranked[: embedding_config.top_k]
                    if hit.similarity >= cutoffs.cosine_similarity
                )
            cases_by_strategy[strategy.name].append(
                StrategyRetrievedCase(
                    case_id=query.case_id,
                    ranked_hits=ranked,
                    accepted_hits=accepted,
                )
            )

    retrievals: list[StrategyRetrieval] = []
    for strategy in strategy_tuple:
        if RetrievalStage.RERANK in strategy.stages:
            cutoff_kind = StrategyCutoffKind.RANK_LIMIT
            cutoff_value: float | int = cutoffs.rerank_top_n
        elif RetrievalStage.HYBRID in strategy.stages:
            cutoff_kind = StrategyCutoffKind.RRF_SCORE
            cutoff_value = cutoffs.rrf_score
        else:
            cutoff_kind = StrategyCutoffKind.COSINE_SIMILARITY
            cutoff_value = cutoffs.cosine_similarity
        retrievals.append(
            StrategyRetrieval(
                strategy=strategy,
                cutoff_kind=cutoff_kind,
                cutoff_value=cutoff_value,
                cases=tuple(cases_by_strategy[strategy.name]),
            )
        )
    return tuple(retrievals)


def score_retrieval(
    retrieved: tuple[RetrievedCase, ...],
    labels: tuple[RetrievalLabel, ...],
    *,
    top_k: int,
    cutoff: float,
) -> RetrievalScore:
    """전체 순위와 독립 정답 라벨을 지정 컷으로 채점한다.

    정답이 빈 문의는 recall 분모에서 빠지지만 precision에는 들어간다. 정답이 있는데
    채택 결과가 0건이면 precision 분모에서만 빠지고 recall은 0이다.
    """
    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 한다")
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("cutoff는 0.0 이상 1.0 이하여야 한다")

    label_by_id = {label.id: label for label in labels}
    retrieved_ids = {case.case_id for case in retrieved}
    label_ids = set(label_by_id)
    if retrieved_ids != label_ids:
        missing = ", ".join(sorted(label_ids - retrieved_ids)) or "없음"
        extra = ", ".join(sorted(retrieved_ids - label_ids)) or "없음"
        raise ValueError(f"검색 결과와 라벨 ID가 다르다(누락={missing}, 추가={extra})")

    cases: list[CaseScore] = []
    for result in retrieved:
        relevant = label_by_id[result.case_id].relevant_evidence_ids
        accepted = tuple(hit for hit in result.ranked_hits[:top_k] if hit.similarity >= cutoff)

        if accepted:
            accepted_ids = {hit.evidence_id for hit in accepted}
            precision = len(accepted_ids & relevant) / len(accepted)
        elif not relevant:
            precision = 1.0
        else:
            precision = None

        accepted_ids = {hit.evidence_id for hit in accepted}
        accepted_recall = len(accepted_ids & relevant) / len(relevant) if relevant else None
        cases.append(
            CaseScore(
                case_id=result.case_id,
                relevant_evidence_ids=relevant,
                ranked_hits=result.ranked_hits,
                accepted_hits=accepted,
                recall_at_1=_recall_at(result.ranked_hits, relevant, 1),
                recall_at_3=_recall_at(result.ranked_hits, relevant, 3),
                recall_at_5=_recall_at(result.ranked_hits, relevant, 5),
                accepted_precision=precision,
                accepted_recall=accepted_recall,
            )
        )

    aggregate = AggregateMetrics(
        recall_at_1=_mean([case.recall_at_1 for case in cases if case.recall_at_1 is not None]),
        recall_at_3=_mean([case.recall_at_3 for case in cases if case.recall_at_3 is not None]),
        recall_at_5=_mean([case.recall_at_5 for case in cases if case.recall_at_5 is not None]),
        accepted_precision=_mean(
            [case.accepted_precision for case in cases if case.accepted_precision is not None]
        ),
        accepted_recall=_mean(
            [case.accepted_recall for case in cases if case.accepted_recall is not None]
        ),
        precision_case_count=sum(case.accepted_precision is not None for case in cases),
        recall_case_count=sum(case.accepted_recall is not None for case in cases),
    )
    return RetrievalScore(cutoff=cutoff, top_k=top_k, cases=tuple(cases), aggregate=aggregate)


def score_strategy_ladder(
    retrieved: Sequence[StrategyRetrieval], labels: Sequence[RetrievalLabel]
) -> tuple[StrategyEvaluation, ...]:
    """라벨 없는 전략 산출을 검색 완료 뒤 독립 채점한다."""
    labels_tuple = tuple(labels)
    label_by_id = {label.id: label for label in labels_tuple}
    evaluations: list[StrategyEvaluation] = []
    for strategy_result in retrieved:
        retrieved_ids = {case.case_id for case in strategy_result.cases}
        label_ids = set(label_by_id)
        if retrieved_ids != label_ids:
            missing = ", ".join(sorted(label_ids - retrieved_ids)) or "없음"
            extra = ", ".join(sorted(retrieved_ids - label_ids)) or "없음"
            raise ValueError(f"검색 결과와 라벨 ID가 다르다(누락={missing}, 추가={extra})")

        cases: list[StrategyCaseScore] = []
        for result in strategy_result.cases:
            relevant = label_by_id[result.case_id].relevant_evidence_ids
            accepted_ids = {hit.evidence_id for hit in result.accepted_hits}
            if result.accepted_hits:
                precision = len(accepted_ids & relevant) / len(result.accepted_hits)
            elif not relevant:
                precision = 1.0
            else:
                precision = None
            accepted_recall = len(accepted_ids & relevant) / len(relevant) if relevant else None
            cases.append(
                StrategyCaseScore(
                    case_id=result.case_id,
                    relevant_evidence_ids=relevant,
                    ranked_hits=result.ranked_hits,
                    accepted_hits=result.accepted_hits,
                    recall_at_1=_recall_at(result.ranked_hits, relevant, 1),
                    recall_at_3=_recall_at(result.ranked_hits, relevant, 3),
                    recall_at_5=_recall_at(result.ranked_hits, relevant, 5),
                    accepted_precision=precision,
                    accepted_recall=accepted_recall,
                )
            )

        aggregate = AggregateMetrics(
            recall_at_1=_mean([case.recall_at_1 for case in cases if case.recall_at_1 is not None]),
            recall_at_3=_mean([case.recall_at_3 for case in cases if case.recall_at_3 is not None]),
            recall_at_5=_mean([case.recall_at_5 for case in cases if case.recall_at_5 is not None]),
            accepted_precision=_mean(
                [case.accepted_precision for case in cases if case.accepted_precision is not None]
            ),
            accepted_recall=_mean(
                [case.accepted_recall for case in cases if case.accepted_recall is not None]
            ),
            precision_case_count=sum(case.accepted_precision is not None for case in cases),
            recall_case_count=sum(case.accepted_recall is not None for case in cases),
        )
        evaluations.append(
            StrategyEvaluation(
                strategy=strategy_result.strategy,
                cutoff_kind=strategy_result.cutoff_kind,
                cutoff_value=strategy_result.cutoff_value,
                cases=tuple(cases),
                aggregate=aggregate,
            )
        )
    return tuple(evaluations)


def evaluate_strategy_ladder(
    *,
    documents: Sequence[PolicyDocument],
    cases: Sequence[GoldenCase],
    labels: Sequence[RetrievalLabel],
    rewritten_queries: Mapping[str, str] | None,
    embedder: EmbeddingClient,
    embedding_config: RetrievalEvalConfig,
    cutoffs: StrategyCutoffs,
    reranker: GenerationClient | None,
    rerank_model: str,
    cache_dir: Path,
    rerank_cache_dir: Path,
    rrf_k: int = 60,
    ngram_size: int = 2,
    strategies: Sequence[StrategyDefinition] = default_strategy_ladder(),
) -> StrategyComparison:
    """정책·문의에 네 전략을 적용한 뒤에만 라벨로 채점한다."""
    queries = tuple(RetrievalQuery(case_id=case.id, text=case.content) for case in cases)
    policy_texts = tuple(
        (chunk.evidence_id, chunk.embedding_text)
        for document in documents
        for chunk in document.chunks
    )
    retrieved = retrieve_strategy_ladder(
        queries=queries,
        policy_texts=policy_texts,
        rewritten_queries=rewritten_queries,
        embedder=embedder,
        embedding_config=embedding_config,
        cutoffs=cutoffs,
        reranker=reranker,
        rerank_model=rerank_model,
        cache_dir=cache_dir,
        rerank_cache_dir=rerank_cache_dir,
        rrf_k=rrf_k,
        ngram_size=ngram_size,
        strategies=strategies,
    )
    return StrategyComparison(
        embedding_config=embedding_config,
        cutoffs=cutoffs,
        rrf_k=rrf_k,
        ngram_size=ngram_size,
        rerank_model=rerank_model,
        strategies=score_strategy_ladder(retrieved, labels),
    )


def _macro_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_retrieval(
    *,
    documents: Sequence[PolicyDocument],
    cases: Sequence[GoldenCase],
    labels: Sequence[RetrievalLabel],
    embedder: EmbeddingClient,
    config: RetrievalEvalConfig,
    cache_dir: Path,
) -> RetrievalEvaluation:
    """정책·문의 로더 산출을 DB 없이 벡터화하고, 검색 완료 뒤 독립 라벨로 채점한다."""
    queries = tuple(RetrievalQuery(case_id=case.id, text=case.content) for case in cases)
    policy_texts = tuple(
        (chunk.evidence_id, chunk.embedding_text)
        for document in documents
        for chunk in document.chunks
    )
    # 이 호출은 라벨을 받지 않는다. 검색 전략이 정답을 보지 않는 경계다.
    retrieved = retrieve_cases(
        queries=queries,
        policy_texts=policy_texts,
        embedder=embedder,
        config=config,
        cache_dir=cache_dir,
    )
    labels_tuple = tuple(labels)
    score = score_retrieval(retrieved, labels_tuple, top_k=config.top_k, cutoff=config.cutoff)
    sweep: list[CutoffSweepPoint] = []
    for cutoff in config.cutoff_sweep:
        aggregate = score_retrieval(
            retrieved, labels_tuple, top_k=config.top_k, cutoff=cutoff
        ).aggregate
        sweep.append(
            CutoffSweepPoint(
                cutoff=cutoff,
                accepted_precision=aggregate.accepted_precision,
                accepted_recall=aggregate.accepted_recall,
                macro_f1=_macro_f1(aggregate.accepted_precision, aggregate.accepted_recall),
                precision_case_count=aggregate.precision_case_count,
                recall_case_count=aggregate.recall_case_count,
            )
        )
    eligible = [point for point in sweep if point.macro_f1 is not None]
    best = (
        max(
            eligible,
            key=lambda point: (
                cast(float, point.macro_f1),
                cast(float, point.accepted_precision),
                cast(float, point.accepted_recall),
                point.cutoff,
            ),
        )
        if eligible
        else None
    )
    return RetrievalEvaluation(
        config=config,
        score=score,
        sweep=tuple(sweep),
        best_cutoff=None if best is None else best.cutoff,
    )


def _number(value: float | None) -> str:
    return "제외" if value is None else f"{value:.4f}"


def _report_slug(config: RetrievalEvalConfig) -> str:
    mode = "stub" if config.is_stub else "live"
    model = re.sub(r"[^a-z0-9]+", "-", config.model.lower()).strip("-") or "model"
    cutoff = f"{round(config.cutoff * 100):03d}"
    return f"vector-{mode}-{model}-d{config.dimensions}-k{config.top_k}-c{cutoff}"


def _report_paths(output_dir: Path, config: RetrievalEvalConfig, suffix: int | None) -> ReportPaths:
    ending = "" if suffix is None else f"-{suffix}"
    stem = f"retrieval-{_report_slug(config)}{ending}"
    return ReportPaths(markdown=output_dir / f"{stem}.md", json=output_dir / f"{stem}.json")


def _next_report_paths(output_dir: Path, config: RetrievalEvalConfig) -> ReportPaths:
    suffix: int | None = None
    while True:
        paths = _report_paths(output_dir, config, suffix)
        if not paths.markdown.exists() and not paths.json.exists():
            return paths
        suffix = 2 if suffix is None else suffix + 1


def _metrics_json(metrics: AggregateMetrics) -> dict[str, float | int | None]:
    return {
        "recall_at_1": metrics.recall_at_1,
        "recall_at_3": metrics.recall_at_3,
        "recall_at_5": metrics.recall_at_5,
        "accepted_precision": metrics.accepted_precision,
        "accepted_recall": metrics.accepted_recall,
        "precision_case_count": metrics.precision_case_count,
        "recall_case_count": metrics.recall_case_count,
    }


def _json_report(evaluation: RetrievalEvaluation) -> dict[str, object]:
    config = evaluation.config
    return {
        "configuration": {
            "strategy": "vector_only",
            "model": config.model,
            "dimensions": config.dimensions,
            "top_k": config.top_k,
            "cutoff": config.cutoff,
            "cutoff_sweep": list(config.cutoff_sweep),
            "stub": config.is_stub,
        },
        "run_conditions": {
            "database_used": False,
            "labels_used_for_retrieval": False,
            "warning": (
                "결정론 어휘 임베딩 대역 수치는 실제 검색 품질이 아니다. "
                "외부 호출 없는 배관 검증용이다."
                if config.is_stub
                else None
            ),
        },
        "aggregate": _metrics_json(evaluation.score.aggregate),
        "sweep": [
            {
                "cutoff": point.cutoff,
                "accepted_precision": point.accepted_precision,
                "accepted_recall": point.accepted_recall,
                "macro_f1": point.macro_f1,
                "precision_case_count": point.precision_case_count,
                "recall_case_count": point.recall_case_count,
            }
            for point in evaluation.sweep
        ],
        "best_cutoff": evaluation.best_cutoff,
        "best_cutoff_selection": ("macro F1 최대, 동률이면 precision, recall, 높은 cutoff 순"),
        "cases": [
            {
                "id": case.case_id,
                "relevant_evidence_ids": sorted(case.relevant_evidence_ids),
                "recall_at_1": case.recall_at_1,
                "recall_at_3": case.recall_at_3,
                "recall_at_5": case.recall_at_5,
                "accepted_precision": case.accepted_precision,
                "accepted_recall": case.accepted_recall,
                "accepted_hits": [
                    {
                        "rank": hit.rank,
                        "evidence_id": hit.evidence_id,
                        "similarity": hit.similarity,
                    }
                    for hit in case.accepted_hits
                ],
                "ranked_hits": [
                    {
                        "rank": hit.rank,
                        "evidence_id": hit.evidence_id,
                        "similarity": hit.similarity,
                    }
                    for hit in case.ranked_hits
                ],
            }
            for case in evaluation.score.cases
        ],
    }


def _markdown_report(evaluation: RetrievalEvaluation) -> str:
    config = evaluation.config
    aggregate = evaluation.score.aggregate
    lines = [
        "# 정책 검색 비교 — 벡터 단독",
        "",
        "## 실행 조건",
        "",
        "- 전략: 벡터 단독 (DB 미사용, 메모리 코사인 유사도)",
        f"- 임베딩: `{config.model}` ({config.dimensions}차원)",
        f"- top_k: {config.top_k}",
        f"- 채택 코사인 컷: {config.cutoff:.2f}",
        "- 라벨 사용 경계: 검색 완료 뒤 채점 단계에서만 사용",
    ]
    if config.is_stub:
        lines.extend(
            [
                "- **경고: 결정론 어휘 임베딩 대역 수치는 실제 검색 품질이 아니다. "
                "외부 호출 없는 배관 검증용이다.**",
                "- 대역 실행 조건: `testing.LexicalEmbeddingClient`, 과금 0회, DB 0회",
            ]
        )
    lines.extend(
        [
            "",
            "## 기본 컷 집계",
            "",
            "| metric | macro value | denominator cases |",
            "|---|---:|---:|",
            f"| recall@1 | {_number(aggregate.recall_at_1)} | {aggregate.recall_case_count} |",
            f"| recall@3 | {_number(aggregate.recall_at_3)} | {aggregate.recall_case_count} |",
            f"| recall@5 | {_number(aggregate.recall_at_5)} | {aggregate.recall_case_count} |",
            "| accepted precision | "
            f"{_number(aggregate.accepted_precision)} | {aggregate.precision_case_count} |",
            "| accepted recall | "
            f"{_number(aggregate.accepted_recall)} | {aggregate.recall_case_count} |",
            "",
            "## 컷 스윕",
            "",
            "최적점은 macro F1 최대, 동률이면 precision, recall, 높은 cutoff 순이다.",
            f"선택된 최적 컷: {_number(evaluation.best_cutoff)}",
            "",
            "| cutoff | precision | recall | macro F1 | precision n | recall n |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for point in evaluation.sweep:
        lines.append(
            f"| {point.cutoff:.2f} | {_number(point.accepted_precision)} | "
            f"{_number(point.accepted_recall)} | {_number(point.macro_f1)} | "
            f"{point.precision_case_count} | {point.recall_case_count} |"
        )
    lines.extend(
        [
            "",
            "## 케이스별 결과",
            "",
            "`전체 순위`는 임계값과 top_k를 적용하기 전 26개 조항의 순위다. "
            "`컷 통과`는 top_k 안에서 구성 컷을 넘은 조항이다.",
            "",
            "| case | relevant | r@1/3/5 | cutoff precision/recall | 컷 통과 | 전체 순위 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for case in evaluation.score.cases:
        relevant = ", ".join(sorted(case.relevant_evidence_ids)) or "없음"
        accepted = (
            "<br>".join(
                f"{hit.rank}. {hit.evidence_id} ({hit.similarity:.6f})"
                for hit in case.accepted_hits
            )
            or "없음"
        )
        ranking = "<br>".join(
            f"{hit.rank}. {hit.evidence_id} ({hit.similarity:.6f})" for hit in case.ranked_hits
        )
        lines.append(
            f"| {case.case_id} | {relevant} | {_number(case.recall_at_1)} / "
            f"{_number(case.recall_at_3)} / {_number(case.recall_at_5)} | "
            f"{_number(case.accepted_precision)} / {_number(case.accepted_recall)} | "
            f"{accepted} | {ranking} |"
        )
    return "\n".join(lines) + "\n"


def write_report(evaluation: RetrievalEvaluation, *, output_dir: Path) -> ReportPaths:
    """두 형식 보고서를 다음 사용 가능한 retrieval 이름에 써 기존 산출물을 보존한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _next_report_paths(output_dir, evaluation.config)
    markdown = _markdown_report(evaluation)
    json_text = json.dumps(_json_report(evaluation), ensure_ascii=False, indent=2) + "\n"
    # 경로 선택 뒤에도 독점 생성 모드로 재확인해 스크립트 재실행/동시 실행의 덮어쓰기를 막는다.
    created_markdown = False
    try:
        with paths.markdown.open("x", encoding="utf-8") as handle:
            handle.write(markdown)
        created_markdown = True
        with paths.json.open("x", encoding="utf-8") as handle:
            handle.write(json_text)
    except FileExistsError:
        # 동시 실행이 사이에 선점했다면 그 산출물은 건드리지 않고 다음 이름을 다시 찾는다.
        # 이 실행이 먼저 만든 Markdown만 있는 경우에는 불완전한 쌍을 남기지 않는다.
        if created_markdown:
            paths.markdown.unlink(missing_ok=True)
        return write_report(evaluation, output_dir=output_dir)
    return paths


def _strategy_hit_json(hit: RankedHit) -> dict[str, object]:
    return {
        "rank": hit.rank,
        "evidence_id": hit.evidence_id,
        "vector_similarity": hit.similarity,
        "vector_rank": hit.vector_rank,
        "bm25_score": hit.bm25_score,
        "bm25_rank": hit.bm25_rank,
        "rrf_score": hit.rrf_score,
    }


def _strategy_json_report(comparison: StrategyComparison) -> dict[str, object]:
    config = comparison.embedding_config
    uses_rewrite = any(
        RetrievalStage.REWRITE in result.strategy.stages for result in comparison.strategies
    )
    warning = (
        "결정론 어휘 임베딩·리랭크 대역 수치는 실제 검색 품질이 아니다. "
        "외부 호출 없는 배관 검증용이다."
        if config.is_stub
        else None
    )
    return {
        "configuration": {
            "embedding_model": config.model,
            "dimensions": config.dimensions,
            "top_k": config.top_k,
            "cosine_similarity_cutoff": comparison.cutoffs.cosine_similarity,
            "rrf_score_cutoff": comparison.cutoffs.rrf_score,
            "rerank_top_n": comparison.cutoffs.rerank_top_n,
            "rrf_k": comparison.rrf_k,
            "ngram_size": comparison.ngram_size,
            "rerank_model": comparison.rerank_model,
            "stub": config.is_stub,
        },
        "run_conditions": {
            "database_used": False,
            "labels_used_for_retrieval": False,
            "rewrite_source": "caller_injected" if uses_rewrite else "not_used",
            "warning": warning,
        },
        "strategies": [
            {
                "name": result.strategy.name,
                "stages": [stage.value for stage in result.strategy.stages],
                "cutoff": {
                    "kind": result.cutoff_kind.value,
                    "value": result.cutoff_value,
                },
                "aggregate": _metrics_json(result.aggregate),
                "cases": [
                    {
                        "id": case.case_id,
                        "strategy": result.strategy.name,
                        "cutoff": {
                            "kind": result.cutoff_kind.value,
                            "value": result.cutoff_value,
                        },
                        "relevant_evidence_ids": sorted(case.relevant_evidence_ids),
                        "recall_at_1": case.recall_at_1,
                        "recall_at_3": case.recall_at_3,
                        "recall_at_5": case.recall_at_5,
                        "accepted_precision": case.accepted_precision,
                        "accepted_recall": case.accepted_recall,
                        "accepted_hits": [_strategy_hit_json(hit) for hit in case.accepted_hits],
                        "ranked_hits": [_strategy_hit_json(hit) for hit in case.ranked_hits],
                    }
                    for case in result.cases
                ],
            }
            for result in comparison.strategies
        ],
    }


def _strategy_markdown_report(comparison: StrategyComparison) -> str:
    config = comparison.embedding_config
    uses_rewrite = any(
        RetrievalStage.REWRITE in result.strategy.stages for result in comparison.strategies
    )
    lines = [
        "# 정책 검색 전략 비교",
        "",
        "## 실행 조건",
        "",
        "- 코퍼스: 주입된 정책 파일 조항 (DB 미사용)",
        (
            "- 질의 재작성: 호출자 주입 (원문 검색을 항상 함께 유지)"
            if uses_rewrite
            else "- 질의 재작성: 미사용"
        ),
        f"- 임베딩: `{config.model}` ({config.dimensions}차원)",
        f"- 벡터/재작성 코사인 컷: {comparison.cutoffs.cosine_similarity:.6f}",
        f"- 하이브리드 RRF 컷: {comparison.cutoffs.rrf_score:.6f}",
        f"- 리랭크 최종 순위 상한 n: {comparison.cutoffs.rerank_top_n}",
        f"- RRF k: {comparison.rrf_k}",
        f"- BM25 문자 n-gram: {comparison.ngram_size}",
        f"- 리랭크 모델: `{comparison.rerank_model}`",
        "- 라벨 사용 경계: 검색·컷·리랭크 완료 뒤 채점 단계에서만 사용",
    ]
    if config.is_stub:
        lines.extend(
            [
                "- **경고: 결정론 어휘 임베딩·리랭크 대역 수치는 실제 검색 품질이 아니다. "
                "외부 호출 없는 배관 검증용이다.**",
                "- 대역 실행 조건: 과금 0회, DB 0회",
            ]
        )
    lines.extend(
        [
            "",
            "## 전략별 집계",
            "",
            "| strategy | stages | cutoff kind/value | r@1 | r@3 | r@5 | precision | recall |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in comparison.strategies:
        aggregate = result.aggregate
        stages = " + ".join(stage.value for stage in result.strategy.stages)
        lines.append(
            f"| {result.strategy.name} | {stages} | {result.cutoff_kind.value}="
            f"{result.cutoff_value} | {_number(aggregate.recall_at_1)} | "
            f"{_number(aggregate.recall_at_3)} | {_number(aggregate.recall_at_5)} | "
            f"{_number(aggregate.accepted_precision)} | {_number(aggregate.accepted_recall)} |"
        )
    lines.extend(["", "## 케이스별 결과", ""])
    for result in comparison.strategies:
        lines.extend(
            [
                f"### {result.strategy.name}",
                "",
                f"컷: `{result.cutoff_kind.value}={result.cutoff_value}`",
                "",
                "| case | 채택 결과 | 순위 |",
                "|---|---|---|",
            ]
        )
        for case in result.cases:
            accepted = (
                "<br>".join(f"{hit.rank}. {hit.evidence_id}" for hit in case.accepted_hits)
                or "없음"
            )
            ranking = "<br>".join(
                f"{hit.rank}. {hit.evidence_id} "
                f"(cos={hit.similarity:.6f}, bm25={_number(hit.bm25_score)}, "
                f"rrf={_number(hit.rrf_score)})"
                for hit in case.ranked_hits
            )
            lines.append(f"| {case.case_id} | {accepted} | {ranking} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _next_strategy_report_paths(output_dir: Path, comparison: StrategyComparison) -> ReportPaths:
    config = comparison.embedding_config
    mode = "stub" if config.is_stub else "live"
    model = re.sub(r"[^a-z0-9]+", "-", config.model.lower()).strip("-") or "model"
    stem = f"retrieval-strategies-{mode}-{model}-d{config.dimensions}"
    suffix: int | None = None
    while True:
        ending = "" if suffix is None else f"-{suffix}"
        paths = ReportPaths(
            markdown=output_dir / f"{stem}{ending}.md",
            json=output_dir / f"{stem}{ending}.json",
        )
        if not paths.markdown.exists() and not paths.json.exists():
            return paths
        suffix = 2 if suffix is None else suffix + 1


def write_strategy_report(comparison: StrategyComparison, *, output_dir: Path) -> ReportPaths:
    """전략 사다리 비교를 기존 산출물과 충돌하지 않는 두 형식으로 쓴다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _next_strategy_report_paths(output_dir, comparison)
    markdown = _strategy_markdown_report(comparison)
    json_text = json.dumps(_strategy_json_report(comparison), ensure_ascii=False, indent=2) + "\n"
    created_markdown = False
    try:
        with paths.markdown.open("x", encoding="utf-8") as handle:
            handle.write(markdown)
        created_markdown = True
        with paths.json.open("x", encoding="utf-8") as handle:
            handle.write(json_text)
    except FileExistsError:
        if created_markdown:
            paths.markdown.unlink(missing_ok=True)
        return write_strategy_report(comparison, output_dir=output_dir)
    return paths


def _cutoff_sweep(*, start: float, end: float, step: float) -> tuple[float, ...]:
    start_decimal = Decimal(str(start))
    end_decimal = Decimal(str(end))
    step_decimal = Decimal(str(step))
    if step_decimal <= 0:
        raise RetrievalConfigurationError("sweep_step은 0보다 커야 한다")
    if end_decimal < start_decimal:
        raise RetrievalConfigurationError("sweep_end는 sweep_start 이상이어야 한다")
    span = end_decimal - start_decimal
    if span % step_decimal != 0:
        raise RetrievalConfigurationError("sweep 범위는 step으로 정확히 나누어져야 한다")
    values = tuple(
        float(start_decimal + step_decimal * index) for index in range(int(span / step_decimal) + 1)
    )
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise RetrievalConfigurationError("sweep 컷은 0.0 이상 1.0 이하여야 한다")
    return values


def run_retrieval_comparison(
    *,
    live: bool,
    dimensions: int | None = None,
    top_k: int = 5,
    cutoff: float | None = None,
    sweep_start: float = 0.10,
    sweep_end: float = 0.70,
    sweep_step: float = 0.05,
    policy_dir: Path = DEFAULT_POLICY_DIR,
    golden_set_path: Path = DEFAULT_GOLDEN_SET_PATH,
    labels_path: Path = DEFAULT_RETRIEVAL_LABELS_PATH,
    output_dir: Path = DEFAULT_RETRIEVAL_REPORT_DIR,
    cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR,
    rewritten_queries: Mapping[str, str] | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    rrf_cutoff: float = DEFAULT_RRF_CUTOFF,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
    rerank_top_n: int = 5,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    strategies: Sequence[StrategyDefinition] = default_strategy_ladder(),
) -> ReportPaths:
    """CLI가 호출하는 전체 오프라인 사다리. 실제 모드만 외부 호출을 과금한다."""
    strategy_tuple = tuple(strategies)
    if not strategy_tuple:
        raise RetrievalConfigurationError("검색 전략은 하나 이상이어야 한다")
    cases = load_golden_set(golden_set_path)
    queries = tuple(RetrievalQuery(case_id=case.id, text=case.content) for case in cases)
    injected_rewrites = _validate_rewritten_queries(
        queries=queries,
        rewritten_queries=rewritten_queries,
        strategies=strategy_tuple,
    )

    settings = get_settings()
    resolved_dimensions = settings.embedding_dimensions if dimensions is None else dimensions
    sweep = _cutoff_sweep(start=sweep_start, end=sweep_end, step=sweep_step)
    if live:
        if not settings.openai_api_key:
            raise RetrievalConfigurationError(
                "OPENAI_API_KEY가 없다 — 실제 임베딩 비교는 과금되므로 키를 선검사한다"
            )
        model = settings.embedding_model
        resolved_cutoff = settings.vector_similarity_threshold if cutoff is None else cutoff
        embedder: EmbeddingClient = OpenAIEmbeddingClient(
            api_key=settings.openai_api_key,
            model=model,
            dimensions=resolved_dimensions,
        )
        reranker: GenerationClient | None = (
            OpenAIGenerationClient(
                api_key=settings.openai_api_key,
                model=rerank_model,
            )
            if any(RetrievalStage.RERANK in strategy.stages for strategy in strategy_tuple)
            else None
        )
    else:
        model = STUB_EMBEDDING_MODEL
        resolved_cutoff = STUB_DEFAULT_CUTOFF if cutoff is None else cutoff
        embedder = LexicalEmbeddingClient(dimensions=resolved_dimensions)
        reranker = (
            _IdentityRerankClient()
            if any(RetrievalStage.RERANK in strategy.stages for strategy in strategy_tuple)
            else None
        )

    config = RetrievalEvalConfig(
        model=model,
        dimensions=resolved_dimensions,
        top_k=top_k,
        cutoff=resolved_cutoff,
        cutoff_sweep=sweep,
        is_stub=not live,
    )
    # 세 입력은 각각의 단독 소유 로더에서 독립적으로 읽는다.
    documents = load_policy_documents(policy_dir)
    labels = load_retrieval_labels(labels_path)
    comparison = evaluate_strategy_ladder(
        documents=documents,
        cases=cases,
        labels=labels,
        rewritten_queries=injected_rewrites if injected_rewrites else None,
        embedder=embedder,
        embedding_config=config,
        cutoffs=StrategyCutoffs(
            cosine_similarity=resolved_cutoff,
            rrf_score=rrf_cutoff,
            rerank_top_n=rerank_top_n,
        ),
        reranker=reranker,
        rerank_model=(
            rerank_model
            if live
            else "stub-identity-reranker"
            if reranker is not None
            else "not_used"
        ),
        cache_dir=cache_dir,
        rerank_cache_dir=cache_dir / "rerank",
        rrf_k=rrf_k,
        ngram_size=ngram_size,
        strategies=strategy_tuple,
    )
    return write_strategy_report(comparison, output_dir=output_dir)
