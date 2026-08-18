"""DB를 거치지 않는 정책 검색 전략 사다리 비교와 채점."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from typing import Any, Final, cast

from reply_gate.config import get_settings
from reply_gate.evaluation import DEFAULT_GOLDEN_SET_PATH, GoldenCase, load_golden_set
from reply_gate.llm import (
    BgeM3EmbeddingClient,
    EmbeddingClient,
    GenerationClient,
    JsonCompletion,
    OpenAIEmbeddingClient,
    OpenAIGenerationClient,
    OptionalEmbeddingDependencyError,
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
    merge_rewritten_bm25,
    merge_rewritten_rankings,
    reciprocal_rank_fusion,
)
from reply_gate.testing import LexicalEmbeddingClient

__all__ = [
    "DEFAULT_EMBEDDING_CACHE_DIR",
    "DEFAULT_EMBEDDING_CANDIDATES",
    "DEFAULT_FUSION_POOL_SIZE",
    "DEFAULT_NGRAM_SIZE",
    "DEFAULT_ORACLE_REWRITTEN_QUERIES_PATH",
    "DEFAULT_RETRIEVAL_REPORT_DIR",
    "DEFAULT_REWRITTEN_QUERIES_PATH",
    "DEFAULT_RRF_CUTOFF",
    "DEFAULT_RRF_K",
    "AggregateMetrics",
    "CaseScore",
    "CutoffSweepPoint",
    "EmbeddingAxisResult",
    "EmbeddingAxisRow",
    "EmbeddingCandidate",
    "EmbeddingProvider",
    "RankedHit",
    "ReportPaths",
    "RerankStats",
    "RetrievalConfigurationError",
    "RetrievalEvalConfig",
    "RetrievalEvaluation",
    "RetrievalQuery",
    "RetrievalScore",
    "RetrievedCase",
    "RewriteCondition",
    "StrategyComparison",
    "StrategyCutoffs",
    "StrategyEvaluation",
    "StrategyLadderRetrieval",
    "StrategyRetrieval",
    "StrategyRetrievedCase",
    "UnmeasuredStage",
    "evaluate_retrieval",
    "evaluate_strategy_ladder",
    "load_rewritten_queries",
    "retrieve_cases",
    "retrieve_strategy_ladder",
    "run_embedding_model_axis",
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
DEFAULT_RRF_K: Final = 60
#: RRF 점수는 순위 기반이라 절대 관련성을 표현하지 못한다. 기본값은 무필터(0.0)이고,
#: 0 이 아닌 값은 이 코퍼스에서 실제로 걸러낼 수 있는지 검사한 뒤에만 쓰인다.
DEFAULT_RRF_CUTOFF: Final = 0.0
#: 융합 후보 풀. 리랭크 최종 채택 상한보다 커야 리랭크가 채택 집합을 바꿀 수 있다.
DEFAULT_FUSION_POOL_SIZE: Final = 10
DEFAULT_NGRAM_SIZE: Final = 2
DEFAULT_REWRITTEN_QUERIES_PATH: Final = _ROOT / "data" / "rewritten_queries.jsonl"
DEFAULT_ORACLE_REWRITTEN_QUERIES_PATH: Final = _ROOT / "data" / "rewritten_queries_oracle.jsonl"
_REWRITTEN_QUERY_ROW_KEYS: Final = frozenset({"id", "original", "rewritten", "note"})


class RetrievalConfigurationError(ValueError):
    """외부 호출 전 발견한 실행 구성 오류."""


class RewriteCondition(StrEnum):
    """재작성 비교 입력의 생성 조건."""

    BLIND = "blind"
    ORACLE = "oracle_upper_bound"
    CALLER_INJECTED = "caller_injected"
    NOT_USED = "not_used"


class EmbeddingProvider(StrEnum):
    """비교 임베딩의 실행 위치."""

    OPENAI = "openai"
    LOCAL = "local"


@dataclass(frozen=True)
class EmbeddingCandidate:
    """임베딩 모델 축 한 행."""

    key: str
    model: str
    dimensions: int
    provider: EmbeddingProvider


DEFAULT_EMBEDDING_CANDIDATES: Final = (
    EmbeddingCandidate(
        key="3-small-1536",
        model="text-embedding-3-small",
        dimensions=1536,
        provider=EmbeddingProvider.OPENAI,
    ),
    EmbeddingCandidate(
        key="3-large-1536",
        model="text-embedding-3-large",
        dimensions=1536,
        provider=EmbeddingProvider.OPENAI,
    ),
    EmbeddingCandidate(
        key="3-large-3072",
        model="text-embedding-3-large",
        dimensions=3072,
        provider=EmbeddingProvider.OPENAI,
    ),
    EmbeddingCandidate(
        key="bge-m3-1024",
        model=BgeM3EmbeddingClient.MODEL,
        dimensions=BgeM3EmbeddingClient.DIMENSIONS,
        provider=EmbeddingProvider.LOCAL,
    ),
)


@dataclass(frozen=True)
class EmbeddingAxisRow:
    """모델 축 한 행의 실행 결과. 미실행은 0이 아니라 사유를 가진다."""

    candidate: EmbeddingCandidate
    measured: bool
    reports: ReportPaths | None
    reason: str | None


@dataclass(frozen=True)
class EmbeddingAxisResult:
    """다른 행의 미실행에 막히지 않는 임베딩 모델 축 결과."""

    rows: tuple[EmbeddingAxisRow, ...]


def load_rewritten_queries(
    path: Path = DEFAULT_REWRITTEN_QUERIES_PATH,
    *,
    golden_set_path: Path = DEFAULT_GOLDEN_SET_PATH,
) -> dict[str, str]:
    """골든셋 전용 재작성 질의를 읽고 원문과의 1:1 계약을 검증한다."""
    rows: list[tuple[int, str, str, str]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                raw: object = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} JSON 파싱 실패: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{lineno} 각 줄은 JSON 객체여야 한다")
            row = cast(dict[str, object], raw)
            row_keys = set(row)
            if row_keys != _REWRITTEN_QUERY_ROW_KEYS:
                missing = ", ".join(sorted(_REWRITTEN_QUERY_ROW_KEYS - row_keys)) or "없음"
                extra = ", ".join(sorted(row_keys - _REWRITTEN_QUERY_ROW_KEYS)) or "없음"
                raise ValueError(
                    f"{path}:{lineno} 행 키가 계약과 다르다(누락={missing}, 추가={extra})"
                )

            case_id = row["id"]
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"{path}:{lineno} id는 비어 있지 않은 문자열이어야 한다")
            if case_id in seen:
                raise ValueError(f"재작성 ID가 중복된다: {case_id}")
            seen.add(case_id)

            original = row["original"]
            if not isinstance(original, str):
                raise ValueError(f"{path}:{lineno} original은 문자열이어야 한다")
            rewritten = row["rewritten"]
            if not isinstance(rewritten, str) or not rewritten.strip():
                raise ValueError(
                    f"{path}:{lineno} rewritten은 비어 있지 않은 문자열이어야 한다: {case_id}"
                )
            note = row["note"]
            if not isinstance(note, str):
                raise ValueError(f"{path}:{lineno} note는 문자열이어야 한다")
            rows.append((lineno, case_id, original, rewritten))

    cases = load_golden_set(golden_set_path)
    golden_by_id = {case.id: case for case in cases}
    fixture_ids = {case_id for _, case_id, _, _ in rows}
    golden_ids = set(golden_by_id)
    unknown_ids = fixture_ids - golden_ids
    if unknown_ids:
        raise ValueError(
            "골든셋에 없는 ID가 재작성 픽스처에 있다: " + ", ".join(sorted(unknown_ids))
        )
    missing_ids = golden_ids - fixture_ids
    if missing_ids:
        raise ValueError(
            "재작성 픽스처에서 빠진 골든셋 ID가 있다: " + ", ".join(sorted(missing_ids))
        )

    for lineno, case_id, original, _ in rows:
        if original != golden_by_id[case_id].content:
            raise ValueError(f"{path}:{lineno} original이 골든셋 content와 다르다: {case_id}")
    return {case_id: rewritten for _, case_id, _, rewritten in rows}


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
    """문의 한 건에 대한 조항의 전체 순위와 코사인 유사도.

    `similarity` 가 None 이면 그 조항은 벡터 순위에 없어 코사인이 측정되지 않았다는 뜻이다.
    0.0 과 같은 값이 아니다 — 절대 관련성 게이트는 측정되지 않은 조항을 통과시키지 않는다.
    """

    rank: int
    evidence_id: str
    similarity: float | None
    bm25_score: float | None = None
    rrf_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None


@dataclass(frozen=True)
class RetrievedCase:
    """라벨을 보지 않고 만든 문의 한 건의 전체 벡터 순위."""

    case_id: str
    ranked_hits: tuple[RankedHit, ...]


@dataclass(frozen=True)
class StrategyCutoffs:
    """전략 사다리의 채택 축과 후보 풀 크기.

    채택 판정은 **모든 전략이 같은 절대 축**(코사인 유사도)을 쓴다. RRF 점수는 순위 기반이라
    "이 조항이 애초에 관련 있는가"를 표현할 수 없고, 그 축으로 채택을 옮기면 근거 없는 문의에서
    기권이 구조적으로 불가능해져 전략 간 precision 이 비교 불가가 된다.
    """

    cosine_similarity: float = 0.30
    rrf_score: float = 0.0
    rerank_top_n: int = 5
    fusion_pool_size: int = 10

    def __post_init__(self) -> None:
        if not 0.0 <= self.cosine_similarity <= 1.0:
            raise ValueError("cosine_similarity는 0.0 이상 1.0 이하여야 한다")
        if self.rrf_score < 0.0:
            raise ValueError("rrf_score는 0.0 이상이어야 한다")
        if self.rerank_top_n <= 0:
            raise ValueError("rerank_top_n은 1 이상이어야 한다")
        if self.fusion_pool_size <= 0:
            raise ValueError("fusion_pool_size는 1 이상이어야 한다")


@dataclass(frozen=True)
class UnmeasuredStage:
    """실행되지 않은 사다리 단. 0 이 아니라 사유를 가진다."""

    stage: str
    reason: str


@dataclass(frozen=True)
class RerankStats:
    """리랭크 단의 관측 기록. 조용한 폴백을 막는다(docs/business-rules.md "검색 단계 실패")."""

    calls: int
    fallbacks: int
    fallback_reasons: tuple[str, ...]
    cache_hits: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class StrategyRetrievedCase:
    """라벨을 보지 않고 검색까지 끝낸 전략별 케이스.

    `ranked_hits` 는 컷을 무시한 전체 순위다(순위 품질 축). `accept_candidates` 는 채택
    판정 대상 후보를 그 전략의 순서로 담는다(채택 축). 두 축은 서로 다른 질문이므로
    분리해서 보관하고, 코사인 컷은 채점 단계에서 적용한다 — 그래야 컷 스윕이 검색을
    다시 돌리지 않는다.
    """

    case_id: str
    ranked_hits: tuple[RankedHit, ...]
    accept_candidates: tuple[RankedHit, ...]


@dataclass(frozen=True)
class StrategyRetrieval:
    """한 전략의 라벨 없는 검색 결과."""

    strategy: StrategyDefinition
    accept_limit: int
    cases: tuple[StrategyRetrievedCase, ...]


@dataclass(frozen=True)
class StrategyLadderRetrieval:
    """사다리 전체의 라벨 없는 검색 결과와 리랭크 관측 기록."""

    retrievals: tuple[StrategyRetrieval, ...]
    rerank: RerankStats


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
    """한 누적 전략의 기본 컷 결과와 그 전략 자신의 컷 스윕."""

    strategy: StrategyDefinition
    accept_limit: int
    cutoff: float
    cases: tuple[StrategyCaseScore, ...]
    aggregate: AggregateMetrics
    sweep: tuple[CutoffSweepPoint, ...]
    best_cutoff: float | None


@dataclass(frozen=True)
class StrategyComparison:
    """동일 입력에 대한 네 단계 누적 전략 비교."""

    embedding_config: RetrievalEvalConfig
    cutoffs: StrategyCutoffs
    rrf_k: int
    ngram_size: int
    rerank_model: str
    rewrite_condition: RewriteCondition
    rewrite_source: str
    strategies: tuple[StrategyEvaluation, ...]
    rerank: RerankStats
    unmeasured_stages: tuple[UnmeasuredStage, ...] = ()


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


def _ranked_fused_hits(hits: Sequence[FusedHit], *, start_rank: int = 1) -> tuple[RankedHit, ...]:
    return tuple(
        RankedHit(
            rank=rank,
            evidence_id=hit.evidence_id,
            # 벡터 순위에 없으면 None 을 유지한다. 0.0 으로 바꾸면 "유사도 0"과 구분되지 않는다.
            similarity=hit.vector_similarity,
            bm25_score=hit.bm25_score,
            rrf_score=hit.rrf_score,
            vector_rank=hit.vector_rank,
            bm25_rank=hit.bm25_rank,
        )
        for rank, hit in enumerate(hits, start=start_rank)
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
) -> StrategyLadderRetrieval:
    """정답 라벨 없이 누적 전략 검색을 돌린다. 코사인 컷은 채점 단계에서 적용한다."""
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
        if cutoffs.fusion_pool_size <= cutoffs.rerank_top_n:
            raise ValueError(
                "리랭크 후보 풀은 최종 채택 상한보다 커야 한다 "
                f"(fusion_pool_size={cutoffs.fusion_pool_size} <= "
                f"rerank_top_n={cutoffs.rerank_top_n}) — 같거나 작으면 리랭크가 채택 집합을 "
                "바꿀 수 없어 하이브리드와 항등이 되고, 리랭크 효과를 측정할 수 없다"
            )
    if uses_hybrid and cutoffs.rrf_score > 0.0:
        # 두 순위가 코퍼스 전량을 덮으므로 모든 조항이 양쪽에 있고 최소 RRF는 2/(k+N)이다.
        # 그보다 작은 컷은 아무것도 걸러내지 못하면서 걸러낸 척한다 — 조용히 두지 않는다.
        reachable_floor = 2.0 / (rrf_k + len(policy_texts))
        if cutoffs.rrf_score <= reachable_floor:
            raise ValueError(
                f"RRF 컷 {cutoffs.rrf_score:.6f}은 조항 {len(policy_texts)}개·rrf_k={rrf_k}에서 "
                f"아무것도 걸러내지 못한다(최소 도달 가능 {reachable_floor:.6f}). "
                "0으로 두거나 그보다 큰 값을 준다"
            )

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
    rerank_calls = 0
    rerank_fallbacks = 0
    rerank_cache_hits = 0
    rerank_input_tokens = 0
    rerank_output_tokens = 0
    fallback_reasons: set[str] = set()

    for query in queries:
        rewritten_text = injected_rewrites[query.case_id] if uses_rewrite else None
        original_vector = tuple(
            VectorHit(hit.rank, hit.evidence_id, hit.similarity)
            for hit in original_by_id[query.case_id].ranked_hits
            if hit.similarity is not None
        )
        rewritten_vector = (
            tuple(
                VectorHit(hit.rank, hit.evidence_id, hit.similarity)
                for hit in rewritten_by_id[query.case_id].ranked_hits
                if hit.similarity is not None
            )
            if uses_rewrite
            else ()
        )
        merged_vector = (
            merge_rewritten_rankings(original=original_vector, rewritten=rewritten_vector)
            if uses_rewrite
            else original_vector
        )
        if uses_hybrid:
            original_bm25 = bm25_rank(
                query=query.text, documents=policy_texts, ngram_size=ngram_size
            )
            # 재작성 단이 켜져 있으면 어휘 다리도 재작성문을 받는다. 벡터 다리만 재작성을
            # 쓰면 사다리가 누적이 아니게 되고 재작성 기여가 상위 단에서 과소평가된다.
            bm25_hits = (
                merge_rewritten_bm25(
                    original=original_bm25,
                    rewritten=bm25_rank(
                        query=cast(str, rewritten_text),
                        documents=policy_texts,
                        ngram_size=ngram_size,
                    ),
                )
                if uses_rewrite
                else original_bm25
            )
            fused = reciprocal_rank_fusion(vector=merged_vector, bm25=bm25_hits, rrf_k=rrf_k)
            pool = tuple(
                hit
                for hit in fused[: cutoffs.fusion_pool_size]
                if hit.rrf_score >= cutoffs.rrf_score
            )
        else:
            fused = ()
            pool = ()

        if uses_rerank and reranker is not None:
            outcome = llm_rerank(
                query=query.text,
                rewritten_query=rewritten_text,
                candidates=pool,
                policy_texts=policy_by_id,
                client=reranker,
                model=rerank_model,
                cache_dir=rerank_cache_dir,
            )
            rerank_calls += 1
            rerank_input_tokens += outcome.input_tokens
            rerank_output_tokens += outcome.output_tokens
            if outcome.served_from_cache:
                rerank_cache_hits += 1
            if outcome.fell_back:
                rerank_fallbacks += 1
                if outcome.fallback_reason is not None:
                    fallback_reasons.add(outcome.fallback_reason)
            reranked = outcome.hits
        else:
            reranked = ()

        for strategy in strategy_tuple:
            if RetrievalStage.RERANK in strategy.stages:
                ranked = _rerank_full_ranking(reranked=reranked, fused=fused)
                candidates = _ranked_fused_hits(reranked)
            elif RetrievalStage.HYBRID in strategy.stages:
                ranked = _ranked_fused_hits(fused)
                candidates = _ranked_fused_hits(pool)
            elif RetrievalStage.REWRITE in strategy.stages:
                ranked = _ranked_vector_hits(merged_vector)
                candidates = ranked
            else:
                ranked = _ranked_vector_hits(original_vector)
                candidates = ranked
            cases_by_strategy[strategy.name].append(
                StrategyRetrievedCase(
                    case_id=query.case_id,
                    ranked_hits=ranked,
                    accept_candidates=candidates,
                )
            )

    retrievals = tuple(
        StrategyRetrieval(
            strategy=strategy,
            accept_limit=(
                cutoffs.rerank_top_n
                if RetrievalStage.RERANK in strategy.stages
                else embedding_config.top_k
            ),
            cases=tuple(cases_by_strategy[strategy.name]),
        )
        for strategy in strategy_tuple
    )
    return StrategyLadderRetrieval(
        retrievals=retrievals,
        rerank=RerankStats(
            calls=rerank_calls,
            fallbacks=rerank_fallbacks,
            fallback_reasons=tuple(sorted(fallback_reasons)),
            cache_hits=rerank_cache_hits,
            input_tokens=rerank_input_tokens,
            output_tokens=rerank_output_tokens,
        ),
    )


def _rerank_full_ranking(
    *, reranked: Sequence[FusedHit], fused: Sequence[FusedHit]
) -> tuple[RankedHit, ...]:
    """리랭크된 후보 뒤에 풀 밖 조항을 융합 순서로 이어 전체 순위를 복원한다.

    순위 품질(recall@k)은 임계값·풀 크기와 다른 축이다. 리랭크 행만 후보 수로 잘린 순위를
    쓰면 recall@5 가 다른 행과 같은 정의가 아니게 된다.
    """
    pooled = {hit.evidence_id for hit in reranked}
    tail = tuple(hit for hit in fused if hit.evidence_id not in pooled)
    return _ranked_fused_hits((*reranked, *tail))


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
        accepted = tuple(
            hit
            for hit in result.ranked_hits[:top_k]
            if hit.similarity is not None and hit.similarity >= cutoff
        )

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


def _accepted_at(
    case: StrategyRetrievedCase, *, limit: int, cutoff: float
) -> tuple[RankedHit, ...]:
    """전략 후보에 절대 관련성 게이트와 채택 상한을 적용한다.

    코사인이 측정되지 않은 후보(어휘 다리에만 있던 조항)는 게이트를 통과하지 못한다.
    """
    return tuple(
        hit
        for hit in case.accept_candidates[:limit]
        if hit.similarity is not None and hit.similarity >= cutoff
    )


def _score_cases(
    strategy_result: StrategyRetrieval,
    label_by_id: Mapping[str, RetrievalLabel],
    *,
    cutoff: float,
) -> tuple[StrategyCaseScore, ...]:
    cases: list[StrategyCaseScore] = []
    for result in strategy_result.cases:
        relevant = label_by_id[result.case_id].relevant_evidence_ids
        accepted = _accepted_at(result, limit=strategy_result.accept_limit, cutoff=cutoff)
        accepted_ids = {hit.evidence_id for hit in accepted}
        if accepted:
            precision = len(accepted_ids & relevant) / len(accepted)
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
                accepted_hits=accepted,
                recall_at_1=_recall_at(result.ranked_hits, relevant, 1),
                recall_at_3=_recall_at(result.ranked_hits, relevant, 3),
                recall_at_5=_recall_at(result.ranked_hits, relevant, 5),
                accepted_precision=precision,
                accepted_recall=accepted_recall,
            )
        )
    return tuple(cases)


def _aggregate_cases(cases: Sequence[StrategyCaseScore]) -> AggregateMetrics:
    return AggregateMetrics(
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


def score_strategy_ladder(
    retrieved: Sequence[StrategyRetrieval],
    labels: Sequence[RetrievalLabel],
    *,
    cutoff: float,
    cutoff_sweep: Sequence[float] = DEFAULT_CUTOFF_SWEEP,
) -> tuple[StrategyEvaluation, ...]:
    """라벨 없는 전략 산출을 검색 완료 뒤 독립 채점하고, 전략마다 컷을 훑는다.

    스윕은 검색을 다시 돌리지 않는다 — 임베딩·리랭크는 이미 끝났고 컷만 바뀐다.
    전략마다 컷의 실효 강도가 다르므로 하나의 고정 컷으로만 비교하면 불공정하다.
    """
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("cutoff는 0.0 이상 1.0 이하여야 한다")
    sweep_values = tuple(cutoff_sweep)
    if not sweep_values:
        raise ValueError("cutoff_sweep은 비어 있을 수 없다")
    if any(not 0.0 <= value <= 1.0 for value in sweep_values):
        raise ValueError("cutoff_sweep의 모든 값은 0.0 이상 1.0 이하여야 한다")

    label_by_id = {label.id: label for label in labels}
    evaluations: list[StrategyEvaluation] = []
    for strategy_result in retrieved:
        retrieved_ids = {case.case_id for case in strategy_result.cases}
        label_ids = set(label_by_id)
        if retrieved_ids != label_ids:
            missing = ", ".join(sorted(label_ids - retrieved_ids)) or "없음"
            extra = ", ".join(sorted(retrieved_ids - label_ids)) or "없음"
            raise ValueError(f"검색 결과와 라벨 ID가 다르다(누락={missing}, 추가={extra})")

        cases = _score_cases(strategy_result, label_by_id, cutoff=cutoff)
        sweep = tuple(
            _sweep_point(strategy_result, label_by_id, cutoff=value) for value in sweep_values
        )
        evaluations.append(
            StrategyEvaluation(
                strategy=strategy_result.strategy,
                accept_limit=strategy_result.accept_limit,
                cutoff=cutoff,
                cases=cases,
                aggregate=_aggregate_cases(cases),
                sweep=sweep,
                best_cutoff=_best_sweep_cutoff(sweep),
            )
        )
    return tuple(evaluations)


def _sweep_point(
    strategy_result: StrategyRetrieval,
    label_by_id: Mapping[str, RetrievalLabel],
    *,
    cutoff: float,
) -> CutoffSweepPoint:
    aggregate = _aggregate_cases(_score_cases(strategy_result, label_by_id, cutoff=cutoff))
    return CutoffSweepPoint(
        cutoff=cutoff,
        accepted_precision=aggregate.accepted_precision,
        accepted_recall=aggregate.accepted_recall,
        macro_f1=_macro_f1(aggregate.accepted_precision, aggregate.accepted_recall),
        precision_case_count=aggregate.precision_case_count,
        recall_case_count=aggregate.recall_case_count,
    )


def _best_sweep_cutoff(sweep: Sequence[CutoffSweepPoint]) -> float | None:
    eligible = [point for point in sweep if point.macro_f1 is not None]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda point: (
            cast(float, point.macro_f1),
            cast(float, point.accepted_precision),
            cast(float, point.accepted_recall),
            point.cutoff,
        ),
    ).cutoff


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
    rewrite_condition: RewriteCondition | None = None,
    rewrite_source: str | None = None,
    cache_dir: Path,
    rerank_cache_dir: Path,
    rrf_k: int = 60,
    ngram_size: int = 2,
    strategies: Sequence[StrategyDefinition] = default_strategy_ladder(),
    unmeasured_stages: Sequence[UnmeasuredStage] = (),
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
    uses_rewrite = any(
        RetrievalStage.REWRITE in result.strategy.stages for result in retrieved.retrievals
    )
    if uses_rewrite:
        resolved_condition = rewrite_condition or RewriteCondition.CALLER_INJECTED
        resolved_source = rewrite_source or "caller_injected"
    else:
        resolved_condition = RewriteCondition.NOT_USED
        resolved_source = "not_used"
    return StrategyComparison(
        embedding_config=embedding_config,
        cutoffs=cutoffs,
        rrf_k=rrf_k,
        ngram_size=ngram_size,
        rerank_model=rerank_model,
        rewrite_condition=resolved_condition,
        rewrite_source=resolved_source,
        strategies=score_strategy_ladder(
            retrieved.retrievals,
            labels,
            cutoff=cutoffs.cosine_similarity,
            cutoff_sweep=embedding_config.cutoff_sweep,
        ),
        rerank=retrieved.rerank,
        unmeasured_stages=tuple(unmeasured_stages),
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
    """집계·케이스 지표 표기. None 은 **평균 분모에서 제외**됐다는 뜻이다."""
    return "제외" if value is None else f"{value:.4f}"


def _score(value: float | None) -> str:
    """검색기 원점수 표기. None 은 그 검색기 순위에 없어 **측정되지 않았다**는 뜻이다."""
    return "미측정" if value is None else f"{value:.6f}"


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
    # 조항 수는 실제 주입된 코퍼스에서 읽는다 — 하드코딩하면 축소 코퍼스 실행에서 거짓이 된다.
    corpus_size = max((len(case.ranked_hits) for case in evaluation.score.cases), default=0)
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
            f"`전체 순위`는 임계값과 top_k를 적용하기 전 {corpus_size}개 조항의 순위다. "
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
                f"{hit.rank}. {hit.evidence_id} ({_score(hit.similarity)})"
                for hit in case.accepted_hits
            )
            or "없음"
        )
        ranking = "<br>".join(
            f"{hit.rank}. {hit.evidence_id} ({_score(hit.similarity)})" for hit in case.ranked_hits
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
            "accept_axis": "cosine_similarity",
            "cosine_similarity_cutoff": comparison.cutoffs.cosine_similarity,
            "cutoff_sweep": list(config.cutoff_sweep),
            "rrf_score_cutoff": comparison.cutoffs.rrf_score,
            "rerank_top_n": comparison.cutoffs.rerank_top_n,
            "fusion_pool_size": comparison.cutoffs.fusion_pool_size,
            "rrf_k": comparison.rrf_k,
            "ngram_size": comparison.ngram_size,
            "rerank_model": comparison.rerank_model,
            "stub": config.is_stub,
        },
        "run_conditions": {
            "database_used": False,
            "labels_used_for_retrieval": False,
            "rewrite_condition": comparison.rewrite_condition.value,
            "rewrite_source": comparison.rewrite_source,
            "rewrite_applied_to": (
                ["vector", "bm25", "llm_rerank"]
                if comparison.rewrite_condition is not RewriteCondition.NOT_USED
                else []
            ),
            "warning": warning,
        },
        "rerank_observability": {
            "calls": comparison.rerank.calls,
            "fallbacks": comparison.rerank.fallbacks,
            "fallback_reasons": list(comparison.rerank.fallback_reasons),
            "cache_hits": comparison.rerank.cache_hits,
            "input_tokens": comparison.rerank.input_tokens,
            "output_tokens": comparison.rerank.output_tokens,
        },
        "unmeasured_stages": [
            {"stage": item.stage, "reason": item.reason} for item in comparison.unmeasured_stages
        ],
        "best_cutoff_selection": "macro F1 최대, 동률이면 precision, recall, 높은 cutoff 순",
        "strategies": [
            {
                "name": result.strategy.name,
                "stages": [stage.value for stage in result.strategy.stages],
                "accept_limit": result.accept_limit,
                "cutoff": {
                    "kind": "cosine_similarity",
                    "value": result.cutoff,
                },
                "aggregate": _metrics_json(result.aggregate),
                "sweep": [
                    {
                        "cutoff": point.cutoff,
                        "accepted_precision": point.accepted_precision,
                        "accepted_recall": point.accepted_recall,
                        "macro_f1": point.macro_f1,
                        "precision_case_count": point.precision_case_count,
                        "recall_case_count": point.recall_case_count,
                    }
                    for point in result.sweep
                ],
                "best_cutoff": result.best_cutoff,
                "cases": [
                    {
                        "id": case.case_id,
                        "strategy": result.strategy.name,
                        "cutoff": {
                            "kind": "cosine_similarity",
                            "value": result.cutoff,
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
    uses_rewrite = comparison.rewrite_condition is not RewriteCondition.NOT_USED
    if comparison.rewrite_condition is RewriteCondition.BLIND:
        rewrite_line = (
            "- 질의 재작성: blind/deployable — 정책·라벨을 본 적 없는 생성 모델이 "
            "문의 원문만 보고 만든 고정 입력 "
            f"(`{comparison.rewrite_source}`; 원문 검색을 항상 함께 유지)"
        )
    elif comparison.rewrite_condition is RewriteCondition.ORACLE:
        rewrite_line = (
            "- 질의 재작성: oracle/curated upper bound "
            f"(`{comparison.rewrite_source}`; **배포 가능 개선폭이 아님**)"
        )
    elif uses_rewrite:
        rewrite_line = (
            f"- 질의 재작성: 호출자 주입 (`{comparison.rewrite_source}`; "
            "원문 검색을 항상 함께 유지)"
        )
    else:
        rewrite_line = "- 질의 재작성: 미사용"
    corpus_size = max(
        (len(case.ranked_hits) for result in comparison.strategies for case in result.cases),
        default=0,
    )
    sweep_values = config.cutoff_sweep
    lines = [
        "# 정책 검색 전략 비교",
        "",
        "## 실행 조건",
        "",
        f"- 코퍼스: 주입된 정책 파일 조항 {corpus_size}개 (DB 미사용)",
        rewrite_line,
    ]
    if uses_rewrite:
        lines.append("- 재작성 적용 범위: 벡터 다리·BM25 어휘 다리·리랭크 입력 (누적 사다리 계약)")
    lines.extend(
        [
            f"- 임베딩: `{config.model}` ({config.dimensions}차원)",
            f"- top_k: {config.top_k}",
            "- **채택 축: 코사인 유사도 (모든 전략 동일)** — 순위 기반 점수는 절대 관련성을 "
            "표현할 수 없으므로 채택·기권 판정은 전 전략이 같은 축을 쓴다",
            f"- 채택 코사인 컷: {comparison.cutoffs.cosine_similarity:.6f}",
            f"- 컷 스윕 범위: {min(sweep_values):.2f} ~ {max(sweep_values):.2f} "
            f"({len(sweep_values)}점)",
            f"- 하이브리드 후보 풀: 융합 상위 {comparison.cutoffs.fusion_pool_size}건"
            f" (RRF 컷 {comparison.cutoffs.rrf_score:.6f})",
            f"- 리랭크 최종 채택 상한 n: {comparison.cutoffs.rerank_top_n}",
            f"- RRF k: {comparison.rrf_k}",
            f"- BM25 문자 n-gram: {comparison.ngram_size}",
            f"- 리랭크 모델: `{comparison.rerank_model}`",
            "- 라벨 사용 경계: 검색·컷·리랭크 완료 뒤 채점 단계에서만 사용",
        ]
    )
    if config.is_stub:
        lines.extend(
            [
                "- **경고: 결정론 어휘 임베딩·리랭크 대역 수치는 실제 검색 품질이 아니다. "
                "외부 호출 없는 배관 검증용이다.**",
                "- 대역 실행 조건: 과금 0회, DB 0회",
            ]
        )
    if comparison.unmeasured_stages:
        lines.extend(["", "## 미측정 단", "", "0 이 아니라 사유를 남긴다.", ""])
        lines.extend(
            f"- `{item.stage}`: 미측정 — {item.reason}" for item in comparison.unmeasured_stages
        )
    if comparison.rerank.calls or comparison.rerank.fallbacks:
        rerank = comparison.rerank
        lines.extend(
            [
                "",
                "## 리랭크 관측",
                "",
                f"- 호출: {rerank.calls}건 (캐시 재사용 {rerank.cache_hits}건)",
                f"- **폴백: {rerank.fallbacks}건** — 폴백은 직전 순위를 유지하지만 "
                "지표는 그 사실을 숨기지 않는다",
                f"- 토큰: 입력 {rerank.input_tokens} · 출력 {rerank.output_tokens}",
            ]
        )
        if rerank.fallback_reasons:
            lines.append("- 폴백 사유:")
            lines.extend(f"  - {reason}" for reason in rerank.fallback_reasons)
    lines.extend(
        [
            "",
            "## 전략별 집계",
            "",
            "`precision n`·`recall n`은 그 지표의 macro 평균 분모다. 채택 0건 + 정답 있음은 "
            "precision 분모에서 빠지므로, 분모를 함께 읽지 않으면 전략 간 precision 을 "
            "비교할 수 없다.",
            "",
            "| strategy | stages | accept n | cutoff | r@1 | r@3 | r@5 | "
            "precision | precision n | recall | recall n | best cutoff |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in comparison.strategies:
        aggregate = result.aggregate
        stages = " + ".join(stage.value for stage in result.strategy.stages)
        lines.append(
            f"| {result.strategy.name} | {stages} | {result.accept_limit} | "
            f"{result.cutoff:.2f} | {_number(aggregate.recall_at_1)} | "
            f"{_number(aggregate.recall_at_3)} | {_number(aggregate.recall_at_5)} | "
            f"{_number(aggregate.accepted_precision)} | {aggregate.precision_case_count} | "
            f"{_number(aggregate.accepted_recall)} | {aggregate.recall_case_count} | "
            f"{_number(result.best_cutoff)} |"
        )
    lines.extend(
        [
            "",
            "## 전략별 컷 스윕",
            "",
            "전략마다 컷의 실효 강도가 다르므로 고정 컷 한 점 비교는 불공정하다. 같은 검색 "
            "산출에 컷만 바꿔 재채점한다(추가 임베딩·리랭크 호출 없음). 최적점은 macro F1 "
            "최대, 동률이면 precision, recall, 높은 cutoff 순이다.",
            "",
        ]
    )
    for result in comparison.strategies:
        lines.extend(
            [
                f"### {result.strategy.name}",
                "",
                f"선택된 최적 컷: {_number(result.best_cutoff)}",
                "",
                "| cutoff | precision | recall | macro F1 | precision n | recall n |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for point in result.sweep:
            lines.append(
                f"| {point.cutoff:.2f} | {_number(point.accepted_precision)} | "
                f"{_number(point.accepted_recall)} | {_number(point.macro_f1)} | "
                f"{point.precision_case_count} | {point.recall_case_count} |"
            )
        lines.append("")
    lines.extend(["## 케이스별 결과", ""])
    for result in comparison.strategies:
        lines.extend(
            [
                f"### {result.strategy.name}",
                "",
                f"채택: 코사인 ≥ {result.cutoff:.6f} 인 상위 {result.accept_limit}건. "
                "`순위`는 컷을 무시한 전체 순위다.",
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
                f"(cos={_score(hit.similarity)}, bm25={_score(hit.bm25_score)}, "
                f"rrf={_score(hit.rrf_score)})"
                for hit in case.ranked_hits
            )
            lines.append(f"| {case.case_id} | {accepted} | {ranking} |")
        lines.append("")
    return "\n".join(lines) + "\n"


_CONDITION_SLUG: Final = {
    RewriteCondition.BLIND: "blind",
    RewriteCondition.ORACLE: "oracle",
    RewriteCondition.CALLER_INJECTED: "injected",
    RewriteCondition.NOT_USED: "norewrite",
}


def _next_strategy_report_paths(output_dir: Path, comparison: StrategyComparison) -> ReportPaths:
    config = comparison.embedding_config
    mode = "stub" if config.is_stub else "live"
    model = re.sub(r"[^a-z0-9]+", "-", config.model.lower()).strip("-") or "model"
    # 재작성 조건·컷·top_k 를 이름에 넣는다. blind 산출물과 oracle 산출물이 접미사 숫자로만
    # 갈리면 결정 기록이 인용할 파일을 이름으로 식별할 수 없다.
    condition = _CONDITION_SLUG[comparison.rewrite_condition]
    cutoff = f"{round(comparison.cutoffs.cosine_similarity * 100):03d}"
    stem = (
        f"retrieval-strategies-{mode}-{model}-d{config.dimensions}"
        f"-{condition}-k{config.top_k}-c{cutoff}"
    )
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


def _build_embedding_candidate_client(
    candidate: EmbeddingCandidate, *, api_key: str
) -> EmbeddingClient:
    if candidate.provider is EmbeddingProvider.OPENAI:
        if not api_key:
            raise RetrievalConfigurationError(f"{candidate.key} 미실행 — OPENAI_API_KEY가 없다")
        return OpenAIEmbeddingClient(
            api_key=api_key,
            model=candidate.model,
            dimensions=candidate.dimensions,
        )
    return BgeM3EmbeddingClient()


def run_embedding_model_axis(
    *,
    api_key: str,
    evaluate: Callable[[EmbeddingCandidate, EmbeddingClient], ReportPaths],
    candidates: Sequence[EmbeddingCandidate] = DEFAULT_EMBEDDING_CANDIDATES,
    client_factory: Callable[[EmbeddingCandidate, str], EmbeddingClient] | None = None,
) -> EmbeddingAxisResult:
    """후속 격자 실행이 쓰는 모델 축.

    BGE-M3 선택 의존성이 없으면 그 행만 사유와 함께 미실행으로 남기고 나머지 행은 계속
    돈다. API 키 부재나 실제 비교 실패는 전체 실행의 구성/실행 오류이므로 숨기지 않는다.
    """
    factory = client_factory or (
        lambda candidate, key: _build_embedding_candidate_client(candidate, api_key=key)
    )
    rows: list[EmbeddingAxisRow] = []
    for candidate in candidates:
        try:
            client = factory(candidate, api_key)
        except OptionalEmbeddingDependencyError as exc:
            rows.append(
                EmbeddingAxisRow(
                    candidate=candidate,
                    measured=False,
                    reports=None,
                    reason=str(exc),
                )
            )
            continue
        if client.dimensions != candidate.dimensions:
            raise RetrievalConfigurationError(
                f"{candidate.key} 클라이언트 차원 불일치: "
                f"{client.dimensions} != {candidate.dimensions}"
            )
        rows.append(
            EmbeddingAxisRow(
                candidate=candidate,
                measured=True,
                reports=evaluate(candidate, client),
                reason=None,
            )
        )
    return EmbeddingAxisResult(rows=tuple(rows))


_CANONICAL_REWRITE_CONDITIONS: Final = {
    DEFAULT_REWRITTEN_QUERIES_PATH.resolve(): RewriteCondition.BLIND,
    DEFAULT_ORACLE_REWRITTEN_QUERIES_PATH.resolve(): RewriteCondition.ORACLE,
}


def _resolved_rewrite_condition(
    *,
    declared: RewriteCondition | None,
    rewritten_queries_path: Path,
    caller_injected: bool,
    uses_rewrite: bool,
) -> RewriteCondition:
    """재작성 조건을 실제 입력과 결속한다.

    조건이 산문이 아니라 코드로 강제되는 지점이다. 저장소 픽스처 경로는 조건을 스스로
    결정하고, 선언과 다르면 거부한다. 그 밖의 경로는 조건을 반드시 명시해야 한다 —
    검증할 수 없는 출처를 리포트가 사실로 인쇄하지 않게 한다.
    """
    if not uses_rewrite:
        return RewriteCondition.NOT_USED
    if declared is not None and not isinstance(declared, RewriteCondition):
        raise RetrievalConfigurationError("rewrite_condition은 정해진 조건 enum이어야 한다")
    if caller_injected:
        if declared is not None and declared is not RewriteCondition.CALLER_INJECTED:
            raise RetrievalConfigurationError(
                "호출자가 직접 주입한 재작성문의 조건은 caller_injected 뿐이다"
            )
        return RewriteCondition.CALLER_INJECTED
    if declared in {RewriteCondition.CALLER_INJECTED, RewriteCondition.NOT_USED}:
        raise RetrievalConfigurationError(
            "파일 기반 비교의 rewrite_condition은 blind 또는 oracle_upper_bound여야 한다"
        )
    canonical = _CANONICAL_REWRITE_CONDITIONS.get(rewritten_queries_path.resolve())
    if canonical is not None:
        if declared is not None and declared is not canonical:
            raise RetrievalConfigurationError(
                f"재작성 픽스처 경로와 선언된 조건이 다르다: "
                f"{rewritten_queries_path.name}은 {canonical.value} 입력인데 "
                f"{declared.value}로 선언됐다"
            )
        return canonical
    if declared is None:
        raise RetrievalConfigurationError(
            "저장소 픽스처가 아닌 재작성 입력에는 rewrite_condition을 명시해야 한다 "
            f"({rewritten_queries_path})"
        )
    return declared


def run_retrieval_comparison(
    *,
    live: bool,
    embedding_model: str | None = None,
    embedding_client: EmbeddingClient | None = None,
    dimensions: int | None = None,
    top_k: int = 5,
    cutoff: float | None = None,
    sweep_start: float = 0.10,
    sweep_end: float = 0.70,
    sweep_step: float = 0.05,
    policy_dir: Path = DEFAULT_POLICY_DIR,
    golden_set_path: Path = DEFAULT_GOLDEN_SET_PATH,
    labels_path: Path = DEFAULT_RETRIEVAL_LABELS_PATH,
    rewritten_queries_path: Path = DEFAULT_REWRITTEN_QUERIES_PATH,
    output_dir: Path = DEFAULT_RETRIEVAL_REPORT_DIR,
    cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR,
    rewritten_queries: Mapping[str, str] | None = None,
    rewrite_condition: RewriteCondition | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    rrf_cutoff: float = DEFAULT_RRF_CUTOFF,
    fusion_pool_size: int = DEFAULT_FUSION_POOL_SIZE,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
    rerank_top_n: int = 5,
    rerank_model: str | None = None,
    paid_rerank: bool | None = None,
    strategies: Sequence[StrategyDefinition] = default_strategy_ladder(),
) -> ReportPaths:
    """CLI가 호출하는 전체 오프라인 사다리. 실제 모드만 외부 호출을 과금한다.

    `rewrite_condition` 에 기본값이 없는 것은 의도적이다. 기본값이 있으면 oracle 픽스처를
    넘기고 조건을 잊었을 때 리포트가 그것을 blind/deployable 이라고 인쇄한다.
    `paid_rerank` 는 `live` 와 별개 축이다 — 로컬 임베딩 실행이 OpenAI 리랭크를 조용히
    과금하지 않게 한다.
    """
    strategy_tuple = tuple(strategies)
    if not strategy_tuple:
        raise RetrievalConfigurationError("검색 전략은 하나 이상이어야 한다")
    resolved_paid_rerank = live if paid_rerank is None else paid_rerank
    if resolved_paid_rerank and not live:
        raise RetrievalConfigurationError("대역 모드에서는 유료 리랭크를 켤 수 없다")

    unmeasured: list[UnmeasuredStage] = []
    if live and not resolved_paid_rerank:
        # 실제 임베딩 실행인데 리랭크 과금이 승인되지 않았다. 대역 리랭커로 채우면 실제
        # 수치와 대역 수치가 한 리포트에 섞이므로, 그 단만 사유와 함께 미측정으로 남긴다.
        without_rerank = tuple(
            strategy for strategy in strategy_tuple if RetrievalStage.RERANK not in strategy.stages
        )
        if len(without_rerank) != len(strategy_tuple):
            if not without_rerank:
                raise RetrievalConfigurationError(
                    "리랭크 단만 요청했는데 리랭크 과금이 승인되지 않았다"
                )
            unmeasured.append(
                UnmeasuredStage(
                    stage=RetrievalStage.RERANK.value,
                    reason=("OpenAI 리랭크 과금 미승인 — 실행하려면 유료 리랭크를 명시적으로 켠다"),
                )
            )
            strategy_tuple = without_rerank

    rewrite_condition = _resolved_rewrite_condition(
        declared=rewrite_condition,
        rewritten_queries_path=rewritten_queries_path,
        caller_injected=rewritten_queries is not None,
        uses_rewrite=any(RetrievalStage.REWRITE in strategy.stages for strategy in strategy_tuple),
    )
    cases = load_golden_set(golden_set_path)
    queries = tuple(RetrievalQuery(case_id=case.id, text=case.content) for case in cases)
    uses_rewrite = any(RetrievalStage.REWRITE in strategy.stages for strategy in strategy_tuple)
    if not uses_rewrite and rewrite_condition is RewriteCondition.ORACLE:
        raise RetrievalConfigurationError("oracle 재작성과 벡터 단독 전략을 함께 요청할 수 없다")
    rewrite_input = (
        load_rewritten_queries(
            rewritten_queries_path,
            golden_set_path=golden_set_path,
        )
        if uses_rewrite and rewritten_queries is None
        else rewritten_queries
    )
    injected_rewrites = _validate_rewritten_queries(
        queries=queries,
        rewritten_queries=rewrite_input,
        strategies=strategy_tuple,
    )
    if not uses_rewrite:
        resolved_rewrite_source = "not_used"
    elif rewritten_queries is not None:
        resolved_rewrite_source = "caller_injected"
    else:
        try:
            resolved_rewrite_source = rewritten_queries_path.relative_to(_ROOT).as_posix()
        except ValueError:
            resolved_rewrite_source = str(rewritten_queries_path)

    settings = get_settings()
    resolved_dimensions = (
        embedding_client.dimensions
        if embedding_client is not None and dimensions is None
        else settings.embedding_dimensions
        if dimensions is None
        else dimensions
    )
    sweep = _cutoff_sweep(start=sweep_start, end=sweep_end, step=sweep_step)
    resolved_rerank_model = rerank_model or settings.rerank_model
    uses_rerank = any(RetrievalStage.RERANK in strategy.stages for strategy in strategy_tuple)
    if live:
        # 로컬 임베딩을 주입했고 유료 리랭크도 승인하지 않았다면 OpenAI 키가 필요 없다.
        needs_openai = embedding_client is None or (uses_rerank and resolved_paid_rerank)
        if needs_openai and not settings.openai_api_key:
            raise RetrievalConfigurationError(
                "OPENAI_API_KEY가 없다 — 실제 임베딩·리랭크 비교는 과금되므로 키를 선검사한다"
            )
        model = embedding_model or settings.embedding_model
        resolved_cutoff = settings.vector_similarity_threshold if cutoff is None else cutoff
        embedder: EmbeddingClient = embedding_client or OpenAIEmbeddingClient(
            api_key=settings.openai_api_key, model=model, dimensions=resolved_dimensions
        )
        reranker: GenerationClient | None = (
            OpenAIGenerationClient(
                api_key=settings.openai_api_key,
                model=resolved_rerank_model,
            )
            if uses_rerank and resolved_paid_rerank
            else None
        )
    else:
        if embedding_client is not None or embedding_model is not None:
            raise RetrievalConfigurationError(
                "대역 모드에는 실제 embedding_client/embedding_model을 주입할 수 없다"
            )
        model = STUB_EMBEDDING_MODEL
        resolved_cutoff = STUB_DEFAULT_CUTOFF if cutoff is None else cutoff
        embedder = LexicalEmbeddingClient(dimensions=resolved_dimensions)
        reranker = _IdentityRerankClient() if uses_rerank else None

    config = RetrievalEvalConfig(
        model=model,
        dimensions=resolved_dimensions,
        top_k=top_k,
        cutoff=resolved_cutoff,
        cutoff_sweep=sweep,
        is_stub=not live,
    )
    # 정책·라벨·재작성문은 각각의 로더에서 읽고, 검색에는 라벨 없이 재작성 mapping만 주입한다.
    # 라벨 교차검증은 **실제로 검색할 코퍼스와 문의**를 기준으로 한다 — 기본 경로로 검사하면
    # 축소된 코퍼스로 돌릴 때 없는 조항이 정답으로 남아 recall 이 조용히 0 으로 계상된다.
    documents = load_policy_documents(policy_dir)
    labels = load_retrieval_labels(
        labels_path,
        golden_set_path=golden_set_path,
        policy_dir=policy_dir,
    )
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
            fusion_pool_size=fusion_pool_size,
        ),
        reranker=reranker,
        rerank_model=(
            resolved_rerank_model
            if live and resolved_paid_rerank
            else "stub-identity-reranker"
            if reranker is not None
            else "not_used"
        ),
        rewrite_condition=rewrite_condition,
        rewrite_source=resolved_rewrite_source,
        cache_dir=cache_dir,
        rerank_cache_dir=cache_dir / "rerank",
        rrf_k=rrf_k,
        ngram_size=ngram_size,
        strategies=strategy_tuple,
        unmeasured_stages=unmeasured,
    )
    return write_strategy_report(comparison, output_dir=output_dir)
