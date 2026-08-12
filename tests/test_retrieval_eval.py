"""DB 없는 벡터 검색 비교 하네스의 공개 동작."""

import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from reply_gate.evaluation import load_golden_set
from reply_gate.llm import (
    EmbeddingClient,
    EmbeddingResult,
    GenerationClient,
    JsonCompletion,
    OptionalEmbeddingDependencyError,
)
from reply_gate.policy_index import load_policy_documents
from reply_gate.retrieval_eval import (
    DEFAULT_EMBEDDING_CANDIDATES,
    DEFAULT_ORACLE_REWRITTEN_QUERIES_PATH,
    STUB_EMBEDDING_MODEL,
    EmbeddingCandidate,
    EmbeddingProvider,
    RankedHit,
    ReportPaths,
    RetrievalConfigurationError,
    RetrievalEvalConfig,
    RetrievalQuery,
    RetrievedCase,
    RewriteCondition,
    StrategyCutoffs,
    StrategyLadderRetrieval,
    evaluate_retrieval,
    evaluate_strategy_ladder,
    retrieve_cases,
    retrieve_strategy_ladder,
    run_embedding_model_axis,
    run_retrieval_comparison,
    score_retrieval,
    score_strategy_ladder,
    write_report,
    write_strategy_report,
)
from reply_gate.retrieval_labels import RetrievalLabel, load_retrieval_labels
from reply_gate.retrieval_strategies import RetrievalStage, StrategyDefinition
from reply_gate.testing import LexicalEmbeddingClient


class _FixedEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    @property
    def dimensions(self) -> int:
        return 2

    def embed(self, *, stage: str, texts: Sequence[str]) -> EmbeddingResult:
        self.calls.append((stage, tuple(texts)))
        return EmbeddingResult(vectors=[self._vectors[text] for text in texts], total_tokens=0)


class _IdentityReranker:
    def complete_json(self, **kwargs: Any) -> JsonCompletion:
        payload = json.loads(cast(str, kwargs["user"]))
        candidates = cast(list[dict[str, str]], payload["candidates"])
        return JsonCompletion(
            data={"evidence_ids": [candidate["evidence_id"] for candidate in candidates]},
            input_tokens=0,
            output_tokens=0,
        )


def test_score_retrieval_uses_documented_empty_label_and_zero_hit_denominators() -> None:
    retrieved = (
        RetrievedCase(
            case_id="A",
            ranked_hits=(
                RankedHit(rank=1, evidence_id="p1", similarity=0.9),
                RankedHit(rank=2, evidence_id="p3", similarity=0.8),
                RankedHit(rank=3, evidence_id="p2", similarity=0.7),
            ),
        ),
        RetrievedCase(
            case_id="B",
            ranked_hits=(RankedHit(rank=1, evidence_id="p4", similarity=0.6),),
        ),
        RetrievedCase(
            case_id="C",
            ranked_hits=(RankedHit(rank=1, evidence_id="p4", similarity=0.7),),
        ),
    )
    labels = (
        RetrievalLabel(id="A", relevant_evidence_ids=frozenset({"p1", "p2"}), note=""),
        RetrievalLabel(id="B", relevant_evidence_ids=frozenset(), note=""),
        RetrievalLabel(id="C", relevant_evidence_ids=frozenset({"p5"}), note=""),
    )

    scored = score_retrieval(retrieved, labels, top_k=2, cutoff=0.75)

    assert scored.aggregate.recall_at_1 == 0.25
    assert scored.aggregate.recall_at_3 == 0.5
    assert scored.aggregate.recall_at_5 == 0.5
    assert scored.aggregate.accepted_precision == 0.75
    assert scored.aggregate.accepted_recall == 0.25
    assert scored.aggregate.precision_case_count == 2
    assert scored.aggregate.recall_case_count == 2

    by_id = {case.case_id: case for case in scored.cases}
    assert by_id["B"].accepted_precision == 1.0
    assert by_id["B"].accepted_recall is None
    assert by_id["C"].accepted_precision is None
    assert by_id["C"].accepted_recall == 0.0


def test_retrieve_cases_ranks_all_policy_vectors_in_memory_without_labels(
    tmp_path: Path,
) -> None:
    embedder = _FixedEmbedder(
        {
            "정책 A": [1.0, 0.0],
            "정책 B": [0.0, 1.0],
            "문의": [0.8, 0.2],
        }
    )
    config = RetrievalEvalConfig(model="fixed", dimensions=2, top_k=1)

    retrieved = retrieve_cases(
        queries=(RetrievalQuery(case_id="G", text="문의"),),
        policy_texts=(("policy:a:1", "정책 A"), ("policy:b:1", "정책 B")),
        embedder=cast(EmbeddingClient, embedder),
        config=config,
        cache_dir=tmp_path,
    )

    assert [hit.evidence_id for hit in retrieved[0].ranked_hits] == [
        "policy:a:1",
        "policy:b:1",
    ]
    assert retrieved[0].ranked_hits[0].similarity == 0.9701425001453318
    assert embedder.calls == [
        ("retrieval_policy", ("정책 A", "정책 B")),
        ("retrieval_query", ("문의",)),
    ]


def test_embedding_cache_reuses_same_text_and_invalidates_changed_policy(
    tmp_path: Path,
) -> None:
    embedder = _FixedEmbedder(
        {
            "정책 A": [1.0, 0.0],
            "정책 B": [0.0, 1.0],
            "정책 B 변경": [0.2, 0.8],
            "문의": [0.8, 0.2],
            "문의 변경": [0.7, 0.3],
        }
    )
    config = RetrievalEvalConfig(model="fixed", dimensions=2)
    queries = (RetrievalQuery(case_id="G", text="문의"),)
    client = cast(EmbeddingClient, embedder)

    retrieve_cases(
        queries=queries,
        policy_texts=(("policy:a:1", "정책 A"), ("policy:b:1", "정책 B")),
        embedder=client,
        config=config,
        cache_dir=tmp_path,
    )
    retrieve_cases(
        queries=queries,
        policy_texts=(("policy:a:1", "정책 A"), ("policy:b:1", "정책 B")),
        embedder=client,
        config=config,
        cache_dir=tmp_path,
    )
    retrieve_cases(
        queries=queries,
        policy_texts=(("policy:a:1", "정책 A"), ("policy:b:1", "정책 B 변경")),
        embedder=client,
        config=config,
        cache_dir=tmp_path,
    )
    retrieve_cases(
        queries=(RetrievalQuery(case_id="G", text="문의 변경"),),
        policy_texts=(("policy:a:1", "정책 A"), ("policy:b:1", "정책 B 변경")),
        embedder=client,
        config=config,
        cache_dir=tmp_path,
    )
    retrieve_cases(
        queries=queries,
        policy_texts=(("policy:a:1", "정책 A"), ("policy:b:1", "정책 B 변경")),
        embedder=client,
        config=RetrievalEvalConfig(model="fixed-v2", dimensions=2),
        cache_dir=tmp_path,
    )

    assert embedder.calls == [
        ("retrieval_policy", ("정책 A", "정책 B")),
        ("retrieval_query", ("문의",)),
        ("retrieval_policy", ("정책 B 변경",)),
        ("retrieval_query", ("문의 변경",)),
        ("retrieval_policy", ("정책 A", "정책 B 변경")),
        ("retrieval_query", ("문의",)),
    ]


def test_stub_completes_30_cases_writes_both_reports_and_never_overwrites(
    tmp_path: Path,
) -> None:
    config = RetrievalEvalConfig(
        model="lexical-2gram-v1",
        dimensions=64,
        top_k=5,
        cutoff=0.10,
        is_stub=True,
    )
    evaluation = evaluate_retrieval(
        documents=load_policy_documents(),
        cases=load_golden_set(),
        labels=load_retrieval_labels(),
        embedder=LexicalEmbeddingClient(dimensions=64),
        config=config,
        cache_dir=tmp_path / "cache",
    )

    assert len(evaluation.score.cases) == 30
    assert all(len(case.ranked_hits) == 26 for case in evaluation.score.cases)
    assert [point.cutoff for point in evaluation.sweep] == [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]
    assert evaluation.best_cutoff in config.cutoff_sweep

    first = write_report(evaluation, output_dir=tmp_path / "reports")
    markdown_before = first.markdown.read_text(encoding="utf-8")
    json_before = first.json.read_text(encoding="utf-8")
    second = write_report(evaluation, output_dir=tmp_path / "reports")

    assert first.markdown.exists() and first.json.exists()
    assert second.markdown.exists() and second.json.exists()
    assert second.markdown.stem == f"{first.markdown.stem}-2"
    assert second.json.stem == f"{first.json.stem}-2"
    assert first.markdown.read_text(encoding="utf-8") == markdown_before
    assert first.json.read_text(encoding="utf-8") == json_before
    assert "실제 검색 품질이 아니다" in markdown_before
    assert "결정론 어휘 임베딩 대역" in markdown_before
    assert "G17" in markdown_before
    assert "policy:support:4-1" in markdown_before
    for case_id in ("G21", "G22", "G23", "G24"):
        assert case_id in markdown_before
    assert '"database_used": false' in json_before
    assert '"labels_used_for_retrieval": false' in json_before


def test_전략_리포트는_전략마다_컷_스윕과_최적점을_싣는다(tmp_path: Path) -> None:
    """컷 스윕이 산출물에 없으면 고정 컷 한 점 비교가 유일한 근거가 된다."""
    paths = run_retrieval_comparison(
        live=False,
        dimensions=32,
        top_k=5,
        cutoff=0.10,
        sweep_start=0.10,
        sweep_end=0.30,
        sweep_step=0.05,
        output_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
    )

    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    assert payload["configuration"]["cutoff_sweep"] == [0.10, 0.15, 0.20, 0.25, 0.30]
    assert payload["configuration"]["accept_axis"] == "cosine_similarity"
    for strategy in payload["strategies"]:
        assert [point["cutoff"] for point in strategy["sweep"]] == [0.10, 0.15, 0.20, 0.25, 0.30]
        assert strategy["best_cutoff"] in {0.10, 0.15, 0.20, 0.25, 0.30, None}
        assert {"macro_f1", "precision_case_count"} <= set(strategy["sweep"][0])

    markdown = paths.markdown.read_text(encoding="utf-8")
    assert "## 전략별 컷 스윕" in markdown
    # 사람이 읽는 표가 분모와 top_k 를 함께 보여야 전략 간 precision 을 비교할 수 있다.
    assert "top_k: 5" in markdown
    assert "precision n" in markdown
    assert "채택 축: 코사인 유사도 (모든 전략 동일)" in markdown


def test_전략_리포트_파일명은_재작성_조건과_컷을_구분한다(tmp_path: Path) -> None:
    """blind 산출물과 oracle 산출물이 접미사 숫자로만 갈리면 결정 기록이 인용할 수 없다."""
    common = {
        "live": False,
        "dimensions": 32,
        "cutoff": 0.10,
        "sweep_start": 0.10,
        "sweep_end": 0.20,
        "sweep_step": 0.05,
        "output_dir": tmp_path / "reports",
        "cache_dir": tmp_path / "cache",
    }
    blind = run_retrieval_comparison(**cast(Any, common))
    oracle = run_retrieval_comparison(
        **cast(Any, common),
        rewritten_queries_path=DEFAULT_ORACLE_REWRITTEN_QUERIES_PATH,
    )

    assert "blind" in blind.markdown.stem
    assert "oracle" in oracle.markdown.stem
    assert blind.markdown.stem != oracle.markdown.stem
    assert "-k5-c010" in blind.markdown.stem


def test_리랭크_과금_미승인이면_그_단만_사유와_함께_미측정으로_남는다(tmp_path: Path) -> None:
    """무과금이라고 안내한 실행이 조용히 과금하지 않고, 빈 자리를 0 으로 채우지도 않는다."""
    paths = run_retrieval_comparison(
        live=True,
        embedding_model="local-fake",
        embedding_client=LexicalEmbeddingClient(dimensions=32),
        paid_rerank=False,
        cutoff=0.10,
        sweep_start=0.10,
        sweep_end=0.20,
        sweep_step=0.05,
        output_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
    )

    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    assert [strategy["name"] for strategy in payload["strategies"]] == [
        "vector",
        "vector_rewrite",
        "vector_rewrite_hybrid",
    ]
    assert payload["unmeasured_stages"] == [
        {
            "stage": "llm_rerank",
            "reason": "OpenAI 리랭크 과금 미승인 — 실행하려면 유료 리랭크를 명시적으로 켠다",
        }
    ]
    assert payload["rerank_observability"]["calls"] == 0
    markdown = paths.markdown.read_text(encoding="utf-8")
    assert "## 미측정 단" in markdown
    assert "`llm_rerank`: 미측정" in markdown


def test_대역_리랭크_실행은_폴백_건수와_토큰을_리포트에_남긴다(tmp_path: Path) -> None:
    paths = run_retrieval_comparison(
        live=False,
        dimensions=32,
        cutoff=0.10,
        sweep_start=0.10,
        sweep_end=0.20,
        sweep_step=0.05,
        output_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
    )

    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    observability = payload["rerank_observability"]
    assert observability["calls"] == 30
    assert observability["fallbacks"] == 0
    assert observability["fallback_reasons"] == []
    assert "## 리랭크 관측" in paths.markdown.read_text(encoding="utf-8")


def test_run_retrieval_comparison_loads_committed_rewrites_for_default_ladder(
    tmp_path: Path,
) -> None:
    paths = run_retrieval_comparison(
        live=False,
        dimensions=32,
        top_k=5,
        cutoff=0.10,
        sweep_start=0.10,
        sweep_end=0.20,
        sweep_step=0.05,
        output_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
    )

    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    assert [strategy["name"] for strategy in payload["strategies"]] == [
        "vector",
        "vector_rewrite",
        "vector_rewrite_hybrid",
        "vector_rewrite_hybrid_rerank",
    ]
    assert all(len(strategy["cases"]) == 30 for strategy in payload["strategies"])
    assert payload["run_conditions"]["rewrite_condition"] == "blind"
    assert payload["run_conditions"]["rewrite_source"] == "data/rewritten_queries.jsonl"
    markdown = paths.markdown.read_text(encoding="utf-8")
    assert "정책·라벨을 본 적 없는 생성 모델이 문의 원문만 보고 만든" in markdown
    # 리포트만 보고도 blind 와 oracle 을 혼동할 수 없어야 한다.
    assert "oracle" not in markdown


def test_run_retrieval_comparison_can_report_oracle_upper_bound_condition(
    tmp_path: Path,
) -> None:
    paths = run_retrieval_comparison(
        live=False,
        dimensions=32,
        top_k=5,
        cutoff=0.10,
        sweep_start=0.10,
        sweep_end=0.20,
        sweep_step=0.05,
        output_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
        rewritten_queries_path=DEFAULT_ORACLE_REWRITTEN_QUERIES_PATH,
        rewrite_condition=RewriteCondition.ORACLE,
    )

    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    assert payload["run_conditions"]["rewrite_condition"] == "oracle_upper_bound"
    assert payload["run_conditions"]["rewrite_source"] == "data/rewritten_queries_oracle.jsonl"
    assert "배포 가능 개선폭이 아님" in paths.markdown.read_text(encoding="utf-8")


def test_fixture_and_rewrite_configuration_fail_before_any_embedding_call(tmp_path: Path) -> None:
    embedder = _FixedEmbedder({})

    with pytest.raises(FileNotFoundError):
        run_retrieval_comparison(
            live=True,
            embedding_client=cast(EmbeddingClient, embedder),
            rewritten_queries_path=tmp_path / "missing.jsonl",
            rewrite_condition=RewriteCondition.BLIND,
            paid_rerank=False,
            output_dir=tmp_path / "reports",
            cache_dir=tmp_path / "cache",
        )
    assert embedder.calls == []

    with pytest.raises(RetrievalConfigurationError, match="rewrite_condition"):
        run_retrieval_comparison(
            live=True,
            embedding_client=cast(EmbeddingClient, embedder),
            rewrite_condition=cast(Any, "oracle"),
            output_dir=tmp_path / "reports",
            cache_dir=tmp_path / "cache",
        )
    assert embedder.calls == []


def test_oracle_픽스처를_blind로_선언하면_거부한다(tmp_path: Path) -> None:
    """조건 표기는 산문이 아니라 코드로 결속된다 — 상한 수치가 배포 가능 개선폭으로 인쇄되지 않게."""
    with pytest.raises(RetrievalConfigurationError, match="선언된 조건이 다르다"):
        run_retrieval_comparison(
            live=False,
            dimensions=32,
            output_dir=tmp_path / "reports",
            cache_dir=tmp_path / "cache",
            rewritten_queries_path=DEFAULT_ORACLE_REWRITTEN_QUERIES_PATH,
            rewrite_condition=RewriteCondition.BLIND,
        )


def test_저장소_픽스처_경로는_조건을_스스로_결정한다(tmp_path: Path) -> None:
    """기본값 BLIND 를 없앴으므로 조건을 생략해도 oracle 입력이 blind 로 찍히지 않는다."""
    paths = run_retrieval_comparison(
        live=False,
        dimensions=32,
        cutoff=0.10,
        sweep_start=0.10,
        sweep_end=0.20,
        sweep_step=0.05,
        output_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
        rewritten_queries_path=DEFAULT_ORACLE_REWRITTEN_QUERIES_PATH,
    )

    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    assert payload["run_conditions"]["rewrite_condition"] == "oracle_upper_bound"


def test_저장소_밖_재작성_입력은_조건_명시를_요구한다(tmp_path: Path) -> None:
    fixture = tmp_path / "custom.jsonl"
    fixture.write_text("", encoding="utf-8")

    with pytest.raises(RetrievalConfigurationError, match="rewrite_condition을 명시"):
        run_retrieval_comparison(
            live=False,
            dimensions=32,
            output_dir=tmp_path / "reports",
            cache_dir=tmp_path / "cache",
            rewritten_queries_path=fixture,
        )


def test_run_vector_only_without_rewrites_reports_rewrite_not_used(tmp_path: Path) -> None:
    paths = run_retrieval_comparison(
        live=False,
        dimensions=32,
        top_k=5,
        cutoff=0.10,
        sweep_start=0.10,
        sweep_end=0.20,
        sweep_step=0.05,
        output_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
        strategies=(StrategyDefinition("vector", (RetrievalStage.VECTOR,)),),
    )

    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    assert [strategy["name"] for strategy in payload["strategies"]] == ["vector"]
    assert payload["run_conditions"]["rewrite_source"] == "not_used"
    assert payload["run_conditions"]["rewrite_condition"] == "not_used"
    assert "질의 재작성: 미사용" in paths.markdown.read_text(encoding="utf-8")


def test_strategy_ladder_completes_four_stub_combinations_and_reports_separate_cuts(
    tmp_path: Path,
) -> None:
    cases = load_golden_set()
    comparison = evaluate_strategy_ladder(
        documents=load_policy_documents(),
        cases=cases,
        labels=load_retrieval_labels(),
        rewritten_queries={case.id: f"{case.content} 재작성" for case in cases},
        embedder=LexicalEmbeddingClient(dimensions=32),
        embedding_config=RetrievalEvalConfig(
            model="lexical-2gram-v1",
            dimensions=32,
            top_k=5,
            cutoff=0.10,
            is_stub=True,
        ),
        cutoffs=StrategyCutoffs(
            cosine_similarity=0.10,
            rrf_score=0.0,
            rerank_top_n=3,
        ),
        reranker=cast(GenerationClient, _IdentityReranker()),
        rerank_model="stub-reranker",
        cache_dir=tmp_path / "embedding-cache",
        rerank_cache_dir=tmp_path / "rerank-cache",
    )

    assert [result.strategy.name for result in comparison.strategies] == [
        "vector",
        "vector_rewrite",
        "vector_rewrite_hybrid",
        "vector_rewrite_hybrid_rerank",
    ]
    # 채택 축은 전 전략 동일하다. 다른 것은 채택 상한뿐이다.
    assert [result.cutoff for result in comparison.strategies] == [0.10, 0.10, 0.10, 0.10]
    assert [result.accept_limit for result in comparison.strategies] == [5, 5, 5, 3]
    assert all(len(result.cases) == 30 for result in comparison.strategies)
    # 순위 품질은 컷·후보 풀과 다른 축이다 — 전 전략이 전체 코퍼스 순위를 갖는다.
    ranking_lengths = {
        len(case.ranked_hits) for result in comparison.strategies for case in result.cases
    }
    assert ranking_lengths == {26}

    paths = write_strategy_report(comparison, output_dir=tmp_path / "reports")
    report = json.loads(paths.json.read_text(encoding="utf-8"))
    assert report["run_conditions"]["labels_used_for_retrieval"] is False
    assert report["run_conditions"]["rewrite_source"] == "caller_injected"
    assert report["run_conditions"]["rewrite_condition"] == "caller_injected"
    assert "실제 검색 품질이 아니다" in report["run_conditions"]["warning"]
    case = report["strategies"][3]["cases"][0]
    assert case["strategy"] == "vector_rewrite_hybrid_rerank"
    assert case["cutoff"] == {"kind": "cosine_similarity", "value": 0.10}
    assert {"rank", "evidence_id", "vector_similarity", "bm25_score", "rrf_score"} <= set(
        case["ranked_hits"][0]
    )
    assert "accepted_hits" in case


@pytest.mark.parametrize(
    ("rewritten_queries", "message"),
    [
        ({}, "누락=G"),
        ({"G": "재작성", "EXTRA": "추가"}, "추가=EXTRA"),
        ({"G": "   "}, "비어"),
    ],
)
def test_rewrite_input_is_validated_before_embedding(
    tmp_path: Path, rewritten_queries: dict[str, str], message: str
) -> None:
    embedder = _FixedEmbedder({"정책 A": [1.0, 0.0], "문의": [1.0, 0.0], "재작성": [0.0, 1.0]})

    with pytest.raises(RetrievalConfigurationError, match=message):
        retrieve_strategy_ladder(
            queries=(RetrievalQuery(case_id="G", text="문의"),),
            policy_texts=(("policy:a:1", "정책 A"),),
            rewritten_queries=rewritten_queries,
            embedder=cast(EmbeddingClient, embedder),
            embedding_config=RetrievalEvalConfig(model="fixed", dimensions=2),
            cutoffs=StrategyCutoffs(),
            reranker=cast(GenerationClient, _IdentityReranker()),
            rerank_model="stub-reranker",
            cache_dir=tmp_path / "embedding-cache",
            rerank_cache_dir=tmp_path / "rerank-cache",
        )

    assert embedder.calls == []


def test_vector_rewrite_uses_distinct_injected_text(tmp_path: Path) -> None:
    embedder = _FixedEmbedder(
        {
            "정책 A": [1.0, 0.0],
            "정책 B": [0.0, 1.0],
            "원문 문의": [1.0, 0.0],
            "실제 재작성 문의": [0.0, 1.0],
        }
    )

    retrieved = retrieve_strategy_ladder(
        queries=(RetrievalQuery(case_id="G", text="원문 문의"),),
        policy_texts=(("policy:a:1", "정책 A"), ("policy:b:1", "정책 B")),
        rewritten_queries={"G": "실제 재작성 문의"},
        embedder=cast(EmbeddingClient, embedder),
        embedding_config=RetrievalEvalConfig(model="fixed", dimensions=2, top_k=2),
        cutoffs=StrategyCutoffs(cosine_similarity=0.5),
        reranker=None,
        rerank_model="",
        cache_dir=tmp_path / "embedding-cache",
        rerank_cache_dir=tmp_path / "rerank-cache",
        strategies=(
            StrategyDefinition("vector", (RetrievalStage.VECTOR,)),
            StrategyDefinition("vector_rewrite", (RetrievalStage.VECTOR, RetrievalStage.REWRITE)),
        ),
    )

    scored = score_strategy_ladder(
        retrieved.retrievals,
        (RetrievalLabel(id="G", relevant_evidence_ids=frozenset({"policy:a:1"}), note=""),),
        cutoff=0.5,
        cutoff_sweep=(0.5,),
    )
    by_name = {result.strategy.name: result.cases[0] for result in scored}
    assert [hit.evidence_id for hit in by_name["vector"].accepted_hits] == ["policy:a:1"]
    assert [hit.evidence_id for hit in by_name["vector_rewrite"].accepted_hits] == [
        "policy:a:1",
        "policy:b:1",
    ]
    assert ("retrieval_query", ("실제 재작성 문의",)) in embedder.calls


def test_strategy_retrieval_boundary_does_not_accept_or_import_labels() -> None:
    assert "labels" not in inspect.signature(retrieve_strategy_ladder).parameters
    retrieval_eval_module = inspect.getmodule(retrieve_strategy_ladder)
    assert retrieval_eval_module is not None
    source = inspect.getsource(retrieval_eval_module)
    strategy_source = inspect.getsource(
        __import__("reply_gate.retrieval_strategies", fromlist=["retrieval_strategies"])
    )

    assert (
        "labels="
        not in source[source.index("def retrieve_strategy_ladder") :][
            : source[source.index("def retrieve_strategy_ladder") :].index("\ndef ")
        ]
    )
    assert "retrieval_labels" not in strategy_source
    assert "RetrievalLabel" not in strategy_source


def _two_clause_ladder(
    tmp_path: Path,
    *,
    cutoffs: StrategyCutoffs,
    rrf_k: int = 60,
) -> StrategyLadderRetrieval:
    embedder = _FixedEmbedder(
        {
            "정책 A": [1.0, 0.0],
            "정책 B": [0.0, 1.0],
            "문의": [0.8, 0.2],
            "재작성": [0.2, 0.8],
        }
    )
    return retrieve_strategy_ladder(
        queries=(RetrievalQuery(case_id="G", text="문의"),),
        policy_texts=(("policy:a:1", "정책 A"), ("policy:b:1", "정책 B")),
        rewritten_queries={"G": "재작성"},
        embedder=cast(EmbeddingClient, embedder),
        embedding_config=RetrievalEvalConfig(model="fixed", dimensions=2, top_k=2),
        cutoffs=cutoffs,
        reranker=cast(GenerationClient, _IdentityReranker()),
        rerank_model="stub-reranker",
        cache_dir=tmp_path / "embedding-cache",
        rerank_cache_dir=tmp_path / "rerank-cache",
        rrf_k=rrf_k,
    )


def test_하이브리드와_리랭크도_절대_게이트로_기권한다(tmp_path: Path) -> None:
    """근거 없는 문의에서 기권이 전 전략에서 가능해야 precision 을 비교할 수 있다.

    채택을 RRF 점수 축으로 옮기면 순위 기반 점수에는 "관련 없음"이 없으므로 하이브리드
    이후 단은 구조적으로 기권할 수 없고, 빈 정답 케이스의 precision 이 0 으로 고정된다.
    """
    retrieved = _two_clause_ladder(
        tmp_path,
        cutoffs=StrategyCutoffs(cosine_similarity=0.99, rerank_top_n=1, fusion_pool_size=2),
    )

    scored = score_strategy_ladder(
        retrieved.retrievals,
        (RetrievalLabel(id="G", relevant_evidence_ids=frozenset(), note="정답 없음"),),
        cutoff=0.99,
        cutoff_sweep=(0.99,),
    )

    assert [result.cases[0].accepted_hits for result in scored] == [(), (), (), ()]
    # 기권했으므로 빈 정답 케이스의 precision 은 전 전략에서 1.0 이고 분모가 같다.
    assert [result.aggregate.accepted_precision for result in scored] == [1.0, 1.0, 1.0, 1.0]
    assert {result.aggregate.precision_case_count for result in scored} == {1}


def test_도달_불가능한_RRF_컷은_거부된다(tmp_path: Path) -> None:
    """걸러내지 못하는 컷이 걸러낸 척하면 하이브리드가 항상 top_k 를 채택한다."""
    with pytest.raises(ValueError, match="아무것도 걸러내지 못한다"):
        _two_clause_ladder(
            tmp_path,
            cutoffs=StrategyCutoffs(
                cosine_similarity=0.1, rrf_score=0.02, rerank_top_n=1, fusion_pool_size=2
            ),
        )


def test_리랭크_후보_풀이_채택_상한보다_작으면_거부된다(tmp_path: Path) -> None:
    """풀과 상한이 같으면 리랭크 채택 집합이 하이브리드와 항등이 되어 효과를 측정할 수 없다."""
    with pytest.raises(ValueError, match="리랭크 후보 풀은 최종 채택 상한보다 커야"):
        _two_clause_ladder(
            tmp_path,
            cutoffs=StrategyCutoffs(cosine_similarity=0.1, rerank_top_n=2, fusion_pool_size=2),
        )


def test_리랭크는_후보_풀에서_골라_하이브리드와_다른_집합을_채택할_수_있다(tmp_path: Path) -> None:
    """리랭크가 순서를 바꾸면 채택 집합이 실제로 달라져야 한다."""

    class _ReverseReranker:
        def complete_json(self, **kwargs: Any) -> JsonCompletion:
            payload = json.loads(cast(str, kwargs["user"]))
            ids = [candidate["evidence_id"] for candidate in payload["candidates"]]
            return JsonCompletion(
                data={"evidence_ids": list(reversed(ids))}, input_tokens=3, output_tokens=2
            )

    embedder = _FixedEmbedder(
        {
            "정책 A": [1.0, 0.0],
            "정책 B": [0.0, 1.0],
            "문의": [0.8, 0.2],
            "재작성": [0.7, 0.3],
        }
    )
    retrieved = retrieve_strategy_ladder(
        queries=(RetrievalQuery(case_id="G", text="문의"),),
        policy_texts=(("policy:a:1", "정책 A"), ("policy:b:1", "정책 B")),
        rewritten_queries={"G": "재작성"},
        embedder=cast(EmbeddingClient, embedder),
        embedding_config=RetrievalEvalConfig(model="fixed", dimensions=2, top_k=2),
        cutoffs=StrategyCutoffs(cosine_similarity=0.0, rerank_top_n=1, fusion_pool_size=2),
        reranker=cast(GenerationClient, _ReverseReranker()),
        rerank_model="stub-reverse",
        cache_dir=tmp_path / "embedding-cache",
        rerank_cache_dir=tmp_path / "rerank-cache",
    )

    scored = score_strategy_ladder(
        retrieved.retrievals,
        (RetrievalLabel(id="G", relevant_evidence_ids=frozenset({"policy:a:1"}), note=""),),
        cutoff=0.0,
        cutoff_sweep=(0.0,),
    )
    by_name = {result.strategy.name: result.cases[0] for result in scored}

    hybrid_top = [hit.evidence_id for hit in by_name["vector_rewrite_hybrid"].accepted_hits][:1]
    rerank_top = [hit.evidence_id for hit in by_name["vector_rewrite_hybrid_rerank"].accepted_hits]
    assert hybrid_top != rerank_top
    # 리랭크 행도 전체 순위를 유지하므로 recall@k 정의가 다른 행과 같다.
    assert len(by_name["vector_rewrite_hybrid_rerank"].ranked_hits) == 2
    assert retrieved.rerank.calls == 1
    assert retrieved.rerank.fallbacks == 0
    assert (retrieved.rerank.input_tokens, retrieved.rerank.output_tokens) == (3, 2)


class _CandidateEmbedder:
    def __init__(self, dimensions: int) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, *, stage: str, texts: Sequence[str]) -> EmbeddingResult:
        del stage
        return EmbeddingResult(vectors=[[0.0] * self._dimensions for _ in texts], total_tokens=0)


def test_embedding_model_axis_has_four_documented_candidates() -> None:
    assert [
        (candidate.model, candidate.dimensions, candidate.provider)
        for candidate in DEFAULT_EMBEDDING_CANDIDATES
    ] == [
        ("text-embedding-3-small", 1536, EmbeddingProvider.OPENAI),
        ("text-embedding-3-large", 1536, EmbeddingProvider.OPENAI),
        ("text-embedding-3-large", 3072, EmbeddingProvider.OPENAI),
        ("BAAI/bge-m3", 1024, EmbeddingProvider.LOCAL),
    ]


def test_missing_BGE_dependency_marks_only_that_axis_row_unmeasured(tmp_path: Path) -> None:
    evaluated: list[str] = []

    def factory(candidate: EmbeddingCandidate, api_key: str) -> EmbeddingClient:
        assert api_key == "test-key"
        if candidate.provider is EmbeddingProvider.LOCAL:
            raise OptionalEmbeddingDependencyError("미측정 — 로컬 의존성 미설치")
        return _CandidateEmbedder(candidate.dimensions)

    def evaluate(candidate: EmbeddingCandidate, client: EmbeddingClient) -> ReportPaths:
        assert client.dimensions == candidate.dimensions
        evaluated.append(candidate.key)
        return ReportPaths(
            markdown=tmp_path / f"{candidate.key}.md",
            json=tmp_path / f"{candidate.key}.json",
        )

    result = run_embedding_model_axis(api_key="test-key", evaluate=evaluate, client_factory=factory)

    assert evaluated == ["3-small-1536", "3-large-1536", "3-large-3072"]
    assert [row.measured for row in result.rows] == [True, True, True, False]
    assert result.rows[-1].reports is None
    assert result.rows[-1].reason == "미측정 — 로컬 의존성 미설치"


def test_대역_임베딩_구현은_모델_버전_문자열에_고정된다() -> None:
    """대역 캐시 키는 모델 이름만 담는다 — 알고리즘을 바꾸면 이 테스트가 먼저 깨진다.

    이 값이 바뀌었다면 `STUB_EMBEDDING_MODEL` 의 버전을 올려야 한다. 올리지 않으면
    `.retrieval-cache/` 의 옛 벡터가 새 구현의 결과로 조용히 재사용된다.
    """
    vectors = LexicalEmbeddingClient(dimensions=8).embed(stage="t", texts=["환불 규정"]).vectors

    assert STUB_EMBEDDING_MODEL == "lexical-2gram-v1"
    assert [round(value, 9) for value in vectors[0]] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.577350269,
        0.577350269,
        0.577350269,
    ]
