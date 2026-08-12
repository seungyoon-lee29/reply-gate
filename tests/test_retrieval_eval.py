"""DB 없는 벡터 검색 비교 하네스의 공개 동작."""

from pathlib import Path
from typing import cast

from reply_gate.evaluation import load_golden_set
from reply_gate.llm import EmbeddingClient, EmbeddingResult
from reply_gate.policy_index import load_policy_documents
from reply_gate.retrieval_eval import (
    RankedHit,
    RetrievalEvalConfig,
    RetrievalQuery,
    RetrievedCase,
    evaluate_retrieval,
    retrieve_cases,
    run_retrieval_comparison,
    score_retrieval,
    write_report,
)
from reply_gate.retrieval_labels import RetrievalLabel, load_retrieval_labels
from reply_gate.testing import LexicalEmbeddingClient


class _FixedEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    @property
    def dimensions(self) -> int:
        return 2

    def embed(self, *, stage: str, texts: list[str]) -> EmbeddingResult:
        self.calls.append((stage, tuple(texts)))
        return EmbeddingResult(vectors=[self._vectors[text] for text in texts], total_tokens=0)


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


def test_run_retrieval_comparison_is_a_complete_free_stub_entrypoint(
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

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "G17" in paths.markdown.read_text(encoding="utf-8")
