"""청킹 전략 비교 격자 — 코퍼스 변형과 청크→조항 매핑 채점의 공개 동작.

이 격자는 **측정 전용**이다. 제품 경로·DB·근거 ID 체계는 건드리지 않는다(결정 0015).
그래서 테스트가 보는 것도 둘이다: 매핑 규칙이 손계산과 일치하는가, 격자가 대역 임베딩으로
완주하며 편향·민감도를 리포트에 남기는가.

**채점자 쪽과 코퍼스 쪽의 경계**를 테스트가 함께 못박는다 — 코퍼스 변형
(`build_policy_sources`·`chunk_policy_sources`)은 라벨을 받지 않고, 라벨을 보는 것은
채점(`score_chunking`)뿐이다.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path

import pytest

from reply_gate.evaluation import load_golden_set
from reply_gate.llm import EmbeddingResult
from reply_gate.policy_index import PlantedKind, PolicyChunk, PolicyDocument, load_policy_documents
from reply_gate.retrieval_eval import (
    DEFAULT_CHUNKING_STRATEGIES,
    DEFAULT_CONTAINMENT_SWEEP,
    ChunkingKind,
    ChunkingStrategy,
    CorpusUnit,
    RetrievalConfigurationError,
    RetrievalEvalConfig,
    RetrievalQuery,
    build_policy_sources,
    chunk_policy_sources,
    map_units_to_clauses,
    retrieve_cases,
    run_chunking_comparison,
    run_chunking_grid,
    score_chunking,
    score_retrieval,
    write_chunking_report,
)
from reply_gate.retrieval_labels import load_retrieval_labels
from reply_gate.testing import LexicalEmbeddingClient

_CLAUSE_ONLY = ChunkingStrategy(key="clause", kind=ChunkingKind.CLAUSE)


def _clause(article: str, title: str, content: str) -> PolicyChunk:
    return PolicyChunk(
        evidence_id=f"policy:doc:{article}",
        document_slug="doc",
        document_title="손계산 문서",
        article=article,
        article_title=title,
        content=content,
        planted=PlantedKind.NONE,
        planted_note=None,
    )


def _handcalc_document() -> PolicyDocument:
    """손계산 픽스처 — 본문을 10·20·10 자로 못박아 포함 비율을 손으로 셀 수 있게 한다."""
    return PolicyDocument(
        slug="doc",
        title="손계산 문서",
        chunks=(
            _clause("1-1", "가", "가" * 10),
            _clause("1-2", "나", "나" * 20),
            _clause("1-3", "다", "다" * 10),
        ),
    )


def _unit(unit_id: str, slug: str, start: int, end: int) -> CorpusUnit:
    return CorpusUnit(
        unit_id=unit_id,
        document_slug=slug,
        embedding_text="",
        start=start,
        end=end,
    )


class _RecordingEmbedder:
    """어휘 대역을 감싸 임베딩 대상 텍스트를 기록한다."""

    def __init__(self, dimensions: int = 64) -> None:
        self._inner = LexicalEmbeddingClient(dimensions=dimensions)
        self.texts: list[str] = []

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    def embed(self, *, stage: str, texts: Sequence[str]) -> EmbeddingResult:
        self.texts.extend(texts)
        return self._inner.embed(stage=stage, texts=texts)


# --- 코퍼스 변형 (라벨을 보지 않는 쪽) -------------------------------------------------


def test_원문_스트림은_조항_본문_span_을_글자_단위로_들고_있다() -> None:
    document = _handcalc_document()

    sources = build_policy_sources((document,))

    assert len(sources) == 1
    source = sources[0]
    bodies = {chunk.evidence_id: chunk.content for chunk in document.chunks}
    for clause in source.clauses:
        # span 이 가리키는 구간이 실제 본문과 글자 단위로 같아야 매핑이 성립한다.
        assert source.text[clause.start : clause.end] == bodies[clause.evidence_id]
    assert [clause.end - clause.start for clause in source.clauses] == [10, 20, 10]


def test_고정_크기_청킹은_선언된_크기와_보폭으로_자른다() -> None:
    sources = build_policy_sources((_handcalc_document(),))
    strategy = ChunkingStrategy(key="fixed30o0", kind=ChunkingKind.FIXED, chunk_size=30, overlap=0)

    units = chunk_policy_sources(sources, strategy)

    text = sources[0].text
    assert [unit.end - unit.start for unit in units[:-1]] == [30] * (len(units) - 1)
    assert units[0].start == 0
    assert units[-1].end == len(text)
    # 오버랩 0 이면 구간이 겹치지 않고 원문을 정확히 덮는다.
    assert "".join(text[unit.start : unit.end] for unit in units) == text


def test_오버랩은_보폭을_줄이고_구간이_겹친다() -> None:
    sources = build_policy_sources((_handcalc_document(),))
    strategy = ChunkingStrategy(
        key="fixed30o10", kind=ChunkingKind.FIXED, chunk_size=30, overlap=10
    )

    units = chunk_policy_sources(sources, strategy)

    starts = [unit.start for unit in units]
    assert starts[1] - starts[0] == 20  # 보폭 = 30 - 10
    assert units[0].end > units[1].start


def test_오버랩이_크기_이상이면_구성_오류다() -> None:
    with pytest.raises(RetrievalConfigurationError):
        ChunkingStrategy(key="bad", kind=ChunkingKind.FIXED, chunk_size=100, overlap=100)


def test_조항_단위_전략은_근거_ID_를_그대로_단위_ID_로_쓴다() -> None:
    document = _handcalc_document()
    sources = build_policy_sources((document,))

    units = chunk_policy_sources(sources, _CLAUSE_ONLY)

    assert [unit.unit_id for unit in units] == [
        "policy:doc:1-1",
        "policy:doc:1-2",
        "policy:doc:1-3",
    ]
    # 임베딩 텍스트가 현행 인덱싱(`PolicyChunk.embedding_text`)과 같아야 캐시가 그대로 산다.
    assert units[0].embedding_text == document.chunks[0].embedding_text


def test_코퍼스_변형은_라벨을_인자로_받지_않는다() -> None:
    """코퍼스 쪽과 채점자 쪽의 경계 — 서명에 정답이 들어올 자리가 없어야 한다."""
    for function in (build_policy_sources, chunk_policy_sources):
        parameters = set(inspect.signature(function).parameters)
        assert not parameters & {"labels", "relevant_evidence_ids", "golden_set", "label"}

    assert "labels" in set(inspect.signature(score_chunking).parameters)


# --- 청크→조항 매핑 (채점자 쪽) -------------------------------------------------------


def test_매핑은_손계산_픽스처와_일치한다() -> None:
    """본문 10·20·10 자에 대해 포함 비율을 손으로 센 결과와 같아야 한다."""
    sources = build_policy_sources((_handcalc_document(),))
    source = sources[0]
    first, second, third = source.clauses

    units = (
        # 1-2 본문(20자)의 앞 10자만 = 0.50.
        _unit("u1", source.slug, second.start, second.start + 10),
        # 1-1 본문 전체(1.00) + 1-2 본문 앞 5자(0.25).
        _unit("u2", source.slug, first.start, second.start + 5),
        # 세 조항 본문을 모두 덮는 구간.
        _unit("u3", source.slug, first.start, third.end),
    )

    mapping = map_units_to_clauses(units, sources, min_containment=0.5)

    assert mapping["u1"] == ("policy:doc:1-2",)
    assert mapping["u2"] == ("policy:doc:1-1",)
    assert mapping["u3"] == ("policy:doc:1-1", "policy:doc:1-2", "policy:doc:1-3")


def test_조항_번호만_언급하는_청크는_적중이_아니다() -> None:
    """음성 대조 — 제목줄만 덮은 청크는 그 조항을 적중시키지 않는다."""
    sources = build_policy_sources((_handcalc_document(),))
    source = sources[0]
    first, second = source.clauses[0], source.clauses[1]

    # 1-1 본문 전체 + `## 1-2 나` 제목줄. 1-2 **본문**은 한 글자도 들어 있지 않다.
    heading_only = _unit("heading", source.slug, first.start, second.start)

    assert "1-2" in source.text[heading_only.start : heading_only.end]
    # 최소 비율을 바닥까지 내려도 적중이 아니다 — 번호 언급은 본문 포함이 아니다.
    assert map_units_to_clauses((heading_only,), sources, min_containment=0.01) == {
        "heading": ("policy:doc:1-1",)
    }


def test_실제_코퍼스에도_번호만_언급하는_청크가_있고_매핑이_배제한다() -> None:
    """손계산 픽스처가 아니라 실제 정책 원문에서도 음성 대조가 성립하는지 본다."""
    sources = build_policy_sources(load_policy_documents())
    units = chunk_policy_sources(
        sources,
        ChunkingStrategy(key="fixed120o0", kind=ChunkingKind.FIXED, chunk_size=120, overlap=0),
    )
    mapping = map_units_to_clauses(units, sources, min_containment=0.5)
    by_slug = {source.slug: source for source in sources}

    mentioned_not_hit = 0
    for unit in units:
        source = by_slug[unit.document_slug]
        text = source.text[unit.start : unit.end]
        for clause in source.clauses:
            if clause.article in text and clause.evidence_id not in mapping[unit.unit_id]:
                mentioned_not_hit += 1

    assert mentioned_not_hit > 0


def test_최소_포함_비율이_경계에서_적중을_가른다() -> None:
    sources = build_policy_sources((_handcalc_document(),))
    second = sources[0].clauses[1]
    half = _unit("half", "doc", second.start, second.start + 10)  # 20자 중 10자 = 0.50

    assert map_units_to_clauses((half,), sources, min_containment=0.50) == {
        "half": ("policy:doc:1-2",)
    }
    assert map_units_to_clauses((half,), sources, min_containment=0.51) == {"half": ()}


def test_최소_포함_비율은_0_초과_1_이하만_받는다() -> None:
    sources = build_policy_sources((_handcalc_document(),))

    with pytest.raises(RetrievalConfigurationError):
        map_units_to_clauses((), sources, min_containment=0.0)
    with pytest.raises(RetrievalConfigurationError):
        map_units_to_clauses((), sources, min_containment=1.5)


def test_조항_단위_청킹의_매핑은_어떤_비율에서도_항등이다() -> None:
    """조항 청크는 자기 본문을 100% 담으므로 비율을 올려도 매핑이 흔들리지 않는다."""
    sources = build_policy_sources(load_policy_documents())
    units = chunk_policy_sources(sources, _CLAUSE_ONLY)

    for containment in DEFAULT_CONTAINMENT_SWEEP:
        mapping = map_units_to_clauses(units, sources, min_containment=containment)
        assert all(mapping[unit.unit_id] == (unit.unit_id,) for unit in units)


# --- 채점 ---------------------------------------------------------------------------


def test_조항_단위_격자는_기존_벡터_채점과_같은_수치를_낸다(tmp_path: Path) -> None:
    """항등 매핑이면 이 격자의 채점이 `score_retrieval` 과 한 자리도 다르지 않아야 한다.

    다르면 비교표의 기준 열이 기존 실측과 다른 것을 재고 있다는 뜻이다.
    """
    documents = load_policy_documents()
    cases = load_golden_set()
    labels = load_retrieval_labels()
    config = RetrievalEvalConfig(model="lexical-2gram-v1", dimensions=64, top_k=5, cutoff=0.10)
    embedder = LexicalEmbeddingClient(dimensions=64)
    queries = tuple(RetrievalQuery(case_id=case.id, text=case.content) for case in cases)
    sources = build_policy_sources(documents)
    units = chunk_policy_sources(sources, _CLAUSE_ONLY)

    retrieved = retrieve_cases(
        queries=queries,
        policy_texts=tuple((unit.unit_id, unit.embedding_text) for unit in units),
        embedder=embedder,
        config=config,
        cache_dir=tmp_path / "cache",
    )
    baseline = score_retrieval(retrieved, labels, top_k=5, cutoff=0.10)
    chunked = score_chunking(
        retrieved,
        labels,
        mapping=map_units_to_clauses(units, sources, min_containment=0.5),
        top_k=5,
        cutoff=0.10,
    )

    assert chunked.aggregate.recall_at_1 == baseline.aggregate.recall_at_1
    assert chunked.aggregate.recall_at_3 == baseline.aggregate.recall_at_3
    assert chunked.aggregate.recall_at_5 == baseline.aggregate.recall_at_5
    assert chunked.aggregate.accepted_precision == baseline.aggregate.accepted_precision
    assert chunked.aggregate.accepted_recall == baseline.aggregate.accepted_recall
    assert chunked.aggregate.precision_case_count == baseline.aggregate.precision_case_count


def test_채점은_케이스_부분집합으로_좁힐_수_있다(tmp_path: Path) -> None:
    """편향 케이스를 뺀 열이 같은 채점 코드에서 나와야 두 수치를 나란히 읽을 수 있다."""
    cases = load_golden_set()
    labels = load_retrieval_labels()
    config = RetrievalEvalConfig(model="lexical-2gram-v1", dimensions=64, top_k=5, cutoff=0.10)
    sources = build_policy_sources(load_policy_documents())
    units = chunk_policy_sources(sources, _CLAUSE_ONLY)
    retrieved = retrieve_cases(
        queries=tuple(RetrievalQuery(case_id=case.id, text=case.content) for case in cases),
        policy_texts=tuple((unit.unit_id, unit.embedding_text) for unit in units),
        embedder=LexicalEmbeddingClient(dimensions=64),
        config=config,
        cache_dir=tmp_path / "cache",
    )
    mapping = map_units_to_clauses(units, sources, min_containment=0.5)

    narrowed = score_chunking(
        retrieved,
        labels,
        mapping=mapping,
        top_k=5,
        cutoff=0.10,
        exclude_case_ids=frozenset({"G01", "G02"}),
    )

    assert {case.case_id for case in narrowed.cases}.isdisjoint({"G01", "G02"})
    assert len(narrowed.cases) == len(cases) - 2


# --- 격자와 리포트 --------------------------------------------------------------------


def test_격자는_대역_임베딩으로_완주하고_리포트를_쓴다(tmp_path: Path) -> None:
    paths = run_chunking_comparison(
        live=False,
        output_dir=tmp_path,
        cache_dir=tmp_path / "cache",
    )

    assert paths.markdown.exists() and paths.json.exists()
    markdown = paths.markdown.read_text(encoding="utf-8")
    # 선언된 조정 인자가 리포트에 그대로 있어야 수치를 읽을 수 있다.
    assert "최소 포함 비율" in markdown
    for strategy in DEFAULT_CHUNKING_STRATEGIES:
        assert strategy.label in markdown
    assert "대역" in markdown


def test_리포트는_인접_조항_편향을_수치_옆에_적는다(tmp_path: Path) -> None:
    paths = run_chunking_comparison(live=False, output_dir=tmp_path, cache_dir=tmp_path / "cache")

    markdown = paths.markdown.read_text(encoding="utf-8")
    assert "G01" in markdown and "G02" in markdown
    assert "인접" in markdown


def test_리포트는_최소_포함_비율_민감도를_함께_싣는다(tmp_path: Path) -> None:
    paths = run_chunking_comparison(live=False, output_dir=tmp_path, cache_dir=tmp_path / "cache")

    markdown = paths.markdown.read_text(encoding="utf-8")
    for containment in DEFAULT_CONTAINMENT_SWEEP:
        assert f"{containment:.2f}" in markdown


def test_리포트는_처분을_비교_결과와_함께_적는다(tmp_path: Path) -> None:
    paths = run_chunking_comparison(live=False, output_dir=tmp_path, cache_dir=tmp_path / "cache")

    markdown = paths.markdown.read_text(encoding="utf-8")
    assert "비교만" in markdown
    assert "policy:<slug>:<조항>" in markdown


def test_리포트는_기존_산출물을_덮어쓰지_않는다(tmp_path: Path) -> None:
    first = run_chunking_comparison(live=False, output_dir=tmp_path, cache_dir=tmp_path / "cache")
    marker = first.markdown.read_text(encoding="utf-8")

    second = run_chunking_comparison(live=False, output_dir=tmp_path, cache_dir=tmp_path / "cache")

    assert second.markdown != first.markdown
    assert first.markdown.read_text(encoding="utf-8") == marker


def test_격자는_고정_청크만_새로_임베딩한다(tmp_path: Path) -> None:
    """조항 단위 열은 현행 인덱싱과 같은 텍스트라 캐시가 그대로 산다 — 과금 경계."""
    embedder = _RecordingEmbedder()
    documents = load_policy_documents()
    clause_texts = {chunk.embedding_text for document in documents for chunk in document.chunks}

    grid = run_chunking_grid(
        documents=documents,
        cases=load_golden_set(),
        labels=load_retrieval_labels(),
        embedder=embedder,
        model="lexical-2gram-v1",
        dimensions=64,
        is_stub=True,
        cache_dir=tmp_path / "cache",
        strategies=(
            _CLAUSE_ONLY,
            ChunkingStrategy(key="fixed240o0", kind=ChunkingKind.FIXED, chunk_size=240, overlap=0),
        ),
    )

    assert grid.embedding_cost.cache_misses == len(set(embedder.texts))
    # 조항 단위 코퍼스의 임베딩 텍스트가 현행 인덱싱과 한 글자도 다르지 않다.
    clause_stat = next(stat for stat in grid.corpus if stat.strategy_key == "clause")
    assert clause_stat.unit_count == len(clause_texts)


def test_write_chunking_report_는_출력_디렉터리를_만든다(tmp_path: Path) -> None:
    grid = run_chunking_grid(
        documents=load_policy_documents(),
        cases=load_golden_set(),
        labels=load_retrieval_labels(),
        embedder=LexicalEmbeddingClient(dimensions=64),
        model="lexical-2gram-v1",
        dimensions=64,
        is_stub=True,
        cache_dir=tmp_path / "cache",
    )

    paths = write_chunking_report(grid, output_dir=tmp_path / "nested" / "reports")

    assert paths.markdown.exists()
    assert paths.json.exists()


def test_인접_판정은_원문_순서에서_계산된다(tmp_path: Path) -> None:
    """G01·G02 는 인접, G18 은 인접이 아니다 — 편향 서술의 근거가 여기서 나온다."""
    grid = run_chunking_grid(
        documents=load_policy_documents(),
        cases=load_golden_set(),
        labels=load_retrieval_labels(),
        embedder=LexicalEmbeddingClient(dimensions=64),
        model="lexical-2gram-v1",
        dimensions=64,
        is_stub=True,
        cache_dir=tmp_path / "cache",
        strategies=(_CLAUSE_ONLY,),
    )

    adjacency = {case.case_id: case.adjacent for case in grid.multi_clause_cases}

    assert adjacency["G01"] is True
    assert adjacency["G02"] is True
    assert adjacency["G18"] is False


# --- CLI --------------------------------------------------------------------------------


def test_CLI_는_대역으로_청킹_격자를_돌린다(tmp_path: Path) -> None:
    from scripts import compare_retrieval

    exit_code = compare_retrieval.main(
        [
            "--stub-embedding",
            "--chunking-grid",
            "--out-dir",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    assert exit_code == 0
    assert list(tmp_path.glob("retrieval-chunking-stub-*.md"))


def test_CLI_는_청킹_축을_다른_축과_섞지_않는다(tmp_path: Path) -> None:
    """청킹만이 변수여야 비교가 성립한다 — 섞인 요청은 실행 전에 거부한다."""
    from scripts import compare_retrieval

    exit_code = compare_retrieval.main(
        ["--stub-embedding", "--chunking-grid", "--vector-only", "--out-dir", str(tmp_path)]
    )

    assert exit_code == 2
    assert not list(tmp_path.glob("*.md"))


def test_CLI_는_청킹_격자가_읽지_않는_인자를_거부한다(tmp_path: Path) -> None:
    """조용히 버리면 준 값이 반영됐다고 읽히고 리포트 조건이 실제 실행과 갈린다."""
    from scripts import compare_retrieval

    exit_code = compare_retrieval.main(
        [
            "--stub-embedding",
            "--chunking-grid",
            "--sweep-step",
            "0.01",
            "--out-dir",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    assert exit_code == 2
    assert not list(tmp_path.glob("*.md"))


def test_CLI_는_기본값_그대로인_인자는_통과시킨다(tmp_path: Path) -> None:
    """거부는 **명시적으로 다른 값을 준** 경우에만이다 — 기본값 동작을 막지 않는다."""
    from scripts import compare_retrieval

    exit_code = compare_retrieval.main(
        [
            "--stub-embedding",
            "--chunking-grid",
            "--sweep-step",
            "0.05",
            "--out-dir",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    assert exit_code == 0


def test_CLI_는_잘못된_최소_포함_비율을_실행_전에_거부한다(tmp_path: Path) -> None:
    from scripts import compare_retrieval

    exit_code = compare_retrieval.main(
        [
            "--stub-embedding",
            "--chunking-grid",
            "--min-containment",
            "0",
            "--out-dir",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    assert exit_code == 2
    assert not list(tmp_path.glob("*.md"))


def test_리포트는_결정_기록을_가리킨다(tmp_path: Path) -> None:
    """수치와 서술이 갈려 있으므로 리포트가 서술의 자리를 알려줘야 한다."""
    paths = run_chunking_comparison(live=False, output_dir=tmp_path, cache_dir=tmp_path / "cache")

    markdown = paths.markdown.read_text(encoding="utf-8")
    assert "docs/tracking/decisions/0015-" in markdown


def test_캐시_미적중_0_을_공짜로_읽지_않게_적는다(tmp_path: Path) -> None:
    """두 번째 실행은 미적중 0 이다 — 그 0 이 '구매가 없었다'로 읽히면 비용 서사가 거짓이 된다."""
    cache = tmp_path / "cache"
    first = run_chunking_comparison(live=False, output_dir=tmp_path, cache_dir=cache)
    second = run_chunking_comparison(live=False, output_dir=tmp_path, cache_dir=cache)

    assert "미적중 0 은" not in first.markdown.read_text(encoding="utf-8")
    assert "미적중 0 은" in second.markdown.read_text(encoding="utf-8")
