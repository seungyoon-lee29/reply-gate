"""검색 비교 CLI가 공개하는 전략·재작성 조건 계약."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from scripts import compare_retrieval

from reply_gate.retrieval_eval import (
    DEFAULT_EMBEDDING_CANDIDATES,
    EmbeddingAxisResult,
    EmbeddingAxisRow,
    ReportPaths,
    RewriteCondition,
)
from reply_gate.retrieval_strategies import RetrievalStage, StrategyDefinition


def _capture_run(monkeypatch: Any, tmp_path: Path) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs: object) -> ReportPaths:
        calls.append(kwargs)
        return ReportPaths(tmp_path / "report.md", tmp_path / "report.json")

    monkeypatch.setattr(compare_retrieval, "run_retrieval_comparison", fake_run)
    return calls


def test_표준_CLI는_blind_재작성으로_네_전략_전체를_실행한다(
    monkeypatch: Any, tmp_path: Path
) -> None:
    calls = _capture_run(monkeypatch, tmp_path)

    assert compare_retrieval.main(["--stub-embedding", "--out-dir", str(tmp_path)]) == 0

    strategies = cast(Sequence[StrategyDefinition], calls[0]["strategies"])
    assert len(strategies) == 4
    assert any(RetrievalStage.REWRITE in strategy.stages for strategy in strategies)
    assert calls[0]["rewrite_condition"] is RewriteCondition.BLIND


def test_벡터_단독과_oracle은_명시적_옵션으로만_선택한다(monkeypatch: Any, tmp_path: Path) -> None:
    calls = _capture_run(monkeypatch, tmp_path)

    assert compare_retrieval.main(["--stub-embedding", "--vector-only"]) == 0
    assert len(cast(Sequence[StrategyDefinition], calls[0]["strategies"])) == 1
    assert calls[0]["rewrite_condition"] is RewriteCondition.BLIND

    assert compare_retrieval.main(["--stub-embedding", "--oracle-rewrite"]) == 0
    assert calls[1]["rewrite_condition"] is RewriteCondition.ORACLE
    assert cast(Path, calls[1]["rewritten_queries_path"]).name == "rewritten_queries_oracle.jsonl"


def test_oracle과_벡터_단독을_동시에_요청하면_실행_전에_거부한다(tmp_path: Path) -> None:
    parser = compare_retrieval.build_parser()

    try:
        parser.parse_args(["--oracle-rewrite", "--vector-only"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("상충 옵션이 파싱 단계에서 거부되지 않았다")


def test_live는_유료_리랭크를_함께_승인한다(monkeypatch: Any, tmp_path: Path) -> None:
    calls = _capture_run(monkeypatch, tmp_path)

    assert compare_retrieval.main(["--live", "--out-dir", str(tmp_path)]) == 0

    assert calls[0]["live"] is True
    assert calls[0]["paid_rerank"] is True


def test_로컬_임베딩은_유료_리랭크를_켜지_않는다(monkeypatch: Any, tmp_path: Path) -> None:
    """`--bge-m3` 는 로컬 임베딩이다. 리랭크 과금은 별개 축으로 명시해야 켜진다."""
    calls = _capture_run(monkeypatch, tmp_path)
    monkeypatch.setattr(compare_retrieval, "BgeM3EmbeddingClient", _FakeLocalEmbedder, raising=True)

    assert compare_retrieval.main(["--bge-m3", "--out-dir", str(tmp_path)]) == 0

    assert calls[0]["live"] is True
    assert calls[0]["paid_rerank"] is False


def test_로컬_임베딩도_명시하면_유료_리랭크를_켠다(monkeypatch: Any, tmp_path: Path) -> None:
    calls = _capture_run(monkeypatch, tmp_path)
    monkeypatch.setattr(compare_retrieval, "BgeM3EmbeddingClient", _FakeLocalEmbedder, raising=True)

    assert (
        compare_retrieval.main(["--bge-m3", "--rerank-with-openai", "--out-dir", str(tmp_path)])
        == 0
    )

    assert calls[0]["paid_rerank"] is True


def test_임베딩_모델_축은_live_없이는_거부된다(tmp_path: Path) -> None:
    assert compare_retrieval.main(["--embedding-axis", "--out-dir", str(tmp_path)]) == 2


def test_임베딩_모델_축은_행마다_리포트를_내고_미설치_행은_사유를_남긴다(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    _capture_run(monkeypatch, tmp_path)
    rows: list[str] = []

    def fake_axis(**kwargs: Any) -> Any:
        evaluate = cast(Any, kwargs["evaluate"])
        measured = DEFAULT_EMBEDDING_CANDIDATES[0]
        missing = DEFAULT_EMBEDDING_CANDIDATES[-1]
        rows.append(measured.key)
        return EmbeddingAxisResult(
            rows=(
                EmbeddingAxisRow(
                    candidate=measured,
                    measured=True,
                    reports=evaluate(measured, _FakeLocalEmbedder()),
                    reason=None,
                ),
                EmbeddingAxisRow(
                    candidate=missing,
                    measured=False,
                    reports=None,
                    reason="BGE-M3 미측정 — 로컬 의존성 미설치",
                ),
            )
        )

    monkeypatch.setattr(compare_retrieval, "run_embedding_model_axis", fake_axis)

    assert compare_retrieval.main(["--live", "--embedding-axis", "--out-dir", str(tmp_path)]) == 0

    printed = capsys.readouterr().out
    assert "3-small-1536" in printed
    assert "bge-m3-1024: 미측정 — BGE-M3 미측정" in printed


def test_축_실행도_live면_리랭크를_함께_과금한다(monkeypatch: Any, tmp_path: Path) -> None:
    """플래그의 뜻이 자리마다 달라지면 4단 사다리를 기대한 실행이 조용히 3단이 된다.

    실제로 그랬다 — 축 경로만 `--live` 를 무시해 리랭크 단이 "미측정 + 사유"로 남았고,
    `--rerank-with-openai` 의 "--live 는 이미 포함한다"는 안내가 거짓이었다.
    """
    calls = _capture_run(monkeypatch, tmp_path)

    def fake_axis(**kwargs: Any) -> Any:
        evaluate = cast(Any, kwargs["evaluate"])
        candidate = DEFAULT_EMBEDDING_CANDIDATES[0]
        return EmbeddingAxisResult(
            rows=(
                EmbeddingAxisRow(
                    candidate=candidate,
                    measured=True,
                    reports=evaluate(candidate, None),
                    reason=None,
                ),
            )
        )

    monkeypatch.setattr(compare_retrieval, "run_embedding_model_axis", fake_axis)

    assert compare_retrieval.main(["--live", "--embedding-axis", "--out-dir", str(tmp_path)]) == 0

    assert calls[0]["paid_rerank"] is True


class _FakeLocalEmbedder:
    MODEL = "BAAI/bge-m3"
    DIMENSIONS = 1024

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS
