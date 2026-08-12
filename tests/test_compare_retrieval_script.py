"""검색 비교 CLI가 공개하는 전략·재작성 조건 계약."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from scripts import compare_retrieval

from reply_gate.retrieval_eval import ReportPaths, RewriteCondition
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
