"""골든셋 전용 재작성 질의 로더와 런타임 격리 하드 게이트."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from reply_gate.evaluation import DEFAULT_GOLDEN_SET_PATH, load_golden_set
from reply_gate.retrieval_eval import (
    DEFAULT_REWRITTEN_QUERIES_PATH,
    load_rewritten_queries,
)

_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_MODULES = (
    "api.py",
    "config.py",
    "contracts.py",
    "db.py",
    "draft.py",
    "evidence.py",
    "gate.py",
    "judge.py",
    "llm.py",
    "order_ref.py",
    "pipeline.py",
    "policy_index.py",
    "records.py",
    "sql_guard.py",
)
_FORBIDDEN_MODULES = frozenset(
    {
        "reply_gate.retrieval_eval",
        "reply_gate.rewritten_queries",
    }
)
_FORBIDDEN_SYMBOLS = frozenset(
    {
        "DEFAULT_REWRITTEN_QUERIES_PATH",
        "load_rewritten_queries",
    }
)
_FORBIDDEN_PATH = "rewritten_queries.jsonl"


def _rows() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in DEFAULT_REWRITTEN_QUERIES_PATH.read_text(encoding="utf-8").splitlines()
    ]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _normalized_from_module(node: ast.ImportFrom) -> str:
    """테스트 대상 패키지 파일의 상대 import를 절대 모듈명으로 바꾼다."""
    if node.level == 0:
        return node.module or ""
    package_parts = ["reply_gate"]
    ascents = node.level - 1
    if ascents > len(package_parts):
        return node.module or ""
    base_parts = package_parts[: len(package_parts) - ascents]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def _fold_path_string(node: ast.AST) -> str | None:
    """금지 픽스처 경로 판정에 필요한 문자열·Path 조립만 접는다."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _fold_path_string(node.left)
        right = _fold_path_string(node.right)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.Call):
        function = node.func
        is_path = (isinstance(function, ast.Name) and function.id == "Path") or (
            isinstance(function, ast.Attribute) and function.attr == "Path"
        )
        if is_path and len(node.args) == 1 and not node.keywords:
            return _fold_path_string(node.args[0])
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _fold_path_string(node.left)
        right = _fold_path_string(node.right)
        if left is not None and right is not None:
            return f"{left.rstrip('/')}/{right.lstrip('/')}"
    return None


def _runtime_isolation_violations(sources: Mapping[str, str]) -> tuple[str, ...]:
    """AST에서 금지 import·호출·상수·픽스처 경로 참조를 찾는다."""
    violations: list[str] = []
    for filename, source in sources.items():
        tree = ast.parse(source, filename=filename)
        forbidden_module_aliases: set[str] = set()
        forbidden_symbol_aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_MODULES:
                        forbidden_module_aliases.add(alias.asname or alias.name.split(".")[-1])
                        violations.append(f"{filename}: 금지 모듈 import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = _normalized_from_module(node)
                if module in _FORBIDDEN_MODULES:
                    violations.append(f"{filename}: 금지 모듈 import {module}")
                    for alias in node.names:
                        local_name = alias.asname or alias.name
                        if alias.name in _FORBIDDEN_SYMBOLS:
                            forbidden_symbol_aliases[local_name] = alias.name
                elif module == "reply_gate":
                    for alias in node.names:
                        imported = f"reply_gate.{alias.name}"
                        if imported in _FORBIDDEN_MODULES:
                            forbidden_module_aliases.add(alias.asname or alias.name)
                            violations.append(f"{filename}: 금지 모듈 import {imported}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function = node.func
                if isinstance(function, ast.Name) and (
                    function.id in _FORBIDDEN_SYMBOLS
                    or function.id in forbidden_symbol_aliases
                    or function.id in forbidden_module_aliases
                ):
                    violations.append(f"{filename}: 금지 로더 호출 {function.id}")
                elif isinstance(function, ast.Attribute) and (
                    function.attr in _FORBIDDEN_SYMBOLS
                    or (
                        isinstance(function.value, ast.Name)
                        and function.value.id in forbidden_module_aliases
                    )
                ):
                    violations.append(f"{filename}: 금지 로더 호출 {function.attr}")
            elif isinstance(node, ast.Name) and (
                node.id in _FORBIDDEN_SYMBOLS or node.id in forbidden_symbol_aliases
            ):
                symbol = forbidden_symbol_aliases.get(node.id, node.id)
                violations.append(f"{filename}: 금지 상수/로더 참조 {symbol}")
            elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_SYMBOLS:
                violations.append(f"{filename}: 금지 상수/로더 참조 {node.attr}")
            folded_path = _fold_path_string(node)
            if folded_path is not None and _FORBIDDEN_PATH in folded_path:
                violations.append(f"{filename}: 금지 픽스처 경로 참조")
    return tuple(dict.fromkeys(violations))


def _assert_runtime_isolated(sources: Mapping[str, str]) -> None:
    assert not _runtime_isolation_violations(sources)


def test_저장소_재작성_질의_30건을_골든셋과_정확히_대응해_읽는다() -> None:
    cases = load_golden_set(DEFAULT_GOLDEN_SET_PATH)
    rewrites = load_rewritten_queries(DEFAULT_REWRITTEN_QUERIES_PATH)

    assert len(rewrites) == len(cases) == 30
    assert list(rewrites) == [case.id for case in cases]
    assert set(_rows()[0]) == {"id", "original", "rewritten", "note"}
    assert rewrites["G17"] == (
        "고객센터 전화 상담을 위해 상담원과 통화할 수 있는 전화번호와 운영 안내"
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.pop(), r"재작성 픽스처에서 빠진 골든셋 ID.*G30"),
        (
            lambda rows: rows.append(
                {"id": "G99", "original": "없는 문의", "rewritten": "없는 문의", "note": "추가"}
            ),
            r"골든셋에 없는 ID가 재작성 픽스처에 있다.*G99",
        ),
    ],
)
def test_골든셋과_재작성_ID가_어느_방향으로든_다르면_거부한다(
    tmp_path: Path,
    mutate: Callable[[list[dict[str, object]]], object],
    message: str,
) -> None:
    path = tmp_path / "id-mismatch.jsonl"
    rows = _rows()
    mutate(rows)
    _write_rows(path, rows)

    with pytest.raises(ValueError, match=message):
        load_rewritten_queries(path)


def test_재작성_ID가_중복되면_거부한다(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-id.jsonl"
    rows = _rows()
    rows[-1]["id"] = "G01"
    _write_rows(path, rows)

    with pytest.raises(ValueError, match=r"재작성 ID가 중복.*G01"):
        load_rewritten_queries(path)


def test_재작성_행의_키_집합이_계약과_다르면_거부한다(tmp_path: Path) -> None:
    path = tmp_path / "extra-key.jsonl"
    rows = _rows()
    rows[0]["category"] = "normal"
    _write_rows(path, rows)

    with pytest.raises(ValueError, match=r"행 키가 계약과 다르다.*category"):
        load_rewritten_queries(path)


def test_original이_골든셋_content와_다르면_거부한다(tmp_path: Path) -> None:
    path = tmp_path / "wrong-original.jsonl"
    rows = _rows()
    rows[0]["original"] = "원문을 바꿨다"
    _write_rows(path, rows)

    with pytest.raises(ValueError, match=r"original이 골든셋 content와 다르다.*G01"):
        load_rewritten_queries(path)


@pytest.mark.parametrize("bad_value", ["", "   ", None])
def test_rewritten이_비어_있거나_문자열이_아니면_거부한다(
    tmp_path: Path, bad_value: object
) -> None:
    path = tmp_path / "empty-rewritten.jsonl"
    rows = _rows()
    rows[0]["rewritten"] = bad_value
    _write_rows(path, rows)

    with pytest.raises(ValueError, match=r"rewritten은 비어 있지 않은 문자열.*G01"):
        load_rewritten_queries(path)


def test_재작성이_불필요하면_원문과_정확히_같은_문자열을_허용한다() -> None:
    cases = {case.id: case for case in load_golden_set()}
    rewrites = load_rewritten_queries()

    assert rewrites["G21"] == cases["G21"].content
    assert rewrites["G22"] == cases["G22"].content
    assert rewrites["G23"] == cases["G23"].content
    assert rewrites["G24"] == cases["G24"].content


def test_런타임_실행_경로는_재작성_평가_픽스처와_격리된다() -> None:
    package_dir = _ROOT / "src" / "reply_gate"
    sources = {
        filename: (package_dir / filename).read_text(encoding="utf-8")
        for filename in _RUNTIME_MODULES
    }

    _assert_runtime_isolated(sources)


@pytest.mark.parametrize(
    "mutant",
    [
        "from reply_gate.retrieval_eval import load_rewritten_queries\nload_rewritten_queries()\n",
        "from reply_gate import retrieval_eval\nretrieval_eval.load_rewritten_queries()\n",
        "DEFAULT_REWRITTEN_QUERIES_PATH = 'data/elsewhere.jsonl'\n",
        "fixture = Path('data/rewritten_queries.jsonl')\n",
        "from .retrieval_eval import load_rewritten_queries as load\nload()\n",
        "from .retrieval_eval import DEFAULT_REWRITTEN_QUERIES_PATH as path\nfixture = path\n",
        'fixture = Path("data") / ("rewritten_" + "queries.jsonl")\n',
    ],
)
def test_런타임_격리_검사는_금지_참조_변이를_RED로_잡는다(mutant: str) -> None:
    with pytest.raises(AssertionError):
        _assert_runtime_isolated({"pipeline.py": mutant})
