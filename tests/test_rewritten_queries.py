"""골든셋 전용 재작성 질의 로더와 런타임 격리 하드 게이트."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from reply_gate.evaluation import DEFAULT_GOLDEN_SET_PATH, load_golden_set
from reply_gate.retrieval_eval import (
    DEFAULT_ORACLE_REWRITTEN_QUERIES_PATH,
    DEFAULT_REWRITTEN_QUERIES_PATH,
    load_rewritten_queries,
)

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_DIR = _ROOT / "src" / "reply_gate"
#: 평가 입력을 **읽는 것이 일인** 모듈만 면제한다. 나머지 패키지 파일은 전부 검사 대상이며,
#: 목록을 손으로 관리하지 않는다 — 새 런타임 모듈이 조용히 검사 밖으로 나가지 않게 한다.
_EVALUATION_MODULES = frozenset(
    {
        "__init__.py",
        "evaluation.py",
        "retrieval_eval.py",
        "retrieval_labels.py",
    }
)


def _runtime_module_names() -> tuple[str, ...]:
    """패키지 디렉터리에서 검사 대상 런타임 모듈을 유도한다."""
    return tuple(
        sorted(
            path.name for path in _PACKAGE_DIR.glob("*.py") if path.name not in _EVALUATION_MODULES
        )
    )


_FORBIDDEN_MODULES = frozenset(
    {
        "reply_gate.retrieval_eval",
        "reply_gate.retrieval_labels",
        "reply_gate.rewritten_queries",
    }
)
_FORBIDDEN_SYMBOLS = frozenset(
    {
        "DEFAULT_ORACLE_REWRITTEN_QUERIES_PATH",
        "DEFAULT_RETRIEVAL_LABELS_PATH",
        "DEFAULT_REWRITTEN_QUERIES_PATH",
        "RetrievalLabel",
        "load_retrieval_labels",
        "load_rewritten_queries",
    }
)
_FORBIDDEN_PATHS = (
    "retrieval_labels.jsonl",
    "rewritten_queries.jsonl",
    "rewritten_queries_oracle.jsonl",
)
_DYNAMIC_IMPORTERS = frozenset({"import_module", "__import__"})
#: 정답 조항에만 있는 값 — 어휘 정규화로는 나올 수 없고, 정답을 봐야만 재작성에 실린다.
_POLICY_ONLY_VALUES = (
    "policy:",
    "7일",
    "14일",
    "30,000",
    "50,000",
    "3,000",
    "4,000",
    "6,000",
    "09:00",
    "18:00",
    "12:00",
    "13:00",
    "2영업일",
    "3영업일",
    "1%",
    "1년",
)
_CLAUSE_NUMBER = re.compile(r"[1-4]-[1-7]")
#: f-string 의 치환부는 알 수 없는 값이다. 상수 조각만 이 표식으로 이어 붙여 우연한 결합을 막는다.
_UNKNOWN_SEGMENT = "\x00"


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
    if isinstance(node, ast.JoinedStr):
        # f-string: 상수 조각만 모은다. 치환부가 섞여 있어도 상수 조각에 금지 파일명이
        # 남아 있으면 잡힌다.
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                parts.append(_UNKNOWN_SEGMENT)
        return "".join(parts)
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
                # 동적 import 로 금지 모듈을 우회하는 경로를 막는다.
                importer = (
                    function.id
                    if isinstance(function, ast.Name)
                    else function.attr
                    if isinstance(function, ast.Attribute)
                    else ""
                )
                if importer in _DYNAMIC_IMPORTERS:
                    for argument in node.args:
                        folded = _fold_path_string(argument)
                        if folded is not None and folded in _FORBIDDEN_MODULES:
                            violations.append(f"{filename}: 금지 모듈 동적 import {folded}")
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
            if folded_path is not None and any(path in folded_path for path in _FORBIDDEN_PATHS):
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
    # G17 은 이 사이클의 표적 케이스다 — 픽스처가 조용히 재생성되면 여기서 걸린다.
    assert rewrites["G17"] == "고객센터 상담원 전화 연결 방법 및 대표번호"


def test_기본_blind는_생성_모델_산출이고_oracle은_별도_조건으로_보존한다() -> None:
    """blind 는 사람이 다듬은 문장이 아니다 — 정답을 모르는 재작성기의 실제 산출이다.

    사람이 쓰면 "어디까지가 의미 보존이고 어디부터가 정답 어휘인가"를 사람이 판정하게 되고,
    그 판정이 blind 조건의 이득에 그대로 실린다(결정 0010).
    """
    blind_rows = {row["id"]: row for row in _rows()}
    blind = load_rewritten_queries(DEFAULT_REWRITTEN_QUERIES_PATH)
    oracle = load_rewritten_queries(DEFAULT_ORACLE_REWRITTEN_QUERIES_PATH)

    assert len(blind) == len(oracle) == 30
    notes = {row["note"] for row in blind_rows.values()}
    assert len(notes) == 1, "blind 픽스처는 한 번의 생성 실행 산출이어야 한다"
    note = next(iter(notes))
    assert isinstance(note, str)
    assert "생성 모델" in note and "라벨을 입력으로 받지 않았다" in note

    # 두 조건이 실제로 갈라져 있어야 blind 대비 oracle gap 이 의미를 갖는다.
    assert sum(blind[case_id] != oracle[case_id] for case_id in blind) >= 25
    assert oracle["G03"] == "상품 교환 가능 사유와 반품 후 재주문 조건"
    # 정답이 빈 케이스의 상한은 "아무것도 채택하지 않는 것" 이므로 원문을 유지한다.
    assert oracle["G28"] == "환불 처리해 주세요."


@pytest.mark.parametrize(
    ("case_id", "forbidden"),
    [
        ("G12", "카드"),
        ("G16", "운영시간"),
        ("G17", "운영 안내"),
        ("G25", "주문번호"),
        ("G26", "영업일"),
        ("G28", "기간"),
        ("G29", "본인 확인"),
        ("G30", "본인 확인"),
    ],
)
def test_blind_재작성은_원문에_없는_답이나_전제를_주입하지_않는다(
    case_id: str, forbidden: str
) -> None:
    rewrites = load_rewritten_queries()

    assert forbidden not in rewrites[case_id]


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


def test_blind_재작성에는_정답_조항_고유값이_없다() -> None:
    """어휘 정규화는 blind 재작성기가 실제로 하는 일이지만, 고유값은 정답을 봐야만 나온다.

    구어체를 문서체로 옮기며 정답 조항과 같은 낱말을 쓰는 것 자체는 누출이 아니다 —
    이 픽스처를 만든 모델은 정책 문서를 입력으로 받은 적이 없다. 갈라야 하는 것은
    **정답 조항에만 있는 값**(조항 번호·근거 ID·정책 고유 수치)이고, 그것만 여기서 막는다.
    """
    rewrites = load_rewritten_queries()

    for case_id, text in rewrites.items():
        for value in _POLICY_ONLY_VALUES:
            assert value not in text, f"{case_id}: 정책 고유값 {value!r} 이 재작성에 있다"
        assert not _CLAUSE_NUMBER.search(text), f"{case_id}: 조항 번호가 재작성에 있다"


def test_blind_재작성_생성기는_정답과_정책_원문을_읽지_않는다() -> None:
    """blind 조건은 생성기의 입력 목록이 지킨다 — 픽스처를 다시 만들 때도 같아야 한다."""
    source = (_ROOT / "scripts" / "generate_blind_rewrites.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    referenced: set[str] = set()
    for node in ast.walk(tree):
        # import 만 하고 쓰지 않아도 잡는다 — 쓰임(Name)만 보면 반입 자체가 빠져나간다.
        if isinstance(node, ast.ImportFrom):
            referenced.add(node.module or "")
            referenced.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            referenced.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
    assert not referenced & _FORBIDDEN_SYMBOLS
    assert not referenced & _FORBIDDEN_MODULES
    assert not referenced & {
        "DEFAULT_POLICY_DIR",
        "load_policy_documents",
        "reply_gate.policy_index",
    }

    for node in ast.walk(tree):
        folded = _fold_path_string(node)
        if folded is None:
            continue
        assert "retrieval_labels" not in folded
        assert "rewritten_queries_oracle" not in folded
        assert "policies" not in folded


def test_런타임_실행_경로는_재작성_평가_픽스처와_격리된다() -> None:
    sources = {
        filename: (_PACKAGE_DIR / filename).read_text(encoding="utf-8")
        for filename in _runtime_module_names()
    }

    _assert_runtime_isolated(sources)


def test_격리_검사_대상은_패키지에서_유도되어_새_모듈을_자동으로_덮는다() -> None:
    """검사 대상이 손으로 관리되는 목록이면 새 런타임 모듈이 조용히 빠져나간다."""
    checked = set(_runtime_module_names())
    present = {path.name for path in _PACKAGE_DIR.glob("*.py")}

    # 면제 목록에 있는 파일만 빠지고, 그 밖의 모든 패키지 파일이 검사된다.
    assert checked == present - _EVALUATION_MODULES
    # 런타임에서 같은 구현을 재사용한다고 선언한 전략 모듈도 대상에 들어 있어야 한다.
    assert "retrieval_strategies.py" in checked
    assert "gate.py" in checked
    # 면제 목록은 실제로 존재하는 파일만 담는다 — 오타로 검사 구멍을 만들지 않는다.
    assert present >= _EVALUATION_MODULES


@pytest.mark.parametrize(
    "mutant",
    [
        "from reply_gate.retrieval_eval import load_rewritten_queries\nload_rewritten_queries()\n",
        "from reply_gate import retrieval_eval\nretrieval_eval.load_rewritten_queries()\n",
        "DEFAULT_REWRITTEN_QUERIES_PATH = 'data/elsewhere.jsonl'\n",
        "fixture = Path('data/rewritten_queries.jsonl')\n",
        "from .retrieval_eval import load_rewritten_queries as load\nload()\n",
        "from .retrieval_eval import DEFAULT_REWRITTEN_QUERIES_PATH as path\nfixture = path\n",
        "from .retrieval_eval import DEFAULT_ORACLE_REWRITTEN_QUERIES_PATH as path\nfixture = path\n",
        'fixture = Path("data") / ("rewritten_" + "queries.jsonl")\n',
        'fixture = Path("data") / ("rewritten_queries_" + "oracle.jsonl")\n',
        # 정답 라벨은 재작성문보다 강한 치트다 — 같은 자격으로 막는다.
        "from reply_gate.retrieval_labels import load_retrieval_labels\nload_retrieval_labels()\n",
        "from reply_gate.retrieval_labels import DEFAULT_RETRIEVAL_LABELS_PATH as p\nx = p\n",
        "fixture = Path('data/retrieval_labels.jsonl')\n",
        # f-string 과 동적 import 우회.
        'fixture = Path(f"data/{prefix}retrieval_labels.jsonl")\n',
        'fixture = f"data/rewritten_queries.jsonl"\n',
        'module = importlib.import_module("reply_gate.retrieval_eval")\n',
        'module = __import__("reply_gate.retrieval_labels")\n',
    ],
)
def test_런타임_격리_검사는_금지_참조_변이를_RED로_잡는다(mutant: str) -> None:
    with pytest.raises(AssertionError):
        _assert_runtime_isolated({"pipeline.py": mutant})
