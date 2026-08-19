"""정답에 닿는 경로를 런타임·전략 코드에서 통째로 끊는다 — 치트 경계의 구조 테스트.

`tests/test_rewritten_queries.py` 가 **검색 정답 라벨과 재작성 픽스처**를 막았지만
그것만으로는 치트가 닫히지 않는다. 정답은 세 곳에 있다:

1. `data/retrieval_labels.jsonl` — 검색 정답 조항.
2. `data/golden_set.jsonl` — 케이스별 기대 결과.
3. **정책 원문의 심은 주석** — `data/policies/shipping.md`·`refund.md` 의
   `<!-- planted: conflicting; note: ... -->` 는 **짝의 조항 번호까지 적는다.** 즉 주석
   자체가 정답이고, `policy_index` 가 `PolicyChunk.planted`·`planted_note` 로 1급 노출한다.
   라벨만 막으면 이 문이 열린 채로 남는다.

그래서 이 테스트는 **패키지에서 유도한 런타임 모듈 전부**를 AST 로 훑어 셋 다 막는다.
면제는 둘뿐이고 각각 이유가 다르다:

- **채점자 모듈** — 정답을 읽는 것이 일이다. 채점자는 정답을 봐도 되고 전략만 못 본다
  (`docs/standards.md` 의 "검색 관련성과 답변 충분성은 다른 계약이다").
- **`policy_index.py`** — 주석을 **파싱하는** 소유자다. 심은 주석 계열만 면제되고
  골든셋·라벨 계열은 그대로 걸린다. 그리고 파서가 문을 열어 두지 않는다는 것은 면제가
  아니라 별도 검사가 지킨다: 런타임 검색이 돌려받는 `PolicySearchHit` 에도, 적재 테이블
  DDL 에도 `planted` 가 없다.

음성 대조를 함께 둔다 — 항상 통과하는 빈 검사는 게이트가 아니다.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path

import pytest

from reply_gate.policy_index import PolicySearchHit, load_policy_documents

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_DIR = _ROOT / "src" / "reply_gate"
_POLICY_DIR = _ROOT / "data" / "policies"
_SCHEMA = _ROOT / "db" / "schema.sql"

#: 정답을 **읽는 것이 일인** 채점자 모듈. 손으로 관리하는 목록이지만 아래 테스트가
#: "목록의 파일이 실제로 존재하고, 그 밖의 모든 패키지 파일이 검사된다" 를 못박는다.
_SCORER_MODULES = frozenset(
    {
        "__init__.py",
        "adoption_axis.py",
        "evaluation.py",
        "retrieval_eval.py",
        "retrieval_labels.py",
    }
)

#: 심은 주석의 **파서**. 이 파일만 심은 주석 계열이 면제된다.
_ANNOTATION_OWNER = "policy_index.py"

#: 골든셋·검색 정답 라벨 — 어느 모듈에서도(파서 포함) 런타임 경로가 만지면 안 된다.
_ANSWER_SYMBOLS = frozenset(
    {
        "DEFAULT_GOLDEN_SET_PATH",
        "DEFAULT_RETRIEVAL_LABELS_PATH",
        "RetrievalLabel",
        "load_golden_set",
        "load_retrieval_labels",
    }
)
_ANSWER_MODULES = frozenset({"reply_gate.evaluation", "reply_gate.retrieval_labels"})
_ANSWER_PATHS = ("golden_set.jsonl", "retrieval_labels.jsonl")

#: 심은 장치 — 주석 그 자체가 정답이다. 파서만 면제된다.
_PLANTED_SYMBOLS = frozenset(
    {
        "DEFAULT_POLICY_DIR",
        "PlantedKind",
        "PolicyChunk",
        "PolicyDocument",
        "load_policy_documents",
        "parse_policy_document",
        "planted",
        "planted_note",
    }
)
_PLANTED_PATHS = ("data/policies", "policies/")
#: 주석을 직접 다시 파싱하는 경로도 막는다 — 문자열 상수에 표식이 있으면 잡는다.
_PLANTED_MARKER = "planted"

_DYNAMIC_IMPORTERS = frozenset({"import_module", "__import__"})
#: f-string 치환부는 알 수 없는 값이다. 상수 조각만 이 표식으로 이어 붙인다.
_UNKNOWN_SEGMENT = "\x00"


def _runtime_module_names() -> tuple[str, ...]:
    """검사 대상을 패키지 디렉터리에서 유도한다 — 새 런타임 모듈이 조용히 빠지지 않게."""
    return tuple(
        sorted(path.name for path in _PACKAGE_DIR.glob("*.py") if path.name not in _SCORER_MODULES)
    )


def _normalized_from_module(node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    if node.level > 1:
        return node.module or ""
    return f"reply_gate.{node.module}" if node.module else "reply_gate"


def _fold_path_string(node: ast.AST) -> str | None:
    """금지 경로 판정에 필요한 문자열·Path 조립만 접는다."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
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


def _violations(sources: Mapping[str, str]) -> tuple[str, ...]:
    """AST 에서 정답 계열 import·호출·속성·상수·경로 참조를 찾는다."""
    found: list[str] = []
    for filename, source in sources.items():
        planted_exempt = filename == _ANNOTATION_OWNER
        symbols = _ANSWER_SYMBOLS if planted_exempt else _ANSWER_SYMBOLS | _PLANTED_SYMBOLS
        paths = _ANSWER_PATHS if planted_exempt else _ANSWER_PATHS + _PLANTED_PATHS

        tree = ast.parse(source, filename=filename)
        module_aliases: set[str] = set()
        symbol_aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _ANSWER_MODULES:
                        module_aliases.add(alias.asname or alias.name.split(".")[-1])
                        found.append(f"{filename}: 금지 모듈 import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = _normalized_from_module(node)
                if module in _ANSWER_MODULES:
                    found.append(f"{filename}: 금지 모듈 import {module}")
                    for alias in node.names:
                        module_aliases.add(alias.asname or alias.name)
                elif module == "reply_gate":
                    for alias in node.names:
                        if f"reply_gate.{alias.name}" in _ANSWER_MODULES:
                            module_aliases.add(alias.asname or alias.name)
                            found.append(f"{filename}: 금지 모듈 import reply_gate.{alias.name}")
                for alias in node.names:
                    if alias.name in symbols:
                        symbol_aliases[alias.asname or alias.name] = alias.name
                        found.append(f"{filename}: 금지 심볼 import {alias.name}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function = node.func
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
                        if folded is not None and folded in _ANSWER_MODULES:
                            found.append(f"{filename}: 금지 모듈 동적 import {folded}")
            if isinstance(node, ast.Name) and (node.id in symbols or node.id in symbol_aliases):
                found.append(f"{filename}: 금지 참조 {symbol_aliases.get(node.id, node.id)}")
            elif isinstance(node, ast.Attribute) and (
                node.attr in symbols
                or (isinstance(node.value, ast.Name) and node.value.id in module_aliases)
            ):
                found.append(f"{filename}: 금지 참조 {node.attr}")
            elif isinstance(node, ast.keyword) and node.arg in symbols:
                found.append(f"{filename}: 금지 참조 {node.arg}")
            folded_path = _fold_path_string(node)
            if folded_path is None:
                continue
            if any(candidate in folded_path for candidate in paths):
                found.append(f"{filename}: 금지 경로 참조")
            if not planted_exempt and _PLANTED_MARKER in folded_path.lower():
                found.append(f"{filename}: 심은 주석 표식 문자열")
    return tuple(dict.fromkeys(found))


def _assert_isolated(sources: Mapping[str, str]) -> None:
    assert not _violations(sources)


def test_런타임과_전략_코드는_라벨_골든셋_심은_주석_어디에도_닿지_않는다() -> None:
    sources = {
        filename: (_PACKAGE_DIR / filename).read_text(encoding="utf-8")
        for filename in _runtime_module_names()
    }

    _assert_isolated(sources)


def test_검사_대상은_패키지에서_유도되고_면제_목록에_구멍이_없다() -> None:
    checked = set(_runtime_module_names())
    present = {path.name for path in _PACKAGE_DIR.glob("*.py")}

    assert checked == present - _SCORER_MODULES
    # 전략·런타임의 핵심 파일이 실제로 검사 안에 있어야 이 가드가 의미를 갖는다.
    assert {"retrieval_strategies.py", "evidence.py", "pipeline.py", "gate.py"} <= checked
    assert _ANNOTATION_OWNER in checked
    # 면제 목록은 존재하는 파일만 담는다 — 오타 하나가 검사 구멍이 된다.
    assert present >= _SCORER_MODULES


def test_심은_주석은_파서_밖으로_나가지_못한다() -> None:
    """면제는 파서 한 곳뿐이고, 그 파서도 주석을 밖으로 흘리지 않는다."""
    hit_fields = {field.name for field in fields(PolicySearchHit)}

    # 런타임 검색이 돌려받는 것에는 심은 장치가 없다.
    assert not {name for name in hit_fields if "planted" in name}
    # 적재 테이블에도 없다 — DB 를 경유하는 우회로도 막혀 있다.
    schema = _SCHEMA.read_text(encoding="utf-8")
    table = schema.split("CREATE TABLE IF NOT EXISTS policy_chunks", 1)[1].split(");", 1)[0]
    assert "planted" not in table


def test_채점자_모듈은_런타임_경로에서_import_되지_않는다() -> None:
    """정답을 아는 채점자가 런타임에 끌려 들어오면 면제가 그대로 구멍이 된다."""
    offenders: list[str] = []
    for filename in _runtime_module_names():
        source = (_PACKAGE_DIR / filename).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source, filename=filename)):
            if isinstance(node, ast.ImportFrom):
                module = _normalized_from_module(node)
            elif isinstance(node, ast.Import):
                module = node.names[0].name
            else:
                continue
            scorer = module.removeprefix("reply_gate.") + ".py"
            if scorer in _SCORER_MODULES:
                offenders.append(f"{filename} -> {module}")

    assert not offenders


def test_심은_상충_주석이_짝의_조항_번호를_그대로_들고_있다() -> None:
    """가드가 지키는 대상이 실제로 정답인지 확인한다 — 동시에 평가 장치 보존 검사다."""
    notes = [
        chunk.planted_note
        for document in load_policy_documents(_POLICY_DIR)
        for chunk in document.chunks
        if chunk.planted_note is not None and chunk.planted.value == "conflicting"
    ]

    assert len(notes) >= 2
    # 주석이 짝의 조항 번호를 적는다 = 주석을 읽으면 상충쌍 정답을 그냥 얻는다.
    assert any(note is not None and "1-3" in note for note in notes)
    assert any(note is not None and "2-1" in note for note in notes)


@pytest.mark.parametrize(
    "mutant",
    [
        # 골든셋 — 케이스별 기대 결과 그 자체.
        "from reply_gate.evaluation import load_golden_set\nload_golden_set()\n",
        "from reply_gate.evaluation import DEFAULT_GOLDEN_SET_PATH as p\nx = p\n",
        "from reply_gate import evaluation\nevaluation.load_golden_set()\n",
        "fixture = Path('data/golden_set.jsonl')\n",
        'fixture = Path(f"data/{prefix}golden_set.jsonl")\n',
        'module = importlib.import_module("reply_gate.evaluation")\n',
        # 검색 정답 라벨.
        "from reply_gate.retrieval_labels import load_retrieval_labels\nload_retrieval_labels()\n",
        "fixture = Path('data') / 'retrieval_labels.jsonl'\n",
        # 심은 주석 — 라벨 금지만으로는 안 막히던 문.
        "from reply_gate.policy_index import PlantedKind\nx = PlantedKind\n",
        "from .policy_index import load_policy_documents\nload_policy_documents()\n",
        "from .policy_index import PolicyChunk\nx = PolicyChunk\n",
        "hint = chunk.planted_note\n",
        "if chunk.planted is not None:\n    pass\n",
        "boost = score(planted=True)\n",
        "corpus = Path('data') / 'policies'\n",
        "PLANTED = re.compile(r'<!--\\\\s*planted:')\n",
        "from .policy_index import DEFAULT_POLICY_DIR as d\nx = d\n",
    ],
)
def test_가드는_금지_참조_변이를_RED로_잡는다(mutant: str) -> None:
    """음성 대조 — 전략이 정답을 만졌다면 이 테스트가 **실제로** 깨진다."""
    with pytest.raises(AssertionError):
        _assert_isolated({"retrieval_strategies.py": mutant})


@pytest.mark.parametrize(
    "mutant",
    [
        "from reply_gate.policy_index import PlantedKind\nx = PlantedKind\n",
        "hint = chunk.planted_note\n",
        "corpus = Path('data') / 'policies'\n",
    ],
)
def test_파서_면제는_심은_주석_계열에만_적용된다(mutant: str) -> None:
    """같은 변이가 파서에서는 통과하고 전략에서는 걸린다 — 면제의 범위가 이것이다."""
    _assert_isolated({_ANNOTATION_OWNER: mutant})

    with pytest.raises(AssertionError):
        _assert_isolated({"evidence.py": mutant})


def test_파서_면제도_골든셋과_라벨은_막는다() -> None:
    """면제는 심은 주석까지다 — 파서가 정답 라벨을 읽는 것은 여전히 금지다."""
    with pytest.raises(AssertionError):
        _assert_isolated({_ANNOTATION_OWNER: "from reply_gate.evaluation import load_golden_set\n"})
