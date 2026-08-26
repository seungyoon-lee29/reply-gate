"""구조 검사 — 패턴형 개인정보 정의의 **단독 소유자는 게이트 모듈 하나**다.

접기가 층마다 달라 L1 이 뚫렸다. 근거 렌더는 접기 **전**으로 걸러 계산 컬럼의 전각 번호를
`evidence_text` 에 남겼고, 게이트는 그것을 접어 반각 번호와 맞춰 지어낸 값을 통과시켰다.
그 사고의 원인은 접기 함수가 아니라 **정의가 층마다 따로 서 있었다는 것**이고, 그래서
정의를 한 곳으로 모은 뒤에는 그 상태를 실행이 아니라 **코드 모양**으로 지켜야 한다.

소비자는 셋이다 — 게이트(소유자) · 근거 렌더의 개인정보 필터 · 조회 가드. 소비자가 자기
정규식이나 자기 접기를 새로 만들면 저장소에 개인정보 정의가 여럿이 되고, 한쪽만 넓혀지는
순간 기준이 다시 갈린다. 이 파일은 그 형태를 AST 로 막는다.

`tests/AGENTS.md` 의 구조 검사 규율을 그대로 따른다: 검사 대상 목록은 **재귀로 유도**하고
목록이 비면 실패한다(불변식 4) · `node.names` 를 전부 훑는다(불변식 5) · 소스를 문자열로
스캔해 가드를 켜고 끄지 않는다(불변식 7) · 음성 대조를 같은 파일에 둔다(불변식 3).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from reply_gate import gate

_PACKAGE_DIR = Path(gate.__file__).resolve().parent
_GATE = "gate.py"
_GATE_MODULE = "reply_gate.gate"

#: 게이트가 단독으로 정의하는 이름들. 소비자는 **가져다 쓰기만** 한다.
CANONICAL_NAMES: frozenset[str] = frozenset(
    {
        "DEFAULT_PII_PATTERNS",
        "NUMERIC_SEPARATOR_VARIANTS",
        "PiiPattern",
        "fold_for_detection",
        "fold_numeric_for_detection",
        "pii_shaped",
    }
)

#: 탐지 전 접기를 짜려면 반드시 거쳐야 하는 모듈. 게이트 밖에서 이것을 부르는 것이
#: "자기 접기를 새로 만든다"의 가장 흔한 모양이다.
_FOLDING_MODULE = "unicodedata"


def _package_sources() -> list[tuple[str, str]]:
    """`(패키지 기준 상대 경로, 소스)` — **재귀**로 유도한다.

    비재귀 `glob("*.py")` 은 하위 패키지를 통째로 놓쳐, 하나 생기는 순간 그 안의 코드가
    조용히 검사 밖으로 나간다(`tests/AGENTS.md` 불변식 4).
    """
    return [
        (path.relative_to(_PACKAGE_DIR).as_posix(), path.read_text(encoding="utf-8"))
        for path in sorted(_PACKAGE_DIR.rglob("*.py"))
    ]


# ── 검사기 (음성 대조가 같은 함수를 부른다) ─────────────────────────────────


def defined_names(source: str) -> set[str]:
    """모듈이 **스스로 정의하는** 최상위·중첩 이름 전부 (함수·클래스·대입).

    중첩까지 보는 이유: 함수 안에 `def pii_shaped(...)` 를 다시 두는 것이 정의를 늘리는
    가장 조용한 방법이다 — 실제로 근거 필터가 그 모양의 지역 함수를 들고 있었다.
    import 로 들여온 이름은 정의가 아니므로 세지 않는다.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in ast.walk(node):
                if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def redefined_canonical(source: str) -> set[str]:
    """모듈이 다시 정의한 정본 이름. **앞의 밑줄은 벗겨서 본다.**

    실제 사고가 그 모양이었다 — 근거 필터가 `_pii_shaped` 라는 지역 함수로 같은 판정을
    한 벌 더 들고 있었고, 이름이 한 글자 달라 어떤 가드에도 걸리지 않았다. 비공개
    표기는 정의를 늘리지 않았다는 뜻이 아니다.
    """
    return {name for name in defined_names(source) if name.lstrip("_") in CANONICAL_NAMES}


def constructed_patterns(source: str) -> int:
    """`PiiPattern(...)` 을 몇 번 만드는가. 소유자 밖에서 1회라도 만들면 정의가 둘이다."""
    return sum(
        1
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "PiiPattern")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "PiiPattern")
        )
    )


def imported_modules(source: str) -> set[str]:
    """import 한 모듈 전체 경로. `import a, b` 의 둘째 이름도 놓치지 않는다."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def names_imported_from_gate(source: str) -> set[str]:
    """`from reply_gate.gate import …` 로 들여온 이름 (별칭이 붙어도 원래 이름을 센다)."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module == _GATE_MODULE:
            names.update(alias.name for alias in node.names)
    return names


def referenced_names(source: str) -> set[str]:
    """소스가 **이름으로** 언급하는 식별자. 주석·docstring 은 AST 에 없어 애초에 안 걸린다."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


# ── 양성 검사 — 저장소의 실제 상태 ──────────────────────────────────────────


def test_검사_대상_목록이_비어_있지_않다() -> None:
    """목록이 비면 아래 가드들이 전부 조용히 초록이 된다."""
    sources = _package_sources()

    assert len(sources) > 1
    assert _GATE in {name for name, _source in sources}


def test_PII_패턴은_게이트_모듈에서만_만들어진다() -> None:
    """소유권 — 다른 층이 자기 패턴을 만들면 저장소에 개인정보 정의가 둘이 된다."""
    offenders = {
        name: constructed_patterns(source)
        for name, source in _package_sources()
        if name != _GATE and constructed_patterns(source)
    }

    assert offenders == {}
    assert constructed_patterns(Path(gate.__file__).read_text(encoding="utf-8")) == len(
        gate.DEFAULT_PII_PATTERNS
    )


def test_정본_이름은_게이트_밖에서_다시_정의되지_않는다() -> None:
    """소비자는 **가져다 쓰기만** 한다 — 같은 이름을 자기 모듈에서 다시 짜면 기준이 갈린다."""
    offenders = {
        name: sorted(redefined_canonical(source))
        for name, source in _package_sources()
        if name != _GATE and redefined_canonical(source)
    }

    assert offenders == {}


def test_게이트는_정본_이름을_전부_들고_공개한다() -> None:
    """반대 방향 — 소유자가 실제로 그 이름들을 갖고 있어야 위 가드가 의미를 갖는다."""
    source = Path(gate.__file__).read_text(encoding="utf-8")

    assert defined_names(source) >= CANONICAL_NAMES
    assert set(gate.__all__) >= CANONICAL_NAMES


def test_정본_이름을_쓰는_모듈은_게이트에서_가져온다() -> None:
    """소비자 목록을 손으로 관리하지 않는다 — 이름을 언급하는 모듈에서 **유도**한다.

    지역에서 같은 이름을 만들어 쓰면 위 가드가 잡고, 다른 모듈을 거쳐 우회하면 이 가드가
    잡는다. 두 방향을 함께 막아야 "정의는 하나"가 성립한다.
    """
    consumers: dict[str, list[str]] = {}
    for name, source in _package_sources():
        if name == _GATE:
            continue
        used = referenced_names(source) & CANONICAL_NAMES
        if not used:
            continue
        missing = sorted(used - names_imported_from_gate(source))
        consumers[name] = missing

    assert consumers, "정본 이름을 쓰는 소비자가 하나도 없으면 계약이 성립하지 않는다"
    assert {name: missing for name, missing in consumers.items() if missing} == {}


def test_탐지_전_접기는_게이트_모듈에서만_구현된다() -> None:
    """접기를 짜려면 유니코드 범주를 봐야 한다 — 그 도구가 게이트 밖으로 나가면 층이 는다."""
    users = {
        name for name, source in _package_sources() if _FOLDING_MODULE in imported_modules(source)
    }

    assert users == {_GATE}


# ── 음성 대조 — 검사기가 실제로 잡는다 ──────────────────────────────────────

_MUTANT_OWN_PATTERN = """
import re
from reply_gate.gate import PiiPattern

MY_PATTERNS = (PiiPattern(name="phone", regex=re.compile(r"010"), fold=str, normalize=str),)
"""

_MUTANT_REDEFINED_FOLD = """
import unicodedata


def fold_numeric_for_detection(text: str) -> str:
    return unicodedata.normalize("NFKC", text)
"""

_MUTANT_NESTED_HELPER = """
from reply_gate.gate import DEFAULT_PII_PATTERNS


def render(rows):
    def pii_shaped(text: str) -> bool:
        return any(p.regex.search(text) for p in DEFAULT_PII_PATTERNS)

    return [row for row in rows if not pii_shaped(row)]
"""

#: 실제로 뚫렸던 모양 — 이름 앞에 밑줄만 붙여 같은 판정을 한 벌 더 들고 있었다.
_MUTANT_PRIVATE_HELPER = """
from reply_gate.gate import DEFAULT_PII_PATTERNS, fold_for_detection


def render(rows):
    def _pii_shaped(text: str) -> bool:
        folded = fold_for_detection(text)
        return any(p.regex.search(folded) for p in DEFAULT_PII_PATTERNS)

    return [row for row in rows if not _pii_shaped(row)]
"""

_MUTANT_LOCAL_ALIAS = """
from reply_gate.contracts import Evidence

DEFAULT_PII_PATTERNS = ()


def check(text):
    return any(p.regex.search(text) for p in DEFAULT_PII_PATTERNS)
"""

_MUTANT_IMPORT_PAIR = """
import json, unicodedata
"""


def test_가드는_소유자_밖의_패턴_생성을_RED로_잡는다() -> None:
    assert constructed_patterns(_MUTANT_OWN_PATTERN) == 1


def test_가드는_접기_재정의를_RED로_잡는다() -> None:
    assert "fold_numeric_for_detection" in redefined_canonical(_MUTANT_REDEFINED_FOLD)
    assert _FOLDING_MODULE in imported_modules(_MUTANT_REDEFINED_FOLD)


def test_가드는_함수_안에_숨은_재정의도_RED로_잡는다() -> None:
    """중첩을 안 보면 지역 함수 하나로 정의가 조용히 둘이 된다 — 실제로 그 모양이었다."""
    assert redefined_canonical(_MUTANT_NESTED_HELPER) == {"pii_shaped"}
    assert redefined_canonical(_MUTANT_PRIVATE_HELPER) == {"_pii_shaped"}


def test_가드는_게이트에서_가져오지_않은_동명_이름을_RED로_잡는다() -> None:
    used = referenced_names(_MUTANT_LOCAL_ALIAS) & CANONICAL_NAMES

    assert used == {"DEFAULT_PII_PATTERNS"}
    assert used - names_imported_from_gate(_MUTANT_LOCAL_ALIAS) == {"DEFAULT_PII_PATTERNS"}


def test_가드는_한_줄에_묶인_import_의_둘째_이름도_본다() -> None:
    """`import a, b` 에서 첫 이름만 보면 둘째부터가 가드 밖이다(`tests/AGENTS.md` 불변식 5)."""
    assert imported_modules(_MUTANT_IMPORT_PAIR) == {"json", _FOLDING_MODULE}


def test_가드는_정상_소비자를_통과시킨다() -> None:
    """양성 대조 — 전부 걸러내는 검사기는 검사기가 아니다."""
    healthy = "from reply_gate.gate import pii_shaped\n\n\ndef f(t):\n    return pii_shaped(t)\n"

    assert constructed_patterns(healthy) == 0
    assert redefined_canonical(healthy) == set()
    assert names_imported_from_gate(healthy) >= referenced_names(healthy) & CANONICAL_NAMES
    assert _FOLDING_MODULE not in imported_modules(healthy)


@pytest.mark.parametrize(
    "mutant",
    [
        pytest.param(_MUTANT_OWN_PATTERN, id="자기_패턴_생성"),
        pytest.param(_MUTANT_REDEFINED_FOLD, id="접기_재정의"),
        pytest.param(_MUTANT_NESTED_HELPER, id="중첩_재정의"),
        pytest.param(_MUTANT_PRIVATE_HELPER, id="밑줄_붙인_재정의"),
        pytest.param(_MUTANT_LOCAL_ALIAS, id="지역_동명_이름"),
    ],
)
def test_변이는_적어도_한_가드에_걸린다(mutant: str) -> None:
    """가드 넷을 한 판정으로 묶어, 어느 하나가 느슨해져도 이 검사가 남는다."""
    caught = (
        constructed_patterns(mutant) > 0
        or bool(redefined_canonical(mutant))
        or bool((referenced_names(mutant) & CANONICAL_NAMES) - names_imported_from_gate(mutant))
        or _FOLDING_MODULE in imported_modules(mutant)
    )

    assert caught
