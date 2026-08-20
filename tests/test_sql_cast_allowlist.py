"""조회 가드의 **형변환 대상 타입 허용 목록** — 그리고 그 목록이 생성 안내와 같다는 구조 검사.

형변환은 허용 함수인데 **대상 타입을 한 번도 읽지 않았다.** 시스템 카탈로그 참조 타입
(`regclass`·`regtype`·`regproc`·`oid`)으로 변환하면 화이트리스트 밖의 **이름과 존재 여부**가
결과 컬럼으로 나온다 — `cast('inquiries' AS regclass)` 는 그 테이블이 있다는 사실을 돌려주고,
없으면 오류로 그것을 알려준다. 읽기 전용 계정은 `pg_catalog` 을 읽을 수 있으므로 **이 경로를
막는 층은 가드 하나뿐**이다(아래 `db` 마커 검사가 그 사실을 실측으로 든다).

허용 목록은 **가드가 단독 소유**하고 생성 안내가 그 문면을 그대로 싣는다. 갈리면 생성기가
목록 밖 타입을 쓸 때마다 거부 → 재시도 1회 → **인계**가 되고, 그 인계가 기준선 수치가 된다.
그래서 "같다"를 실행이 아니라 **구조로** 지킨다(`tests/AGENTS.md` 불변식 3·4·7).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import DictRow

from reply_gate import sql_guard
from reply_gate.evidence import build_sql_user_prompt
from reply_gate.sql_guard import (
    ALLOWED_CAST_TYPES,
    SqlGuardRejection,
    describe_allowed_cast_types,
    describe_allowed_functions,
    validate_sql,
)

MAX_ROWS = 50
ORDER_NO = "ORD-20260201-0001"
SCOPE = f"WHERE order_no = '{ORDER_NO}'"

_PACKAGE_DIR = Path(sql_guard.__file__).resolve().parent
_GUARD = "sql_guard.py"

#: 가드 밖에서 다시 정의하면 목록이 둘이 되는 이름들.
CANONICAL_NAMES: frozenset[str] = frozenset({"ALLOWED_CAST_TYPES", "describe_allowed_cast_types"})


def _reject(sql: str) -> SqlGuardRejection:
    with pytest.raises(SqlGuardRejection) as excinfo:
        validate_sql(sql, order_no=ORDER_NO, max_rows=MAX_ROWS)
    return excinfo.value


# ── 양성 — 안내가 광고하는 용도(숫자·일자·문자열)는 전부 통과한다 ───────────


def test_허용_목록이_비어_있지_않다() -> None:
    """목록이 비면 아래 검사들이 전부 조용히 초록이 된다."""
    assert ALLOWED_CAST_TYPES
    assert len(ALLOWED_CAST_TYPES) == len(set(ALLOWED_CAST_TYPES))


@pytest.mark.parametrize("type_name", ALLOWED_CAST_TYPES)
def test_허용_목록의_모든_타입은_통과한다(type_name: str) -> None:
    """전부 거부하는 검사는 검사가 아니다 — 목록에서 **유도한** 양성 대조다."""
    validate_sql(
        f"SELECT cast(quantity AS {type_name}) AS q, order_no FROM orders {SCOPE}",
        order_no=ORDER_NO,
        max_rows=MAX_ROWS,
    )


def test_허용_타입의_축약_표기와_길이_지정도_통과한다() -> None:
    for sql in (
        f"SELECT quantity::text AS q, order_no FROM orders {SCOPE}",
        f"SELECT cast(product_name AS varchar(20)) AS n, order_no FROM orders {SCOPE}",
        f"SELECT cast(unit_price_krw AS numeric(12, 2)) AS p, order_no FROM orders {SCOPE}",
        f"SELECT cast(cast(ordered_at AS date) AS text) AS d, order_no FROM orders {SCOPE}",
    ):
        validate_sql(sql, order_no=ORDER_NO, max_rows=MAX_ROWS)


# ── 음성 — 목록 밖 타입은 fail-closed 로 거부된다 ───────────────────────────


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            f"SELECT cast('inquiries' AS regclass) AS x, order_no FROM orders {SCOPE}",
            id="regclass_테이블_존재_확인",
        ),
        pytest.param(
            f"SELECT quantity::oid::regclass AS x, order_no FROM orders {SCOPE}",
            id="oid_regclass_연쇄",
        ),
        pytest.param(
            f"SELECT cast(quantity AS regtype) AS x, order_no FROM orders {SCOPE}",
            id="regtype",
        ),
        pytest.param(
            f"SELECT cast(quantity AS regproc) AS x, order_no FROM orders {SCOPE}",
            id="regproc",
        ),
        pytest.param(
            f"SELECT cast(quantity AS regprocedure) AS x, order_no FROM orders {SCOPE}",
            id="regprocedure",
        ),
        pytest.param(
            f"SELECT cast(quantity AS oid) AS x, order_no FROM orders {SCOPE}",
            id="oid",
        ),
        pytest.param(
            f"SELECT cast(product_name AS name) AS x, order_no FROM orders {SCOPE}",
            id="name",
        ),
        pytest.param(
            f"SELECT cast(quantity AS xid) AS x, order_no FROM orders {SCOPE}",
            id="xid",
        ),
        pytest.param(
            f"SELECT cast(product_name AS pg_catalog.regclass) AS x, order_no FROM orders {SCOPE}",
            id="스키마_한정_regclass",
        ),
        pytest.param(
            f"SELECT cast(product_name AS citext) AS x, order_no FROM orders {SCOPE}",
            id="사용자_정의_타입",
        ),
        pytest.param(
            f"SELECT cast(product_name AS bytea) AS x, order_no FROM orders {SCOPE}",
            id="바이너리",
        ),
        pytest.param(
            f"SELECT cast(product_option AS jsonb) AS x, order_no FROM orders {SCOPE}",
            id="jsonb",
        ),
        pytest.param(
            f"SELECT order_no FROM orders {SCOPE} AND cast('policy_chunks' AS regclass) IS NOT NULL",
            id="WHERE_절_경유",
        ),
        pytest.param(
            "WITH t AS (SELECT cast('inquiries' AS regclass) AS x, order_no "
            f"FROM orders {SCOPE}) SELECT x FROM t",
            id="임시테이블_경유",
        ),
        pytest.param(
            "SELECT coalesce(product_name, cast('inquiries' AS regclass)) AS x, order_no "
            f"FROM orders {SCOPE}",
            id="빈칸채우기_경유",
        ),
        pytest.param(
            f"SELECT order_no FROM orders {SCOPE} ORDER BY cast('orders' AS regclass)",
            id="ORDER_BY_경유",
        ),
    ],
)
def test_허용_목록_밖_대상_타입은_거부한다(sql: str) -> None:
    """목록에 없으면 거부 — 거부 목록이 아니라 허용 목록이다(fail-closed)."""
    rejection = _reject(sql)

    assert rejection.rule == "unsupported_cast_type"


def test_거부_사유가_재생성_프롬프트에_실을_안내를_들고_나온다() -> None:
    """사유는 **무엇을 어떻게 고쳐야 하는지**를 담아야 재시도 1회가 살아난다."""
    rejection = _reject(f"SELECT cast('inquiries' AS regclass) AS x, order_no FROM orders {SCOPE}")

    assert describe_allowed_cast_types() in rejection.detail
    # 무엇을 썼다가 거부됐는지도 들고 나와야 모델이 그 자리를 고칠 수 있다.
    assert "REGCLASS" in rejection.detail


# ── 구조 검사 — 안내에 실린 목록이 가드의 목록과 같다 ───────────────────────


def types_present(text: str) -> tuple[str, ...]:
    """`text` 가 언급하는 허용 타입 표기 — **가드의 목록에서 유도한다.**

    손으로 관리하는 목록을 대조하면 가드에 타입을 더할 때 검사가 따라오지 않는다
    (`tests/AGENTS.md` 불변식 4). 낱말 경계로 본다 — 맨 부분 문자열로 찾으면 `bigint` 가
    `int` 를 대신 켜 줘서, 목록에서 빠진 타입을 가드가 못 잡는다.
    """
    return tuple(
        name for name in ALLOWED_CAST_TYPES if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text)
    )


def _sql_prompt() -> str:
    return build_sql_user_prompt(inquiry="배송 언제 되나요", order_no=ORDER_NO, max_rows=MAX_ROWS)


def test_생성_안내가_가드의_목록을_문면_그대로_싣는다() -> None:
    """가드와 안내가 갈리면 목록 밖 타입 → 거부 → 재시도 → **인계**가 수치에 남는다."""
    prompt = _sql_prompt()

    assert describe_allowed_cast_types() in prompt
    assert types_present(prompt) == ALLOWED_CAST_TYPES


def test_허용_함수_설명이_캐스트_목록을_함께_들고_다닌다() -> None:
    """생성 안내의 `[허용 함수]` 절이 곧 이 문자열이다 — 안내 쪽에 목록을 따로 두지 않는다."""
    described = describe_allowed_functions()

    assert "cast" in described
    assert describe_allowed_cast_types() in described
    assert "regclass" not in described


def _package_sources() -> list[tuple[str, str]]:
    """`(패키지 기준 상대 경로, 소스)` — **재귀**로 유도하고, 비면 검사가 실패한다."""
    return [
        (path.relative_to(_PACKAGE_DIR).as_posix(), path.read_text(encoding="utf-8"))
        for path in sorted(_PACKAGE_DIR.rglob("*.py"))
    ]


def defined_names(source: str) -> set[str]:
    """모듈이 스스로 정의하는 최상위·중첩 이름 (함수·클래스·대입)."""
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
    """정본 이름을 다시 정의했나. **앞의 밑줄은 벗겨서 본다.**"""
    return {name for name in defined_names(source) if name.lstrip("_") in CANONICAL_NAMES}


def test_검사_대상_목록이_비어_있지_않다() -> None:
    sources = _package_sources()

    assert len(sources) > 1
    assert _GUARD in {name for name, _source in sources}


def test_캐스트_허용_목록은_가드_밖에서_다시_정의되지_않는다() -> None:
    """소유자는 가드 하나다 — 안내는 그 문면을 부르기만 한다."""
    offenders = {
        name: sorted(redefined_canonical(source))
        for name, source in _package_sources()
        if name != _GUARD and redefined_canonical(source)
    }

    assert offenders == {}
    assert defined_names(Path(sql_guard.__file__).read_text(encoding="utf-8")) >= CANONICAL_NAMES
    assert set(sql_guard.__all__) >= CANONICAL_NAMES


# ── 음성 대조 — 검사기가 실제로 무언가를 잡는다 ─────────────────────────────

#: 안내가 목록을 손으로 옮겨 적다 한 줄을 빠뜨린 모양.
_MUTANT_PROMPT = "[허용 캐스트 대상 타입]\n" + ", ".join(ALLOWED_CAST_TYPES[:-1])

#: 안내 쪽 모듈이 자기 목록을 새로 든 모양.
_MUTANT_OWN_LIST = """
ALLOWED_CAST_TYPES = ("text", "regclass")
"""

#: 밑줄 하나로 이름을 비껴간 모양 — 실제 사고가 그 모양이었다.
_MUTANT_PRIVATE_LIST = """
_ALLOWED_CAST_TYPES = ("text", "regclass")
"""


def test_가드는_목록에서_빠진_타입을_RED로_잡는다() -> None:
    assert types_present(_MUTANT_PROMPT) != ALLOWED_CAST_TYPES
    assert set(ALLOWED_CAST_TYPES) - set(types_present(_MUTANT_PROMPT)) == {ALLOWED_CAST_TYPES[-1]}


def test_가드는_안내_쪽의_자기_목록을_RED로_잡는다() -> None:
    assert redefined_canonical(_MUTANT_OWN_LIST) == {"ALLOWED_CAST_TYPES"}
    assert redefined_canonical(_MUTANT_PRIVATE_LIST) == {"_ALLOWED_CAST_TYPES"}


def test_가드는_정상_소비자를_통과시킨다() -> None:
    """양성 대조 — 전부 걸러내는 검사기는 검사기가 아니다."""
    healthy = (
        "from reply_gate.sql_guard import describe_allowed_cast_types\n\n\n"
        "def section():\n    return describe_allowed_cast_types()\n"
    )

    assert redefined_canonical(healthy) == set()
    assert types_present(describe_allowed_cast_types()) == ALLOWED_CAST_TYPES


# ── 안전장치 1은 이 경로를 막지 못한다 (실측) ───────────────────────────────


@pytest.mark.db
def test_읽기_전용_계정은_카탈로그_참조_타입을_막지_못한다(
    ro_conn: psycopg.Connection[DictRow],
) -> None:
    """**막는 층이 가드 하나뿐**이라는 것을 실측으로 든다 (docs/security.md 1층 서술의 근거).

    read-only 계정은 `orders` 밖 **업무 테이블**에는 권한이 없지만
    (`tests/test_db_readonly.py`), 시스템 카탈로그는 PUBLIC 권한이라 읽힌다. 그래서
    화이트리스트 밖 테이블의 **이름과 존재 여부**가 이 캐스트로 결과에 실린다.
    """
    row = ro_conn.execute("SELECT cast('inquiries' AS regclass) AS leaked").fetchone()

    assert row is not None
    assert row["leaked"] == "inquiries"
