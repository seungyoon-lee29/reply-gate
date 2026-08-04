"""text-to-SQL 안전장치 2·3 단위 테스트 (spec "text-to-SQL 안전장치" 절).

여기서 검증하는 것은 **실행 전에 끝나는 코드 검증**이다. DB 권한(안전장치 1)은
`tests/test_db_readonly.py` 가 따로 검증한다 — 두 층은 서로를 대체하지 않는다.

전부 거부하는 검증기는 검증기가 아니므로 양성 대조(정상 SELECT 통과)를 함께 둔다.
"""

from __future__ import annotations

import psycopg
import pytest
from psycopg.rows import DictRow

from reply_gate.sql_guard import (
    ORDERS_TABLE,
    SCHEMA_WHITELIST,
    SqlGuardRejection,
    describe_whitelist,
    validate_sql,
)

MAX_ROWS = 50
ORDER_NO = "ORD-20260201-0001"

OK_SQL = (
    "SELECT order_no, status, courier, tracking_no, shipped_at "
    f"FROM orders WHERE order_no = '{ORDER_NO}'"
)


def _reject(sql: str, *, max_rows: int = MAX_ROWS) -> SqlGuardRejection:
    with pytest.raises(SqlGuardRejection) as excinfo:
        validate_sql(sql, max_rows=max_rows)
    return excinfo.value


# ── 양성 대조 ───────────────────────────────────────────────────────────────


def test_정상_SELECT_는_통과한다() -> None:
    validated = validate_sql(OK_SQL, max_rows=MAX_ROWS)

    assert "orders" in validated.sql
    assert ORDER_NO in validated.sql
    assert validated.tables == ("orders",)
    assert validated.max_rows == MAX_ROWS


def test_별표는_화이트리스트_전체_컬럼과_같으므로_통과한다() -> None:
    """`*` 는 orders 의 전체 컬럼으로 펼쳐지고 그 집합이 곧 화이트리스트다."""
    validated = validate_sql(f"SELECT * FROM orders WHERE order_no = '{ORDER_NO}'", max_rows=10)

    assert validated.tables == ("orders",)


def test_집계와_별칭과_서브쿼리도_화이트리스트_안이면_통과한다() -> None:
    validate_sql("SELECT count(*) AS n FROM orders", max_rows=MAX_ROWS)
    validate_sql("SELECT order_no AS 주문번호 FROM orders ORDER BY 주문번호", max_rows=MAX_ROWS)
    validate_sql(
        "SELECT order_no FROM orders WHERE order_no IN "
        f"(SELECT order_no FROM orders WHERE status = '배송중' AND order_no = '{ORDER_NO}')",
        max_rows=MAX_ROWS,
    )
    validate_sql(
        f"WITH t AS (SELECT order_no, status FROM orders WHERE order_no = '{ORDER_NO}')"
        " SELECT order_no, status FROM t",
        max_rows=MAX_ROWS,
    )


def test_스키마_한정_public_은_통과한다() -> None:
    validate_sql(
        f"SELECT o.order_no FROM public.orders o WHERE o.order_no = '{ORDER_NO}'", max_rows=5
    )


# ── 비SELECT 거부 ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO orders (order_no) VALUES ('ORD-20260201-0002')",
        "UPDATE orders SET status = '취소'",
        "DELETE FROM orders",
        "DROP TABLE orders",
        "ALTER TABLE orders ADD COLUMN x text",
        "TRUNCATE TABLE orders",
        "CREATE TABLE t (a int)",
        "GRANT SELECT ON orders TO someone",
        "COPY orders TO STDOUT",
    ],
    ids=[
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "GRANT",
        "COPY",
    ],
)
def test_SELECT_가_아니면_거부한다(sql: str) -> None:
    assert _reject(sql).rule == "not_select"


def test_SELECT_INTO_는_테이블을_만들므로_거부한다() -> None:
    """루트는 SELECT 지만 Postgres 에서 `SELECT ... INTO` 는 테이블을 생성한다."""
    assert _reject("SELECT * INTO t FROM orders").rule == "forbidden_statement"


def test_UNION_은_단일_SELECT_가_아니므로_거부한다() -> None:
    assert _reject("SELECT order_no FROM orders UNION SELECT article FROM policy_chunks").rule == (
        "not_select"
    )


def test_CTE_안에_숨긴_INSERT_도_거부한다() -> None:
    """Postgres 는 data-modifying CTE 를 지원한다 — 루트가 SELECT 라고 안전한 게 아니다."""
    sql = (
        "WITH x AS (INSERT INTO orders (order_no) VALUES ('ORD-20260201-0002') RETURNING order_no)"
        " SELECT order_no FROM x"
    )

    assert _reject(sql).rule == "forbidden_statement"


# ── 다중문·주석 우회 거부 ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; DROP TABLE orders",
        f"SELECT order_no FROM orders WHERE order_no = '{ORDER_NO}'; DELETE FROM orders",
    ],
)
def test_다중문은_거부한다(sql: str) -> None:
    assert _reject(sql).rule == "multiple_statements"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders -- ; DROP TABLE orders",
        "SELECT * FROM orders /* ; DROP TABLE orders */",
        "SELECT /* inquiries */ order_no FROM orders",
        "-- 주석으로 시작\nSELECT order_no FROM orders",
    ],
)
def test_주석은_거부한다(sql: str) -> None:
    """주석 안에 문장을 숨기면 파서가 못 보므로, 주석 자체를 거부해 우회로를 닫는다."""
    assert _reject(sql).rule == "comment"


# ── 화이트리스트 (안전장치 2) ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM inquiries",
        "SELECT content FROM policy_chunks",
        "SELECT tablename FROM pg_catalog.pg_tables",
        "SELECT table_name FROM information_schema.tables",
    ],
)
def test_화이트리스트_밖_테이블은_거부한다(sql: str) -> None:
    assert _reject(sql).rule == "unknown_table"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT order_no FROM orders WHERE order_no IN (SELECT order_no FROM inquiries)",
        "WITH x AS (SELECT content FROM policy_chunks) SELECT content FROM x",
        "SELECT o.order_no FROM orders o JOIN inquiries i ON i.order_no = o.order_no",
    ],
)
def test_CTE_서브쿼리_조인에_숨긴_테이블도_거부한다(sql: str) -> None:
    assert _reject(sql).rule == "unknown_table"


def test_화이트리스트_밖_컬럼은_거부한다() -> None:
    assert _reject("SELECT order_no, secret_col FROM orders").rule == "unknown_column"


def test_서브쿼리_안의_화이트리스트_밖_컬럼도_거부한다() -> None:
    sql = (
        "SELECT order_no FROM orders WHERE order_no IN "
        "(SELECT order_no FROM orders WHERE secret_col = 1)"
    )

    assert _reject(sql).rule == "unknown_column"


def test_화이트리스트_테이블을_하나도_안_쓰면_거부한다() -> None:
    """`SELECT pg_sleep(5)` 처럼 테이블 없는 쿼리로 DB 를 건드리는 경로를 닫는다."""
    assert _reject("SELECT 1").rule == "no_whitelisted_table"
    assert _reject("SELECT pg_sleep(5)").rule == "no_whitelisted_table"


@pytest.mark.parametrize(
    "sql",
    ["SELECT * FROM generate_series(1, 10)", "SELECT * FROM pg_sleep(5)"],
)
def test_FROM_절의_테이블_함수는_거부한다(sql: str) -> None:
    assert _reject(sql).rule == "unknown_table"


@pytest.mark.parametrize(
    "sql",
    ["SELECT FROM WHERE ~~~ (((", "not sql at all", "SELECT order_no FROM"],
)
def test_파싱되지_않는_문자열은_거부한다(sql: str) -> None:
    assert _reject(sql).rule == "parse_error"


def test_빈_SQL_은_거부한다() -> None:
    assert _reject("   ").rule == "empty"


# ── 결과 행 수 상한 (안전장치 3) ────────────────────────────────────────────


def test_LIMIT_이_없으면_상한을_붙인다() -> None:
    """거부가 아니라 **강제**다 — 상한 없는 쿼리를 그대로 실행시키지 않는다."""
    validated = validate_sql(OK_SQL, max_rows=7)

    assert validated.limit == 7
    assert "LIMIT 7" in validated.sql.upper()


def test_상한_이하의_LIMIT_은_그대로_둔다() -> None:
    validated = validate_sql(f"{OK_SQL} LIMIT 3", max_rows=50)

    assert validated.limit == 3
    assert "LIMIT 3" in validated.sql.upper()


def test_상한을_넘는_LIMIT_은_상한으로_낮춘다() -> None:
    validated = validate_sql(f"{OK_SQL} LIMIT 5000", max_rows=50)

    assert validated.limit == 50
    assert "5000" not in validated.sql


def test_정수_리터럴이_아닌_LIMIT_은_거부한다() -> None:
    assert _reject(f"{OK_SQL} LIMIT (SELECT count(*) FROM orders)").rule == "unsupported_limit"


def test_상한이_0_이하이면_설정_오류다() -> None:
    with pytest.raises(ValueError):
        validate_sql(OK_SQL, max_rows=0)


# ── 화이트리스트 정의 자체 ──────────────────────────────────────────────────


def test_화이트리스트는_orders_한_테이블뿐이다() -> None:
    """read-only 계정이 SELECT 권한을 가진 테이블도 orders 하나다 — 두 층이 같아야 한다."""
    assert set(SCHEMA_WHITELIST) == {ORDERS_TABLE}


def test_프롬프트용_설명에_테이블과_모든_컬럼이_들어간다() -> None:
    described = describe_whitelist()

    assert ORDERS_TABLE in described
    for column in SCHEMA_WHITELIST[ORDERS_TABLE]:
        assert column in described


@pytest.mark.db
def test_화이트리스트가_실제_orders_컬럼과_일치한다(app_conn: psycopg.Connection[DictRow]) -> None:
    """스키마가 바뀌었는데 화이트리스트가 안 따라오면 정상 쿼리가 조용히 거부된다."""
    rows = app_conn.execute(
        "SELECT column_name FROM information_schema.columns"
        " WHERE table_schema = 'public' AND table_name = %s",
        (ORDERS_TABLE,),
    ).fetchall()

    assert {row["column_name"] for row in rows} == set(SCHEMA_WHITELIST[ORDERS_TABLE])
