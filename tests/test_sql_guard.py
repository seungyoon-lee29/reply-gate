"""text-to-SQL 안전장치 2·3 단위 테스트 (spec "text-to-SQL 안전장치" 절).

여기서 검증하는 것은 **실행 전에 끝나는 코드 검증**이다. DB 권한(안전장치 1)은
`tests/test_db_readonly.py` 가, 실행 시간 상한(read-only 커넥션의 `statement_timeout`)은
`tests/test_evidence.py` 가 따로 검증한다 — 세 층은 서로를 대체하지 않는다.

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
    describe_allowed_functions,
    describe_whitelist,
    validate_sql,
)

MAX_ROWS = 50
ORDER_NO = "ORD-20260201-0001"
OTHER_ORDER_NO = "ORD-20260201-0002"

#: 선검사를 통과한 주문 1건으로 한정된 WHERE 절 — 모든 정상 쿼리에 붙는다.
SCOPE = f"WHERE order_no = '{ORDER_NO}'"

OK_SQL = f"SELECT order_no, status, courier, tracking_no, shipped_at FROM orders {SCOPE}"


def _ok(sql: str, *, max_rows: int = MAX_ROWS) -> None:
    validate_sql(sql, order_no=ORDER_NO, max_rows=max_rows)


def _reject(sql: str, *, max_rows: int = MAX_ROWS) -> SqlGuardRejection:
    with pytest.raises(SqlGuardRejection) as excinfo:
        validate_sql(sql, order_no=ORDER_NO, max_rows=max_rows)
    return excinfo.value


# ── 양성 대조 ───────────────────────────────────────────────────────────────


def test_정상_SELECT_는_통과한다() -> None:
    validated = validate_sql(OK_SQL, order_no=ORDER_NO, max_rows=MAX_ROWS)

    assert "orders" in validated.sql
    assert ORDER_NO in validated.sql
    assert validated.tables == ("orders",)
    assert validated.max_rows == MAX_ROWS
    assert validated.order_no == ORDER_NO


def test_별표는_화이트리스트_전체_컬럼과_같으므로_통과한다() -> None:
    """`*` 는 orders 의 전체 컬럼으로 펼쳐지고 그 집합이 곧 화이트리스트다."""
    validated = validate_sql(f"SELECT * FROM orders {SCOPE}", order_no=ORDER_NO, max_rows=10)

    assert validated.tables == ("orders",)


def test_집계와_별칭과_서브쿼리도_화이트리스트_안이면_통과한다() -> None:
    _ok(f"SELECT count(*) AS n FROM orders {SCOPE}")
    _ok(f"SELECT order_no AS 주문번호 FROM orders {SCOPE} ORDER BY 주문번호")
    _ok(
        f"SELECT order_no FROM orders {SCOPE} AND status IN "
        f"(SELECT status FROM orders WHERE order_no = '{ORDER_NO}')"
    )
    _ok(f"WITH t AS (SELECT order_no, status FROM orders {SCOPE}) SELECT order_no, status FROM t")
    _ok(f"SELECT a.order_no FROM (SELECT order_no FROM orders {SCOPE}) a")


def test_스키마_한정_public_은_통과한다() -> None:
    validate_sql(
        f"SELECT o.order_no FROM public.orders o WHERE o.order_no = '{ORDER_NO}'",
        order_no=ORDER_NO,
        max_rows=5,
    )


def test_선검사를_통과하지_않은_주문번호로_호출하는_것은_설정_오류다() -> None:
    """`order_no` 는 존재성 선검사를 통과한 정규화된 값이어야 한다 (호출 측 계약)."""
    with pytest.raises(ValueError):
        validate_sql(OK_SQL, order_no="주문번호아님", max_rows=MAX_ROWS)


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


# ── 잠금 절 (finding 4) ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "clause",
    ["FOR UPDATE", "FOR SHARE", "FOR NO KEY UPDATE", "FOR KEY SHARE"],
)
def test_행_잠금_절은_읽기_전용_SELECT_가_아니므로_거부한다(clause: str) -> None:
    """실행 시점의 권한 오류로 미루면 단 1회뿐인 재시도를 헛되이 태운다."""
    rejection = _reject(f"{OK_SQL} {clause}")

    assert rejection.rule == "locking_clause"
    assert "잠금" in rejection.detail


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
    assert _reject(f"SELECT order_no, secret_col FROM orders {SCOPE}").rule == "unknown_column"


def test_서브쿼리_안의_화이트리스트_밖_컬럼도_거부한다() -> None:
    sql = (
        f"SELECT order_no FROM orders {SCOPE} AND status IN "
        f"(SELECT status FROM orders WHERE order_no = '{ORDER_NO}' AND secret_col = 1)"
    )

    assert _reject(sql).rule == "unknown_column"


def test_FROM_절에_없는_한정자는_거부한다() -> None:
    assert _reject(f"SELECT x.order_no FROM orders {SCOPE}").rule == "unknown_table"


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


def test_재귀_CTE_는_실행_시간을_예측할_수_없으므로_거부한다() -> None:
    sql = "WITH RECURSIVE t(n) AS (SELECT 1 AS n) SELECT n FROM t"

    assert _reject(sql).rule == "unsupported_source"


@pytest.mark.parametrize(
    "sql",
    ["SELECT FROM WHERE ~~~ (((", "not sql at all", "SELECT order_no FROM"],
)
def test_파싱되지_않는_문자열은_거부한다(sql: str) -> None:
    assert _reject(sql).rule == "parse_error"


def test_빈_SQL_은_거부한다() -> None:
    assert _reject("   ").rule == "empty"


# ── 별칭이 컬럼 화이트리스트를 우회하지 못한다 (finding 2) ──────────────────


def test_같은_SELECT_목록의_별칭이_원본_컬럼_검사를_무력화하지_못한다() -> None:
    """리뷰가 실제로 통과시킨 문자열 — `ctid` 가 그대로 결과에 나왔다."""
    rejection = _reject("SELECT 1 AS ctid, ctid, order_no FROM orders LIMIT 3")

    assert rejection.rule == "unknown_column"
    assert "ctid" in rejection.detail


def test_테이블_컬럼_별칭_목록으로_화이트리스트를_갈아끼우지_못한다() -> None:
    """`orders AS o(ctid)` — 화이트리스트 컬럼을 임의 이름으로 바꿔 검사를 비껴간다."""
    rejection = _reject("SELECT o.ctid FROM orders AS o(ctid) LIMIT 3")

    assert rejection.rule == "unsupported_source"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT ctid FROM orders",
        "SELECT o.ctid FROM orders o",
        "SELECT xmin, order_no FROM orders",
        "SELECT tableoid FROM orders",
    ],
)
def test_시스템_컬럼은_한정_여부와_무관하게_거부한다(sql: str) -> None:
    assert _reject(sql).rule == "unknown_column"


def test_USING_절로도_화이트리스트_밖_컬럼을_내보내지_못한다() -> None:
    """PostgreSQL 은 USING 컬럼을 `SELECT *` 출력에 합쳐서 내보낸다."""
    sql = (
        "SELECT * FROM orders a JOIN orders b USING (ctid)"
        f" WHERE a.order_no = '{ORDER_NO}' AND b.order_no = '{ORDER_NO}'"
    )

    assert _reject(sql).rule == "unknown_column"


def test_양쪽_모두_묶인_자기_조인은_통과한다() -> None:
    """조인 자체를 막지는 않는다 — 막는 것은 묶이지 않은 쪽이다."""
    _ok(
        "SELECT a.order_no, b.status FROM orders a JOIN orders b ON b.order_no = a.order_no"
        f" WHERE a.order_no = '{ORDER_NO}' AND b.order_no = '{ORDER_NO}'"
    )


def test_부질의_안쪽_별칭은_바깥_스코프를_열지_않는다() -> None:
    sql = (
        f"SELECT ctid FROM orders {SCOPE} AND status IN "
        f"(SELECT status AS ctid FROM orders WHERE order_no = '{ORDER_NO}')"
    )

    assert _reject(sql).rule == "unknown_column"


def test_정상_별칭_사용은_통과한다() -> None:
    """PostgreSQL 이 출력 이름을 실제로 해석해 주는 자리에서는 별칭을 받아준다."""
    _ok(f"SELECT order_no AS 주문번호 FROM orders {SCOPE} ORDER BY 주문번호")
    _ok(f"SELECT status AS 상태, count(*) AS n FROM orders {SCOPE} GROUP BY 상태 ORDER BY n DESC")


def test_별칭은_ORDER_BY_GROUP_BY_밖에서는_이름으로_쓸_수_없다() -> None:
    """WHERE·HAVING 에서 맨 이름은 별칭이 아니라 입력 컬럼으로 해석된다 (PostgreSQL)."""
    rejection = _reject(
        f"SELECT status AS ctid FROM orders {SCOPE} GROUP BY status HAVING ctid IS NOT NULL"
    )

    assert rejection.rule == "unknown_column"


# ── 함수 허용 목록 (finding 3b) ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT order_no FROM orders {SCOPE} AND pg_sleep(2) IS NULL LIMIT 3",
        f"SELECT pg_sleep(2)::text AS s, order_no FROM orders {SCOPE}",
        f"SELECT order_no FROM orders {SCOPE} AND random() < 1",
        f"SELECT pg_read_file('/etc/passwd') AS f, order_no FROM orders {SCOPE}",
        f"SELECT order_no FROM orders {SCOPE} ORDER BY pg_sleep(2)",
    ],
    ids=["where", "select", "random", "pg_read_file", "order_by"],
)
def test_허용_목록_밖_함수는_실행_전에_거부한다(sql: str) -> None:
    """`orders` 를 끼워 넣으면 테이블·컬럼 검사만으로는 함수 호출을 못 잡는다."""
    rejection = _reject(sql)

    assert rejection.rule == "forbidden_function"
    # 사유는 재시도 프롬프트에 실려 모델이 고칠 수 있어야 한다.
    assert "count" in rejection.detail


def test_조회에_필요한_함수는_통과한다() -> None:
    """전부 거부하는 함수 검사는 검사가 아니다 — 양성 대조."""
    _ok(f"SELECT count(*) AS n FROM orders {SCOPE}")
    _ok(f"SELECT sum(quantity) AS q, max(ordered_at) AS last FROM orders {SCOPE}")
    _ok(f"SELECT to_char(ordered_at, 'YYYY-MM-DD') AS d FROM orders {SCOPE}")
    _ok(f"SELECT date_trunc('day', ordered_at) AS d FROM orders {SCOPE}")
    _ok(f"SELECT coalesce(courier, '미지정') AS c FROM orders {SCOPE}")
    _ok(f"SELECT upper(status) AS s, length(order_no) AS n FROM orders {SCOPE}")
    _ok(f"SELECT CASE WHEN status = '배송중' THEN 1 ELSE 0 END AS s FROM orders {SCOPE}")
    _ok(f"SELECT quantity::text AS q FROM orders {SCOPE}")
    _ok(f"SELECT order_no FROM orders {SCOPE} AND ordered_at > now() - interval '3 days'")


def test_허용_함수_설명은_프롬프트에_실을_수_있는_형태다() -> None:
    described = describe_allowed_functions()

    assert "count" in described
    assert "to_char" in described
    assert "pg_sleep" not in described


# ── 주문 1건 한정 (finding 1) ───────────────────────────────────────────────


def test_WHERE_가_없으면_거부한다() -> None:
    """리뷰가 실제로 통과·실행시킨 문자열 — 무관한 고객 50명의 연락처가 근거가 됐다."""
    rejection = _reject(
        "SELECT order_no, customer_name, customer_phone, customer_email FROM orders LIMIT 3"
    )

    assert rejection.rule == "order_scope"
    assert ORDER_NO in rejection.detail


def test_다른_주문번호를_가리키면_거부한다() -> None:
    assert _reject(f"SELECT order_no FROM orders WHERE order_no = '{OTHER_ORDER_NO}'").rule == (
        "order_scope"
    )


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT order_no FROM orders WHERE order_no = '{ORDER_NO}' OR order_no = '{OTHER_ORDER_NO}'",
        f"SELECT order_no FROM orders WHERE order_no = '{ORDER_NO}' OR 1 = 1",
        f"SELECT order_no FROM orders WHERE (order_no = '{ORDER_NO}' OR status = '배송중')",
        f"SELECT order_no FROM orders WHERE order_no IN ('{ORDER_NO}', '{OTHER_ORDER_NO}')",
        f"SELECT order_no FROM orders WHERE NOT (order_no <> '{ORDER_NO}')",
    ],
    ids=["or-다른주문", "or-항진", "괄호-or", "in-목록", "not-부등호"],
)
def test_OR_우회는_거부한다(sql: str) -> None:
    assert _reject(sql).rule == "order_scope"


def test_조인_한쪽만_묶으면_거부한다() -> None:
    """자기 조인의 반대편이 안 묶이면 500행이 그대로 딸려 나온다."""
    sql = (
        "SELECT o1.order_no, o2.customer_phone FROM orders o1, orders o2"
        f" WHERE o1.order_no = '{ORDER_NO}'"
    )

    assert _reject(sql).rule == "order_scope"


def test_묶이지_않은_부질의_스코프도_거부한다() -> None:
    sql = f"SELECT order_no FROM orders {SCOPE} AND status IN (SELECT status FROM orders)"

    assert _reject(sql).rule == "order_scope"


def test_CTE_안쪽이_묶이지_않으면_거부한다() -> None:
    sql = "WITH t AS (SELECT order_no, customer_phone FROM orders) SELECT order_no FROM t"

    assert _reject(sql).rule == "order_scope"


def test_주문_1건으로_묶인_쿼리는_통과한다() -> None:
    """양성 대조 — 정상 형태를 막으면 재시도 1회를 헛되이 태운다."""
    _ok(f"SELECT order_no, status FROM orders WHERE order_no = '{ORDER_NO}'")
    _ok(f"SELECT order_no FROM orders WHERE '{ORDER_NO}' = order_no")
    _ok(f"SELECT o.order_no FROM orders o WHERE o.order_no = '{ORDER_NO}'")
    _ok(f"SELECT order_no FROM orders WHERE order_no IN ('{ORDER_NO}')")
    _ok(f"SELECT order_no FROM orders WHERE status = '배송중' AND order_no = '{ORDER_NO}'")
    _ok(
        f"SELECT order_no FROM orders WHERE (status = '배송중' OR status = '배송완료') AND order_no = '{ORDER_NO}'"
    )


def test_한정_범위는_검증_결과에_남는다() -> None:
    validated = validate_sql(OK_SQL, order_no=ORDER_NO, max_rows=MAX_ROWS)

    assert validated.order_no == ORDER_NO


# ── 결과 행 수 상한 (안전장치 3) ────────────────────────────────────────────


def test_LIMIT_이_없으면_상한을_붙인다() -> None:
    """거부가 아니라 **강제**다 — 상한 없는 쿼리를 그대로 실행시키지 않는다."""
    validated = validate_sql(OK_SQL, order_no=ORDER_NO, max_rows=7)

    assert validated.limit == 7
    assert "LIMIT 7" in validated.sql.upper()


def test_상한_이하의_LIMIT_은_그대로_둔다() -> None:
    validated = validate_sql(f"{OK_SQL} LIMIT 3", order_no=ORDER_NO, max_rows=50)

    assert validated.limit == 3
    assert "LIMIT 3" in validated.sql.upper()


def test_상한을_넘는_LIMIT_은_상한으로_낮춘다() -> None:
    validated = validate_sql(f"{OK_SQL} LIMIT 5000", order_no=ORDER_NO, max_rows=50)

    assert validated.limit == 50
    assert "5000" not in validated.sql


def test_FETCH_FIRST_도_행_수_상한으로_받아들인다() -> None:
    """finding 5 — 예전에는 "LIMIT 은 정수 리터럴이어야 한다" 라는 엉뚱한 사유로 거부됐다."""
    validated = validate_sql(f"{OK_SQL} FETCH FIRST 5 ROWS ONLY", order_no=ORDER_NO, max_rows=50)

    assert validated.limit == 5
    assert "LIMIT 5" in validated.sql.upper()


def test_FETCH_FIRST_상한도_설정값으로_낮춘다() -> None:
    validated = validate_sql(f"{OK_SQL} FETCH FIRST 500 ROWS ONLY", order_no=ORDER_NO, max_rows=50)

    assert validated.limit == 50


@pytest.mark.parametrize(
    ("sql", "expected_in_detail"),
    [
        (f"{OK_SQL} FETCH FIRST 5 PERCENT ROWS ONLY", "PERCENT"),
        (f"{OK_SQL} FETCH FIRST 5 ROWS WITH TIES", "WITH TIES"),
    ],
    ids=["percent", "with-ties"],
)
def test_행_수를_확정할_수_없는_FETCH_는_정확한_사유로_거부한다(
    sql: str, expected_in_detail: str
) -> None:
    """사유가 정확해야 재시도가 올바른 방향으로 고친다."""
    rejection = _reject(sql)

    assert rejection.rule == "unsupported_limit"
    assert expected_in_detail in rejection.detail


def test_정수_리터럴이_아닌_LIMIT_은_거부한다() -> None:
    sql = f"{OK_SQL} LIMIT (SELECT count(*) FROM orders WHERE order_no = '{ORDER_NO}')"

    assert _reject(sql).rule == "unsupported_limit"


def test_상한이_0_이하이면_설정_오류다() -> None:
    with pytest.raises(ValueError):
        validate_sql(OK_SQL, order_no=ORDER_NO, max_rows=0)


# ── 한정된 별표 (finding 5) ─────────────────────────────────────────────────


def test_한정된_별표도_통과한다() -> None:
    """화이트리스트 밖 테이블은 이미 먼저 거부되므로 `o.*` 는 `*` 와 같은 범위다."""
    validated = validate_sql(
        f"SELECT o.* FROM orders o WHERE o.order_no = '{ORDER_NO}'",
        order_no=ORDER_NO,
        max_rows=MAX_ROWS,
    )

    assert validated.tables == ("orders",)


def test_파생_테이블의_한정된_별표도_통과한다() -> None:
    _ok(f"SELECT a.* FROM (SELECT order_no, status FROM orders {SCOPE}) a")


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
