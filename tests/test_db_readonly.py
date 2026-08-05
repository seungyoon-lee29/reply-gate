"""text-to-SQL 안전장치 1 — read-only DB 계정 (docs/security.md "text-to-SQL 안전장치" 절).

SQL 실행 계정이 SELECT 밖의 무엇도 못 한다는 것을 DB 에게 직접 물어본다. 코드의 검증
로직(안전장치 2·3)이 뚫려도 **권한 계층이 마지막으로 막는다**는 것이 이 계정의 존재 이유이므로,
여기서 검증하는 것은 코드가 아니라 권한 자체다.

전제: `docker compose up -d --wait`. 없으면 `tests/conftest.py` 가 사유를 붙여 skip 한다.
"""

from __future__ import annotations

import psycopg
import pytest
from psycopg.rows import DictRow
from scripts.seed_orders import ORDER_COUNT

pytestmark = pytest.mark.db


def test_readonly_account_can_select_orders(ro_conn: psycopg.Connection[DictRow]) -> None:
    """양성 대조 — 이게 실패하면 아래 거부 테스트들이 공허해진다."""
    row = ro_conn.execute("SELECT count(*) AS total FROM orders").fetchone()
    assert row is not None
    assert row["total"] == ORDER_COUNT


def test_readonly_account_is_a_different_user_than_the_app(
    ro_conn: psycopg.Connection[DictRow], app_conn: psycopg.Connection[DictRow]
) -> None:
    ro_user = ro_conn.execute("SELECT current_user AS u").fetchone()
    app_user = app_conn.execute("SELECT current_user AS u").fetchone()
    assert ro_user is not None and app_user is not None
    assert ro_user["u"] != app_user["u"]


@pytest.mark.parametrize(
    "sql",
    [
        """INSERT INTO orders (order_no, customer_name, customer_phone, customer_email,
                               shipping_address, product_name, quantity, unit_price_krw,
                               total_price_krw, status, ordered_at)
           VALUES ('ORD-20260101-0001', '무단', '010-0000-0000', 'x@example.com', '서울',
                   '상품', 1, 1000, 1000, '결제완료', now())""",
        "UPDATE orders SET status = '취소'",
        "DELETE FROM orders",
        "TRUNCATE orders",
    ],
    ids=["INSERT", "UPDATE", "DELETE", "TRUNCATE"],
)
def test_readonly_account_cannot_write(ro_conn: psycopg.Connection[DictRow], sql: str) -> None:
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        ro_conn.execute(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE ro_probe (x integer)",
        "CREATE TEMP TABLE ro_probe_tmp (x integer)",
        "DROP TABLE orders",
        "ALTER TABLE orders ADD COLUMN ro_probe integer",
        "CREATE INDEX ro_probe_idx ON orders (status)",
    ],
    ids=["CREATE TABLE", "CREATE TEMP TABLE", "DROP TABLE", "ALTER TABLE", "CREATE INDEX"],
)
def test_readonly_account_cannot_run_ddl(ro_conn: psycopg.Connection[DictRow], sql: str) -> None:
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        ro_conn.execute(sql)


@pytest.mark.parametrize(
    "table", ["inquiries", "inquiry_attempts", "inquiry_evidence", "inquiry_sql_failures"]
)
def test_readonly_account_cannot_read_processing_records(
    ro_conn: psycopg.Connection[DictRow], table: str
) -> None:
    """방어층 하나 더 — SQL 실행 계정에는 주문 외 테이블 권한 자체를 주지 않는다.

    text-to-SQL 의 조회 대상은 주문뿐이다(docs/security.md "text-to-SQL 안전장치" 절). 안전장치 2(스키마
    화이트리스트)가 코드에서 막는 것을 권한이 한 번 더 막는다.
    """
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        ro_conn.execute(f"SELECT * FROM {table}")


def test_readonly_account_cannot_read_policy_chunks(
    ro_conn: psycopg.Connection[DictRow],
) -> None:
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        ro_conn.execute("SELECT * FROM policy_chunks")


def test_readonly_account_has_no_table_privileges_beyond_select_on_orders(
    ro_conn: psycopg.Connection[DictRow], app_conn: psycopg.Connection[DictRow]
) -> None:
    """권한 목록을 직접 뒤져서 SELECT 외의 권한이 하나도 없는지 확인한다."""
    row = ro_conn.execute("SELECT current_user AS u").fetchone()
    assert row is not None
    grants = app_conn.execute(
        """SELECT table_name, privilege_type
           FROM information_schema.table_privileges
           WHERE grantee IN (%s, 'reply_gate_readers')
           ORDER BY table_name, privilege_type""",
        (row["u"],),
    ).fetchall()
    assert {(g["table_name"], g["privilege_type"]) for g in grants} == {("orders", "SELECT")}
