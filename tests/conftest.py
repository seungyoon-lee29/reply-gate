"""공용 pytest 픽스처 — DB 통합 테스트의 접속·시딩 준비.

`db` 마커가 붙은 테스트는 Postgres 가 필요하다. 접속되지 않으면 **사유를 담아 skip** 하고,
접속되면 반드시 **실행된다**. 조용히 늘 skip 되는 테스트는 검증이 아니므로, skip 사유에는
접속 대상과 원인, 복구 명령을 함께 싣는다.

    docker compose up -d --wait   # DB 기동
    uv run pytest -m db           # DB 통합 테스트만
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from psycopg.rows import DictRow
from scripts.seed_orders import seed_orders

from reply_gate.config import Settings, get_settings
from reply_gate.db import connect, database_unavailable_reason, readonly_connect

#: 접속 가능 여부는 세션당 1회만 확인한다(테스트마다 붙었다 떼면 느리다).
_skip_reason_cache: list[str | None] = []


def _db_skip_reason() -> str | None:
    if not _skip_reason_cache:
        _skip_reason_cache.append(database_unavailable_reason())
    return _skip_reason_cache[0]


def pytest_runtest_setup(item: pytest.Item) -> None:
    """`db` 마커 테스트는 DB 가 없을 때만 skip 한다."""
    if item.get_closest_marker("db") is None:
        return
    reason = _db_skip_reason()
    if reason is not None:
        pytest.skip(reason)


@pytest.fixture(scope="session")
def db_settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="session")
def seeded_order_count() -> int:
    """스키마 적용 + 주문 시딩을 세션당 1회. 반환값은 시딩 후 주문 건수다."""
    return seed_orders()


@pytest.fixture
def app_conn(seeded_order_count: int) -> Iterator[psycopg.Connection[DictRow]]:
    """애플리케이션 계정 커넥션. 테스트가 쓴 행은 롤백으로 되돌린다."""
    del seeded_order_count  # 스키마·시딩 선행만 보장하면 된다.
    with connect() as conn:
        try:
            yield conn
        finally:
            conn.rollback()


@pytest.fixture
def ro_conn(seeded_order_count: int) -> Iterator[psycopg.Connection[DictRow]]:
    """text-to-SQL 실행 전용(SELECT 만) 계정 커넥션."""
    del seeded_order_count
    with readonly_connect() as conn:
        yield conn
