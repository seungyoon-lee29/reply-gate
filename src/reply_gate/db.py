"""Postgres 커넥션 헬퍼 — 애플리케이션 계정과 read-only 계정을 분리해서 연다.

docs/security.md "text-to-SQL 안전장치" 1번(read-only DB 계정)의 코드 쪽 진입점이다. text-to-SQL
실행 경로는 반드시 `readonly_connect()` 를 쓴다 — 권한 자체가 SELECT 뿐이라 실수로
쓰기 경로를 타도 DB 가 거부한다.

DSN 은 `reply_gate.config.Settings` 가 만들고 **비밀 전용 타입으로 내놓는다**. 그래서 이
모듈에 평문으로 꺼내는 자리가 정확히 **한 줄** 생긴다 — psycopg 에 넘기기 직전이다. 그 한
줄이 이 모듈에서 비밀번호가 문자열이 되는 유일한 자리이고, 그 밖의 어디에도 DSN 을 그대로
싣지 않는다. 오류 메시지·로그에 쓰는 접속 대상 설명은 비밀번호를 아예 담지 않는다
(`describe_target`).
"""

from __future__ import annotations

from pathlib import Path

import psycopg
from psycopg.rows import DictRow, dict_row

from reply_gate.config import Settings, get_settings

__all__ = [
    "SCHEMA_SQL_PATH",
    "apply_schema",
    "connect",
    "database_unavailable_reason",
    "describe_target",
    "readonly_connect",
]

#: 저장소 루트의 `db/schema.sql` — `src/reply_gate/db.py` 기준으로 두 단계 위가 루트다.
SCHEMA_SQL_PATH = Path(__file__).resolve().parents[2] / "db" / "schema.sql"

#: 접속 시도가 매달리지 않도록 하는 상한(초). DB 가 없을 때 테스트가 빨리 skip 되도록.
CONNECT_TIMEOUT_SECONDS = 5


def connect(
    *,
    readonly: bool = False,
    settings: Settings | None = None,
    autocommit: bool = False,
    connect_timeout: int = CONNECT_TIMEOUT_SECONDS,
    statement_timeout_ms: int | None = None,
) -> psycopg.Connection[DictRow]:
    """Postgres 커넥션을 연다. `with` 블록에 넣으면 정상 종료 시 커밋된다.

    `readonly=True` 면 SELECT 권한만 가진 계정으로 연다.

    `statement_timeout_ms` 는 **접속 옵션**으로 건다(`SET` 문이 아니라). 세션이 열리는 순간부터
    적용되고 롤백·`RESET ALL` 로 되돌아가지 않아야, 실행 경로 어디서도 상한 없는 쿼리가
    생기지 않는다.
    """
    settings = settings if settings is not None else get_settings()
    # 이 모듈에서 비밀번호가 평문 문자열이 되는 **유일한 줄**이다 — psycopg 에 넘기기 직전.
    dsn = (settings.readonly_database_url if readonly else settings.database_url).get_secret_value()
    options = (
        None if statement_timeout_ms is None else f"-c statement_timeout={statement_timeout_ms:d}"
    )
    return psycopg.connect(
        dsn,
        row_factory=dict_row,
        autocommit=autocommit,
        connect_timeout=connect_timeout,
        options=options,
    )


def readonly_connect(
    *,
    settings: Settings | None = None,
    autocommit: bool = True,
    connect_timeout: int = CONNECT_TIMEOUT_SECONDS,
) -> psycopg.Connection[DictRow]:
    """text-to-SQL 실행 전용 커넥션 (SELECT 권한만 + 실행 시간 상한).

    기본이 `autocommit=True` 인 것은 의도적이다 — 조회만 하는 경로에서 트랜잭션을
    열어둔 채 방치하지 않기 위해서다.

    `statement_timeout` 을 반드시 건다: LLM 이 만든 쿼리는 코드가 검증하지만, 검증을 통과한
    쿼리라도 실행 시간까지 코드가 예측할 수는 없다. 상한 값은 `Settings.sql_statement_timeout_ms`
    의 조정 가능 기본값이고, 상한을 넘긴 쿼리는 `psycopg.errors.QueryCanceled` 로 끊겨
    SQL 실패 경로(실행 오류 → 1회 재시도)를 탄다.
    """
    settings = settings if settings is not None else get_settings()
    return connect(
        readonly=True,
        settings=settings,
        autocommit=autocommit,
        connect_timeout=connect_timeout,
        statement_timeout_ms=settings.sql_statement_timeout_ms,
    )


def describe_target(*, readonly: bool = False, settings: Settings | None = None) -> str:
    """오류 메시지·로그에 쓸 접속 대상 설명. **비밀번호를 담지 않는다.**"""
    settings = settings if settings is not None else get_settings()
    user = settings.postgres_ro_user if readonly else settings.postgres_app_user
    return f"{user}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"


def database_unavailable_reason(
    *, readonly: bool = False, settings: Settings | None = None
) -> str | None:
    """접속 가능하면 `None`, 아니면 **사람이 읽을 수 있는 사유 문자열**.

    테스트 수집 단계에서 DB 통합 테스트를 skip 할지 판단하는 데 쓴다. 조용히 건너뛰지
    않도록 사유에 접속 대상과 원인을 함께 담는다(비밀번호는 제외).
    """
    target = describe_target(readonly=readonly, settings=settings)
    try:
        with connect(readonly=readonly, settings=settings, autocommit=True) as conn:
            conn.execute("SELECT 1")
    except psycopg.Error as exc:
        detail = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return (
            f"Postgres 에 접속할 수 없다 ({target}): {detail}. "
            "`docker compose up -d --wait` 로 DB 를 띄운 뒤 다시 실행한다."
        )
    return None


def apply_schema(conn: psycopg.Connection[DictRow] | None = None) -> None:
    """`db/schema.sql` 을 적용한다. 재실행 안전(멱등).

    반드시 **애플리케이션 계정**으로 실행한다 — 테이블 소유자가 앱 계정이어야
    read-only 그룹에 SELECT 를 부여할 수 있다.
    """
    script = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
    if conn is not None:
        conn.execute(script)
        return
    with connect() as owned:
        owned.execute(script)
