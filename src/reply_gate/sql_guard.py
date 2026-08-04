"""text-to-SQL 안전장치 2·3 — 스키마 화이트리스트 + 쿼리 검증 (spec "text-to-SQL 안전장치").

검증은 **실행 전에 전부 끝난다.** LLM 은 SQL 문자열을 만들 뿐이고, 그 문자열이 DB 에 닿을지는
이 모듈이 정한다. 통과한 쿼리도 실행은 SELECT 권한만 가진 계정(`db.readonly_connect`)으로만
한다 — 안전장치 1은 이 코드가 뚫려도 남는 마지막 층이고, 이 모듈이 그 층을 대체하지 않는다.

파싱은 `sqlglot` 으로 한다. 정규식으로 훑으면 주석 안에 숨긴 문장(`-- ; DROP ...`)과
data-modifying CTE(`WITH x AS (INSERT ...) SELECT ...`)를 놓친다 — 둘 다 여기서 실제로 막는다.

거부 규칙은 `SqlGuardRule` 이 전부이고, 거부 사유는 SQL 재생성 프롬프트에 그대로 실린다
(spec "SQL 실패 경로" — 오류 내용을 피드백으로 1회 재시도).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

__all__ = [
    "ALLOWED_SCHEMAS",
    "ORDERS_TABLE",
    "SCHEMA_WHITELIST",
    "SqlGuardRejection",
    "SqlGuardRule",
    "ValidatedQuery",
    "describe_whitelist",
    "validate_sql",
]

#: 파싱·생성 대상 방언. 실행 대상이 Postgres 이므로 검증도 같은 방언으로 읽는다.
DIALECT: Final = "postgres"

#: text-to-SQL 의 유일한 조회 대상 (spec "근거 수집" 절 — 주문 근거만 SQL 로 모은다).
ORDERS_TABLE: Final = "orders"

#: 조회 가능한 테이블·컬럼. **`db/schema.sql` 의 orders 컬럼과 같아야 한다**
#: (`tests/test_sql_guard.py` 가 information_schema 와 대조한다).
#:
#: orders 하나뿐인 것은 의도된 설계다: read-only 계정도 orders 에만 SELECT 권한이 있으므로
#: 코드 화이트리스트와 DB 권한이 같은 경계를 가리킨다.
SCHEMA_WHITELIST: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        ORDERS_TABLE: (
            "order_no",
            "customer_name",
            "customer_phone",
            "customer_email",
            "shipping_address",
            "product_name",
            "product_option",
            "quantity",
            "unit_price_krw",
            "total_price_krw",
            "status",
            "ordered_at",
            "shipped_at",
            "delivered_at",
            "courier",
            "tracking_no",
        )
    }
)

#: 테이블 이름 앞에 붙일 수 있는 스키마. 그 밖(`pg_catalog`·`information_schema`)은 거부된다.
ALLOWED_SCHEMAS: Final = frozenset({"", "public"})


class SqlGuardRule(StrEnum):
    """거부 규칙 코드. 처리 기록의 실패 내역과 재생성 피드백이 이 값을 쓴다."""

    EMPTY = "empty"
    PARSE_ERROR = "parse_error"
    COMMENT = "comment"
    MULTIPLE_STATEMENTS = "multiple_statements"
    NOT_SELECT = "not_select"
    FORBIDDEN_STATEMENT = "forbidden_statement"
    UNKNOWN_TABLE = "unknown_table"
    NO_WHITELISTED_TABLE = "no_whitelisted_table"
    UNKNOWN_COLUMN = "unknown_column"
    UNSUPPORTED_LIMIT = "unsupported_limit"


class SqlGuardRejection(ValueError):
    """안전장치가 실행 **전에** 거부했다. 재생성 피드백에 그대로 실린다."""

    def __init__(self, *, rule: SqlGuardRule, detail: str) -> None:
        super().__init__(f"[{rule.value}] {detail}")
        self.rule = rule
        self.detail = detail


@dataclass(frozen=True)
class ValidatedQuery:
    """검증을 통과해 **실행해도 되는** 쿼리.

    `sql` 은 원문이 아니라 안전장치가 LIMIT 을 강제해 다시 쓴 문장이다 — 처리 기록의
    스냅샷에도 이 문장이 남아야 실제로 실행된 것과 일치한다.
    """

    sql: str
    #: 실제로 붙은 LIMIT 값 (`<= max_rows`).
    limit: int
    #: 설정된 결과 행 수 상한 (`Settings.sql_max_rows`). 실행 측이 fetch 상한으로도 쓴다.
    max_rows: int
    #: 이 쿼리가 참조하는 화이트리스트 테이블 이름들.
    tables: tuple[str, ...]


def describe_whitelist(whitelist: Mapping[str, tuple[str, ...]] = SCHEMA_WHITELIST) -> str:
    """SQL 생성 프롬프트에 싣는 화이트리스트 설명 (테이블 + 전체 컬럼)."""
    return "\n".join(f"- {table}({', '.join(columns)})" for table, columns in whitelist.items())


def _reject(rule: SqlGuardRule, detail: str) -> SqlGuardRejection:
    return SqlGuardRejection(rule=rule, detail=detail)


def _parse_single_statement(text: str) -> exp.Expr:
    """주석·다중문을 먼저 걷어내고 단일 문장을 파싱한다.

    주석은 **파싱 전에 토큰 단위로** 잡는다. 파서는 주석을 버리므로
    `SELECT ... -- ; DROP TABLE orders` 가 정상 단일문으로 보이기 때문이다.
    """
    try:
        tokens = sqlglot.tokenize(text, read=DIALECT)
    except SqlglotError as exc:
        raise _reject(SqlGuardRule.PARSE_ERROR, f"토큰화 실패: {exc}") from exc

    if any(token.comments for token in tokens):
        raise _reject(
            SqlGuardRule.COMMENT,
            "SQL 에 주석(`--`, `/* */`)이 있다. 주석 없이 SELECT 문 하나만 낸다.",
        )

    try:
        statements = sqlglot.parse(text, read=DIALECT)
    except SqlglotError as exc:
        raise _reject(SqlGuardRule.PARSE_ERROR, f"SQL 파싱 실패: {exc}") from exc

    if len(statements) != 1:
        raise _reject(
            SqlGuardRule.MULTIPLE_STATEMENTS,
            f"문장이 {len(statements)}개다. 세미콜론으로 잇지 말고 SELECT 문 하나만 낸다.",
        )
    statement = statements[0]
    if statement is None:
        raise _reject(SqlGuardRule.PARSE_ERROR, "빈 문장으로 파싱되었다")
    return statement


def _require_read_only_select(statement: exp.Expr) -> exp.Select:
    """루트가 단일 SELECT 이고, 트리 어디에도 DML/DDL 이 없어야 한다."""
    if not isinstance(statement, exp.Select):
        raise _reject(
            SqlGuardRule.NOT_SELECT,
            f"SELECT 단일문만 허용된다 (받은 문장: {type(statement).__name__}).",
        )

    for node in statement.walk():
        # Postgres 의 data-modifying CTE 는 루트가 SELECT 여도 쓰기를 수행한다.
        if isinstance(node, exp.DML | exp.DDL | exp.Command):
            raise _reject(
                SqlGuardRule.FORBIDDEN_STATEMENT,
                f"SELECT 안에 {type(node).__name__} 문이 있다. 조회만 하는 SELECT 여야 한다.",
            )
        # `SELECT ... INTO t` 는 Postgres 에서 테이블을 만든다.
        if isinstance(node, exp.Into):
            raise _reject(SqlGuardRule.FORBIDDEN_STATEMENT, "SELECT ... INTO 는 허용되지 않는다.")
    return statement


def _referenced_tables(
    statement: exp.Select, whitelist: Mapping[str, tuple[str, ...]]
) -> tuple[str, ...]:
    """화이트리스트 밖 테이블을 거부하고, 참조된 화이트리스트 테이블을 돌려준다."""
    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    referenced: list[str] = []

    for table in statement.find_all(exp.Table):
        name = table.name.lower()
        schema = table.db.lower()
        catalog = table.catalog.lower()

        if not name:
            # `FROM generate_series(...)` 같은 테이블 함수 — 이름 없는 FROM 소스다.
            raise _reject(
                SqlGuardRule.UNKNOWN_TABLE,
                "FROM 절에 테이블이 아닌 소스(함수 등)가 있다. 화이트리스트 테이블만 조회한다.",
            )
        if name in cte_names and not schema and not catalog:
            continue
        if catalog:
            raise _reject(
                SqlGuardRule.UNKNOWN_TABLE, f"다른 데이터베이스를 참조할 수 없다: {catalog}.{name}"
            )
        if schema not in ALLOWED_SCHEMAS:
            raise _reject(
                SqlGuardRule.UNKNOWN_TABLE, f"화이트리스트 밖 스키마를 참조했다: {schema}.{name}"
            )
        if name not in whitelist:
            raise _reject(
                SqlGuardRule.UNKNOWN_TABLE,
                f"화이트리스트 밖 테이블을 참조했다: {name}. "
                f"조회 가능한 테이블은 {', '.join(whitelist)} 뿐이다.",
            )
        if name not in referenced:
            referenced.append(name)

    if not referenced:
        raise _reject(
            SqlGuardRule.NO_WHITELISTED_TABLE,
            f"화이트리스트 테이블({', '.join(whitelist)})을 하나도 조회하지 않는다.",
        )
    return tuple(referenced)


def _check_columns(
    statement: exp.Select, tables: tuple[str, ...], whitelist: Mapping[str, tuple[str, ...]]
) -> None:
    """화이트리스트 밖 컬럼을 거부한다.

    쿼리 안에서 정의된 별칭(`AS ...`, CTE 컬럼 목록)은 화이트리스트 컬럼에서 유래한 이름이므로
    함께 허용한다 — 별칭의 원본 표현식은 여전히 이 검사를 통과해야 한다.
    """
    allowed = {column.lower() for table in tables for column in whitelist[table]}
    allowed |= {alias.alias.lower() for alias in statement.find_all(exp.Alias) if alias.alias}
    allowed |= {
        column.name.lower()
        for table_alias in statement.find_all(exp.TableAlias)
        for column in table_alias.columns
    }

    for column in statement.find_all(exp.Column):
        if column.name.lower() not in allowed:
            raise _reject(
                SqlGuardRule.UNKNOWN_COLUMN,
                f"화이트리스트 밖 컬럼을 참조했다: {column.sql(dialect=DIALECT)}",
            )


def _enforce_limit(statement: exp.Select, max_rows: int) -> tuple[exp.Select, int]:
    """결과 행 수 상한을 **강제**한다 — 거부가 아니라 LIMIT 을 붙이거나 낮춘다.

    상한 없는 쿼리를 그대로 실행시키지 않는 것이 목적이므로, LIMIT 이 없으면 붙이고
    상한을 넘으면 상한으로 낮춘다. 다만 LIMIT 값이 정수 리터럴이 아니면(부질의·파라미터)
    상한 이하임을 실행 전에 확인할 수 없으므로 거부한다.
    """
    limit_node = statement.args.get("limit")
    effective = max_rows
    if limit_node is not None:
        value = limit_node.expression
        if not isinstance(value, exp.Literal) or not value.is_int:
            raise _reject(
                SqlGuardRule.UNSUPPORTED_LIMIT,
                f"LIMIT 은 정수 리터럴이어야 한다 (받은 값: {limit_node.sql(dialect=DIALECT)}).",
            )
        requested = int(value.name)
        if requested < 1:
            raise _reject(
                SqlGuardRule.UNSUPPORTED_LIMIT, f"LIMIT 은 1 이상이어야 한다: {requested}"
            )
        effective = min(requested, max_rows)
    return statement.limit(effective), effective


def validate_sql(
    sql: str,
    *,
    max_rows: int,
    whitelist: Mapping[str, tuple[str, ...]] = SCHEMA_WHITELIST,
) -> ValidatedQuery:
    """생성된 SQL 을 실행 전에 검증한다. 통과하지 못하면 `SqlGuardRejection`.

    `max_rows` 가 0 이하인 것은 LLM 산출 문제가 아니라 설정 오류이므로 `ValueError` 다.
    """
    if max_rows < 1:
        raise ValueError(f"결과 행 수 상한은 1 이상이어야 한다: {max_rows}")

    text = sql.strip()
    if not text:
        raise _reject(SqlGuardRule.EMPTY, "SQL 이 비어 있다.")

    statement = _require_read_only_select(_parse_single_statement(text))
    tables = _referenced_tables(statement, whitelist)
    _check_columns(statement, tables, whitelist)
    limited, effective = _enforce_limit(statement, max_rows)

    return ValidatedQuery(
        sql=limited.sql(dialect=DIALECT),
        limit=effective,
        max_rows=max_rows,
        tables=tables,
    )
