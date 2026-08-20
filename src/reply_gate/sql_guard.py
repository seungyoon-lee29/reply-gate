"""text-to-SQL 안전장치 2·3 — 스키마 화이트리스트 + 쿼리 검증
(docs/security.md "text-to-SQL 안전장치").

검증은 **실행 전에 전부 끝난다.** LLM 은 SQL 문자열을 만들 뿐이고, 그 문자열이 DB 에 닿을지는
이 모듈이 정한다. 통과한 쿼리도 실행은 SELECT 권한만 가진 계정(`db.readonly_connect`)으로만
한다 — 안전장치 1은 이 코드가 뚫려도 남는 마지막 층이고, 이 모듈이 그 층을 대체하지 않는다.

파싱은 `sqlglot` 으로 한다. 정규식으로 훑으면 주석 안에 숨긴 문장(`-- ; DROP ...`)과
data-modifying CTE(`WITH x AS (INSERT ...) SELECT ...`)를 놓친다 — 둘 다 여기서 실제로 막는다.

이 모듈이 실행 전에 확인하는 것 (docs/standards.md "하드 게이트" — 데이터 접근은 항상 코드가 통제):

1. **단일 읽기 전용 SELECT** — DML/DDL·다중문·주석·`SELECT ... INTO`·잠금 절(`FOR UPDATE`)을
   거부한다. 잠금 절은 read-only 계정 권한으로도 막히지만 그건 실행 시점이라 재시도 1회를
   헛되이 태운다.
2. **스키마 화이트리스트** — 테이블과 **컬럼**을 스코프 단위로 해석해서 대조한다. 별칭은
   화이트리스트 구성원이 아니다: 한정된 컬럼(`o.col`)은 그 한정자가 가리키는 소스의 목록으로
   검사하고, 맨 이름 별칭은 PostgreSQL 이 실제로 출력 이름을 해석하는 자리(ORDER BY·GROUP BY)
   에서만 받아들인다.
3. **함수 허용 목록과 형변환 대상 타입 허용 목록** — `pg_sleep` 같은 호출이 실행 시간을
   무한정 잡는 것을 막고, `cast(... AS regclass)` 같은 **시스템 카탈로그 참조 타입**이
   화이트리스트 밖의 이름과 존재 여부를 결과로 실어 나르는 것을 막는다.
   (권한이 필요한 함수는 안전장치 1이 막지만, 가용성은 권한이 막아주지 않는다. 카탈로그
   참조 타입은 **안전장치 1도 막지 못한다** — 시스템 카탈로그는 read-only 계정도 읽으므로
   막는 층이 이 가드 하나뿐이다.)
4. **주문 1건 한정** — 생성된 쿼리가 **선검사를 통과한 주문번호 1건**으로 한정됨을 AST 로
   확인한다 (docs/architecture.md "대표 흐름" — 선검사를 통과한 주문에 대해 조회).
   어떤 행이 나올지를 LLM 이 정하게 두면 무관한 고객의 연락처가 근거로 채택되고,
   L1 의 PII allowlist 가 그것을 정상 에코로 허용한다. 조인 종류도 여기서 읽는다 —
   **외부 조인의 ON 절은 보존측 테이블을 거르지 않으므로** 한정 조건으로 인정할 수 없다
   (아래 `_check_joins`·`_is_inner_join`).
5. **결과 행 수 상한** — 거부가 아니라 LIMIT 을 강제한다.
6. **PII allowlist 출력 유래** — 어느 결과 컬럼의 값이 `orders` 직접 컬럼에서 왔는지를
   스코프를 타고 내려가며 증명한다(아래 `_projection_provenance`). 증명하지 못한 출력은
   **자동으로 불승인**이다 — 허용을 늘리는 방식이지 거부 목록을 두는 방식이 아니다.

거부 규칙은 `SqlGuardRule` 이 전부이고, 거부 사유는 SQL 재생성 프롬프트에 그대로 실린다
(docs/standards.md "재시도 상한" — 오류 내용을 피드백으로 1회 재시도). 그러므로 사유
메시지는 **무엇을 어떻게 고쳐야 하는지**를 담아야 한다.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import Final, NamedTuple

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from reply_gate.gate import pii_shaped
from reply_gate.order_ref import is_valid_order_no

__all__ = [
    "ALLOWED_CAST_TYPES",
    "ALLOWED_SCHEMAS",
    "ORDERS_TABLE",
    "ORDER_SCOPE_COLUMN",
    "SCHEMA_WHITELIST",
    "SqlGuardRejection",
    "SqlGuardRule",
    "ValidatedQuery",
    "describe_allowed_cast_types",
    "describe_allowed_functions",
    "describe_whitelist",
    "validate_sql",
]

#: 파싱·생성 대상 방언. 실행 대상이 Postgres 이므로 검증도 같은 방언으로 읽는다.
DIALECT: Final = "postgres"

#: text-to-SQL 의 유일한 조회 대상 — 주문 근거만 SQL 로 모은다
#: (docs/security.md "text-to-SQL 안전장치").
ORDERS_TABLE: Final = "orders"

#: 조회 범위를 주문 1건으로 묶는 컬럼. 선검사(`evidence.order_exists`)가 쓰는 것과 같다.
ORDER_SCOPE_COLUMN: Final = "order_no"

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

#: 호출을 허용하는 함수. **여기 없는 함수는 전부 거부한다.**
#:
#: 근거: 주문 1건을 조회해 답변 근거로 쓰는 데 실제로 필요한 것만 남겼다 — 요약(집계),
#: NULL 처리, 조건 분기, 타입 변환, 문자열·날짜 포맷. 그 밖의 함수는 조회에 기여하지 않으면서
#: 실행 시간을 묶을 수 없게 만든다(`pg_sleep`) 거나, 권한 계층에만 의존하게 만든다
#: (`pg_read_file`·`lo_import`). 화이트리스트가 아니라 블랙리스트로 두면 Postgres 의 수천 개
#: 내장 함수를 우리가 따라잡아야 하므로, **모르는 함수는 거부**가 기본값이다.
#:
#: 키는 프롬프트에 싣는 SQL 표기, 값은 sqlglot 이 그 표기를 파싱해 만드는 노드 타입이다
#: (`to_char` → `TimeToStr` 처럼 sqlglot 이 이름을 정규화하므로 표기만으로는 못 맞춘다).
_ALLOWED_FUNCTIONS: Final[tuple[tuple[str, type[exp.Func]], ...]] = (
    # 집계 — "이 주문의 총 수량" 같은 요약.
    ("count", exp.Count),
    ("sum", exp.Sum),
    ("min", exp.Min),
    ("max", exp.Max),
    ("avg", exp.Avg),
    # NULL 처리 — 미배송 주문의 빈 컬럼을 근거 문장에 그대로 싣지 않기 위해.
    ("coalesce", exp.Coalesce),
    ("nullif", exp.Nullif),
    # 조건 분기 — sqlglot 은 CASE 를 함수 노드로 만든다.
    ("case", exp.Case),
    ("case", exp.If),
    # 타입 변환 — 숫자·일자를 문자열로 붙일 때.
    ("cast", exp.Cast),
    # 문자열 포맷.
    ("upper", exp.Upper),
    ("lower", exp.Lower),
    ("trim", exp.Trim),
    ("length", exp.Length),
    ("concat", exp.Concat),
    ("substring", exp.Substring),
    ("replace", exp.Replace),
    # 날짜 포맷 — 배송 일자를 사람이 읽는 형태로.
    ("to_char", exp.TimeToStr),
    ("date_trunc", exp.TimestampTrunc),
    ("extract", exp.Extract),
    ("now", exp.CurrentTimestamp),
    ("current_date", exp.CurrentDate),
)

_ALLOWED_FUNCTION_TYPES: Final[tuple[type[exp.Func], ...]] = tuple(
    node_type for _, node_type in _ALLOWED_FUNCTIONS
)

#: 형변환의 **대상 타입** 허용 목록. **여기 없는 타입은 전부 거부한다.**
#:
#: 근거: 허용 함수 목록이 캐스트를 넣은 이유가 "숫자·일자를 문자열로 붙일 때"이고, 이 목록은
#: 그 용도 — 숫자 · 일자 · 문자열 — 에 정확히 맞춘다. 대상 타입을 읽지 않으면
#: `cast('inquiries' AS regclass)` 처럼 **시스템 카탈로그 참조 타입**으로 화이트리스트 밖
#: 테이블의 이름과 존재 여부를 결과 컬럼에 실을 수 있다. 시스템 카탈로그는 read-only 계정도
#: 읽으므로(PUBLIC 권한) 안전장치 1이 이 경로를 막지 못한다 — **막는 층이 이 목록 하나뿐이다.**
#:
#: 키는 프롬프트·거부 사유에 싣는 SQL 표기, 값은 sqlglot 이 그 표기를 파싱해 만드는 타입이다
#: (`numeric` → `DECIMAL`, `real` → `FLOAT` 처럼 sqlglot 이 이름을 정규화한다). 카탈로그 참조
#: 타입(`oid`·`reg*`)은 `exp.ObjectIdentifier` 로 파싱돼 `this` 가 열거형이 아니므로, 이
#: 집합과의 대조만으로 자동 거부된다.
_ALLOWED_CAST_TYPES: Final = (
    # 문자열 — 근거 문장에 붙이는 형태.
    ("text", exp.DataType.Type.TEXT),
    ("varchar", exp.DataType.Type.VARCHAR),
    ("char", exp.DataType.Type.CHAR),
    # 숫자 — 수량·금액 계산과 표시.
    ("int", exp.DataType.Type.INT),
    ("bigint", exp.DataType.Type.BIGINT),
    ("smallint", exp.DataType.Type.SMALLINT),
    ("numeric", exp.DataType.Type.DECIMAL),
    ("real", exp.DataType.Type.FLOAT),
    ("double precision", exp.DataType.Type.DOUBLE),
    # 일자 — 주문·배송 일자.
    ("date", exp.DataType.Type.DATE),
    ("timestamp", exp.DataType.Type.TIMESTAMP),
    ("timestamptz", exp.DataType.Type.TIMESTAMPTZ),
    ("time", exp.DataType.Type.TIME),
)

#: 프롬프트·거부 사유가 그대로 옮겨 적는 표기 목록. **정의는 여기 하나뿐이다** —
#: SQL 생성 안내는 `describe_allowed_cast_types()` 를 부르기만 한다(구조 검사가 지킨다).
ALLOWED_CAST_TYPES: Final[tuple[str, ...]] = tuple(name for name, _ in _ALLOWED_CAST_TYPES)

_ALLOWED_CAST_TYPE_NODES: Final = frozenset(node_type for _, node_type in _ALLOWED_CAST_TYPES)

#: **빈칸 채우기 계열** — 결과값이 인자 중 하나 **그대로**이고, 조각을 이어 붙이지 않는다.
#: 그래서 인자가 전부 (직접 컬럼 | 개인정보 모양이 아닌 고정값) 이면 출력도 그 성질을
#: 유지한다. 이어 붙이기 계열(`concat`·`||`)은 여기 없다 — 조각 하나하나는 개인정보 모양이
#: 아닌데 합치면 번호가 되기 때문이다(`_projection_provenance`).
#:
#: **둘을 갈라 두는 것은 값이 흐르는 자리가 다르기 때문이다.** `coalesce` 는 어느 인자든
#: 출력이 될 수 있지만 `nullif(a, b)` 의 출력은 `a` 아니면 NULL 이고 `b` 는 비교 상대일
#: 뿐이다 — `b` 자리의 직접 컬럼을 유래로 세면 값이 흐르지 않는 자리로 승인이 난다.
_COALESCE_FUNCTION_TYPES: Final[tuple[type[exp.Func], ...]] = (exp.Coalesce,)
_NULLIF_FUNCTION_TYPES: Final[tuple[type[exp.Func], ...]] = (exp.Nullif,)

#: sqlglot 이 `Func` 로 모델링하지만 **함수 호출이 아닌** 노드들 — 허용 목록의 대상이 아니다.
#:
#: * `Connector` — `AND`/`OR`/`XOR`. 연산자이지 호출이 아니다(WHERE 절을 쓰면 반드시 나온다).
#: * `SubqueryPredicate` — `EXISTS (...)`/`= ANY (...)`. 안쪽 SELECT 는 스코프 검사가 따로
#:   테이블·컬럼·조회 범위까지 본다. 여기서 함수로 취급하면 사유 메시지가 엉뚱해진다.
_OPERATOR_FUNCTION_TYPES: Final[tuple[type[exp.Expr], ...]] = (
    exp.Connector,
    exp.SubqueryPredicate,
)

#: 함수 검사에서 통과시키는 노드 전부 (허용 함수 + 함수가 아닌 것들).
_PERMITTED_FUNCTION_TYPES: Final[tuple[type[exp.Expr], ...]] = (
    _ALLOWED_FUNCTION_TYPES + _OPERATOR_FUNCTION_TYPES
)

#: `Select.args` 중 **조회 소스**를 담는 키. 컬럼 검사는 이 키들을 건너뛰고 따로 해석한다.
#: sqlglot 버전에 따라 `from`/`from_`, `with`/`with_` 를 쓰므로 둘 다 적는다.
_SOURCE_ARGS: Final = frozenset({"from", "from_", "joins", "with", "with_"})

#: PostgreSQL 이 **출력 별칭 이름**을 해석해 주는 절. 그 밖(SELECT 목록·WHERE·HAVING)에서
#: 맨 이름은 언제나 입력 컬럼으로 해석되므로 별칭으로 받아주면 화이트리스트가 뚫린다.
_ALIAS_VISIBLE_ARGS: Final = frozenset({"group", "order"})

#: ON 절이 **양쪽 소스를 모두 거르는** 조인 종류 (sqlglot 의 `Join.args["kind"]` 표기).
#: 빈 값은 `JOIN`(=INNER)과 쉼표 조인(`FROM a, b`)이다.
#:
#: 여기 없는 조인 — 외부 조인(LEFT/RIGHT/FULL, OUTER 포함) — 은 ON 절이 **보존측 테이블을
#: 필터하지 않는다.** 매칭이 없으면 보존측 행은 NULL 로 채워져 그대로 남기 때문에
#: `orders o LEFT JOIN (SELECT 1) d ON o.order_no = '...'` 는 orders **전체**를 돌려준다.
_INNER_JOIN_KINDS: Final = frozenset({"", "INNER", "CROSS"})


class SqlGuardRule(StrEnum):
    """거부 규칙 코드. 처리 기록의 실패 내역과 재생성 피드백이 이 값을 쓴다."""

    EMPTY = "empty"
    PARSE_ERROR = "parse_error"
    COMMENT = "comment"
    MULTIPLE_STATEMENTS = "multiple_statements"
    NOT_SELECT = "not_select"
    FORBIDDEN_STATEMENT = "forbidden_statement"
    LOCKING_CLAUSE = "locking_clause"
    UNKNOWN_TABLE = "unknown_table"
    UNSUPPORTED_SOURCE = "unsupported_source"
    UNSUPPORTED_JOIN = "unsupported_join"
    DUPLICATE_SOURCE = "duplicate_source"
    NO_WHITELISTED_TABLE = "no_whitelisted_table"
    UNKNOWN_COLUMN = "unknown_column"
    FORBIDDEN_FUNCTION = "forbidden_function"
    UNSUPPORTED_CAST_TYPE = "unsupported_cast_type"
    UNSUPPORTED_PROJECTION = "unsupported_projection"
    ORDER_SCOPE = "order_scope"
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
    #: 이 쿼리가 한정된 주문번호 — 선검사를 통과한 그 1건이다.
    order_no: str
    #: PII allowlist 에 써도 되는 최상위 출력명. `orders` 직접 컬럼 projection 만 들어간다.
    pii_safe_output_columns: tuple[str, ...]


def describe_whitelist(whitelist: Mapping[str, tuple[str, ...]] = SCHEMA_WHITELIST) -> str:
    """SQL 생성 프롬프트에 싣는 화이트리스트 설명 (테이블 + 전체 컬럼)."""
    return "\n".join(f"- {table}({', '.join(columns)})" for table, columns in whitelist.items())


def _describe_function_names() -> str:
    """허용 함수 이름만 — 함수 거부 사유가 이 줄만 인용한다."""
    seen: list[str] = []
    for name, _ in _ALLOWED_FUNCTIONS:
        if name not in seen:
            seen.append(name)
    return ", ".join(seen)


def describe_allowed_cast_types() -> str:
    """SQL 생성 프롬프트·거부 사유에 싣는 **형변환 대상 타입** 목록.

    목록의 정의는 이 모듈 하나이고, 생성 안내는 이 문면을 그대로 싣는다. 가드와 안내가
    갈리면 생성기가 목록 밖 타입을 쓸 때마다 거부 → 재시도 1회 → 인계가 되고, 그 인계가
    그대로 평가 수치에 남는다.
    """
    return (
        f"cast 의 대상 타입은 {', '.join(ALLOWED_CAST_TYPES)} 뿐이다 — "
        "그 밖의 타입(시스템 카탈로그 참조 타입 등)은 거부된다."
    )


def describe_allowed_functions() -> str:
    """SQL 생성 프롬프트에 싣는 허용 함수 목록 + 형변환 대상 타입 목록.

    프롬프트의 `[허용 함수]` 절이 곧 이 문자열이다. 캐스트 허용 목록을 여기 함께 싣는 것은
    **정의를 두 곳에 두지 않기 위해서**다 — 안내 쪽 모듈이 자기 목록을 들면 한쪽만 넓혀진다.
    """
    return f"{_describe_function_names()}\n{describe_allowed_cast_types()}"


def _reject(rule: SqlGuardRule, detail: str) -> SqlGuardRejection:
    return SqlGuardRejection(rule=rule, detail=detail)


def _sql_of(node: exp.Expr) -> str:
    return node.sql(dialect=DIALECT)


# ── 1. 단일 읽기 전용 SELECT ────────────────────────────────────────────────


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
    """루트가 단일 SELECT 이고, 트리 어디에도 DML/DDL·잠금 절이 없어야 한다."""
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

    lock = statement.find(exp.Lock)
    if lock is not None:
        # read-only 계정 권한이 실행 시점에 막긴 하지만, 그러면 단 1회뿐인 재시도를 태운다.
        raise _reject(
            SqlGuardRule.LOCKING_CLAUSE,
            "행 잠금 절(`FOR UPDATE`·`FOR SHARE`·`FOR NO KEY UPDATE`·`FOR KEY SHARE`)은 "
            "읽기 전용 단일 SELECT 가 아니므로 허용되지 않는다. 잠금 절을 빼고 조회만 한다.",
        )
    return statement


# ── 조인 종류 (4번 "주문 1건 한정"의 전제) ──────────────────────────────────


def _join_side_kind(join: exp.Join) -> tuple[str, str]:
    """(`LEFT`/`RIGHT`/`FULL` 등 방향, `INNER`/`OUTER`/`CROSS` 등 종류). 없으면 빈 문자열."""
    side = join.args.get("side")
    kind = join.args.get("kind")
    return str(side or "").upper(), str(kind or "").upper()


def _is_inner_join(join: exp.Join) -> bool:
    """이 조인의 ON 절이 **양쪽 소스를 모두 거르는가**.

    주문 1건 한정 판정이 ON 절을 믿어도 되는지가 여기서 갈린다. 외부 조인의 ON 은 보존측을
    거르지 않으므로 `False` 다. 모르는 종류(SEMI·ANTI 등)도 `False` — 안전을 증명하지 못한
    조인은 신뢰하지 않는다.
    """
    side, kind = _join_side_kind(join)
    return not side and kind in _INNER_JOIN_KINDS


def _join_label(join: exp.Join) -> str:
    side, kind = _join_side_kind(join)
    return " ".join(part for part in (side, kind, "JOIN") if part)


def _check_joins(statement: exp.Select) -> None:
    """외부 조인을 통째로 거부한다.

    `orders o LEFT JOIN (SELECT 1) d ON o.order_no = '...'` 는 ON 절이 보존측(orders)을 거르지
    않아 **서로 다른 주문 전부**를 돌려준다 — 주문 1건 한정(4번)을 정면으로 무력화한다.
    한정 조건을 WHERE 로 옮기면 그만이므로 CS 주문 조회에 외부 조인이 필요한 경우는 없다.
    """
    for join in statement.find_all(exp.Join):
        if _is_inner_join(join):
            continue
        raise _reject(
            SqlGuardRule.UNSUPPORTED_JOIN,
            f"허용되지 않은 조인 종류다: {_join_label(join)}. 외부 조인(LEFT/RIGHT/FULL)의 ON "
            "절은 보존측 테이블을 거르지 않아 조회 범위가 주문 1건으로 묶이지 않는다 — "
            "주문 1건 조회에는 외부 조인이 필요하지 않다. INNER JOIN 또는 단일 테이블 조회로 "
            f"다시 쓰고, 주문 한정 조건은 WHERE 절에 `{ORDER_SCOPE_COLUMN} = '...'` 로 넣는다.",
        )


# ── 2. 스코프 해석 (테이블·컬럼 화이트리스트) ───────────────────────────────


class _Exported(NamedTuple):
    """스코프가 바깥에 내놓는 것 — 컬럼 이름들과, 그중 **유래가 증명된** 이름들."""

    #: 이 스코프가 내놓는 출력 이름 전부 (소문자).
    columns: frozenset[str]
    #: 그중 값이 `orders` 직접 컬럼에서 왔음이 증명된 이름들. `columns` 의 부분집합이다.
    pii_safe: frozenset[str]


@dataclass(frozen=True)
class _Source:
    """FROM/JOIN 이 스코프 안에 들여놓은 이름 하나."""

    #: 스코프 안에서 이 소스를 가리키는 이름 (별칭이 있으면 별칭).
    name: str
    #: 이 소스가 내놓는 컬럼 이름들 (소문자).
    columns: frozenset[str]
    #: 화이트리스트 **기저 테이블**이면 그 테이블 이름. CTE·파생 테이블이면 `None`.
    base_table: str | None
    #: `columns` 중 값의 유래가 `orders` 직접 컬럼임이 증명된 것들.
    #:
    #: 기저 테이블이면 전부다. CTE·파생 테이블이면 **안쪽 스코프에서 증명된 부분집합**이고
    #: (그래서 유래 추적이 임시 테이블 한 겹을 건너도 끊기지 않는다), 컬럼 별칭 목록으로
    #: 이름을 갈아 끼운 소스는 자리 대응을 증명할 수 없으므로 비어 있다(fail-closed).
    pii_safe_columns: frozenset[str]


@dataclass
class _Scope:
    """SELECT 하나가 만드는 이름 해석 범위."""

    select: exp.Select
    #: 상관 부질의(correlated subquery)가 바깥 이름을 참조할 수 있으므로 부모를 들고 있는다.
    parent: _Scope | None
    sources: dict[str, _Source]

    @property
    def output_aliases(self) -> frozenset[str]:
        """이 SELECT 목록이 **직접** 정의한 출력 별칭 이름들.

        `find_all` 로 훑지 않는 것이 핵심이다 — 부질의 안쪽 별칭까지 긁어모으면
        별칭이 전역 허용 컬럼이 되어 화이트리스트가 무력화된다.
        """
        return frozenset(
            expression.alias.lower()
            for expression in self.select.expressions
            if isinstance(expression, exp.Alias) and expression.alias
        )


def _as_expressions(value: object) -> Iterator[exp.Expr]:
    if isinstance(value, exp.Expr):
        yield value
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, exp.Expr):
                yield item


def _direct_sources(select: exp.Select) -> Iterator[exp.Expr]:
    """이 SELECT 가 직접 읽는 소스(FROM 1개 + JOIN 들)."""
    for key in ("from_", "from"):
        from_node = select.args.get(key)
        if isinstance(from_node, exp.From):
            yield from_node.this
            break
    for join in select.args.get("joins") or []:
        if isinstance(join, exp.Join) and isinstance(join.this, exp.Expr):
            yield join.this


def _clause_roots(select: exp.Select) -> Iterator[tuple[exp.Expr, bool]]:
    """(절의 최상위 노드, 그 자리에서 출력 별칭이 이름으로 해석되는지)."""
    for key, value in select.args.items():
        if key in _SOURCE_ARGS:
            continue
        for node in _as_expressions(value):
            yield node, key in _ALIAS_VISIBLE_ARGS
    # JOIN 의 ON/USING 은 이 스코프의 식이다 (JOIN 이 들여놓는 소스 자체는 제외).
    for join in select.args.get("joins") or []:
        if not isinstance(join, exp.Join):
            continue
        for key, value in join.args.items():
            if key == "this":
                continue
            for node in _as_expressions(value):
                yield node, False


def _own_nodes(root: exp.Expr) -> Iterator[exp.Expr]:
    """`root` 아래 노드들. **중첩 SELECT 경계에서 멈춘다** (그건 다른 스코프다)."""
    if isinstance(root, exp.Select):
        return
    yield root
    for child in root.iter_expressions():
        yield from _own_nodes(child)


def _boundary_selects(root: exp.Expr) -> Iterator[exp.Select]:
    """`root` 아래에서 처음 만나는 중첩 SELECT 들."""
    if isinstance(root, exp.Select):
        yield root
        return
    for child in root.iter_expressions():
        yield from _boundary_selects(child)


def _alias_columns(node: exp.Expr) -> list[str]:
    alias = node.args.get("alias")
    if not isinstance(alias, exp.TableAlias):
        return []
    return [identifier.name.lower() for identifier in alias.columns]


class _Analyzer:
    """SELECT 트리를 스코프로 나눠 테이블·컬럼·조회 범위를 검사한다."""

    def __init__(self, whitelist: Mapping[str, tuple[str, ...]]) -> None:
        self._whitelist = {
            table.lower(): frozenset(column.lower() for column in columns)
            for table, columns in whitelist.items()
        }
        self._whitelist_names = ", ".join(whitelist)
        self.scopes: list[_Scope] = []
        self.base_tables: list[str] = []
        self._seen_tables: set[int] = set()

    # ── 스코프 구성 ─────────────────────────────────────────────────────────

    def build(
        self,
        select: exp.Select,
        parent: _Scope | None,
        cte_columns: Mapping[str, _Exported],
    ) -> _Scope:
        env = dict(cte_columns)
        self._register_ctes(select, env)

        sources: dict[str, _Source] = {}
        for node in _direct_sources(select):
            source = self._source(node, env)
            if source.name in sources:
                # 덮어쓰면 기저 테이블 `_Source` 가 스코프에서 사라지고, 그 스코프의 조회 범위
                # 검사(`check_order_scope`)가 검사할 대상을 잃어 통째로 스킵된다.
                raise _reject(
                    SqlGuardRule.DUPLICATE_SOURCE,
                    f"같은 FROM/JOIN 절에서 소스 이름이 겹친다: `{source.name}`. "
                    "테이블·부질의마다 서로 다른 별칭을 준다 (`orders o`, `orders o2`).",
                )
            sources[source.name] = source

        scope = _Scope(select=select, parent=parent, sources=sources)
        self.scopes.append(scope)

        # 절 안의 중첩 SELECT 는 바깥 이름을 참조할 수 있으므로 이 스코프를 부모로 준다.
        for root, _ in _clause_roots(select):
            for nested in _boundary_selects(root):
                self.build(nested, scope, env)
        return scope

    def _register_ctes(self, select: exp.Select, env: dict[str, _Exported]) -> None:
        with_node: exp.Expr | None = None
        for key in ("with_", "with"):
            candidate = select.args.get(key)
            if isinstance(candidate, exp.With):
                with_node = candidate
                break
        if with_node is None:
            return
        if with_node.args.get("recursive"):
            raise _reject(
                SqlGuardRule.UNSUPPORTED_SOURCE,
                "`WITH RECURSIVE` 는 허용되지 않는다 (재귀 CTE 는 실행 시간을 예측할 수 없다). "
                "재귀 없는 SELECT 로 만든다.",
            )
        for cte in with_node.expressions:
            if not isinstance(cte, exp.CTE):
                continue
            body = cte.this
            name = cte.alias_or_name.lower()
            if not isinstance(body, exp.Select):
                raise _reject(
                    SqlGuardRule.UNSUPPORTED_SOURCE,
                    f"CTE `{name}` 의 본문이 단일 SELECT 가 아니다 "
                    f"(받은 것: {type(body).__name__}). UNION 등을 쓰지 말고 SELECT 하나만 넣는다.",
                )
            # CTE 는 상관 참조를 할 수 없다 — 부모 스코프를 주지 않는다.
            inner = self.build(body, None, env)
            declared = _alias_columns(cte)
            env[name] = (
                # 컬럼 별칭 목록은 자리로 이름을 갈아 끼운다 — 어느 안쪽 projection 이
                # 어느 이름이 되는지 증명하지 않으므로 유래를 하나도 넘기지 않는다.
                _Exported(frozenset(declared), frozenset()) if declared else _exports(inner)
            )

    def _source(self, node: exp.Expr, env: Mapping[str, _Exported]) -> _Source:
        if isinstance(node, exp.Table):
            return self._table_source(node, env)
        if isinstance(node, exp.Subquery):
            body = node.this
            if not isinstance(body, exp.Select):
                raise _reject(
                    SqlGuardRule.UNSUPPORTED_SOURCE,
                    f"FROM 절의 부질의가 단일 SELECT 가 아니다 (받은 것: {type(body).__name__}).",
                )
            inner = self.build(body, None, env)
            declared = _alias_columns(node)
            exported = _Exported(frozenset(declared), frozenset()) if declared else _exports(inner)
            return _Source(
                name=node.alias.lower(),
                columns=exported.columns,
                base_table=None,
                pii_safe_columns=exported.pii_safe,
            )
        raise _reject(
            SqlGuardRule.UNSUPPORTED_SOURCE,
            f"FROM 절에 허용되지 않은 소스가 있다: {_sql_of(node)}. "
            f"화이트리스트 테이블({self._whitelist_names})이나 부질의만 읽는다.",
        )

    def _table_source(self, node: exp.Table, env: Mapping[str, _Exported]) -> _Source:
        self._seen_tables.add(id(node))
        name = node.name.lower()
        schema = node.db.lower()
        catalog = node.catalog.lower()

        if not name:
            # `FROM generate_series(...)` 같은 테이블 함수 — 이름 없는 FROM 소스다.
            raise _reject(
                SqlGuardRule.UNKNOWN_TABLE,
                "FROM 절에 테이블이 아닌 소스(함수 등)가 있다. 화이트리스트 테이블만 조회한다.",
            )
        declared = _alias_columns(node)
        if name in env and not schema and not catalog:
            exported = _Exported(frozenset(declared), frozenset()) if declared else env[name]
            return _Source(
                name=(node.alias or name).lower(),
                columns=exported.columns,
                base_table=None,
                pii_safe_columns=exported.pii_safe,
            )
        if catalog:
            raise _reject(
                SqlGuardRule.UNKNOWN_TABLE, f"다른 데이터베이스를 참조할 수 없다: {catalog}.{name}"
            )
        if schema not in ALLOWED_SCHEMAS:
            raise _reject(
                SqlGuardRule.UNKNOWN_TABLE, f"화이트리스트 밖 스키마를 참조했다: {schema}.{name}"
            )
        if name not in self._whitelist:
            raise _reject(
                SqlGuardRule.UNKNOWN_TABLE,
                f"화이트리스트 밖 테이블을 참조했다: {name}. "
                f"조회 가능한 테이블은 {self._whitelist_names} 뿐이다.",
            )
        if declared:
            # `orders AS o(ctid)` — 화이트리스트 컬럼을 임의 이름으로 갈아끼워 검사를 무력화한다.
            raise _reject(
                SqlGuardRule.UNSUPPORTED_SOURCE,
                f"테이블에 컬럼 별칭 목록을 붙일 수 없다: {_sql_of(node)}. "
                "컬럼 이름은 SELECT 목록에서 `AS` 로만 바꾼다.",
            )
        if name not in self.base_tables:
            self.base_tables.append(name)
        return _Source(
            name=(node.alias or name).lower(),
            columns=self._whitelist[name],
            base_table=name,
            # 기저 테이블의 컬럼은 정의상 직접 컬럼이다 — 유래 추적의 바닥이다.
            pii_safe_columns=self._whitelist[name],
        )

    # ── 검사 ────────────────────────────────────────────────────────────────

    def verify_all_tables_resolved(self, statement: exp.Select) -> None:
        """검증기가 훑지 않은 자리에 테이블 참조가 숨어 있으면 거부한다.

        스코프 해석이 못 본 위치는 **안전을 증명하지 못한 위치**다. 조용히 통과시키지 않는다.
        """
        for table in statement.find_all(exp.Table):
            if id(table) not in self._seen_tables:
                raise _reject(
                    SqlGuardRule.UNSUPPORTED_SOURCE,
                    f"검증기가 해석하지 못한 자리에서 테이블을 참조했다: {_sql_of(table)}. "
                    "FROM/JOIN 으로만 테이블을 읽는다.",
                )

    def referenced_tables(self) -> tuple[str, ...]:
        if not self.base_tables:
            raise _reject(
                SqlGuardRule.NO_WHITELISTED_TABLE,
                f"화이트리스트 테이블({self._whitelist_names})을 하나도 조회하지 않는다.",
            )
        return tuple(self.base_tables)

    def check_columns(self) -> None:
        for scope in self.scopes:
            for root, alias_visible in _clause_roots(scope.select):
                for node in _own_nodes(root):
                    if isinstance(node, exp.Column):
                        self._check_column(node, scope, alias_visible=alias_visible)
            self._check_using(scope)

    @staticmethod
    def _check_using(scope: _Scope) -> None:
        """`JOIN ... USING (col)` 의 컬럼도 화이트리스트 안이어야 한다.

        USING 은 `Column` 이 아니라 `Identifier` 로 파싱되므로 컬럼 검사에 걸리지 않는다.
        그런데 PostgreSQL 은 USING 컬럼을 `SELECT *` 의 출력에 **합쳐서 내보낸다** —
        `USING (ctid)` 는 화이트리스트 밖 컬럼을 결과에 실어 보내는 우회로다.
        """
        for join in scope.select.args.get("joins") or []:
            if not isinstance(join, exp.Join):
                continue
            for identifier in join.args.get("using") or []:
                if not isinstance(identifier, exp.Expr):
                    continue
                if not _resolve_bare(scope, identifier.name.lower()):
                    raise _reject(
                        SqlGuardRule.UNKNOWN_COLUMN,
                        f"화이트리스트 밖 컬럼을 USING 절에 썼다: {identifier.name}",
                    )

    def _check_column(self, node: exp.Column, scope: _Scope, *, alias_visible: bool) -> None:
        if node.catalog:
            raise _reject(
                SqlGuardRule.UNKNOWN_TABLE, f"다른 데이터베이스를 참조할 수 없다: {_sql_of(node)}"
            )
        if node.db and node.db.lower() not in ALLOWED_SCHEMAS:
            raise _reject(
                SqlGuardRule.UNKNOWN_TABLE, f"화이트리스트 밖 스키마를 참조했다: {_sql_of(node)}"
            )

        qualifier = node.table.lower()
        if qualifier:
            source = _lookup_source(scope, qualifier)
            if source is None:
                raise _reject(
                    SqlGuardRule.UNKNOWN_TABLE,
                    f"FROM 절에 없는 테이블·별칭을 한정자로 썼다: {_sql_of(node)}",
                )
            if isinstance(node.this, exp.Star):
                # `o.*` — 이미 화이트리스트로 확인된 소스의 전체 컬럼이다.
                return
            if node.name.lower() not in source.columns:
                raise _reject(
                    SqlGuardRule.UNKNOWN_COLUMN,
                    f"화이트리스트 밖 컬럼을 참조했다: {_sql_of(node)}",
                )
            return

        if isinstance(node.this, exp.Star):
            return

        name = node.name.lower()
        if _resolve_bare(scope, name):
            return
        if name in scope.output_aliases:
            if alias_visible:
                return
            raise _reject(
                SqlGuardRule.UNKNOWN_COLUMN,
                f"화이트리스트 밖 컬럼을 참조했다: {_sql_of(node)}. "
                f"같은 이름의 별칭(`AS {node.name}`)을 SELECT 목록에 뒀더라도 소용없다 — "
                "별칭 이름은 ORDER BY·GROUP BY 에서만 쓸 수 있고, 그 밖의 자리에서 같은 이름은 "
                "별칭이 아니라 테이블 컬럼으로 해석된다.",
            )
        raise _reject(
            SqlGuardRule.UNKNOWN_COLUMN, f"화이트리스트 밖 컬럼을 참조했다: {_sql_of(node)}"
        )

    def check_order_scope(self, order_no: str) -> None:
        """화이트리스트 테이블을 읽는 **모든 스코프**가 주문 1건으로 묶였는지 확인한다.

        기저 테이블을 읽는 스코프를 하나도 찾지 못하면 **통과가 아니라 거부**다. 그 상황은
        "한정할 것이 없다"가 아니라 "해석이 기저 테이블을 놓쳤다"는 뜻이고
        (`referenced_tables()` 가 이미 기저 테이블이 있음을 확인한 뒤에 불린다),
        검사하지 못한 쿼리를 통과시키면 게이트가 조용히 열린다.
        """
        checked = 0
        for scope in self.scopes:
            bases = [source for source in scope.sources.values() if source.base_table is not None]
            if not bases:
                continue
            checked += 1
            bound = _bound_source_names(scope, order_no)
            for source in bases:
                if source.name in bound:
                    continue
                # 한정자 없는 `order_no = '...'` 는 기저 테이블이 하나뿐일 때만 그것을 가리킨다.
                if len(bases) == 1 and "" in bound:
                    continue
                raise _reject(
                    SqlGuardRule.ORDER_SCOPE,
                    f"조회 범위가 주문 1건으로 한정되지 않았다 (`{source.name}`). "
                    f"WHERE 절에 `{ORDER_SCOPE_COLUMN} = '{order_no}'` 동등 비교를 AND 로 넣는다 "
                    "— 조건을 빼거나, OR 로 묶거나, 다른 주문번호를 쓰면 거부된다.",
                )
        if not checked:
            raise _reject(
                SqlGuardRule.ORDER_SCOPE,
                f"조회 범위를 확인할 기저 테이블({self._whitelist_names})을 스코프에서 찾지 "
                f"못했다. `FROM {ORDERS_TABLE}` 로 직접 읽고 WHERE 절에 "
                f"`{ORDER_SCOPE_COLUMN} = '{order_no}'` 를 넣는다.",
            )


def _lookup_source(scope: _Scope | None, qualifier: str) -> _Source | None:
    while scope is not None:
        source = scope.sources.get(qualifier)
        if source is not None:
            return source
        scope = scope.parent
    return None


def _resolve_bare(scope: _Scope | None, name: str) -> bool:
    while scope is not None:
        if any(name in source.columns for source in scope.sources.values()):
            return True
        scope = scope.parent
    return False


def _resolve_bare_pii_safe(scope: _Scope | None, name: str) -> bool:
    """한정자 없는 이름이 **유래가 증명된** 컬럼을 가리키는가.

    이름을 내놓는 소스가 여럿이면 **전부** 증명돼 있어야 한다. 하나라도 아니면 어느 쪽이
    나올지 코드가 정하지 못하므로 증명이 아니다.
    """
    while scope is not None:
        owners = [source for source in scope.sources.values() if name in source.columns]
        if owners:
            return all(name in source.pii_safe_columns for source in owners)
        scope = scope.parent
    return False


# ── PII allowlist 출력 유래 ─────────────────────────────────────────────────


class _Provenance(Enum):
    """결과 컬럼 값이 어디서 오는가."""

    #: `orders` 직접 컬럼에서 왔음이 증명됐다 (임시 테이블·파생 테이블·빈칸 채우기 경유 포함).
    DIRECT = "direct"
    #: 값이 **개인정보 모양이 아닌 고정값**이다. 빈칸 채우기 자리에서만 의미가 있다.
    FILLER = "filler"
    #: 증명하지 못했다 — 그러므로 승인하지 않는다.
    UNKNOWN = "unknown"


def _unwrap_parens(node: exp.Expr) -> exp.Expr:
    """괄호만 벗긴다. **부질의는 벗기지 않는다** — `unnest()` 는 `Subquery` 도 벗겨서,
    스칼라 부질어를 안쪽 SELECT 로 오인하게 만든다."""
    value = node
    while isinstance(value, exp.Paren) and isinstance(value.this, exp.Expr):
        value = value.this
    return value


def _coalesce_arguments(node: exp.Func) -> Iterator[exp.Expr]:
    """`coalesce` 의 인자 전부 — 어느 것이든 출력이 될 수 있다."""
    if isinstance(node.this, exp.Expr):
        yield node.this
    for item in node.args.get("expressions") or []:
        if isinstance(item, exp.Expr):
            yield item


def _projection_provenance(node: exp.Expr, scope: _Scope) -> _Provenance:
    """이 식의 값이 `orders` 직접 컬럼에서 왔음을 증명할 수 있는가.

    **모르는 표현식은 자동으로 불승인**이다(fail-closed). 넓히는 방향은 허용 목록을 늘리는
    것이지 거부 목록을 두는 것이 아니라, 새 SQL 문법이 생겨도 조용히 승인되지 않는다.

    증명되는 세 가지뿐이다.

    1. **직접 컬럼** — 기저 테이블의 컬럼. 임시 테이블(CTE)·파생 테이블을 거쳐도, 그 안쪽
       스코프가 같은 방식으로 증명했으면 유래가 이어진다(`_Source.pii_safe_columns`).
    2. **개인정보 모양이 아닌 고정값** — `'미정'` 처럼. **"개인정보 모양"의 정의는 이 모듈이
       만들지 않는다** — L1 게이트가 단독 소유하는 패턴 집합(`gate.pii_shaped`)을 부르기만
       한다. 층마다 자기 정규식을 두면 한쪽만 넓혀져 기준이 갈리고, 접기가 층마다 달라
       실제로 그렇게 뚫렸다.
    3. **빈칸 채우기(1·2의 조합)** — `coalesce`·`nullif` 는 값을 **그대로** 내놓고 조각을
       이어 붙이지 않는다. `coalesce` 는 인자가 전부 1이나 2이고 그중 하나라도 1이면 출력의
       가능한 값이 전부 (진짜 컬럼 값 | 개인정보가 아닌 고정값) 이다. `nullif(a, b)` 는
       출력이 `a` 아니면 NULL 이므로 **`a` 가 1이어야** 하고, `b` 는 값이 흐르지 않는
       비교 상대라 유래로 세지 않는다.

    **이어 붙이기 계열(`concat`·`||`)은 여기 없다.** 조각 하나하나는 개인정보 모양이 아닌데
    합치면 번호가 된다(`'01' || '0-9999-9999'`) — 그 길을 열지 않으려면 고정값이 이어 붙이기
    안으로 들어가는 자리가 아예 없어야 한다. 조건 분기(`CASE`)도 넣지 않는다: 넓히는 이유가
    "생성 안내가 빈칸 채우기를 광고한다"는 것이고, 조건 분기는 그 이유에 해당하지 않는다.
    """
    value = _unwrap_parens(node)

    if isinstance(value, exp.Column):
        if isinstance(value.this, exp.Star):
            # `o.*` 는 projection 단위가 아니라 소스 단위로 따로 편다.
            return _Provenance.UNKNOWN
        name = value.name.lower()
        qualifier = value.table.lower()
        if qualifier:
            source = _lookup_source(scope, qualifier)
            proven = source is not None and name in source.pii_safe_columns
        else:
            proven = _resolve_bare_pii_safe(scope, name)
        return _Provenance.DIRECT if proven else _Provenance.UNKNOWN

    if isinstance(value, exp.Literal):
        # `E'...'`·`U&'...'`·`$$...$$`·`N'...'` 는 `Literal` 이 아닌 별도 노드라 여기 오지
        # 않는다 — 이스케이프가 풀린 뒤의 값을 이 모듈이 계산하지 않으므로 그 편이 옳다.
        return _Provenance.UNKNOWN if pii_shaped(str(value.this)) else _Provenance.FILLER

    if isinstance(value, _COALESCE_FUNCTION_TYPES):
        provenances = [_projection_provenance(arg, scope) for arg in _coalesce_arguments(value)]
        if not provenances or _Provenance.UNKNOWN in provenances:
            return _Provenance.UNKNOWN
        # 인자가 전부 고정값이면 유래가 없다 — 승인할 직접 컬럼이 하나도 없다는 뜻이다.
        return _Provenance.DIRECT if _Provenance.DIRECT in provenances else _Provenance.UNKNOWN

    if isinstance(value, _NULLIF_FUNCTION_TYPES):
        compared = value.args.get("expression")
        if not isinstance(compared, exp.Expr) or not isinstance(value.this, exp.Expr):
            return _Provenance.UNKNOWN
        if _projection_provenance(compared, scope) is _Provenance.UNKNOWN:
            return _Provenance.UNKNOWN
        return _projection_provenance(value.this, scope)

    return _Provenance.UNKNOWN


def _star_sources(projection: exp.Expr, scope: _Scope) -> tuple[_Source, ...] | None:
    """이 projection 이 별표라면 그것이 펴는 소스들, 별표가 **아니면** `None`.

    빈 튜플과 `None` 은 다른 상태다 — 빈 튜플은 "별표인데 한정자가 가리키는 소스를 해석하지
    못했다"이고, 그때는 내놓는 이름을 모르므로 아무것도 넘기지 않는다(fail-closed).
    """
    if isinstance(projection, exp.Star):
        return tuple(scope.sources.values())
    value = projection.this if isinstance(projection, exp.Alias) else projection
    if isinstance(value, exp.Column) and isinstance(value.this, exp.Star):
        qualified = scope.sources.get(value.table.lower())
        return () if qualified is None else (qualified,)
    return None


def _exports(scope: _Scope) -> _Exported:
    """스코프가 바깥에 내놓는 컬럼 이름들과, 그중 유래가 증명된 이름들.

    **같은 이름을 내놓는 projection 이 여럿이면 전부 증명돼야 한다.** 바깥 스코프의 출력
    이름 중복 거부(`_check_unique_output_names`)는 안쪽 SELECT 를 보지 않으므로, 안쪽에서
    계산값이 직접 컬럼과 같은 이름을 쓰면 여기서 걸러야 유래가 조용히 넘어가지 않는다.
    """
    columns: set[str] = set()
    proven: set[str] = set()
    unproven: set[str] = set()

    def record(name: str, *, is_direct: bool) -> None:
        columns.add(name)
        (proven if is_direct else unproven).add(name)

    for projection in scope.select.expressions:
        stars = _star_sources(projection, scope)
        if stars is not None:
            for source in stars:
                for column in source.columns:
                    record(column, is_direct=column in source.pii_safe_columns)
            continue
        output = projection.output_name
        if not output:
            continue
        value = projection.this if isinstance(projection, exp.Alias) else projection
        record(
            output.lower(),
            is_direct=_projection_provenance(value, scope) is _Provenance.DIRECT,
        )

    return _Exported(frozenset(columns), frozenset(proven - unproven))


# ── 3. 함수 허용 목록 ───────────────────────────────────────────────────────


def _function_label(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return f"{node.name}()"
    names = node.sql_names()
    return f"{names[0].lower()}()" if names else type(node).__name__.lower()


def _check_functions(statement: exp.Select) -> None:
    """허용 목록 밖 함수 호출을 거부한다.

    `SELECT order_no FROM orders WHERE pg_sleep(2) IS NULL` 처럼 화이트리스트 테이블을 끼워
    넣으면 테이블·컬럼 검사만으로는 함수 호출을 잡지 못한다. 실행 시간 상한은 커넥션의
    `statement_timeout`(`db.readonly_connect`)이 마지막으로 막지만, 여기서 먼저 막는 편이
    재시도 피드백도 정확하고 DB 를 건드리지도 않는다.
    """
    for node in statement.find_all(exp.Func):
        if isinstance(node, _PERMITTED_FUNCTION_TYPES):
            continue
        raise _reject(
            SqlGuardRule.FORBIDDEN_FUNCTION,
            f"허용되지 않은 함수를 호출했다: {_function_label(node)}. "
            f"쓸 수 있는 함수는 {_describe_function_names()} 뿐이다.",
        )


def _check_cast_types(statement: exp.Select) -> None:
    """형변환의 **대상 타입**을 허용 목록으로 판정한다 (fail-closed).

    캐스트가 허용 함수인데 대상 타입을 읽지 않으면 `cast('inquiries' AS regclass)` 한 줄로
    화이트리스트 밖 테이블의 **이름과 존재 여부**가 결과 컬럼이 된다 — 존재하면 이름이
    돌아오고 없으면 오류가 그 사실을 알려주므로, 양방향으로 카탈로그를 훑을 수 있다.
    시스템 카탈로그는 PUBLIC 권한이라 read-only 계정이 그대로 읽는다(실측으로 확인했다 —
    `tests/test_sql_cast_allowlist.py`). **막는 층이 이 검사 하나뿐이다.**

    카탈로그 참조 타입(`oid`·`reg*`)은 sqlglot 이 `exp.ObjectIdentifier` 로 만들어 `this` 가
    타입 열거형이 아니고, 확장 타입은 `USERDEFINED` 다 — 둘 다 허용 집합에 들 수 없으므로
    새 타입 표기가 생겨도 자동으로 거부된다.
    """
    for cast in statement.find_all(exp.Cast):
        target = cast.args.get("to")
        if isinstance(target, exp.DataType) and target.this in _ALLOWED_CAST_TYPE_NODES:
            continue
        label = _sql_of(target) if isinstance(target, exp.Expr) else "(알 수 없음)"
        raise _reject(
            SqlGuardRule.UNSUPPORTED_CAST_TYPE,
            f"허용되지 않은 형변환 대상 타입이다: {label}. {describe_allowed_cast_types()}",
        )


def _check_projections(statement: exp.Select) -> None:
    """최상위 직접 상수 projection 을 중복 방어로 거부한다.

    PII allowlist 의 권위 있는 경계는 ValidatedQuery.pii_safe_output_columns 다. 이 검사는
    정확 재현인 직접 상수 형태를 실행 전에 빠르게 거부하되, 허용 함수·계산식을 막지 않는다.
    계산 결과는 표시용 content 에는 남지만 allowlist 근거 컬럼에는 포함되지 않는다.
    """
    for projection in statement.expressions:
        value = projection.this if isinstance(projection, exp.Alias) else projection
        if not isinstance(value.unnest(), exp.Literal):
            continue
        raise _reject(
            SqlGuardRule.UNSUPPORTED_PROJECTION,
            "상수만 결과 컬럼으로 내보낼 수 없다: "
            f"{_sql_of(projection)}. orders 의 화이트리스트 컬럼을 직접 조회한다.",
        )


def _database_output_name(projection: exp.Expr) -> str:
    """PostgreSQL 결과 메타데이터에 나타나는 projection 이름을 반환한다."""
    if isinstance(projection, exp.Alias):
        alias = projection.args.get("alias")
        if isinstance(alias, exp.Identifier) and alias.args.get("quoted"):
            return projection.output_name
    return projection.output_name.lower()


def _projection_output_names(scope: _Scope) -> tuple[str, ...]:
    """최상위 결과 컬럼 이름을 별표까지 펼쳐 반환한다.

    psycopg ``DictRow`` 는 같은 이름의 결과 컬럼을 dict 로 낮출 때 뒤 값을 남긴다. 따라서
    provenance 를 출력명으로 전달하려면 이름이 유일하다는 전제가 먼저 증명돼야 한다.
    """
    names: list[str] = []
    for projection in scope.select.expressions:
        if isinstance(projection, exp.Star):
            for source in scope.sources.values():
                names.extend(sorted(source.columns))
            continue

        value = projection.this if isinstance(projection, exp.Alias) else projection
        if isinstance(value, exp.Column) and isinstance(value.this, exp.Star):
            qualified_source = scope.sources.get(value.table.lower())
            if qualified_source is not None:
                names.extend(sorted(qualified_source.columns))
            continue
        if projection.output_name:
            names.append(_database_output_name(projection))
    return tuple(names)


def _check_unique_output_names(scope: _Scope) -> None:
    """dict 행에서 provenance 가 다른 값으로 덮이지 않도록 출력 이름 중복을 거부한다."""
    names = _projection_output_names(scope)
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise _reject(
            SqlGuardRule.UNSUPPORTED_PROJECTION,
            "결과 컬럼 이름은 서로 달라야 한다: "
            f"{', '.join(duplicates)}. 각 projection 에 고유한 별칭을 붙인다.",
        )


def _pii_safe_output_columns(scope: _Scope) -> tuple[str, ...]:
    """유래가 `orders` 직접 컬럼으로 증명된 최상위 출력명만 PII allowlist 출처로 승인한다.

    승인되지 않은 출력의 값은 근거 렌더가 개인정보 모양이면 버린다 — DB 에서 돌아온 값이라도
    LLM 이 쓴 리터럴을 합성한 것일 수 있기 때문이다. 반대로 **승인된 값은 마스킹하지 않는다**:
    가리면 정상 에코가 근거와 대조되지 않아 오기각된다.

    유래 판정은 `_projection_provenance` 하나가 한다 — 임시 테이블·파생 테이블 경유와
    빈칸 채우기 관용구까지 거기서 증명하고, 증명하지 못한 것은 전부 불승인이다(fail-closed).
    """
    safe: list[str] = []
    for projection in scope.select.expressions:
        stars = _star_sources(projection, scope)
        if stars is not None:
            for source in stars:
                safe.extend(sorted(source.pii_safe_columns))
            continue

        value = projection.this if isinstance(projection, exp.Alias) else projection
        if _projection_provenance(value, scope) is _Provenance.DIRECT and projection.output_name:
            safe.append(_database_output_name(projection))

    return tuple(dict.fromkeys(safe))


# ── 4. 주문 1건 한정 ────────────────────────────────────────────────────────


def _conjuncts(node: exp.Expr) -> Iterator[exp.Expr]:
    """AND 로 이어진 최상위 조건들. **OR 안쪽으로는 들어가지 않는다.**"""
    unwrapped = node.unnest()
    if isinstance(unwrapped, exp.And):
        yield from _conjuncts(unwrapped.this)
        yield from _conjuncts(unwrapped.expression)
        return
    yield unwrapped


def _matches_order_literal(node: exp.Expr, order_no: str) -> bool:
    literal = node.unnest()
    return isinstance(literal, exp.Literal) and literal.is_string and literal.this == order_no


def _order_scope_qualifier(condition: exp.Expr, order_no: str) -> str | None:
    """이 조건이 주문 1건을 묶는다면 그 한정자(별칭)를, 아니면 `None`."""
    if isinstance(condition, exp.EQ):
        sides = ((condition.this, condition.expression), (condition.expression, condition.this))
        for column_side, literal_side in sides:
            column = column_side.unnest()
            if (
                isinstance(column, exp.Column)
                and column.name.lower() == ORDER_SCOPE_COLUMN
                and _matches_order_literal(literal_side, order_no)
            ):
                return column.table.lower()
        return None
    # `order_no IN ('ORD-...')` — 값이 전부 그 주문번호여야 한다.
    if isinstance(condition, exp.In):
        values = condition.args.get("expressions")
        column = condition.this.unnest() if condition.this is not None else None
        if (
            isinstance(column, exp.Column)
            and column.name.lower() == ORDER_SCOPE_COLUMN
            and isinstance(values, list)
            and values
            and all(_matches_order_literal(value, order_no) for value in values)
        ):
            return column.table.lower()
    return None


def _bound_source_names(scope: _Scope, order_no: str) -> set[str]:
    """이 스코프에서 주문 1건으로 묶인 소스 이름들 (한정자 없으면 빈 문자열).

    바인딩으로 인정하는 자리는 **WHERE 절과 INNER/CROSS 조인의 ON 절뿐이다.** 외부 조인의 ON
    절은 보존측 테이블을 거르지 않으므로 한정 조건이 아니다 — `_check_joins` 가 외부 조인을
    이미 거부하지만, 나중에 외부 조인을 허용하게 되더라도 이 규칙이 없으면 조용히 샌다.
    """
    bound: set[str] = set()
    conditions: list[exp.Expr] = []

    where = scope.select.args.get("where")
    if isinstance(where, exp.Where) and where.this is not None:
        conditions.append(where.this)
    for join in scope.select.args.get("joins") or []:
        if isinstance(join, exp.Join) and _is_inner_join(join):
            on = join.args.get("on")
            if isinstance(on, exp.Expr):
                conditions.append(on)

    for condition in conditions:
        for conjunct in _conjuncts(condition):
            qualifier = _order_scope_qualifier(conjunct, order_no)
            if qualifier is not None:
                bound.add(qualifier)
    return bound


# ── 5. 결과 행 수 상한 ──────────────────────────────────────────────────────


def _limit_count(limit_node: exp.Expr) -> exp.Expr | None:
    """`LIMIT n` 과 `FETCH FIRST n ROWS ONLY` 에서 행 수 식을 꺼낸다."""
    if isinstance(limit_node, exp.Fetch):
        options = limit_node.args.get("limit_options")
        if isinstance(options, exp.LimitOptions) and (
            options.args.get("percent") or options.args.get("with_ties")
        ):
            raise _reject(
                SqlGuardRule.UNSUPPORTED_LIMIT,
                "`FETCH FIRST ... PERCENT` 와 `WITH TIES` 는 결과 행 수를 실행 전에 확정할 수 "
                "없으므로 허용되지 않는다. `LIMIT n` 으로 행 수를 직접 적는다.",
            )
        count = limit_node.args.get("count")
        return count if isinstance(count, exp.Expr) else None
    expression = limit_node.args.get("expression")
    return expression if isinstance(expression, exp.Expr) else None


def _enforce_limit(statement: exp.Select, max_rows: int) -> tuple[exp.Select, int]:
    """결과 행 수 상한을 **강제**한다 — 거부가 아니라 LIMIT 을 붙이거나 낮춘다.

    상한 없는 쿼리를 그대로 실행시키지 않는 것이 목적이므로, 상한이 없으면 붙이고
    상한을 넘으면 상한으로 낮춘다. 다만 행 수가 정수 리터럴이 아니면(부질의·파라미터)
    상한 이하임을 실행 전에 확인할 수 없으므로 거부한다.
    """
    limit_node = statement.args.get("limit")
    effective = max_rows
    if isinstance(limit_node, exp.Expr):
        value = _limit_count(limit_node)
        if not isinstance(value, exp.Literal) or not value.is_int:
            raise _reject(
                SqlGuardRule.UNSUPPORTED_LIMIT,
                "행 수 상한은 정수 리터럴이어야 한다 "
                f"(받은 절: {_sql_of(limit_node)}). `LIMIT n` 형태로 적는다.",
            )
        requested = int(value.name)
        if requested < 1:
            raise _reject(
                SqlGuardRule.UNSUPPORTED_LIMIT, f"행 수 상한은 1 이상이어야 한다: {requested}"
            )
        effective = min(requested, max_rows)
    return statement.limit(effective), effective


# ── 진입점 ──────────────────────────────────────────────────────────────────


def validate_sql(
    sql: str,
    *,
    order_no: str,
    max_rows: int,
    whitelist: Mapping[str, tuple[str, ...]] = SCHEMA_WHITELIST,
) -> ValidatedQuery:
    """생성된 SQL 을 실행 전에 검증한다. 통과하지 못하면 `SqlGuardRejection`.

    `order_no` 는 **선검사(`evidence.order_exists`)를 통과한 정규화된 주문번호**다. 생성된
    쿼리가 그 1건으로 한정됐는지를 코드가 AST 로 확인한다 — 프롬프트 문구에 맡기지 않는다.

    `max_rows` 가 0 이하이거나 `order_no` 가 주문번호 형식이 아닌 것은 LLM 산출 문제가 아니라
    호출 측 오류이므로 `SqlGuardRejection` 이 아니라 `ValueError` 다.
    """
    if max_rows < 1:
        raise ValueError(f"결과 행 수 상한은 1 이상이어야 한다: {max_rows}")
    if not is_valid_order_no(order_no):
        raise ValueError(f"선검사를 통과한 주문번호여야 한다: {order_no!r}")

    text = sql.strip()
    if not text:
        raise _reject(SqlGuardRule.EMPTY, "SQL 이 비어 있다.")

    statement = _require_read_only_select(_parse_single_statement(text))
    _check_joins(statement)

    analyzer = _Analyzer(whitelist)
    root_scope = analyzer.build(statement, None, {})
    analyzer.verify_all_tables_resolved(statement)
    tables = analyzer.referenced_tables()
    analyzer.check_columns()
    _check_functions(statement)
    _check_cast_types(statement)
    _check_projections(statement)
    _check_unique_output_names(root_scope)
    analyzer.check_order_scope(order_no)

    limited, effective = _enforce_limit(statement, max_rows)

    return ValidatedQuery(
        sql=limited.sql(dialect=DIALECT),
        limit=effective,
        max_rows=max_rows,
        tables=tables,
        order_no=order_no,
        pii_safe_output_columns=_pii_safe_output_columns(root_scope),
    )
