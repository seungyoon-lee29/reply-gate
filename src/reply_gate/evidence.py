"""근거 수집 — 의도 해석[LLM] → 코드가 조회 실행 (docs/architecture.md "대표 흐름" 3단계).

**조회 실행 주체는 항상 코드다.** LLM 이 하는 일은 두 가지뿐이다: 필요한 근거 소스를
분류하는 것(`policy`/`order`/`both`)과 주문 조회 SQL **문자열을 만드는 것**. 그 문자열이
DB 에 닿을지, 몇 번 다시 만들지, 결과를 근거로 채택할지는 전부 이 모듈의 코드가 정한다.

이 모듈은 게이트가 아니다 — 근거를 모으기만 하고 초안을 판정하지 않는다(L1 은 `gate.py`).

수집의 결과는 둘 중 하나다: **근거 목록** 또는 **초안 전 인계 사유**. 어느 쪽이든 그 시점까지
모은 근거·토큰 사용량(생성/임베딩 분리)·SQL 실패 내역을 함께 돌려준다 — 처리 기록과 API
응답이 그것을 쓴다(docs/contracts.md "POST /inquiries" — 초안 전 인계라도 수집된 근거는
citations 에 남긴다).

인계 우선순위 (docs/business-rules.md "근거 부족 판정"):

* 주문 측의 **구조적** 사유(`missing_order_ref`·`order_not_found`·`sql_failed`)는 정책 근거를
  이미 확보했더라도 인계가 우선한다.
* 반면 선검사를 통과한 주문에 대해 **SQL 이 정상 실행되고 0건**인 것은 구조적 실패가 아니라
  "주문 소스 근거 0건"이다 — `order` 단독이면 `no_evidence`, `both` 면 정책 근거만으로 진행한다.

실패 정책 (docs/standards.md "재시도 상한" / docs/business-rules.md "인계 사유 6종"):

* **전송 오류**는 래퍼가 이미 1회 재시도했다 — `LLMCallError` 가 오면 어느 단계에서 왔든
  `llm_call_failed` 인계이고, 실패한 단계 이름을 남긴다. SQL 생성의 전송 오류도 여기에 속한다.
* **의도 해석의 형식 불일치**는 이 모듈이 1회 재시도하고, 재실패하면 `llm_call_failed`.
* **SQL 생성의 형식 불일치·안전장치 거부·실행 오류**는 SQL 실패 경로다 — 오류를 피드백으로
  실어 **SQL 생성만 1회 재시도**하고 재실패하면 `sql_failed`. 재시도 상한은 코드가 강제한다.
* **검색 단계(질의 재작성)의 실패는 인계가 아니라 폴백**이다 — 원문 질의로 검색을 계속한다
  (docs/business-rules.md "검색 단계 실패"). 검색은 검증이 아니라 재료 수집이라
  fail-closed 원칙의 적용 대상이 아니다. 대신 폴백 사유와 토큰이 수집 결과에 실려
  처리 기록·리포트에 남는다.
  **미배선은 폴백이 아니라 조립 시점 오류**다(`RetrievalWiringError`).
"""

from __future__ import annotations

import contextlib
import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any, Final

import psycopg
from psycopg.rows import DictRow

from reply_gate.config import Settings, get_settings
from reply_gate.contracts import (
    EscalationReason,
    Evidence,
    EvidenceSource,
    IntentSource,
    sql_evidence_id,
)
from reply_gate.gate import DEFAULT_PII_PATTERNS, fold_for_detection
from reply_gate.llm import (
    EmbeddingClient,
    GenerationClient,
    LLMCallError,
    LLMFormatError,
)
from reply_gate.order_ref import is_valid_order_no, normalize_order_no
from reply_gate.policy_index import PolicySearchHit, search_policy_chunks
from reply_gate.query_rewrite import rewrite_query
from reply_gate.retrieval_strategies import (
    AbstentionGate,
    VectorHit,
    apply_abstention_gate,
    merge_rewritten_rankings,
    truncate_for_gate,
)
from reply_gate.sql_guard import (
    SCHEMA_WHITELIST,
    SqlGuardRejection,
    ValidatedQuery,
    describe_allowed_functions,
    describe_whitelist,
    validate_sql,
)

__all__ = [
    "INQUIRY_EMBEDDING_STAGE",
    "INTENT_JSON_SCHEMA",
    "INTENT_MAX_ATTEMPTS",
    "INTENT_STAGE",
    "INTENT_SYSTEM_PROMPT",
    "ORDER_EXISTS_SQL",
    "SQL_GENERATION_STAGE",
    "SQL_JSON_SCHEMA",
    "SQL_MAX_ATTEMPTS",
    "SQL_SYSTEM_PROMPT",
    "EvidenceCollection",
    "EvidenceCollector",
    "IntentResult",
    "RetrievalWiringError",
    "SqlEvidenceSnapshot",
    "SqlFailure",
    "SqlFailureKind",
    "SqlGenerationResult",
    "adopt_policy_hits",
    "build_intent_user_prompt",
    "build_sql_user_prompt",
    "classify_intent",
    "execute_validated_query",
    "generate_sql",
    "merge_policy_rankings",
    "order_exists",
]


class RetrievalWiringError(RuntimeError):
    """검색 전략이 켜졌는데 그 클라이언트가 배선되지 않았다 — **조립 시점 오류**다.

    `pipeline.PipelineWiringError` 와 같은 자격이다(docs/business-rules.md "검색 단계 실패"):
    기본값·`None` 이 조용히 전략을 끄는 fail-open 배선을 금지한다. 이름이 갈린 것은 자리가 다르기
    때문이다 — 협력자를 받는 조립자가 각자 자기 배선을 검사하고, `evidence` 는
    `pipeline` 을 import 할 수 없다(순환).

    **인계 사유가 아니다.** 업무 판정이 아니라 배선 오류이므로 `llm_call_failed` 로 삼키지
    않고 그대로 올려보낸다. 재작성 **호출의 실패**는 반대로 폴백이다(같은 절) — 배선이
    없는 것과 호출이 실패한 것을 섞으면, 배선을 빠뜨린 실행이 "재작성을 썼는데 매번
    실패했다"로 집계된다.
    """


# ── 단계 이름 (처리 기록·비용 산출·`llm_call_failed` 의 failed_stage) ────────

INTENT_STAGE: Final = "intent"
SQL_GENERATION_STAGE: Final = "sql_generation"
INQUIRY_EMBEDDING_STAGE: Final = "inquiry_embedding"

#: 의도 해석: 최초 호출 + 형식 불일치 재시도 1회 (docs/standards.md "재시도 상한").
INTENT_MAX_ATTEMPTS: Final = 2

#: SQL 생성: 최초 호출 + 실패 피드백 재시도 1회 (docs/standards.md "재시도 상한").
SQL_MAX_ATTEMPTS: Final = 2

#: 표시용 `content` 에 싣는 SQL 결과 행 수. 원문 대조용 `evidence_text` 는 자르지 않는다.
CONTENT_ROW_LIMIT: Final = 10


# ── 구조화 출력 스키마 ──────────────────────────────────────────────────────

#: 의도 해석의 구조화 출력. 값 집합은 `contracts.IntentSource` 와 같아야 한다.
INTENT_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {"source": {"type": "string", "enum": [source.value for source in IntentSource]}},
    "required": ["source"],
    "additionalProperties": False,
}

#: SQL 생성의 구조화 출력. **SQL 의 안전성은 이 스키마가 보장하지 않는다** —
#: 문자열이라는 것만 강제하고, 내용 검증은 전부 `sql_guard` 가 실행 전에 한다.
SQL_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {"sql": {"type": "string"}},
    "required": ["sql"],
    "additionalProperties": False,
}


# ── 프롬프트 (한국어) ───────────────────────────────────────────────────────

INTENT_SYSTEM_PROMPT: Final = """\
너는 한국어 이커머스 고객센터 문의를 읽고, 답변에 **필요한 근거 소스**를 분류한다.
답변을 쓰지 말고 분류만 한다.

- `policy`: 정책 문서만으로 답할 수 있다 (환불 기간, 교환 조건, 배송비 규정 같은 일반 규정).
- `order`: 특정 주문의 데이터만 있으면 답할 수 있다 (내 주문 상태, 송장번호, 배송 일자 등).
- `both`: 정책과 특정 주문 데이터가 **모두** 필요하다 (이 주문이 환불 기간 안인지 등).

산출은 다음 형태의 JSON 이다:
{"source": "policy"}
"""

SQL_SYSTEM_PROMPT: Final = """\
너는 이커머스 고객센터의 주문 조회 SQL 생성기다. PostgreSQL **SELECT 문 하나**만 만든다.
너는 SQL 문자열만 만들고, 검증과 실행은 코드가 한다 — 아래 규칙을 어기면 실행 전에 거부된다.

1. **화이트리스트 밖은 참조하지 않는다.** 아래에 적힌 테이블과 컬럼만 쓴다.
   컬럼 별칭(`AS`)은 화이트리스트를 넓히지 않는다 — 별칭 이름은 ORDER BY·GROUP BY 에서만
   이름으로 쓸 수 있다.
2. **SELECT 단일문만.** INSERT·UPDATE·DELETE·DDL, 세미콜론으로 이은 여러 문장,
   주석(`--`, `/* */`), 행 잠금 절(`FOR UPDATE`·`FOR SHARE`)은 모두 거부된다.
3. **주어진 주문번호 1건만 조회한다.** 주문 테이블을 읽는 모든 SELECT 의 WHERE 절에
   `order_no = '<주어진 주문번호>'` 동등 비교를 **AND 로** 넣는다. 조건을 빼거나, OR 로 묶거나,
   다른 주문번호를 쓰면 거부된다. **외부 조인(LEFT/RIGHT/FULL JOIN)은 쓰지 않는다** — 외부
   조인의 ON 절은 보존측을 거르지 않아 조회 범위가 1건으로 묶이지 않으므로 거부된다.
4. **함수는 허용 목록 안에서만** 호출한다 (아래 [허용 함수]).
5. 문의에 답하는 데 필요한 컬럼을 고른다. 판단이 서지 않으면 넉넉히 고른다.
6. 결과 행 수 상한을 넘는 LIMIT 은 쓰지 않는다.

산출은 다음 형태의 JSON 이다:
{"sql": "SELECT ... FROM orders WHERE order_no = '...'"}
"""


def build_intent_user_prompt(
    *,
    inquiry: str,
    order_no: str | None = None,
    previous_output: str | None = None,
    previous_error: str | None = None,
) -> str:
    """의도 해석 프롬프트. 재시도면 직전 산출과 실패 사유를 피드백으로 싣는다."""
    sections = [
        f"[문의]\n{inquiry}",
        f"[접수된 주문번호]\n{order_no if order_no else '없음'}",
    ]
    if previous_error is not None:
        lines = ["[직전 산출이 형식에 맞지 않았다]", f"- 사유: {previous_error}"]
        if previous_output:
            lines.append(f"- 직전 산출: {previous_output}")
        lines.append('policy · order · both 중 하나만 골라 {"source": "..."} 형태로만 낸다.')
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def build_sql_user_prompt(
    *,
    inquiry: str,
    order_no: str,
    max_rows: int,
    whitelist: Mapping[str, tuple[str, ...]] = SCHEMA_WHITELIST,
    previous_sql: str | None = None,
    previous_error: str | None = None,
) -> str:
    """SQL 생성 프롬프트. 재시도면 직전 쿼리와 거부·실패 사유를 피드백으로 싣는다."""
    sections = [
        f"[문의]\n{inquiry}",
        f"[조회할 주문번호]\n{order_no}",
        f"[조회 가능한 테이블·컬럼 (화이트리스트)]\n{describe_whitelist(whitelist)}",
        f"[허용 함수]\n{describe_allowed_functions()}",
        f"[결과 행 수 상한]\n{max_rows}",
    ]
    if previous_error is not None:
        lines = ["[직전 SQL 이 실행되지 못했다]"]
        if previous_sql:
            lines.append(f"- 직전 쿼리: {previous_sql}")
        lines.append(f"- 사유: {previous_error}")
        lines.append("위 사유를 고쳐 SELECT 문을 다시 만든다.")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


# ── 의도 해석 ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class IntentResult:
    """의도 해석 1회분(재시도 포함)의 결과 + 합산 토큰.

    `source` 가 `None` 이면 재시도까지 형식이 맞지 않은 것이고 `error` 에 사유가 담긴다
    (호출자는 `llm_call_failed` 로 인계한다). **전송 오류는 여기로 오지 않고**
    `LLMCallError` 로 올라간다 — 래퍼가 이미 재시도했으므로 여기서 또 재시도하지 않는다.
    """

    source: IntentSource | None
    input_tokens: int
    output_tokens: int
    error: str | None
    attempts: int


def _parse_intent(data: object) -> IntentSource | None:
    if not isinstance(data, Mapping):
        return None
    value = data.get("source")
    if not isinstance(value, str):
        return None
    try:
        return IntentSource(value)
    except ValueError:
        return None


def classify_intent(
    *,
    client: GenerationClient,
    inquiry: str,
    order_no: str | None = None,
    effort: str | None = None,
) -> IntentResult:
    """필요한 근거 소스를 분류한다. 형식 불일치는 **코드가 1회만** 재시도한다."""
    input_tokens = 0
    output_tokens = 0
    # 실제로 나간 전송 수 — 토큰과 같은 이유로 버리지 않는다(`judge.Judge.judge` 와 같은 형태).
    sent = 0
    error: str | None = None
    previous_output: str | None = None

    for attempt in range(1, INTENT_MAX_ATTEMPTS + 1):
        user = build_intent_user_prompt(
            inquiry=inquiry,
            order_no=order_no,
            previous_output=previous_output,
            previous_error=error,
        )
        try:
            completion = client.complete_json(
                stage=INTENT_STAGE,
                system=INTENT_SYSTEM_PROMPT,
                user=user,
                schema=INTENT_JSON_SCHEMA,
                schema_name="intent",
                effort=effort,
            )
        except LLMFormatError as exc:
            input_tokens += exc.input_tokens
            output_tokens += exc.output_tokens
            sent += exc.transport_attempts
            error = exc.detail
            previous_output = exc.raw_text or None
            continue
        except LLMCallError as exc:
            # 재시도는 하지 않되 **앞선 시도에서 이미 과금된 토큰**을 버리지 않는다
            # (`judge.Judge.judge` 와 같은 형태). 새 예외로 다시 던지는 것은 대역이
            # 같은 예외 객체를 재사용해도 누적이 이중으로 실리지 않게 하기 위해서다.
            raise LLMCallError(
                stage=exc.stage,
                reason=exc.reason,
                attempts=sent + exc.attempts,
                cause=exc.cause,
                input_tokens=input_tokens + exc.input_tokens,
                output_tokens=output_tokens + exc.output_tokens,
            ) from exc

        input_tokens += completion.input_tokens
        output_tokens += completion.output_tokens
        sent += completion.transport_attempts
        source = _parse_intent(completion.data)
        if source is not None:
            return IntentResult(
                source=source,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=None,
                attempts=attempt,
            )
        error = f"source 는 policy·order·both 중 하나여야 한다 (받은 산출: {completion.data!r})"
        previous_output = repr(completion.data)

    return IntentResult(
        source=None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error=error,
        attempts=INTENT_MAX_ATTEMPTS,
    )


# ── SQL 생성 ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SqlGenerationResult:
    """SQL 생성 **1회** 호출의 결과. 재시도 여부는 호출자(수집기)가 정한다.

    `sql` 이 `None` 이면 유효한 SQL 을 얻지 못한 것이고 `error` 에 사유가 담긴다 —
    `sql_failed`(생성 실패) 경로다 — docs/business-rules.md "인계 사유 6종".
    전송 오류는 `LLMCallError` 로 올라간다(그건 `sql_failed` 가 아니라 `llm_call_failed`다).
    """

    sql: str | None
    input_tokens: int
    output_tokens: int
    error: str | None
    #: 이 1회 호출에서 실제로 나간 전송 수. 형식 오류로 SQL 을 못 얻어도 전송은 나갔다 —
    #: 재시도 루프가 토큰과 같은 자격으로 누적한다.
    transport_attempts: int = 1


def _extract_sql(data: object) -> str | None:
    if not isinstance(data, Mapping):
        return None
    value = data.get("sql")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def generate_sql(
    *,
    client: GenerationClient,
    inquiry: str,
    order_no: str,
    max_rows: int,
    whitelist: Mapping[str, tuple[str, ...]] = SCHEMA_WHITELIST,
    previous_sql: str | None = None,
    previous_error: str | None = None,
    effort: str | None = None,
) -> SqlGenerationResult:
    """주문 조회 SQL 문자열을 1회 생성한다. **검증도 실행도 하지 않는다.**"""
    user = build_sql_user_prompt(
        inquiry=inquiry,
        order_no=order_no,
        max_rows=max_rows,
        whitelist=whitelist,
        previous_sql=previous_sql,
        previous_error=previous_error,
    )
    try:
        completion = client.complete_json(
            stage=SQL_GENERATION_STAGE,
            system=SQL_SYSTEM_PROMPT,
            user=user,
            schema=SQL_JSON_SCHEMA,
            schema_name="order_sql",
            effort=effort,
        )
    except LLMFormatError as exc:
        return SqlGenerationResult(
            sql=None,
            input_tokens=exc.input_tokens,
            output_tokens=exc.output_tokens,
            error=f"구조화 출력 형식 불일치: {exc.detail}",
            transport_attempts=exc.transport_attempts,
        )

    sql = _extract_sql(completion.data)
    return SqlGenerationResult(
        sql=sql,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        error=None if sql is not None else "유효한 SQL 문자열을 얻지 못했다",
        transport_attempts=completion.transport_attempts,
    )


# ── 주문 존재성 선검사 (고정 파라미터 쿼리) ─────────────────────────────────

#: 선검사 쿼리. **고정 문장 + 파라미터 바인딩**이라 LLM 이 손댈 여지가 없다.
ORDER_EXISTS_SQL: Final = "SELECT 1 AS present FROM orders WHERE order_no = %s LIMIT 1"


def order_exists(*, conn: psycopg.Connection[DictRow], order_no: str) -> bool:
    """해당 주문이 있는지만 확인한다 — 대표 흐름 3단계의 존재성 선검사
    (docs/architecture.md "구성요소 지도").

    **결과를 근거로 쓰지 않는다** — 주문 근거는 text-to-SQL 경로로만 수집한다. 이 쿼리는
    근거 ID 도 받지 않는다. `order_not_found` 를 판정하는 주체는 이 선검사뿐이다.
    """
    with conn.cursor() as cur:
        cur.execute(ORDER_EXISTS_SQL, (order_no,))
        return cur.fetchone() is not None


# ── SQL 실행 (검증을 통과한 쿼리만) ─────────────────────────────────────────


def _jsonable(value: object) -> Any:
    """스냅샷을 그대로 jsonb 로 넣을 수 있게 원시 타입으로 낮춘다."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return str(value)


def execute_validated_query(
    *, conn: psycopg.Connection[DictRow], query: ValidatedQuery
) -> tuple[dict[str, Any], ...]:
    """검증을 통과한 쿼리를 실행한다. 커넥션은 **read-only 계정**이어야 한다.

    파라미터를 넘기지 않으므로 psycopg 가 쿼리 문자열을 재해석하지 않는다.
    LIMIT 은 이미 강제되어 있지만 fetch 도 상한으로 막아 방어층을 하나 더 둔다.
    """
    with conn.cursor() as cur:
        cur.execute(query.sql)
        rows = cur.fetchmany(query.max_rows)
    return tuple({key: _jsonable(value) for key, value in row.items()} for row in rows)


# ── 근거 렌더링 ─────────────────────────────────────────────────────────────


def _render_value(value: Any) -> str:
    return "(없음)" if value is None else str(value)


def _render_rows(rows: Sequence[Mapping[str, Any]], *, limit: int | None = None) -> str:
    shown = rows if limit is None else rows[:limit]
    lines = [
        f"{index}) " + ", ".join(f"{key}={_render_value(value)}" for key, value in row.items())
        for index, row in enumerate(shown, start=1)
    ]
    if limit is not None and len(rows) > limit:
        lines.append(f"... (이하 {len(rows) - limit}건 생략)")
    return "\n".join(lines)


def _sql_evidence_texts(
    *, sql: str, rows: Sequence[Mapping[str, Any]], pii_safe_output_columns: Sequence[str]
) -> tuple[str, str]:
    """(표시용 `content`, 대조용 `evidence_text`).

    실행 SQL 은 표시용 `content` 에만 둔다. `evidence_text` 는 결과 행을 담되, 계산 출력의
    패턴형 PII 는 LLM 리터럴 합성일 수 있어 제외한다. 비PII 계산 결과는 L2 근거로 유지하고,
    검증기가 `orders` 직접 컬럼으로 증명한 값은 PII 여부와 무관하게 **요약·마스킹하지 않고**
    그대로 둬 정상 에코 계약을 지킨다.

    **컬럼 이름도 값과 같은 자격으로 거른다.** 출력 이름은 LLM 이 정하는 별칭이라
    `SELECT status AS "010-9999-9999"` 처럼 이름 자리에 값을 심을 수 있고, 이름이
    `key=value` 로 렌더되는 순간 allowlist 근거가 된다. 스키마 컬럼 이름은 PII 패턴에
    걸리지 않으므로, 걸리는 이름은 별칭이라는 뜻이다.
    """
    header = f"실행 쿼리: {sql}\n결과 {len(rows)}건"
    safe_names = frozenset(pii_safe_output_columns)

    def _pii_shaped(text: str) -> bool:
        """탐지 전용 — 게이트와 **같은 접기**를 걸고 본다. 렌더는 원문을 유지한다.

        게이트(`_normalized_matches`)는 접은 뒤 대조하는데 여기서 접기 전으로 걸러내면,
        전각 숫자·제로폭 삽입처럼 **접기 전엔 PII 가 아니고 접은 뒤엔 PII 인** 계산 컬럼
        값이 `evidence_text` 로 살아남는다. 게이트가 그것을 접어 반각 번호로 만드는 순간
        지어낸 번호가 allowlist 근거가 된다. 두 층의 기준은 같아야 한다.

        접기는 판정에만 쓰고 값을 갈아 끼우지 않는다 — 근거 문면을 바꾸면 정상 에코 계약과
        L2 근거가 함께 흔들린다.
        """
        folded = fold_for_detection(text)
        return any(pattern.regex.search(folded) for pattern in DEFAULT_PII_PATTERNS)

    evidence_rows = tuple(
        {
            key: value
            for key, value in row.items()
            if not _pii_shaped(key) and (key in safe_names or not _pii_shaped(_render_value(value)))
        }
        for row in rows
    )
    return (
        f"{header}\n{_render_rows(rows, limit=CONTENT_ROW_LIMIT)}",
        _render_rows(evidence_rows),
    )


def _vector_hits(ranking: Sequence[PolicySearchHit]) -> tuple[VectorHit, ...]:
    """검색 결과를 순수 전략 모듈의 순위 자료형으로 옮긴다."""
    return tuple(
        VectorHit(rank=rank, evidence_id=hit.evidence_id, similarity=hit.similarity)
        for rank, hit in enumerate(ranking, start=1)
    )


def merge_policy_rankings(
    *,
    original: Sequence[PolicySearchHit],
    rewritten: Sequence[PolicySearchHit],
    top_k: int,
) -> tuple[PolicySearchHit, ...]:
    """원문·재작성 질의의 검색 결과를 합집합으로 모아 상위 `top_k` **후보**를 돌려준다
    (docs/architecture.md "대표 흐름" 3단계).

    입력도 출력도 **임계값 전**이다. 무엇이 근거가 되는지는 `adopt_policy_hits` 가 정한다.

    같은 조항이 두 질의에 다 걸리면 **더 큰 유사도**를 쓴다 — 합침 규칙은 오프라인 비교
    하네스와 **같은 구현**(`retrieval_strategies.merge_rewritten_rankings`)이다. 여기서
    규칙을 다시 적으면 비교표가 잰 것과 런타임이 하는 것이 갈린다.

    **"각 질의의 top_k 를 합쳐 다시 top_k" 는 "전체 순위를 합친 뒤 top_k" 와 결과가 같다.**
    오프라인 하네스는 조항 26개 전체 순위를 합친 뒤 자르고 런타임은 DB 가 이미 자른 top_k
    를 합치지만, 두 결과는 정확히 일치한다: 조항 X 의 합집합 점수를 `m = max(s₁, s₂)`,
    그 값을 낸 질의를 j 라 하면 — X 가 질의 j 의 top_k 밖이라는 것은 `sⱼ > m` 인 조항이
    k 개 있다는 뜻이고, 그 k 개는 합집합 점수도 `m` 보다 커서 X 는 합집합 top_k 밖이다.
    역방향도 같은 논증이다. 그래서 **재작성 때문에 후보 상한이 2배가 되지 않는다.**

    **컷을 합침 뒤로 옮겨도 채택 집합은 그대로다.** 컷 미만 후보는 컷 이상 후보보다 항상
    아래로 정렬되므로 합집합 상위 `top_k` 자리를 뺏을 수 없고, 뺏을 수 있는 유일한 경우
    (컷 이상 후보가 `top_k` 개보다 적을 때)에는 뒤이어 컷이 다시 지운다. 그래서 순서를
    바꾼 것은 **게이트가 볼 슬라이스**뿐이고 근거 집합은 사이클 3 과 같다.
    """
    if top_k <= 0:
        return ()
    best: dict[str, PolicySearchHit] = {}
    for hit in (*original, *rewritten):
        current = best.get(hit.evidence_id)
        if current is None or hit.similarity > current.similarity:
            best[hit.evidence_id] = hit
    merged = merge_rewritten_rankings(
        original=_vector_hits(original), rewritten=_vector_hits(rewritten)
    )
    return tuple(best[hit.evidence_id] for hit in merged[:top_k])


def adopt_policy_hits(
    *,
    candidates: Sequence[PolicySearchHit],
    top_k: int,
    similarity_threshold: float,
    gate: AbstentionGate | None,
) -> tuple[PolicySearchHit, ...]:
    """상위 `top_k` 후보에서 실제로 근거가 될 것을 고른다 — **질의 축 → 항목 축** 순서다.

    1. **질의 축(기권 게이트)**: 상위 `top_k` 점수 분포의 산포가 τ 미만이면 그 질의의 채택
       집합이 **통째로** 빈다. 어떤 항목을 남길지 고르는 일이 없으므로 항목 간 비교에
       상대 축을 들이지 않는다 — 결정 0009 의 절대 축 원칙이 그대로 남는다.
    2. **항목 축(절대 하한)**: 게이트가 발동하지 않았으면 코사인 ≥ 컷 만 채택한다.

    **통계량 입력은 컷 전 슬라이스다**(`docs/tracking/decisions/0014`).
    `search_policy_chunks` 의 SQL `LIMIT top_k` 가 자른 뒤 여기서 임계값이 걸리는 순서가
    오프라인 격자와 같고, 그래서 두 경로가 같은 수를 낸다. 컷 뒤 슬라이스로 재면 컷 위
    후보가 둘뿐인 케이스(G04)의 산포가 τ 아래로 떨어져 정답 조항을 잃는다.

    **통계량이 미정의면(측정된 후보 2건 미만) 게이트는 열린 채로 남는다.** 미정의를 0 으로
    채우면 모든 양수 τ 에서 기권이 되어, 후보가 하나뿐인 질의가 근거 없이 인계된다.
    원시연산(`retrieval_strategies.apply_abstention_gate`)이 값 대신 사유를 들고 오는 것이
    그 계약이고 여기서 뒤집지 않는다.
    """
    if not candidates:
        return ()
    if gate is not None:
        scores = truncate_for_gate([hit.similarity for hit in candidates], top_k=top_k)
        if apply_abstention_gate(gate, scores).abstains:
            return ()
    return tuple(hit for hit in candidates if hit.similarity >= similarity_threshold)


def _policy_evidence(hit: PolicySearchHit) -> Evidence:
    text = f"[{hit.document_title} {hit.article} {hit.article_title}] {hit.content}"
    return Evidence(
        id=hit.evidence_id,
        source=EvidenceSource.POLICY,
        content=text,
        evidence_text=text,
    )


# ── 수집 결과 ───────────────────────────────────────────────────────────────


class SqlFailureKind(StrEnum):
    """근거 ID 없이 처리 기록에만 남는 SQL 실패의 종류 (`db/schema.sql` 의 enum 과 같다)."""

    #: 안전장치(화이트리스트·쿼리 검증)가 실행 전에 거부했다.
    GUARD_REJECTED = "guard_rejected"
    #: 검증은 통과했지만 DB 가 실행 중 거부했다.
    EXECUTION_ERROR = "execution_error"
    #: LLM 이 쓸 수 있는 SQL 문자열을 내지 못했다.
    GENERATION_FAILED = "generation_failed"


@dataclass(frozen=True)
class SqlFailure:
    """실패한 SQL 시도 1건. **근거 ID 를 받지 않는다** (docs/contracts.md "답변 계약")."""

    attempt_no: int
    kind: SqlFailureKind
    query_sql: str | None
    error: str


@dataclass(frozen=True)
class SqlEvidenceSnapshot:
    """채택된 SQL 근거 1건의 영속화 대상 — 실행된 쿼리문과 결과 행 전체."""

    evidence_id: str
    sequence: int
    query_sql: str
    result_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class EvidenceCollection:
    """근거 수집의 결과 전부.

    `escalation_reason` 이 있으면 초안 생성에 진입하지 않는다. 그 경우에도 `evidence` 에는
    **그 시점까지 수집된 근거**가 들어 있다 — 감사 목적으로 API 응답의 citations 에 실린다.
    """

    intent: IntentSource | None
    evidence: tuple[Evidence, ...]
    escalation_reason: EscalationReason | None
    #: `llm_call_failed` 일 때 실패한 단계 이름 (docs/business-rules.md "인계 사유 6종").
    failed_stage: str | None
    sql_snapshots: tuple[SqlEvidenceSnapshot, ...]
    sql_failures: tuple[SqlFailure, ...]
    #: 생성 LLM 토큰 (의도 해석 + SQL 생성). 임베딩·검색과 섞지 않는다.
    input_tokens: int
    output_tokens: int
    embedding_tokens: int
    #: 검색 단계 생성 호출(질의 재작성)의 토큰. **생성 합산에 넣지 않는다**
    #: (docs/contracts.md "토큰 집계 경계") — 넣으면 초안을 만들지도 않은 문의가 초안
    #: 생성 토큰을 쓴 것으로 찍혀 성공 판정 ②가 거짓이 된다. 실패한 호출의 토큰도 실비용
    #: 이므로 그대로 센다.
    retrieval_input_tokens: int = 0
    retrieval_output_tokens: int = 0
    #: 검색 단계가 폴백한 사유. `None` 은 "폴백하지 않았다"이고, 재작성을 **시도하지 않은**
    #: 경우(스위치 꺼짐·`order` 단독 의도)도 `None` 이다 — 폴백은 시도한 것의 결과다.
    retrieval_fallback_reason: str | None = None

    @property
    def escalated(self) -> bool:
        return self.escalation_reason is not None


@dataclass
class _Ledger:
    """수집 도중 쌓이는 것들. 인계로 끝나도 그대로 결과에 실린다."""

    evidence: list[Evidence] = field(default_factory=list)
    snapshots: list[SqlEvidenceSnapshot] = field(default_factory=list)
    failures: list[SqlFailure] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    #: 검색 단계 생성 호출의 토큰 — 생성 합산과 분리된 계열이다.
    retrieval_input_tokens: int = 0
    retrieval_output_tokens: int = 0
    retrieval_fallback_reason: str | None = None
    #: SQL 생성 루프에서 실제로 나간 전송 수. 전송 오류가 루프 중간에 터지면
    #: `LLMCallError.attempts` 가 이 값을 이어받아야 기록과 실제가 같아진다.
    sql_transport_attempts: int = 0


# ── 수집기 ──────────────────────────────────────────────────────────────────


class EvidenceCollector:
    """의도 해석 → (질의 재작성) → 정책 벡터 검색 → 주문 존재성 선검사 → text-to-SQL 을 조율한다.

    커넥션은 호출자가 연다. 정책 청크는 애플리케이션 계정으로, text-to-SQL 은 **반드시**
    read-only 계정(`db.readonly_connect`)으로 실행해야 한다 — 안전장치 1이다.
    """

    def __init__(
        self,
        *,
        generation_client: GenerationClient,
        embedding_client: EmbeddingClient,
        settings: Settings | None = None,
        rewrite_client: GenerationClient | None = None,
    ) -> None:
        """`query_rewrite_enabled` 가 켜져 있으면 재작성 클라이언트는 **필수**다.

        생성 클라이언트를 자동으로 재사용하지 않는 것이 의도다 — 그러면 배선이 항상
        성공해 미배선 검사가 무의미해지고, 재작성을 어느 계열이 만드는지가 조립부에
        보이지 않는다. 조립부(`pipeline.build_pipeline`)가 명시적으로 넘긴다.
        """
        resolved = settings if settings is not None else get_settings()
        if resolved.query_rewrite_enabled and rewrite_client is None:
            raise RetrievalWiringError(
                "질의 재작성 스위치가 켜져 있는데 재작성 클라이언트가 배선되지 않았다 — "
                "조용히 원문 질의로 돌지 않는다. 클라이언트를 주입하거나 "
                "QUERY_REWRITE_ENABLED=false 로 명시한다."
            )
        self._client = generation_client
        self._embedder = embedding_client
        self._settings = resolved
        self._rewrite_client = rewrite_client
        # 기권 게이트는 설정에서 한 번만 조립한다 — 켜고 끄는 자리가 하나여야 런타임과
        # 리포트의 조건 지문이 갈리지 않는다. `None` 은 "꺼짐"이고 채택은 절대 하한만 남는다.
        self._abstention_gate = resolved.abstention_gate()

    def collect(
        self,
        *,
        inquiry_id: str,
        content: str,
        order_no: str | None,
        app_conn: psycopg.Connection[DictRow],
        readonly_conn: psycopg.Connection[DictRow],
    ) -> EvidenceCollection:
        """문의 1건의 근거를 모은다. 결과는 근거 목록 또는 초안 전 인계 사유다."""
        ledger = _Ledger()
        intent: IntentSource | None = None

        try:
            classification = classify_intent(
                client=self._client,
                inquiry=content,
                order_no=order_no,
                effort=self._settings.generation_effort,
            )
            ledger.input_tokens += classification.input_tokens
            ledger.output_tokens += classification.output_tokens
            if classification.source is None:
                return self._finish(
                    ledger,
                    intent=None,
                    escalation=EscalationReason.LLM_CALL_FAILED,
                    failed_stage=INTENT_STAGE,
                )
            intent = classification.source

            if intent in (IntentSource.POLICY, IntentSource.BOTH):
                self._collect_policy(ledger=ledger, conn=app_conn, content=content)

            if intent in (IntentSource.ORDER, IntentSource.BOTH):
                escalation = self._collect_order(
                    ledger=ledger,
                    inquiry_id=inquiry_id,
                    content=content,
                    order_no=order_no,
                    readonly_conn=readonly_conn,
                )
                # 주문 측 구조적 사유는 정책 근거를 이미 모았더라도 인계가 우선한다.
                if escalation is not None:
                    return self._finish(
                        ledger, intent=intent, escalation=escalation, failed_stage=None
                    )
        except LLMCallError as exc:
            # 전송 오류는 래퍼가 이미 1회 재시도했다 — 어느 단계에서 왔든 llm_call_failed.
            # 실패까지 실제로 과금된 토큰(예: 200 으로 돌아온 거절 응답의 usage)은 예외에
            # 실려 온다 — 초안·판정 경로와 같은 규칙으로 원장에 더한 뒤 종결한다. 여기서만
            # 버리면 같은 거절이 어느 단계에서 났느냐에 따라 실비용 기록이 갈린다.
            ledger.input_tokens += exc.input_tokens
            ledger.output_tokens += exc.output_tokens
            return self._finish(
                ledger,
                intent=intent,
                escalation=EscalationReason.LLM_CALL_FAILED,
                failed_stage=exc.stage,
            )

        if not ledger.evidence:
            return self._finish(
                ledger, intent=intent, escalation=EscalationReason.NO_EVIDENCE, failed_stage=None
            )
        return self._finish(ledger, intent=intent, escalation=None, failed_stage=None)

    # ── 정책 ────────────────────────────────────────────────────────────────

    def _collect_policy(
        self, *, ledger: _Ledger, conn: psycopg.Connection[DictRow], content: str
    ) -> None:
        """(질의 재작성) → 문의 임베딩 → pgvector 유사도 검색 → 상위 k 개 후보 → 채택 판정.

        **원문과 재작성문을 둘 다 검색해 합집합으로 모은다**
        (docs/architecture.md "대표 흐름" 3단계) — 재작성이 주제를
        옮겼을 때 원문이 안전망이다. 재작성을 얻지 못하면 원문만으로 계속한다(폴백).

        검색이 돌려주는 것은 `LIMIT top_k` 후보 그대로이고, **무엇이 근거가 되는지는
        `adopt_policy_hits` 가 정한다** — 질의 단위 기권 게이트 다음에 절대 하한이다.
        """
        queries = [content]
        rewritten = self._rewrite(ledger=ledger, content=content)
        # 재작성이 원문과 같으면 한 번만 검색한다 — 같은 벡터를 두 번 실어 임베딩 토큰을
        # 두 배 쓸 이유가 없고, max 합침이라 결과도 글자 그대로 같다. 픽스처 계약이
        # "재작성이 필요 없다고 판단한 문의는 원문과 같은 문자열"을 허용하므로 실제로 온다.
        if rewritten is not None and rewritten != content:
            queries.append(rewritten)

        embedding = self._embedder.embed(stage=INQUIRY_EMBEDDING_STAGE, texts=queries)
        ledger.embedding_tokens += embedding.total_tokens
        if not embedding.vectors:
            return

        rankings = [
            search_policy_chunks(
                conn=conn,
                query_vector=vector,
                top_k=self._settings.vector_top_k,
                embedding_model=self._embedder.model,
                embedding_dimensions=self._embedder.dimensions,
            )
            # 임베딩 대역이 질의 수보다 적게 돌려주는 경우까지 여기서 감당한다 —
            # 없는 벡터로 검색하지 않고, 있는 만큼만 합친다.
            for vector in embedding.vectors[: len(queries)]
        ]
        candidates = merge_policy_rankings(
            original=rankings[0],
            rewritten=rankings[1] if len(rankings) > 1 else (),
            top_k=self._settings.vector_top_k,
        )
        hits = adopt_policy_hits(
            candidates=candidates,
            top_k=self._settings.vector_top_k,
            similarity_threshold=self._settings.vector_similarity_threshold,
            gate=self._abstention_gate,
        )
        ledger.evidence.extend(_policy_evidence(hit) for hit in hits)

    def _rewrite(self, *, ledger: _Ledger, content: str) -> str | None:
        """검색용 질의를 얻는다. 스위치가 꺼져 있거나 폴백이면 `None`.

        **폴백은 인계가 아니다**(docs/business-rules.md "검색 단계 실패"). 다만 조용히
        지나가지도 않는다 — 사유와 실비용 토큰이 원장에 남아 처리 기록·평가 리포트로
        흘러간다(docs/business-rules.md "조용한 폴백은 금지한다").
        """
        if not self._settings.query_rewrite_enabled:
            return None
        # 조립에서 이미 막혔다. 여기서 다시 보는 것은 타입 좁히기 겸 이중 잠금이다 —
        # `assert` 로 두면 `python -O` 에서 사라져 스위치가 켜진 채 조용히 원문만 도는
        # fail-open 이 된다(`pipeline._draft_loop` 의 판정자 재확인과 같은 형태).
        if self._rewrite_client is None:  # pragma: no cover - 조립에서 이미 막힌다
            raise RetrievalWiringError("질의 재작성 스위치가 켜져 있는데 재작성 클라이언트가 없다.")
        outcome = rewrite_query(
            client=self._rewrite_client,
            inquiry=content,
            effort=self._settings.generation_effort,
        )
        ledger.retrieval_input_tokens += outcome.input_tokens
        ledger.retrieval_output_tokens += outcome.output_tokens
        if outcome.fallback_reason is not None:
            ledger.retrieval_fallback_reason = outcome.fallback_reason
        return outcome.query

    # ── 주문 ────────────────────────────────────────────────────────────────

    def _collect_order(
        self,
        *,
        ledger: _Ledger,
        inquiry_id: str,
        content: str,
        order_no: str | None,
        readonly_conn: psycopg.Connection[DictRow],
    ) -> EscalationReason | None:
        """선검사 → text-to-SQL. 인계 사유가 있으면 그것을, 없으면 `None` 을 돌려준다."""
        if order_no is None or not order_no.strip():
            return EscalationReason.MISSING_ORDER_REF
        normalized = normalize_order_no(order_no)
        # 형식이 깨진 값은 참조가 없는 것과 같다 — `order_not_found` 는 형식이 맞는
        # 주문번호에만 쓴다(docs/business-rules.md "인계 사유 6종").
        # 정상 경로에서는 접수 단계가 먼저 막는다.
        if not is_valid_order_no(normalized):
            return EscalationReason.MISSING_ORDER_REF

        if not order_exists(conn=readonly_conn, order_no=normalized):
            # text-to-SQL 에 진입하지 않는다 — 없는 주문에 SQL 을 생성시킬 이유가 없다.
            return EscalationReason.ORDER_NOT_FOUND

        return self._run_text_to_sql(
            ledger=ledger,
            inquiry_id=inquiry_id,
            content=content,
            order_no=normalized,
            readonly_conn=readonly_conn,
        )

    def _run_text_to_sql(
        self,
        *,
        ledger: _Ledger,
        inquiry_id: str,
        content: str,
        order_no: str,
        readonly_conn: psycopg.Connection[DictRow],
    ) -> EscalationReason | None:
        """생성 → 검증 → 실행. 실패는 피드백을 실어 **1회만** 재시도한다(코드가 상한 강제)."""
        max_rows = self._settings.sql_max_rows
        previous_sql: str | None = None
        previous_error: str | None = None

        for attempt in range(1, SQL_MAX_ATTEMPTS + 1):
            try:
                generation = generate_sql(
                    client=self._client,
                    inquiry=content,
                    order_no=order_no,
                    max_rows=max_rows,
                    previous_sql=previous_sql,
                    previous_error=previous_error,
                    effort=self._settings.generation_effort,
                )
            except LLMCallError as exc:
                # 앞선 시도에서 이미 나간 전송을 이어 센다 — 토큰을 원장에 누적하면서
                # 횟수만 버리면 "1회 재시도"라고 적힌 기록이 실제와 갈린다
                # (`classify_intent`·`judge.Judge.judge` 와 같은 형태).
                raise LLMCallError(
                    stage=exc.stage,
                    reason=exc.reason,
                    attempts=ledger.sql_transport_attempts + exc.attempts,
                    cause=exc.cause,
                    input_tokens=exc.input_tokens,
                    output_tokens=exc.output_tokens,
                ) from exc
            ledger.input_tokens += generation.input_tokens
            ledger.output_tokens += generation.output_tokens
            ledger.sql_transport_attempts += generation.transport_attempts

            if generation.sql is None:
                previous_sql = None
                previous_error = generation.error or "유효한 SQL 을 생성하지 못했다"
                ledger.failures.append(
                    SqlFailure(
                        attempt_no=attempt,
                        kind=SqlFailureKind.GENERATION_FAILED,
                        query_sql=None,
                        error=previous_error,
                    )
                )
                continue

            try:
                # 조회 범위를 **선검사를 통과한 주문 1건**으로 묶는 것도 검증 단계의 일이다
                # (docs/architecture.md "대표 흐름" 3단계). 프롬프트 문구는 그 보증이 되지 못한다.
                query = validate_sql(generation.sql, order_no=order_no, max_rows=max_rows)
            except SqlGuardRejection as rejection:
                previous_sql = generation.sql
                previous_error = f"{rejection.rule.value}: {rejection.detail}"
                ledger.failures.append(
                    SqlFailure(
                        attempt_no=attempt,
                        kind=SqlFailureKind.GUARD_REJECTED,
                        query_sql=generation.sql,
                        error=previous_error,
                    )
                )
                continue

            try:
                rows = execute_validated_query(conn=readonly_conn, query=query)
            except psycopg.Error as exc:
                _reset_after_error(readonly_conn)
                previous_sql = query.sql
                previous_error = _first_line(exc)
                ledger.failures.append(
                    SqlFailure(
                        attempt_no=attempt,
                        kind=SqlFailureKind.EXECUTION_ERROR,
                        query_sql=query.sql,
                        error=previous_error,
                    )
                )
                continue

            if rows:
                self._adopt_sql_evidence(
                    ledger=ledger, inquiry_id=inquiry_id, query=query, rows=rows
                )
            # 정상 실행 0건은 실패가 아니라 "주문 소스 근거 0건"이다 — 재시도하지 않는다.
            return None

        return EscalationReason.SQL_FAILED

    def _adopt_sql_evidence(
        self,
        *,
        ledger: _Ledger,
        inquiry_id: str,
        query: ValidatedQuery,
        rows: tuple[dict[str, Any], ...],
    ) -> None:
        """채택된 쿼리에만 순번과 근거 ID 를 매긴다 (docs/contracts.md "답변 계약")."""
        sequence = len(ledger.snapshots) + 1
        evidence_id = sql_evidence_id(inquiry_id=inquiry_id, sequence=sequence)
        content, evidence_text = _sql_evidence_texts(
            sql=query.sql,
            rows=rows,
            pii_safe_output_columns=query.pii_safe_output_columns,
        )
        ledger.evidence.append(
            Evidence(
                id=evidence_id,
                source=EvidenceSource.SQL,
                content=content,
                evidence_text=evidence_text,
            )
        )
        ledger.snapshots.append(
            SqlEvidenceSnapshot(
                evidence_id=evidence_id,
                sequence=sequence,
                query_sql=query.sql,
                result_rows=rows,
            )
        )

    # ── 마무리 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _finish(
        ledger: _Ledger,
        *,
        intent: IntentSource | None,
        escalation: EscalationReason | None,
        failed_stage: str | None,
    ) -> EvidenceCollection:
        return EvidenceCollection(
            intent=intent,
            evidence=tuple(ledger.evidence),
            escalation_reason=escalation,
            failed_stage=failed_stage,
            sql_snapshots=tuple(ledger.snapshots),
            sql_failures=tuple(ledger.failures),
            input_tokens=ledger.input_tokens,
            output_tokens=ledger.output_tokens,
            embedding_tokens=ledger.embedding_tokens,
            retrieval_input_tokens=ledger.retrieval_input_tokens,
            retrieval_output_tokens=ledger.retrieval_output_tokens,
            retrieval_fallback_reason=ledger.retrieval_fallback_reason,
        )


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


def _reset_after_error(conn: psycopg.Connection[DictRow]) -> None:
    """실패한 트랜잭션을 되돌려 다음 재시도가 같은 커넥션을 쓸 수 있게 한다.

    read-only 커넥션은 기본이 autocommit 이라 대개 할 일이 없지만, 호출자가 트랜잭션
    커넥션을 넘겼을 때 재시도 쿼리가 "current transaction is aborted" 로 죽는 것을 막는다.
    """
    # 커넥션 자체가 끊긴 경우까지 여기서 살릴 수는 없다 — 다음 시도가 같은 오류를 낸다.
    with contextlib.suppress(psycopg.Error):
        conn.rollback()
