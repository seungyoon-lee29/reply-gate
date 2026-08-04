"""근거 수집 테스트 — 의도 분류·벡터 검색·존재성 선검사·text-to-SQL.

생성 LLM 은 전부 목이다. 임베딩은 `LexicalEmbeddingClient` 결정론 대역을 써서 API 키 없이
벡터 검색 배관까지 돌린다. DB 가 필요한 테스트는 `db` 마커가 붙고, 쓰기는 하지 않는다
(정책 인덱싱은 `app_conn` 트랜잭션 안에서만 일어나고 픽스처가 롤백한다).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import psycopg
import pytest
from psycopg.rows import DictRow

from reply_gate.config import Settings
from reply_gate.contracts import (
    EscalationReason,
    EvidenceSource,
    IntentSource,
    Verdict,
)
from reply_gate.db import readonly_connect
from reply_gate.evidence import (
    INQUIRY_EMBEDDING_STAGE,
    INTENT_STAGE,
    SQL_GENERATION_STAGE,
    EvidenceCollector,
    SqlFailureKind,
    classify_intent,
    generate_sql,
    order_exists,
)
from reply_gate.gate import evaluate_draft
from reply_gate.llm import GenerationClient, JsonCompletion, LLMCallError, LLMFormatError
from reply_gate.policy_index import index_policy_documents, load_policy_documents
from reply_gate.testing import LexicalEmbeddingClient

INQUIRY_ID = "11111111-2222-3333-4444-555555555555"
INQUIRY = "환불 신청은 언제까지 가능한가요? 제 주문도 확인해 주세요."
MISSING_ORDER_NO = "ORD-20991231-9999"


# ── 목 / 픽스처 ─────────────────────────────────────────────────────────────


class _ScriptedClient:
    """`GenerationClient` 대역 — 단계별로 미리 정한 결과를 순서대로 돌려준다."""

    def __init__(self, script: Mapping[str, Sequence[Any]]) -> None:
        self._script = {stage: list(outcomes) for stage, outcomes in script.items()}
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        stage = kwargs["stage"]
        queue = self._script.get(stage)
        if not queue:
            raise AssertionError(f"대본에 없는 호출이다: stage={stage!r}")
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def calls_for(self, stage: str) -> list[dict[str, Any]]:
        return [call for call in self.calls if call["stage"] == stage]


def _client(script: Mapping[str, Sequence[Any]]) -> _ScriptedClient:
    return _ScriptedClient(script)


def _intent(value: str, *, input_tokens: int = 10, output_tokens: int = 2) -> JsonCompletion:
    return JsonCompletion(
        data={"source": value}, input_tokens=input_tokens, output_tokens=output_tokens
    )


def _sql(text: str, *, input_tokens: int = 30, output_tokens: int = 12) -> JsonCompletion:
    return JsonCompletion(
        data={"sql": text}, input_tokens=input_tokens, output_tokens=output_tokens
    )


def _collector(
    client: _ScriptedClient,
    *,
    threshold: float = 0.0,
    top_k: int = 5,
    max_rows: int = 50,
) -> EvidenceCollector:
    return EvidenceCollector(
        generation_client=cast(GenerationClient, client),
        embedding_client=LexicalEmbeddingClient(dimensions=1536),
        settings=Settings(
            vector_top_k=top_k,
            vector_similarity_threshold=threshold,
            sql_max_rows=max_rows,
        ),
    )


_NO_CONN = cast(psycopg.Connection[DictRow], None)


@pytest.fixture
def indexed_policies(app_conn: psycopg.Connection[DictRow]) -> None:
    """저장소의 정책 문서를 결정론 임베딩으로 적재한다 (픽스처 롤백으로 되돌아간다)."""
    index_policy_documents(
        conn=app_conn,
        documents=load_policy_documents(),
        embedder=LexicalEmbeddingClient(dimensions=1536),
    )


@pytest.fixture
def sample_order(ro_conn: psycopg.Connection[DictRow]) -> dict[str, Any]:
    row = ro_conn.execute(
        "SELECT order_no, customer_name, customer_phone, customer_email, status"
        " FROM orders ORDER BY order_no LIMIT 1"
    ).fetchone()
    assert row is not None, "시딩된 주문이 있어야 한다"
    return dict(row)


# ── 의도 분류 (DB 불필요) ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("policy", IntentSource.POLICY),
        ("order", IntentSource.ORDER),
        ("both", IntentSource.BOTH),
    ],
)
def test_의도_분류_구조화_출력을_파싱한다(value: str, expected: IntentSource) -> None:
    client = _client({INTENT_STAGE: [_intent(value)]})

    result = classify_intent(client=cast(GenerationClient, client), inquiry=INQUIRY)

    assert result.source is expected
    assert result.error is None
    assert (result.input_tokens, result.output_tokens) == (10, 2)


def test_형식_불일치는_1회_재시도한_뒤_성공한다() -> None:
    broken = LLMFormatError(
        stage=INTENT_STAGE,
        detail="JSON 파싱 실패",
        raw_text="정책이랑 주문 둘 다요",
        input_tokens=7,
        output_tokens=1,
    )
    client = _client({INTENT_STAGE: [broken, _intent("both")]})

    result = classify_intent(client=cast(GenerationClient, client), inquiry=INQUIRY)

    assert result.source is IntentSource.BOTH
    assert len(client.calls_for(INTENT_STAGE)) == 2
    # 재시도 프롬프트는 직전 산출과 실패 사유를 피드백으로 싣는다.
    retry_prompt = client.calls_for(INTENT_STAGE)[1]["user"]
    assert "정책이랑 주문 둘 다요" in retry_prompt
    assert "JSON 파싱 실패" in retry_prompt
    # 실패한 호출의 토큰도 합산에서 빠지지 않는다.
    assert (result.input_tokens, result.output_tokens) == (17, 3)


def test_재시도해도_형식이_안_맞으면_실패로_돌려준다() -> None:
    broken = LLMFormatError(
        stage=INTENT_STAGE, detail="형식 불일치", input_tokens=1, output_tokens=1
    )
    client = _client({INTENT_STAGE: [broken, broken]})

    result = classify_intent(client=cast(GenerationClient, client), inquiry=INQUIRY)

    assert result.source is None
    assert result.error is not None
    # 재시도는 코드가 1회로 상한을 강제한다.
    assert len(client.calls_for(INTENT_STAGE)) == 2


def test_열거값이_아닌_source_도_형식_불일치로_본다() -> None:
    client = _client({INTENT_STAGE: [_intent("shipping"), _intent("policy")]})

    result = classify_intent(client=cast(GenerationClient, client), inquiry=INQUIRY)

    assert result.source is IntentSource.POLICY
    assert len(client.calls_for(INTENT_STAGE)) == 2


def test_전송_오류는_재시도하지_않고_그대로_올린다() -> None:
    """래퍼가 이미 1회 재시도했다 — 여기서 또 재시도하면 상한이 무너진다."""
    client = _client(
        {INTENT_STAGE: [LLMCallError(stage=INTENT_STAGE, reason="transport_error", attempts=2)]}
    )

    with pytest.raises(LLMCallError):
        classify_intent(client=cast(GenerationClient, client), inquiry=INQUIRY)

    assert len(client.calls_for(INTENT_STAGE)) == 1


# ── SQL 생성 (DB 불필요) ────────────────────────────────────────────────────


def test_SQL_생성_프롬프트에_화이트리스트와_주문번호가_실린다() -> None:
    client = _client({SQL_GENERATION_STAGE: [_sql("SELECT order_no FROM orders")]})

    generate_sql(
        client=cast(GenerationClient, client),
        inquiry=INQUIRY,
        order_no="ORD-20260201-0001",
        max_rows=50,
    )

    prompt = client.calls_for(SQL_GENERATION_STAGE)[0]["user"]
    assert "orders" in prompt
    assert "customer_phone" in prompt
    assert "ORD-20260201-0001" in prompt
    assert "50" in prompt


def test_빈_SQL_은_생성_실패로_본다() -> None:
    client = _client({SQL_GENERATION_STAGE: [_sql("   ")]})

    result = generate_sql(
        client=cast(GenerationClient, client),
        inquiry=INQUIRY,
        order_no="ORD-20260201-0001",
        max_rows=50,
    )

    assert result.sql is None
    assert result.error is not None
    # SQL 생성은 여기서 재시도하지 않는다 — 재시도 상한은 수집기가 통제한다.
    assert len(client.calls_for(SQL_GENERATION_STAGE)) == 1


def test_SQL_생성_형식_불일치도_생성_실패로_본다() -> None:
    broken = LLMFormatError(
        stage=SQL_GENERATION_STAGE,
        detail="JSON 아님",
        raw_text="SELECT",
        input_tokens=4,
        output_tokens=2,
    )
    client = _client({SQL_GENERATION_STAGE: [broken]})

    result = generate_sql(
        client=cast(GenerationClient, client),
        inquiry=INQUIRY,
        order_no="ORD-20260201-0001",
        max_rows=50,
    )

    assert result.sql is None
    assert (result.input_tokens, result.output_tokens) == (4, 2)


# ── 인계 경로 중 DB 를 건드리지 않는 것들 ───────────────────────────────────


def test_order_의도인데_주문번호가_없으면_missing_order_ref() -> None:
    """DB 도 SQL 생성도 건드리지 않는다 — 커넥션 자리에 None 을 넣어 그것을 증명한다."""
    client = _client({INTENT_STAGE: [_intent("order")]})

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=None,
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert result.escalation_reason is EscalationReason.MISSING_ORDER_REF
    assert result.evidence == ()
    assert client.calls_for(SQL_GENERATION_STAGE) == []


def test_형식이_깨진_주문번호도_missing_order_ref() -> None:
    client = _client({INTENT_STAGE: [_intent("order")]})

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no="ORD-20260230-0001",  # 달력에 없는 날짜
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert result.escalation_reason is EscalationReason.MISSING_ORDER_REF


def test_의도_분류_실패는_llm_call_failed_와_실패_단계를_남긴다() -> None:
    broken = LLMFormatError(stage=INTENT_STAGE, detail="형식 불일치")
    client = _client({INTENT_STAGE: [broken, broken]})

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=None,
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert result.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert result.failed_stage == INTENT_STAGE
    assert result.intent is None


def test_의도_분류_전송_오류도_llm_call_failed() -> None:
    client = _client(
        {INTENT_STAGE: [LLMCallError(stage=INTENT_STAGE, reason="transport_error", attempts=2)]}
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=None,
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert result.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert result.failed_stage == INTENT_STAGE


# ── 존재성 선검사 (DB 필요) ─────────────────────────────────────────────────


@pytest.mark.db
def test_존재성_선검사는_존재_여부만_판정한다(
    ro_conn: psycopg.Connection[DictRow], sample_order: dict[str, Any]
) -> None:
    assert order_exists(conn=ro_conn, order_no=sample_order["order_no"]) is True
    assert order_exists(conn=ro_conn, order_no=MISSING_ORDER_NO) is False


@pytest.mark.db
def test_없는_주문은_order_not_found_이고_text_to_SQL_에_진입하지_않는다(
    ro_conn: psycopg.Connection[DictRow], app_conn: psycopg.Connection[DictRow]
) -> None:
    client = _client({INTENT_STAGE: [_intent("order")]})

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=MISSING_ORDER_NO,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is EscalationReason.ORDER_NOT_FOUND
    assert client.calls_for(SQL_GENERATION_STAGE) == []
    assert result.sql_snapshots == ()
    assert result.evidence == ()


# ── 정책 벡터 검색 (DB 필요) ────────────────────────────────────────────────


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_정책_의도는_벡터_검색_결과를_근거로_삼는다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    client = _client({INTENT_STAGE: [_intent("policy")]})

    result = _collector(client, threshold=0.0).collect(
        inquiry_id=INQUIRY_ID,
        content="환불 신청 기간이 어떻게 되나요",
        order_no=None,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is None
    assert result.intent is IntentSource.POLICY
    assert result.evidence
    assert all(item.source is EvidenceSource.POLICY for item in result.evidence)
    assert all(item.id.startswith("policy:") for item in result.evidence)
    assert result.embedding_tokens > 0


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_임계값_미달_결과는_버려지고_근거가_없으면_no_evidence(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    client = _client({INTENT_STAGE: [_intent("policy")]})

    result = _collector(client, threshold=0.99).collect(
        inquiry_id=INQUIRY_ID,
        content="환불 신청 기간이 어떻게 되나요",
        order_no=None,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.evidence == ()
    assert result.escalation_reason is EscalationReason.NO_EVIDENCE


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_상위_k_개까지만_근거로_삼는다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    client = _client({INTENT_STAGE: [_intent("policy")]})

    result = _collector(client, threshold=0.0, top_k=2).collect(
        inquiry_id=INQUIRY_ID,
        content="환불 신청 기간이 어떻게 되나요",
        order_no=None,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert len(result.evidence) == 2


# ── text-to-SQL (DB 필요) ───────────────────────────────────────────────────


@pytest.mark.db
def test_채택된_SQL_근거는_순번_ID_와_스냅샷을_받는다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    order_no = sample_order["order_no"]
    client = _client(
        {
            INTENT_STAGE: [_intent("order")],
            SQL_GENERATION_STAGE: [
                _sql(f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'")
            ],
        }
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is None
    assert [item.id for item in result.evidence] == [f"sql:{INQUIRY_ID}:1"]
    assert result.evidence[0].source is EvidenceSource.SQL
    # 선검사 쿼리는 근거가 아니므로 스냅샷도 ID 도 받지 않는다.
    assert len(result.sql_snapshots) == 1
    snapshot = result.sql_snapshots[0]
    assert snapshot.sequence == 1
    assert snapshot.evidence_id == f"sql:{INQUIRY_ID}:1"
    assert order_no in snapshot.query_sql
    assert snapshot.result_rows and snapshot.result_rows[0]["order_no"] == order_no
    assert result.sql_failures == ()


@pytest.mark.db
def test_SQL_근거의_evidence_text_에_연락처_원문이_그대로_들어간다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    """마스킹하면 L1 PII allowlist 가 정상 에코를 오기각한다 (spec PII 정책)."""
    order_no = sample_order["order_no"]
    client = _client(
        {
            INTENT_STAGE: [_intent("order")],
            SQL_GENERATION_STAGE: [
                _sql(
                    "SELECT order_no, customer_phone, customer_email FROM orders"
                    f" WHERE order_no = '{order_no}'"
                )
            ],
        }
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    evidence = result.evidence[0]
    assert sample_order["customer_phone"] in evidence.evidence_text
    assert sample_order["customer_email"] in evidence.evidence_text

    # 실제로 L1 이 정상 에코를 통과시키는지까지 확인한다 — 이게 이 요구사항의 존재 이유다.
    draft = {
        "claims": [
            {
                "text": f"등록된 연락처는 {sample_order['customer_phone']} 입니다.",
                "citation_ids": [evidence.id],
            }
        ]
    }
    assert evaluate_draft(raw_draft=draft, evidences=result.evidence).verdict is Verdict.PASS


@pytest.mark.db
def test_안전장치가_거부하면_피드백으로_1회_재시도해_성공한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    order_no = sample_order["order_no"]
    bad = "SELECT id, content FROM inquiries"
    good = f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'"
    client = _client(
        {INTENT_STAGE: [_intent("order")], SQL_GENERATION_STAGE: [_sql(bad), _sql(good)]}
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is None
    assert [item.id for item in result.evidence] == [f"sql:{INQUIRY_ID}:1"]
    assert len(result.sql_failures) == 1
    failure = result.sql_failures[0]
    assert failure.attempt_no == 1
    assert failure.kind is SqlFailureKind.GUARD_REJECTED
    assert failure.query_sql == bad

    retry_prompt = client.calls_for(SQL_GENERATION_STAGE)[1]["user"]
    assert bad in retry_prompt
    assert "unknown_table" in retry_prompt


@pytest.mark.db
def test_재시도까지_실패하면_sql_failed_이고_근거_ID_는_부여되지_않는다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    client = _client(
        {
            INTENT_STAGE: [_intent("order")],
            SQL_GENERATION_STAGE: [
                _sql("DELETE FROM orders"),
                _sql("SELECT content FROM policy_chunks"),
            ],
        }
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=sample_order["order_no"],
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is EscalationReason.SQL_FAILED
    assert result.evidence == ()
    assert result.sql_snapshots == ()
    assert [failure.attempt_no for failure in result.sql_failures] == [1, 2]
    assert [failure.query_sql for failure in result.sql_failures] == [
        "DELETE FROM orders",
        "SELECT content FROM policy_chunks",
    ]
    # 재시도 상한은 코드가 강제한다 — 3번째 생성 호출은 없다.
    assert len(client.calls_for(SQL_GENERATION_STAGE)) == 2


@pytest.mark.db
def test_주문_범위를_벗어난_SQL_은_거부되고_피드백으로_1회_재시도해_성공한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    """WHERE 없는 쿼리는 무관한 고객 50명의 연락처를 근거로 만든다 — 실행 전에 막아야 한다."""
    order_no = sample_order["order_no"]
    unscoped = "SELECT order_no, customer_name, customer_phone, customer_email FROM orders LIMIT 3"
    good = f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'"
    client = _client(
        {INTENT_STAGE: [_intent("order")], SQL_GENERATION_STAGE: [_sql(unscoped), _sql(good)]}
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is None
    # 거부는 기존 경로를 탄다 — guard_rejected → 재시도 1회.
    assert len(result.sql_failures) == 1
    failure = result.sql_failures[0]
    assert failure.kind is SqlFailureKind.GUARD_REJECTED
    assert failure.query_sql == unscoped
    assert "order_scope" in failure.error

    retry_prompt = client.calls_for(SQL_GENERATION_STAGE)[1]["user"]
    assert "order_scope" in retry_prompt
    assert order_no in retry_prompt

    # 채택된 근거는 그 주문 1건뿐이다.
    assert len(result.sql_snapshots) == 1
    assert {row["order_no"] for row in result.sql_snapshots[0].result_rows} == {order_no}


@pytest.mark.db
def test_외부_조인_우회는_거부되고_피드백으로_1회_재시도해_성공한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    """외부 조인의 ON 절은 보존측을 거르지 않아 무관한 주문 50건이 근거가 된다.

    거부 사유 코드는 **가드 내부 코드**로 남고, 새 인계 사유를 만들지 않는다
    (재시도 상한 1회 그대로).
    """
    order_no = sample_order["order_no"]
    bypass = (
        "SELECT o.order_no, o.customer_name, o.customer_phone FROM orders o"
        f" LEFT JOIN (SELECT 1 AS x) d ON o.order_no = '{order_no}'"
    )
    good = f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'"
    client = _client(
        {INTENT_STAGE: [_intent("order")], SQL_GENERATION_STAGE: [_sql(bypass), _sql(good)]}
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is None
    assert len(result.sql_failures) == 1
    failure = result.sql_failures[0]
    assert failure.kind is SqlFailureKind.GUARD_REJECTED
    assert failure.query_sql == bypass
    assert "unsupported_join" in failure.error

    retry_prompt = client.calls_for(SQL_GENERATION_STAGE)[1]["user"]
    assert "unsupported_join" in retry_prompt
    assert "INNER JOIN" in retry_prompt

    # 채택된 근거는 선검사를 통과한 주문 1건뿐이다.
    assert len(result.sql_snapshots) == 1
    assert {row["order_no"] for row in result.sql_snapshots[0].result_rows} == {order_no}


@pytest.mark.db
def test_다른_주문번호를_계속_가리키면_sql_failed_로_인계한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    """재시도 상한은 1회 그대로다 — 새 인계 사유를 만들지 않는다."""
    order_no = sample_order["order_no"]
    other = "SELECT order_no, customer_phone FROM orders WHERE order_no = 'ORD-20991231-9999'"
    both = (
        f"SELECT order_no, customer_phone FROM orders WHERE order_no = '{order_no}'"
        " OR order_no = 'ORD-20991231-9999'"
    )
    client = _client(
        {INTENT_STAGE: [_intent("order")], SQL_GENERATION_STAGE: [_sql(other), _sql(both)]}
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is EscalationReason.SQL_FAILED
    assert result.evidence == ()
    assert result.sql_snapshots == ()
    assert [failure.attempt_no for failure in result.sql_failures] == [1, 2]
    assert all(failure.kind is SqlFailureKind.GUARD_REJECTED for failure in result.sql_failures)
    assert all("order_scope" in failure.error for failure in result.sql_failures)
    # 세 번째 생성 호출은 없다.
    assert len(client.calls_for(SQL_GENERATION_STAGE)) == 2


@pytest.mark.db
def test_pg_sleep_은_실행되지_않고_안전장치가_먼저_거부한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    """DB 를 건드리기 전에 거부되므로 이 테스트는 2초를 자지 않는다."""
    order_no = sample_order["order_no"]
    sleeping = f"SELECT order_no FROM orders WHERE order_no = '{order_no}' AND pg_sleep(2) IS NULL"
    good = f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'"
    client = _client(
        {INTENT_STAGE: [_intent("order")], SQL_GENERATION_STAGE: [_sql(sleeping), _sql(good)]}
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is None
    assert result.sql_failures[0].kind is SqlFailureKind.GUARD_REJECTED
    assert "forbidden_function" in result.sql_failures[0].error


@pytest.mark.db
def test_실행_오류도_SQL_실패_경로를_탄다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    """안전장치는 통과하지만 DB 가 거부하는 쿼리 — 검증만으로는 못 잡는 층이다."""
    order_no = sample_order["order_no"]
    broken = f"SELECT order_no FROM orders WHERE order_no = '{order_no}' AND quantity = 'abc'"
    good = f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'"
    client = _client(
        {INTENT_STAGE: [_intent("order")], SQL_GENERATION_STAGE: [_sql(broken), _sql(good)]}
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is None
    assert result.sql_failures[0].kind is SqlFailureKind.EXECUTION_ERROR
    assert result.sql_failures[0].error
    assert [item.id for item in result.evidence] == [f"sql:{INQUIRY_ID}:1"]


@pytest.mark.db
def test_유효_SQL_생성_실패도_SQL_실패_경로다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    broken = LLMFormatError(stage=SQL_GENERATION_STAGE, detail="JSON 아님", raw_text="SELECT")
    client = _client({INTENT_STAGE: [_intent("order")], SQL_GENERATION_STAGE: [broken, broken]})

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=sample_order["order_no"],
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is EscalationReason.SQL_FAILED
    assert all(failure.kind is SqlFailureKind.GENERATION_FAILED for failure in result.sql_failures)
    assert result.sql_failures[0].query_sql is None


@pytest.mark.db
def test_SQL_생성의_전송_오류는_llm_call_failed_다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    """전송 오류는 SQL 실패 경로가 아니라 공통 실패 정책 소관이다 (spec)."""
    client = _client(
        {
            INTENT_STAGE: [_intent("order")],
            SQL_GENERATION_STAGE: [
                LLMCallError(stage=SQL_GENERATION_STAGE, reason="transport_error", attempts=2)
            ],
        }
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=sample_order["order_no"],
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert result.failed_stage == SQL_GENERATION_STAGE


# ── 정상 실행 0건 (실패가 아니다) ───────────────────────────────────────────


@pytest.mark.db
def test_SQL_정상_실행_0건은_order_단독이면_no_evidence(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    order_no = sample_order["order_no"]
    client = _client(
        {
            INTENT_STAGE: [_intent("order")],
            SQL_GENERATION_STAGE: [
                _sql(
                    f"SELECT order_no FROM orders WHERE order_no = '{order_no}'"
                    " AND status = '그런상태없음'"
                )
            ],
        }
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    # 실패가 아니라 "주문 소스 근거 0건" 이다 — order_not_found 도 sql_failed 도 아니다.
    assert result.escalation_reason is EscalationReason.NO_EVIDENCE
    assert result.sql_failures == ()
    assert result.sql_snapshots == ()
    assert len(client.calls_for(SQL_GENERATION_STAGE)) == 1


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_both_에서_SQL_0건이면_정책_근거만으로_진행한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    order_no = sample_order["order_no"]
    client = _client(
        {
            INTENT_STAGE: [_intent("both")],
            SQL_GENERATION_STAGE: [
                _sql(
                    f"SELECT order_no FROM orders WHERE order_no = '{order_no}'"
                    " AND status = '그런상태없음'"
                )
            ],
        }
    )

    result = _collector(client, threshold=0.0).collect(
        inquiry_id=INQUIRY_ID,
        content="환불 신청 기간이 어떻게 되나요",
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is None
    assert result.evidence
    assert all(item.source is EvidenceSource.POLICY for item in result.evidence)


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_both_에서_주문_측_구조적_실패는_정책_근거가_있어도_인계가_우선한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    client = _client(
        {
            INTENT_STAGE: [_intent("both")],
            SQL_GENERATION_STAGE: [
                _sql("DROP TABLE orders"),
                _sql("UPDATE orders SET status = 'x'"),
            ],
        }
    )

    result = _collector(client, threshold=0.0).collect(
        inquiry_id=INQUIRY_ID,
        content="환불 신청 기간이 어떻게 되나요",
        order_no=sample_order["order_no"],
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is EscalationReason.SQL_FAILED
    # 그 시점까지 모은 근거는 그대로 실려 나간다 (처리 기록·감사용).
    assert result.evidence
    assert all(item.source is EvidenceSource.POLICY for item in result.evidence)


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_both_에서_주문번호가_없으면_정책_근거가_있어도_missing_order_ref(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    client = _client({INTENT_STAGE: [_intent("both")]})

    result = _collector(client, threshold=0.0).collect(
        inquiry_id=INQUIRY_ID,
        content="환불 신청 기간이 어떻게 되나요",
        order_no=None,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is EscalationReason.MISSING_ORDER_REF
    assert result.evidence


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_모든_소스_근거_0건이면_no_evidence(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    order_no = sample_order["order_no"]
    client = _client(
        {
            INTENT_STAGE: [_intent("both")],
            SQL_GENERATION_STAGE: [
                _sql(
                    f"SELECT order_no FROM orders WHERE order_no = '{order_no}'"
                    " AND status = '그런상태없음'"
                )
            ],
        }
    )

    result = _collector(client, threshold=0.99).collect(
        inquiry_id=INQUIRY_ID,
        content="환불 신청 기간이 어떻게 되나요",
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.evidence == ()
    assert result.escalation_reason is EscalationReason.NO_EVIDENCE


# ── 실행 시간 상한 (read-only 커넥션의 statement_timeout) ───────────────────


@pytest.mark.db
def test_read_only_커넥션에_statement_timeout_이_걸려_있다(
    ro_conn: psycopg.Connection[DictRow], db_settings: Settings
) -> None:
    """가드가 함수를 막아도 실행 시간까지 코드가 예측할 수는 없다 — DB 에 직접 물어본다.

    `0` 은 무제한이다. 무제한이면 쿼리 하나가 워커를 몇 분씩 묶는다.
    """
    row = ro_conn.execute("SHOW statement_timeout").fetchone()

    assert row is not None
    assert row["statement_timeout"] not in ("0", "")
    assert db_settings.sql_statement_timeout_ms > 0


@pytest.mark.db
def test_상한을_넘긴_쿼리는_실제로_끊긴다(db_settings: Settings) -> None:
    """상한이 **동작하는지**까지 확인한다. 250ms 상한 + 2초 sleep 이라 매달리지 않는다."""
    settings = db_settings.model_copy(update={"sql_statement_timeout_ms": 250})

    with readonly_connect(settings=settings) as conn:
        assert conn.execute("SHOW statement_timeout").fetchone() == {"statement_timeout": "250ms"}
        with pytest.raises(psycopg.errors.QueryCanceled):
            conn.execute("SELECT pg_sleep(2)")


# ── 토큰 집계 ───────────────────────────────────────────────────────────────


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_토큰은_생성과_임베딩을_분리해_집계한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    order_no = sample_order["order_no"]
    client = _client(
        {
            INTENT_STAGE: [_intent("both", input_tokens=100, output_tokens=5)],
            SQL_GENERATION_STAGE: [
                _sql(
                    f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'",
                    input_tokens=200,
                    output_tokens=20,
                )
            ],
        }
    )

    result = _collector(client, threshold=0.0).collect(
        inquiry_id=INQUIRY_ID,
        content="환불 신청 기간이 어떻게 되나요",
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    # 생성 토큰은 의도 분류 + SQL 생성의 합이다.
    assert (result.input_tokens, result.output_tokens) == (300, 25)
    # 임베딩 토큰은 생성 토큰과 섞이지 않는다 (건당 비용을 따로 산출한다).
    assert result.embedding_tokens == len("환불 신청 기간이 어떻게 되나요")


@pytest.mark.db
def test_임베딩은_정책_검색이_필요할_때만_호출된다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    order_no = sample_order["order_no"]
    client = _client(
        {
            INTENT_STAGE: [_intent("order")],
            SQL_GENERATION_STAGE: [
                _sql(f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'")
            ],
        }
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    # order 단독 의도는 정책 검색을 하지 않으므로 임베딩 호출 자체가 없다.
    assert result.embedding_tokens == 0
    assert result.escalation_reason is None
    assert INQUIRY_EMBEDDING_STAGE == "inquiry_embedding"
