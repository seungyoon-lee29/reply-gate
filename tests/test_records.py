"""처리 기록 저장·조회 테스트 (DB).

처리 기록은 평가 지표(p50/p95 지연, 건당 토큰)의 원천이므로 **저장한 그대로 복원**되는
것이 계약이다. 모든 쓰기는 `app_conn` 트랜잭션 안에서만 일어나고 픽스처가 롤백한다 —
공유 DB 에 남기는 행이 없다.
"""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from psycopg.rows import DictRow

from reply_gate.contracts import (
    Claim,
    EscalationReason,
    Evidence,
    EvidenceSource,
    InquiryStatus,
    IntentSource,
    RejectReason,
    Verdict,
    policy_evidence_id,
    sql_evidence_id,
)
from reply_gate.draft import DRAFT_STAGE
from reply_gate.evidence import (
    INTENT_STAGE,
    SQL_GENERATION_STAGE,
    SqlEvidenceSnapshot,
    SqlFailure,
    SqlFailureKind,
)
from reply_gate.pipeline import AttemptRecord, ProcessedInquiry, new_inquiry_id
from reply_gate.records import load_inquiry, save_inquiry
from tests.test_pipeline import (
    citing_draft,
    intent_completion,
    live_pipeline,
    run_live,
    scripted_client,
    sql_completion,
)

pytestmark = pytest.mark.db

POLICY_ID = policy_evidence_id(document_slug="refund", article="2.1")
EXECUTED_SQL = "SELECT order_no, status FROM orders WHERE order_no = 'ORD-20260315-0001'"
RESULT_ROWS: tuple[dict[str, Any], ...] = ({"order_no": "ORD-20260315-0001", "status": "배송완료"},)


def _processed(
    inquiry_id: str,
    *,
    status: InquiryStatus = InquiryStatus.ANSWERED,
    answer: str | None = "환불은 수령 후 7일 이내에 신청하실 수 있습니다.",
    escalation: EscalationReason | None = None,
    failed_stage: str | None = None,
    attempts: tuple[AttemptRecord, ...] | None = None,
) -> ProcessedInquiry:
    """정책 근거 1건 + SQL 근거 1건 + 시도 2건 + SQL 실패 1건을 가진 처리 결과."""
    sql_id = sql_evidence_id(inquiry_id=inquiry_id, sequence=1)
    default_attempts = (
        AttemptRecord(
            attempt_no=1,
            verdict=Verdict.REJECT,
            reject_reasons=(RejectReason.MISSING_CITATION, RejectReason.PII_DETECTED),
            draft={"claims": [{"text": "1588-0000 으로 연락 주세요.", "citation_ids": []}]},
        ),
        AttemptRecord(
            attempt_no=2,
            verdict=Verdict.PASS,
            reject_reasons=(),
            draft={
                "claims": [
                    {
                        "text": "환불은 수령 후 7일 이내에 신청하실 수 있습니다.",
                        "citation_ids": [POLICY_ID],
                    }
                ]
            },
        ),
    )
    return ProcessedInquiry(
        inquiry_id=inquiry_id,
        order_no="ORD-20260315-0001",
        content="환불 언제까지 되나요? 제 주문도 확인해 주세요.",
        intent=IntentSource.BOTH,
        status=status,
        answer=answer,
        claims=(
            Claim(
                text="환불은 수령 후 7일 이내에 신청하실 수 있습니다.",
                citation_ids=(POLICY_ID,),
            ),
        )
        if status is InquiryStatus.ANSWERED
        else (),
        escalation_reason=escalation,
        failed_stage=failed_stage,
        evidence=(
            Evidence(
                id=POLICY_ID,
                source=EvidenceSource.POLICY,
                content="환불은 수령 후 7일 이내에 신청할 수 있다.",
                evidence_text="[환불 정책 2.1 환불 기간] 환불은 수령 후 7일 이내에 신청할 수 있다.",
            ),
            Evidence(
                id=sql_id,
                source=EvidenceSource.SQL,
                content=f"실행 쿼리: {EXECUTED_SQL}\n결과 1건",
                evidence_text=f"실행 쿼리: {EXECUTED_SQL}\n결과 1건\n1) order_no=ORD-20260315-0001, status=배송완료",
            ),
        ),
        sql_snapshots=(
            SqlEvidenceSnapshot(
                evidence_id=sql_id,
                sequence=1,
                query_sql=EXECUTED_SQL,
                result_rows=RESULT_ROWS,
            ),
        ),
        sql_failures=(
            SqlFailure(
                attempt_no=1,
                kind=SqlFailureKind.GUARD_REJECTED,
                query_sql="DELETE FROM orders",
                error="non_select_statement: SELECT 단일문만 허용된다",
            ),
        ),
        attempts=default_attempts if attempts is None else attempts,
        latency_ms=1234,
        input_tokens=910,
        output_tokens=210,
        embedding_tokens=57,
    )


# ── 왕복 ────────────────────────────────────────────────────────────────────


def test_저장한_처리_기록이_그대로_복원된다(app_conn: psycopg.Connection[DictRow]) -> None:
    processed = _processed(new_inquiry_id())

    save_inquiry(conn=app_conn, processed=processed)
    loaded = load_inquiry(conn=app_conn, inquiry_id=processed.inquiry_id)

    assert loaded == processed


def test_인계_기록도_그대로_복원된다(app_conn: psycopg.Connection[DictRow]) -> None:
    processed = _processed(
        new_inquiry_id(),
        status=InquiryStatus.ESCALATED,
        answer=None,
        escalation=EscalationReason.LLM_CALL_FAILED,
        failed_stage=DRAFT_STAGE,
        attempts=(),
    )

    save_inquiry(conn=app_conn, processed=processed)
    loaded = load_inquiry(conn=app_conn, inquiry_id=processed.inquiry_id)

    assert loaded == processed
    assert loaded is not None
    assert loaded.failed_stage == DRAFT_STAGE
    # 초안 전 인계라도 그 시점까지의 근거는 남는다 (감사 목적).
    assert [item.id for item in loaded.evidence] == [POLICY_ID, processed.evidence[1].id]


def test_없는_문의는_None_이다(app_conn: psycopg.Connection[DictRow]) -> None:
    assert load_inquiry(conn=app_conn, inquiry_id=new_inquiry_id()) is None


def test_문의_ID_가_uuid_가_아니면_None_이다(app_conn: psycopg.Connection[DictRow]) -> None:
    """경로 파라미터로 아무 문자열이나 들어와도 조회는 '없음'으로 끝난다(오류가 아니다)."""
    assert load_inquiry(conn=app_conn, inquiry_id="이건-uuid-가-아니다") is None


# ── 저장된 행의 모양 ────────────────────────────────────────────────────────


def test_시도는_최대_2건이고_attempt_no_는_1부터_2다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    processed = _processed(new_inquiry_id())
    save_inquiry(conn=app_conn, processed=processed)

    rows = app_conn.execute(
        "SELECT attempt_no, verdict, reject_reasons FROM inquiry_attempts"
        " WHERE inquiry_id = %s ORDER BY attempt_no",
        (processed.inquiry_id,),
    ).fetchall()

    assert [row["attempt_no"] for row in rows] == [1, 2]
    assert [row["verdict"] for row in rows] == [Verdict.REJECT.value, Verdict.PASS.value]
    assert rows[0]["reject_reasons"] == [
        RejectReason.MISSING_CITATION.value,
        RejectReason.PII_DETECTED.value,
    ]
    assert rows[1]["reject_reasons"] == []


def test_SQL_근거는_ID_제약을_만족하고_쿼리문과_결과_행_전체를_남긴다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    processed = _processed(new_inquiry_id())
    save_inquiry(conn=app_conn, processed=processed)

    row = app_conn.execute(
        "SELECT evidence_id, sequence, query_sql, result_rows FROM inquiry_evidence"
        " WHERE inquiry_id = %s AND source = 'sql'",
        (processed.inquiry_id,),
    ).fetchone()

    assert row is not None
    # CHECK: evidence_id = 'sql:' || inquiry_id::text || ':' || sequence::text
    assert row["evidence_id"] == f"sql:{processed.inquiry_id}:1"
    assert row["sequence"] == 1
    assert row["query_sql"] == EXECUTED_SQL
    assert row["result_rows"] == list(RESULT_ROWS)


def test_정책_근거는_스냅샷_컬럼을_비운다(app_conn: psycopg.Connection[DictRow]) -> None:
    processed = _processed(new_inquiry_id())
    save_inquiry(conn=app_conn, processed=processed)

    row = app_conn.execute(
        "SELECT sequence, query_sql, result_rows FROM inquiry_evidence"
        " WHERE inquiry_id = %s AND source = 'policy'",
        (processed.inquiry_id,),
    ).fetchone()

    assert row is not None
    assert (row["sequence"], row["query_sql"], row["result_rows"]) == (None, None, None)


def test_지연과_토큰이_기록되고_임베딩_토큰이_생성_토큰에_섞이지_않는다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    processed = _processed(new_inquiry_id())
    save_inquiry(conn=app_conn, processed=processed)

    row = app_conn.execute(
        "SELECT latency_ms, input_tokens, output_tokens, embedding_tokens, failed_stage"
        " FROM inquiries WHERE id = %s",
        (processed.inquiry_id,),
    ).fetchone()

    assert row is not None
    assert row["latency_ms"] == 1234
    assert (row["input_tokens"], row["output_tokens"]) == (910, 210)
    assert row["embedding_tokens"] == 57


def test_SQL_실패는_근거_ID_없이_쿼리문과_오류만_남는다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    processed = _processed(new_inquiry_id())
    save_inquiry(conn=app_conn, processed=processed)

    row = app_conn.execute(
        "SELECT attempt_no, failure_kind, query_sql, error FROM inquiry_sql_failures"
        " WHERE inquiry_id = %s",
        (processed.inquiry_id,),
    ).fetchone()

    assert row is not None
    assert row["attempt_no"] == 1
    assert row["failure_kind"] == SqlFailureKind.GUARD_REJECTED.value
    assert row["query_sql"] == "DELETE FROM orders"


# ── 실제 파이프라인 산출물의 저장 ───────────────────────────────────────────


def test_실제_SQL_근거의_ID_가_저장_시_CHECK_를_통과한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
) -> None:
    """근거 수집 **전에** 확정한 문의 ID 가 SQL 근거 ID 제약과 맞아떨어지는지 끝까지 확인."""
    order_row = ro_conn.execute("SELECT order_no FROM orders ORDER BY order_no LIMIT 1").fetchone()
    assert order_row is not None
    order_no = str(order_row["order_no"])

    client = scripted_client(
        {
            INTENT_STAGE: [intent_completion("order")],
            SQL_GENERATION_STAGE: [
                sql_completion(f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'")
            ],
            DRAFT_STAGE: [citing_draft(text="주문 상태를 안내드립니다.")],
        }
    )
    processed: ProcessedInquiry = run_live(
        live_pipeline(client),
        app_conn,
        ro_conn,
        content="제 주문 상태를 알려주세요.",
        order_no=order_no,
    )
    assert processed.status is InquiryStatus.ANSWERED
    assert processed.sql_snapshots != ()

    save_inquiry(conn=app_conn, processed=processed)
    loaded = load_inquiry(conn=app_conn, inquiry_id=processed.inquiry_id)

    assert loaded == processed
    assert processed.sql_snapshots[0].evidence_id == f"sql:{processed.inquiry_id}:1"
