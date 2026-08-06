"""처리 기록 저장·조회 테스트 (DB).

처리 기록은 평가 지표(p50/p95 지연, 건당 토큰)의 원천이므로 **저장한 그대로 복원**되는
것이 계약이다. 모든 쓰기는 `app_conn` 트랜잭션 안에서만 일어나고 픽스처가 롤백한다 —
공유 DB 에 남기는 행이 없다.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import psycopg
import pytest
from psycopg.rows import DictRow

from reply_gate.contracts import (
    Claim,
    ClaimJudgment,
    EscalationReason,
    Evidence,
    EvidenceContradiction,
    EvidenceSource,
    GateResult,
    InquiryStatus,
    IntentSource,
    JudgeResult,
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
    """정책 근거 1건 + SQL 근거 1건 + 시도 2건 + SQL 실패 1건을 가진 처리 결과.

    시도 2건이 층별 판정의 두 모양을 모두 덮는다: 시도 1 은 **L1 reject → L2 미실행**(층별
    L2 컬럼 전부 NULL), 시도 2 는 **L1 pass → L2 실행**(claim 판정·모순쌍 포함)이다.
    """
    sql_id = sql_evidence_id(inquiry_id=inquiry_id, sequence=1)
    answered_text = "환불은 수령 후 7일 이내에 신청하실 수 있습니다."
    default_attempts = (
        AttemptRecord(
            attempt_no=1,
            verdict=Verdict.REJECT,
            reject_reasons=(RejectReason.MISSING_CITATION, RejectReason.PII_DETECTED),
            draft={"claims": [{"text": "1588-0000 으로 연락 주세요.", "citation_ids": []}]},
            l1_result=GateResult(
                verdict=Verdict.REJECT,
                reject_reasons=(RejectReason.MISSING_CITATION, RejectReason.PII_DETECTED),
            ),
            l2_result=None,
        ),
        AttemptRecord(
            attempt_no=2,
            verdict=Verdict.PASS,
            reject_reasons=(),
            draft={"claims": [{"text": answered_text, "citation_ids": [POLICY_ID]}]},
            l1_result=GateResult(verdict=Verdict.PASS),
            l2_result=JudgeResult(
                verdict=Verdict.PASS,
                claim_judgments=(
                    ClaimJudgment(
                        claim_text=answered_text,
                        verdict=Verdict.PASS,
                        explanation="환불 정책 2.1 이 7일 기간을 그대로 말한다.",
                    ),
                ),
                contradictions=(
                    EvidenceContradiction(
                        evidence_id_a=POLICY_ID,
                        evidence_id_b=sql_id,
                        explanation="정책은 7일, 주문 근거는 배송완료 후 14일로 읽힌다.",
                    ),
                ),
            ),
        ),
    )
    return ProcessedInquiry(
        inquiry_id=inquiry_id,
        order_no="ORD-20260315-0001",
        content="환불 언제까지 되나요? 제 주문도 확인해 주세요.",
        intent=IntentSource.BOTH,
        status=status,
        answer=answer,
        claims=(Claim(text=answered_text, citation_ids=(POLICY_ID,)),)
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
        judge_input_tokens=433,
        judge_output_tokens=91,
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


def test_층별_판정과_claim_판정과_모순쌍이_그대로_복원된다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """종합 판정만이 아니라 **층별 내역**이 왕복한다 — 데모의 '왜 기각됐는지' 장면의 원천이다."""
    processed = _processed(new_inquiry_id())

    save_inquiry(conn=app_conn, processed=processed)
    loaded = load_inquiry(conn=app_conn, inquiry_id=processed.inquiry_id)

    assert loaded is not None
    first, second = loaded.attempts
    # 시도 1 — L1 reject 라 L2 는 실행되지 않았다: `None` 이 그대로 돌아온다.
    assert first.l1_result == processed.attempts[0].l1_result
    assert first.l2_result is None
    # 시도 2 — L1 pass 뒤 L2 실행: claim 판정과 근거쌍 모순까지 값 그대로.
    assert second.l1_result == GateResult(verdict=Verdict.PASS)
    assert second.l2_result == processed.attempts[1].l2_result
    assert second.l2_result is not None
    assert second.l2_result.claim_judgments[0].verdict is Verdict.PASS
    assert second.l2_result.contradictions[0].evidence_id_b.startswith("sql:")


def test_L2_기각_판정이_사유와_함께_왕복한다(app_conn: psycopg.Connection[DictRow]) -> None:
    """종합 사유 = L1 사유 ∥ L2 사유. L1 pass + L2 reject 조합이 DB CHECK 를 지나 복원된다."""
    inquiry_id = new_inquiry_id()
    judged = JudgeResult(
        verdict=Verdict.REJECT,
        reject_reasons=(RejectReason.UNSUPPORTED_CLAIM,),
        claim_judgments=(
            ClaimJudgment(
                claim_text="교환은 무료입니다.",
                verdict=Verdict.REJECT,
                explanation="인용한 근거에 교환 비용 언급이 없다.",
            ),
        ),
    )
    processed = _processed(
        inquiry_id,
        status=InquiryStatus.ESCALATED,
        answer=None,
        escalation=EscalationReason.REJECTED_TWICE,
        attempts=(
            AttemptRecord(
                attempt_no=1,
                verdict=Verdict.REJECT,
                reject_reasons=(RejectReason.UNSUPPORTED_CLAIM,),
                draft={"claims": [{"text": "교환은 무료입니다.", "citation_ids": [POLICY_ID]}]},
                l1_result=GateResult(verdict=Verdict.PASS),
                l2_result=judged,
            ),
        ),
    )

    save_inquiry(conn=app_conn, processed=processed)
    loaded = load_inquiry(conn=app_conn, inquiry_id=inquiry_id)

    assert loaded == processed
    assert loaded is not None
    assert loaded.attempts[0].l2_result == judged


def test_판정_토큰이_생성_토큰과_섞이지_않고_왕복한다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    processed = _processed(new_inquiry_id())

    save_inquiry(conn=app_conn, processed=processed)
    loaded = load_inquiry(conn=app_conn, inquiry_id=processed.inquiry_id)

    assert loaded is not None
    assert (loaded.judge_input_tokens, loaded.judge_output_tokens) == (433, 91)
    assert (loaded.input_tokens, loaded.output_tokens) == (910, 210)


def test_L2_미실행이면_판정_토큰은_0_으로_왕복한다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """L1 reject 로 끝난 시도만 있는 문의는 판정 호출이 없었으므로 판정 토큰이 0 이다."""
    processed = replace(
        _processed(
            new_inquiry_id(),
            status=InquiryStatus.ESCALATED,
            answer=None,
            escalation=EscalationReason.REJECTED_TWICE,
        ),
        judge_input_tokens=0,
        judge_output_tokens=0,
        attempts=(
            AttemptRecord(
                attempt_no=1,
                verdict=Verdict.REJECT,
                reject_reasons=(RejectReason.PII_DETECTED,),
                draft={"claims": [{"text": "010-1234-5678 로 연락 주세요.", "citation_ids": []}]},
                l1_result=GateResult(
                    verdict=Verdict.REJECT, reject_reasons=(RejectReason.PII_DETECTED,)
                ),
                l2_result=None,
            ),
        ),
    )

    save_inquiry(conn=app_conn, processed=processed)
    loaded = load_inquiry(conn=app_conn, inquiry_id=processed.inquiry_id)

    assert loaded == processed
    assert loaded is not None
    assert (loaded.judge_input_tokens, loaded.judge_output_tokens) == (0, 0)
    assert loaded.attempts[0].l2_result is None


def test_층별_판정이_없는_시도도_그대로_복원된다(app_conn: psycopg.Connection[DictRow]) -> None:
    """층별 내역 없이 종합 판정만 든 시도(층별 컬럼 전부 NULL)도 왕복이 성립한다."""
    processed = _processed(
        new_inquiry_id(),
        attempts=(
            AttemptRecord(
                attempt_no=1,
                verdict=Verdict.PASS,
                reject_reasons=(),
                draft={"claims": [{"text": "안내드립니다.", "citation_ids": [POLICY_ID]}]},
            ),
        ),
    )

    save_inquiry(conn=app_conn, processed=processed)
    loaded = load_inquiry(conn=app_conn, inquiry_id=processed.inquiry_id)

    assert loaded == processed
    assert loaded is not None
    assert (loaded.attempts[0].l1_result, loaded.attempts[0].l2_result) == (None, None)


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


def test_층별_컬럼은_L2_미실행_시도에서만_NULL_이다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """L2 미실행이면 부속(사유·claim 판정·모순쌍)까지 전부 NULL 이고, 실행이면 빈 배열이라도
    값이 남는다 — NULL 을 '실행했지만 결과 0건'과 섞지 않는 것이 CHECK 의 전제다."""
    processed = _processed(new_inquiry_id())
    save_inquiry(conn=app_conn, processed=processed)

    rows = app_conn.execute(
        "SELECT l1_verdict, l1_reject_reasons, l2_verdict, l2_reject_reasons,"
        " claim_verdicts, evidence_contradictions FROM inquiry_attempts"
        " WHERE inquiry_id = %s ORDER BY attempt_no",
        (processed.inquiry_id,),
    ).fetchall()

    # 시도 1 — L1 reject → L2 부속 전부 NULL.
    assert rows[0]["l1_verdict"] == Verdict.REJECT.value
    assert rows[0]["l1_reject_reasons"] == [
        RejectReason.MISSING_CITATION.value,
        RejectReason.PII_DETECTED.value,
    ]
    for column in ("l2_verdict", "l2_reject_reasons", "claim_verdicts", "evidence_contradictions"):
        assert rows[0][column] is None

    # 시도 2 — L2 실행 → 사유가 0건이어도 빈 배열로 남는다.
    assert rows[1]["l1_reject_reasons"] == []
    assert rows[1]["l2_verdict"] == Verdict.PASS.value
    assert rows[1]["l2_reject_reasons"] == []
    assert rows[1]["claim_verdicts"] == [
        {
            "claim_text": "환불은 수령 후 7일 이내에 신청하실 수 있습니다.",
            "verdict": Verdict.PASS.value,
            "explanation": "환불 정책 2.1 이 7일 기간을 그대로 말한다.",
        }
    ]
    assert rows[1]["evidence_contradictions"][0]["evidence_id_a"] == POLICY_ID


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
        "SELECT latency_ms, input_tokens, output_tokens, embedding_tokens,"
        " judge_input_tokens, judge_output_tokens, failed_stage"
        " FROM inquiries WHERE id = %s",
        (processed.inquiry_id,),
    ).fetchone()

    assert row is not None
    assert row["latency_ms"] == 1234
    assert (row["input_tokens"], row["output_tokens"]) == (910, 210)
    assert row["embedding_tokens"] == 57
    # 판정 토큰도 분리 컬럼이다 — 생성 토큰에 합산되지 않는다.
    assert (row["judge_input_tokens"], row["judge_output_tokens"]) == (433, 91)


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


# ── 층별 판정 CHECK 음성 케이스 ─────────────────────────────────────────────
# 층별 컬럼을 **실제로 채워 넣는지**를 이 저장 경로로 확인한다: 채우지 않으면 CHECK 가
# 발화하지 않아 아래 단언들이 전부 실패한다. 모순 상태는 저장 시점에 거부되어야 하고,
# 조용히 들어가면 판정 일치율 지표가 오염된 채 재현된다.


def _saved_layer_violation(
    conn: psycopg.Connection[DictRow], attempt: AttemptRecord
) -> psycopg.errors.CheckViolation:
    processed = _processed(new_inquiry_id(), attempts=(attempt,))
    with pytest.raises(psycopg.errors.CheckViolation) as excinfo:
        save_inquiry(conn=conn, processed=processed)
    return excinfo.value


def test_L1_reject_인데_L2_판정이_있으면_거부된다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """L2 는 L1 통과분에만 돌린다 — L1 이 막은 초안의 L2 판정은 존재할 수 없다."""
    error = _saved_layer_violation(
        app_conn,
        AttemptRecord(
            attempt_no=1,
            verdict=Verdict.REJECT,
            reject_reasons=(RejectReason.PII_DETECTED,),
            draft={"claims": [{"text": "010-1234-5678", "citation_ids": []}]},
            l1_result=GateResult(
                verdict=Verdict.REJECT, reject_reasons=(RejectReason.PII_DETECTED,)
            ),
            l2_result=JudgeResult(verdict=Verdict.PASS),
        ),
    )
    assert error.diag.constraint_name == "inquiry_attempts_l2_requires_l1_pass"


def test_종합_사유가_층별_사유의_합과_다르면_거부된다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """종합 사유 = L1 사유 ∥ L2 사유. 어긋난 기록은 지표 산출의 근거가 될 수 없다."""
    error = _saved_layer_violation(
        app_conn,
        AttemptRecord(
            attempt_no=1,
            verdict=Verdict.REJECT,
            reject_reasons=(RejectReason.PII_DETECTED,),
            draft={"claims": [{"text": "안내드립니다.", "citation_ids": []}]},
            l1_result=GateResult(
                verdict=Verdict.REJECT, reject_reasons=(RejectReason.MISSING_CITATION,)
            ),
            l2_result=None,
        ),
    )
    assert error.diag.constraint_name == "inquiry_attempts_reasons_compose_layers"


def test_종합_pass_인데_층이_reject_면_거부된다(app_conn: psycopg.Connection[DictRow]) -> None:
    """어느 층이든 기각했으면 종합은 pass 일 수 없다 — 기각 장면이 통과로 접히면 안 된다."""
    error = _saved_layer_violation(
        app_conn,
        AttemptRecord(
            attempt_no=1,
            verdict=Verdict.PASS,
            reject_reasons=(),
            draft={"claims": [{"text": "안내드립니다.", "citation_ids": [POLICY_ID]}]},
            l1_result=GateResult(
                verdict=Verdict.REJECT, reject_reasons=(RejectReason.PII_DETECTED,)
            ),
            l2_result=None,
        ),
    )
    # 이 행은 두 CHECK 를 동시에 어긴다(종합 pass ↔ 층 reject, 종합 사유 ↔ 층별 합).
    # 어느 쪽이 먼저 발화하든 거부여야 한다.
    assert error.diag.constraint_name in {
        "inquiry_attempts_pass_needs_no_layer_reject",
        "inquiry_attempts_reasons_compose_layers",
    }


def test_L2_판정이_없으면_부속을_애초에_쓰지_않는다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """'L2 null 인데 claim 판정 존재' 행은 이 저장 경로에서 만들어질 수 없다 — `l2_result`
    가 `None` 이면 부속 컬럼을 통째로 NULL 로 쓰기 때문이다(CHECK 가 그 상태를 금지한다)."""
    processed = _processed(
        new_inquiry_id(),
        attempts=(
            AttemptRecord(
                attempt_no=1,
                verdict=Verdict.REJECT,
                reject_reasons=(RejectReason.PII_DETECTED,),
                draft={"claims": [{"text": "010-1234-5678", "citation_ids": []}]},
                l1_result=GateResult(
                    verdict=Verdict.REJECT, reject_reasons=(RejectReason.PII_DETECTED,)
                ),
                l2_result=None,
            ),
        ),
    )
    save_inquiry(conn=app_conn, processed=processed)

    row = app_conn.execute(
        "SELECT l2_verdict, l2_reject_reasons, claim_verdicts, evidence_contradictions"
        " FROM inquiry_attempts WHERE inquiry_id = %s",
        (processed.inquiry_id,),
    ).fetchone()

    assert row is not None
    assert list(row.values()) == [None, None, None, None]


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

    # 파이프라인이 실은 층별 판정(`l1_result`/`l2_result`)까지 벗기지 않고 통째로 대조한다 —
    # 이 단언이 녹색이라는 것이 층별 내역의 왕복이 실제로 성립한다는 증거다.
    assert loaded == processed
    assert processed.attempts[0].l1_result is not None
    assert processed.sql_snapshots[0].evidence_id == f"sql:{processed.inquiry_id}:1"
