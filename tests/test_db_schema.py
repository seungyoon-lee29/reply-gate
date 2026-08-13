"""DB 통합 테스트 — 시딩 결과와 처리 기록 스키마.

전제: `docker compose up -d --wait` 로 Postgres 가 떠 있어야 한다. 없으면 `tests/conftest.py`
가 사유를 붙여 skip 한다.
"""

from __future__ import annotations

import re
from typing import Any

import psycopg
import pytest
from psycopg.rows import DictRow
from psycopg.types.json import Jsonb
from scripts.seed_orders import ORDER_COUNT, ORDER_STATUSES, load_fixture, seed_orders

from reply_gate.config import Settings
from reply_gate.contracts import (
    EscalationReason,
    EvidenceSource,
    InquiryStatus,
    IntentSource,
    RejectReason,
    Verdict,
    policy_evidence_id,
    sql_evidence_id,
)
from reply_gate.db import connect
from reply_gate.order_ref import ORDER_NO_REGEX, is_valid_order_no

pytestmark = pytest.mark.db

PHONE_PATTERN = re.compile(r"^01\d-\d{4}-\d{4}$")
HANGUL = re.compile(r"[가-힣]")


def _scalar(conn: psycopg.Connection[DictRow], sql: str, params: Any = None) -> Any:
    row = conn.execute(sql, params).fetchone()
    assert row is not None
    return next(iter(row.values()))


# ── 시딩 ─────────────────────────────────────────────────────────────────────


def test_seeded_order_count_is_500(seeded_order_count: int) -> None:
    assert seeded_order_count == ORDER_COUNT


def test_orders_table_holds_exactly_500_rows(app_conn: psycopg.Connection[DictRow]) -> None:
    assert _scalar(app_conn, "SELECT count(*) FROM orders") == ORDER_COUNT


def test_reseeding_is_idempotent(
    app_conn: psycopg.Connection[DictRow], seeded_order_count: int
) -> None:
    """시딩을 한 번 더 돌려도 정확히 500건이다 (행위 계약: 재실행 가능)."""
    assert seeded_order_count == ORDER_COUNT
    assert seed_orders() == ORDER_COUNT
    assert seed_orders() == ORDER_COUNT
    assert _scalar(app_conn, "SELECT count(*) FROM orders") == ORDER_COUNT


def test_reseeding_removes_rows_that_are_not_in_the_fixture(
    seeded_order_count: int,
) -> None:
    """픽스처에 없는 행이 끼어들어도 재시딩이 정확히 픽스처 상태로 되돌린다."""
    del seeded_order_count
    stray = "ORD-20200101-0001"
    with connect() as conn:
        conn.execute(
            """INSERT INTO orders (order_no, customer_name, customer_phone, customer_email,
                                   shipping_address, product_name, quantity, unit_price_krw,
                                   total_price_krw, status, ordered_at)
               VALUES (%s, '침입자', '010-0000-0000', 'stray@example.com', '서울특별시 강남구',
                       '테스트 상품', 1, 1000, 1000, '결제완료', now())""",
            (stray,),
        )
        conn.commit()
        assert _scalar(conn, "SELECT count(*) FROM orders") == ORDER_COUNT + 1

    assert seed_orders() == ORDER_COUNT

    with connect() as conn:
        assert _scalar(conn, "SELECT count(*) FROM orders WHERE order_no = %s", (stray,)) == 0


def test_every_seeded_order_no_passes_the_shared_format_definition(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """시딩과 접수 검증이 같은 정의를 공유한다는 계약의 실증."""
    rows = app_conn.execute("SELECT order_no FROM orders").fetchall()
    bad = [row["order_no"] for row in rows if not is_valid_order_no(row["order_no"])]
    assert bad == []


def test_fixture_matches_the_database(app_conn: psycopg.Connection[DictRow]) -> None:
    """시딩은 픽스처 로드일 뿐이다 — DB 내용이 커밋된 픽스처와 정확히 같아야 한다."""
    fixture = {record.order_no: record for record in load_fixture()}
    assert len(fixture) == ORDER_COUNT
    rows = app_conn.execute(
        "SELECT order_no, customer_name, status, total_price_krw FROM orders"
    ).fetchall()
    assert {row["order_no"] for row in rows} == set(fixture)
    for row in rows:
        record = fixture[row["order_no"]]
        assert (row["customer_name"], row["status"], row["total_price_krw"]) == (
            record.customer_name,
            record.status,
            record.total_price_krw,
        )


def test_orders_carry_the_fields_pii_and_text_to_sql_demos_need(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    rows = app_conn.execute(
        """SELECT customer_name, customer_phone, customer_email, shipping_address,
                  product_name, status, ordered_at
           FROM orders"""
    ).fetchall()
    assert len(rows) == ORDER_COUNT
    for row in rows:
        assert PHONE_PATTERN.fullmatch(row["customer_phone"]), row["customer_phone"]
        assert row["customer_email"].endswith("@example.com")
        assert HANGUL.search(row["customer_name"]), "고객명은 한국어여야 한다"
        assert HANGUL.search(row["shipping_address"]), "배송지는 한국어여야 한다"
        assert HANGUL.search(row["product_name"]), "상품명은 한국어여야 한다"
        assert row["status"] in ORDER_STATUSES
        assert row["ordered_at"] is not None


def test_all_order_statuses_appear_enough_for_a_demo(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    rows = app_conn.execute("SELECT status, count(*) AS n FROM orders GROUP BY status").fetchall()
    counts = {row["status"]: row["n"] for row in rows}
    assert set(counts) == set(ORDER_STATUSES)
    assert min(counts.values()) >= 5, counts


def test_shipping_timestamps_are_consistent_with_status(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    unshipped_with_tracking = _scalar(
        app_conn,
        """SELECT count(*) FROM orders
           WHERE status IN ('결제완료', '상품준비중', '취소')
             AND (shipped_at IS NOT NULL OR tracking_no IS NOT NULL)""",
    )
    assert unshipped_with_tracking == 0
    delivered_without_dates = _scalar(
        app_conn,
        """SELECT count(*) FROM orders
           WHERE status = '배송완료' AND (shipped_at IS NULL OR delivered_at IS NULL)""",
    )
    assert delivered_without_dates == 0


# ── 스키마가 코드의 정의와 어긋나지 않는지 ───────────────────────────────────


@pytest.mark.parametrize("constraint", ["orders_order_no_format", "inquiries_order_no_format"])
def test_db_order_no_check_uses_the_same_regex_as_order_ref(
    app_conn: psycopg.Connection[DictRow], constraint: str
) -> None:
    """DB CHECK 와 `order_ref.ORDER_NO_REGEX` 가 어긋나면 실패한다."""
    definition = _scalar(
        app_conn,
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = %s",
        (constraint,),
    )
    assert ORDER_NO_REGEX in definition, definition


def test_policy_chunk_vector_dimension_matches_settings(
    app_conn: psycopg.Connection[DictRow], db_settings: Settings
) -> None:
    column_type = _scalar(
        app_conn,
        """SELECT format_type(a.atttypid, a.atttypmod)
           FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
           WHERE c.relname = 'policy_chunks' AND a.attname = 'embedding'""",
    )
    assert column_type == f"vector({db_settings.embedding_dimensions})"


def test_policy_chunk_embedding_provenance_is_not_optional(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """출처 없는 벡터는 DB 가 받지 않는다 — 같은 차원의 다른 모델을 가릴 유일한 근거다."""
    with pytest.raises(psycopg.errors.NotNullViolation):
        app_conn.execute(
            """INSERT INTO policy_chunks
                   (evidence_id, document_slug, document_title, article, content)
               VALUES ('policy:delivery:9.9', 'delivery', '배송 정책', '9.9', '조항 본문')"""
        )
    app_conn.rollback()
    with pytest.raises(psycopg.errors.CheckViolation):
        app_conn.execute(
            """INSERT INTO policy_chunks
                   (evidence_id, document_slug, document_title, article, content,
                    embedding_model, embedding_dimensions)
               VALUES ('policy:delivery:9.9', 'delivery', '배송 정책', '9.9', '조항 본문',
                       '', 1536)"""
        )
    app_conn.rollback()
    with pytest.raises(psycopg.errors.CheckViolation):
        app_conn.execute(
            """INSERT INTO policy_chunks
                   (evidence_id, document_slug, document_title, article, content,
                    embedding_model, embedding_dimensions)
               VALUES ('policy:delivery:9.9', 'delivery', '배송 정책', '9.9', '조항 본문',
                       'stub', 0)"""
        )


def test_policy_chunks_have_a_vector_index(app_conn: psycopg.Connection[DictRow]) -> None:
    assert (
        _scalar(
            app_conn,
            "SELECT count(*) FROM pg_indexes WHERE indexname = 'policy_chunks_embedding_idx'",
        )
        == 1
    )


def test_policy_chunk_evidence_id_follows_the_contract(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """`contracts.policy_evidence_id()` 형식을 DB 가 강제한다."""
    good = policy_evidence_id(document_slug="delivery", article="3.1")
    app_conn.execute(
        """INSERT INTO policy_chunks (evidence_id, document_slug, document_title, article, content,
                                      embedding_model, embedding_dimensions)
           VALUES (%s, 'delivery', '배송 정책', '3.1', '조항 본문', 'stub', 1536)""",
        (good,),
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        app_conn.execute(
            """INSERT INTO policy_chunks
                   (evidence_id, document_slug, document_title, article, content,
                    embedding_model, embedding_dimensions)
               VALUES ('policy:wrong:9', 'refund', '환불 정책', '1.1', '조항 본문', 'stub', 1536)"""
        )


# ── 처리 기록 ────────────────────────────────────────────────────────────────


def _insert_inquiry(
    conn: psycopg.Connection[DictRow],
    *,
    status: str = InquiryStatus.ANSWERED.value,
    answer: str | None = "3영업일 내에 발송됩니다.",
    escalation_reason: str | None = None,
    intent_source: str = IntentSource.BOTH.value,
    latency_ms: int = 1234,
    input_tokens: int = 910,
    output_tokens: int = 128,
    embedding_tokens: int = 47,
) -> str:
    row = conn.execute(
        """INSERT INTO inquiries (order_no, content, intent_source, status, answer, claims,
                                  escalation_reason, latency_ms, input_tokens, output_tokens,
                                  embedding_tokens)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           RETURNING id""",
        (
            "ORD-20260202-0001",
            "주문한 상품 언제 오나요?",
            intent_source,
            status,
            answer,
            Jsonb([{"text": "3영업일 내에 발송됩니다.", "citation_ids": ["policy:delivery:3.1"]}]),
            escalation_reason,
            latency_ms,
            input_tokens,
            output_tokens,
            embedding_tokens,
        ),
    ).fetchone()
    assert row is not None
    return str(row["id"])


def _insert_attempt(
    conn: psycopg.Connection[DictRow],
    inquiry_id: str,
    *,
    verdict: str = "pass",
    reject_reasons: list[str] | None = None,
    l1_verdict: str | None = None,
    l1_reject_reasons: list[str] | None = None,
    l2_verdict: str | None = None,
    l2_reject_reasons: list[str] | None = None,
    claim_verdicts: Jsonb | None = None,
    evidence_contradictions: Jsonb | None = None,
) -> None:
    conn.execute(
        """INSERT INTO inquiry_attempts
               (inquiry_id, attempt_no, verdict, reject_reasons, draft, l1_verdict,
                l1_reject_reasons, l2_verdict, l2_reject_reasons, claim_verdicts,
                evidence_contradictions)
           VALUES (%s, 1, %s, %s, '{}'::jsonb, %s, %s, %s, %s, %s, %s)""",
        (
            inquiry_id,
            verdict,
            reject_reasons if reject_reasons is not None else [],
            l1_verdict,
            l1_reject_reasons,
            l2_verdict,
            l2_reject_reasons,
            claim_verdicts,
            evidence_contradictions,
        ),
    )


def test_processing_record_round_trip(app_conn: psycopg.Connection[DictRow]) -> None:
    """문의 1건 + 시도 2건 + 근거 스냅샷 2건 + SQL 실패 1건이 그대로 저장·복원된다."""
    inquiry_id = _insert_inquiry(app_conn)

    app_conn.execute(
        """INSERT INTO inquiry_attempts (inquiry_id, attempt_no, verdict, reject_reasons, draft)
           VALUES (%s, 1, 'reject', %s, %s), (%s, 2, 'pass', '{}', %s)""",
        (
            inquiry_id,
            [RejectReason.MISSING_CITATION.value, RejectReason.PII_DETECTED.value],
            Jsonb({"claims": [{"text": "지어낸 문장", "citation_ids": []}]}),
            inquiry_id,
            Jsonb(
                {
                    "claims": [
                        {
                            "text": "3영업일 내에 발송됩니다.",
                            "citation_ids": ["policy:delivery:3.1"],
                        }
                    ]
                }
            ),
        ),
    )

    policy_id = policy_evidence_id(document_slug="delivery", article="3.1")
    sql_id = sql_evidence_id(inquiry_id=inquiry_id, sequence=1)
    executed_sql = "SELECT status, shipped_at FROM orders WHERE order_no = 'ORD-20260202-0001'"
    result_rows = [{"status": "배송완료", "shipped_at": "2026-02-05T00:36:27+09:00"}]

    app_conn.execute(
        """INSERT INTO inquiry_evidence
               (inquiry_id, evidence_id, source, sequence, content, evidence_text,
                query_sql, result_rows)
           VALUES (%s, %s, 'policy', NULL, %s, %s, NULL, NULL)""",
        (
            inquiry_id,
            policy_id,
            "배송은 결제 후 3영업일 내 발송한다.",
            "제3조 1항 배송은 결제 후 3영업일 내 발송한다.",
        ),
    )
    app_conn.execute(
        """INSERT INTO inquiry_evidence
               (inquiry_id, evidence_id, source, sequence, content, evidence_text,
                query_sql, result_rows)
           VALUES (%s, %s, 'sql', 1, %s, %s, %s, %s)""",
        (inquiry_id, sql_id, "주문 1건 조회", str(result_rows), executed_sql, Jsonb(result_rows)),
    )
    app_conn.execute(
        """INSERT INTO inquiry_sql_failures (inquiry_id, attempt_no, failure_kind, query_sql, error)
           VALUES (%s, 1, 'guard_rejected', %s, %s)""",
        (inquiry_id, "DELETE FROM orders", "화이트리스트 밖 구문: DELETE"),
    )

    inquiry = app_conn.execute("SELECT * FROM inquiries WHERE id = %s", (inquiry_id,)).fetchone()
    assert inquiry is not None
    # 평가 지표의 원천 — 지연과 토큰(생성 계열 / 임베딩 분리).
    assert inquiry["latency_ms"] == 1234
    assert inquiry["input_tokens"] == 910
    assert inquiry["output_tokens"] == 128
    assert inquiry["embedding_tokens"] == 47
    assert inquiry["status"] == InquiryStatus.ANSWERED.value
    assert inquiry["claims"][0]["citation_ids"] == ["policy:delivery:3.1"]

    attempts = app_conn.execute(
        "SELECT * FROM inquiry_attempts WHERE inquiry_id = %s ORDER BY attempt_no", (inquiry_id,)
    ).fetchall()
    assert [a["verdict"] for a in attempts] == [Verdict.REJECT.value, Verdict.PASS.value]
    assert attempts[0]["reject_reasons"] == [
        RejectReason.MISSING_CITATION.value,
        RejectReason.PII_DETECTED.value,
    ]
    assert attempts[1]["reject_reasons"] == []

    evidence = app_conn.execute(
        "SELECT * FROM inquiry_evidence WHERE inquiry_id = %s ORDER BY source", (inquiry_id,)
    ).fetchall()
    by_source = {row["source"]: row for row in evidence}
    assert set(by_source) == {EvidenceSource.POLICY.value, EvidenceSource.SQL.value}
    # SQL 근거는 쿼리문과 결과 행 전체가 스냅샷으로 남는다.
    assert by_source["sql"]["query_sql"] == executed_sql
    assert by_source["sql"]["result_rows"] == result_rows
    assert by_source["sql"]["evidence_id"] == sql_id
    assert by_source["policy"]["query_sql"] is None
    assert by_source["policy"]["evidence_text"].startswith("제3조")

    failures = app_conn.execute(
        "SELECT * FROM inquiry_sql_failures WHERE inquiry_id = %s", (inquiry_id,)
    ).fetchall()
    assert len(failures) == 1
    assert failures[0]["query_sql"] == "DELETE FROM orders"
    assert "DELETE" in failures[0]["error"]


def test_deleting_an_inquiry_cascades_to_its_record(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    inquiry_id = _insert_inquiry(app_conn)
    app_conn.execute(
        """INSERT INTO inquiry_attempts (inquiry_id, attempt_no, verdict, draft)
           VALUES (%s, 1, 'pass', '{}'::jsonb)""",
        (inquiry_id,),
    )
    app_conn.execute("DELETE FROM inquiries WHERE id = %s", (inquiry_id,))
    assert (
        _scalar(
            app_conn,
            "SELECT count(*) FROM inquiry_attempts WHERE inquiry_id = %s",
            (inquiry_id,),
        )
        == 0
    )


@pytest.mark.parametrize("reason", [reason.value for reason in EscalationReason])
def test_every_escalation_reason_from_contracts_is_storable(
    app_conn: psycopg.Connection[DictRow], reason: str
) -> None:
    inquiry_id = _insert_inquiry(
        app_conn,
        status=InquiryStatus.ESCALATED.value,
        answer=None,
        escalation_reason=reason,
    )
    assert (
        _scalar(app_conn, "SELECT escalation_reason FROM inquiries WHERE id = %s", (inquiry_id,))
        == reason
    )


@pytest.mark.parametrize("intent", [intent.value for intent in IntentSource])
def test_every_intent_source_from_contracts_is_storable(
    app_conn: psycopg.Connection[DictRow], intent: str
) -> None:
    inquiry_id = _insert_inquiry(app_conn, intent_source=intent)
    assert (
        _scalar(app_conn, "SELECT intent_source FROM inquiries WHERE id = %s", (inquiry_id,))
        == intent
    )


@pytest.mark.parametrize("reason", [reason.value for reason in RejectReason])
def test_every_reject_reason_from_contracts_is_storable(
    app_conn: psycopg.Connection[DictRow], reason: str
) -> None:
    inquiry_id = _insert_inquiry(app_conn)
    app_conn.execute(
        """INSERT INTO inquiry_attempts (inquiry_id, attempt_no, verdict, reject_reasons, draft)
           VALUES (%s, 1, 'reject', %s, '{}'::jsonb)""",
        (inquiry_id, [reason]),
    )
    assert _scalar(
        app_conn,
        "SELECT reject_reasons FROM inquiry_attempts WHERE inquiry_id = %s",
        (inquiry_id,),
    ) == [reason]


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("status", "in_progress"),
        ("escalation_reason", "because_i_said_so"),
        ("intent_source", "everything"),
    ],
)
def test_unknown_enum_values_are_rejected(
    app_conn: psycopg.Connection[DictRow], column: str, value: str
) -> None:
    kwargs: dict[str, Any] = {}
    if column == "status":
        kwargs = {"status": value, "answer": None}
    elif column == "escalation_reason":
        kwargs = {
            "status": InquiryStatus.ESCALATED.value,
            "answer": None,
            "escalation_reason": value,
        }
    else:
        kwargs = {"intent_source": value}
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_inquiry(app_conn, **kwargs)


def test_unknown_reject_reason_is_rejected(app_conn: psycopg.Connection[DictRow]) -> None:
    inquiry_id = _insert_inquiry(app_conn)
    with pytest.raises(psycopg.errors.CheckViolation):
        app_conn.execute(
            """INSERT INTO inquiry_attempts (inquiry_id, attempt_no, verdict, reject_reasons, draft)
               VALUES (%s, 1, 'reject', ARRAY['made_up_reason'], '{}'::jsonb)""",
            (inquiry_id,),
        )


def test_layered_attempt_round_trip(app_conn: psycopg.Connection[DictRow]) -> None:
    """L1 pass + L2 reject 를 층별 컬럼과 함께 저장하면 종합 사유는 두 층의 병합이다."""
    inquiry_id = _insert_inquiry(app_conn)
    _insert_attempt(
        app_conn,
        inquiry_id,
        verdict="reject",
        reject_reasons=["unsupported_claim"],
        l1_verdict="pass",
        l1_reject_reasons=[],
        l2_verdict="reject",
        l2_reject_reasons=["unsupported_claim"],
        claim_verdicts=Jsonb([{"claim_index": 0, "verdict": "unsupported"}]),
        evidence_contradictions=Jsonb([]),
    )
    row = app_conn.execute(
        "SELECT * FROM inquiry_attempts WHERE inquiry_id = %s", (inquiry_id,)
    ).fetchone()
    assert row is not None
    assert row["l1_verdict"] == "pass"
    assert row["l2_reject_reasons"] == ["unsupported_claim"]
    assert row["claim_verdicts"] == [{"claim_index": 0, "verdict": "unsupported"}]


def test_attempt_without_layer_columns_leaves_them_null(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """층별 컬럼을 모르는 적재 코드의 INSERT 는 그대로 통과하고 전부 NULL 로 남는다."""
    inquiry_id = _insert_inquiry(app_conn)
    app_conn.execute(
        """INSERT INTO inquiry_attempts (inquiry_id, attempt_no, verdict, draft)
           VALUES (%s, 1, 'pass', '{}'::jsonb)""",
        (inquiry_id,),
    )
    row = app_conn.execute(
        "SELECT * FROM inquiry_attempts WHERE inquiry_id = %s", (inquiry_id,)
    ).fetchone()
    assert row is not None
    for column in (
        "l1_verdict",
        "l1_reject_reasons",
        "l2_verdict",
        "l2_reject_reasons",
        "claim_verdicts",
        "evidence_contradictions",
    ):
        assert row[column] is None


@pytest.mark.parametrize(
    "attempt_kwargs",
    [
        pytest.param(
            {
                "verdict": "reject",
                "reject_reasons": ["pii_detected"],
                "l1_verdict": "reject",
                "l1_reject_reasons": ["pii_detected"],
                "l2_verdict": "pass",
                "l2_reject_reasons": [],
            },
            id="l1-reject-with-l2-verdict",
        ),
        pytest.param(
            {"verdict": "pass", "l2_verdict": "pass", "l2_reject_reasons": []},
            id="l2-verdict-without-l1-verdict",
        ),
        pytest.param(
            {
                "verdict": "pass",
                "l1_verdict": "pass",
                "l1_reject_reasons": [],
                "claim_verdicts": Jsonb([]),
            },
            id="l2-null-with-claim-verdicts",
        ),
        pytest.param(
            {
                "verdict": "reject",
                "reject_reasons": ["unsupported_claim"],
                "l1_verdict": "pass",
                "l1_reject_reasons": [],
                "l2_reject_reasons": ["unsupported_claim"],
            },
            id="l2-null-with-l2-reasons",
        ),
        pytest.param(
            {
                "verdict": "pass",
                "l1_verdict": "reject",
                "l1_reject_reasons": ["pii_detected"],
            },
            id="overall-pass-with-l1-reject",
        ),
        pytest.param(
            {
                "verdict": "pass",
                "l1_verdict": "pass",
                "l1_reject_reasons": ["pii_detected"],
            },
            id="l1-pass-with-nonzero-reasons",
        ),
        pytest.param(
            {"verdict": "pass", "l1_verdict": "pass"},
            id="l1-verdict-without-reasons-array",
        ),
        pytest.param(
            {
                "verdict": "reject",
                "reject_reasons": ["unsupported_claim"],
                "l1_verdict": "reject",
                "l1_reject_reasons": ["unsupported_claim"],
            },
            id="l2-reason-inside-l1-array",
        ),
        pytest.param(
            {
                "verdict": "reject",
                "reject_reasons": ["pii_detected"],
                "l1_verdict": "pass",
                "l1_reject_reasons": [],
                "l2_verdict": "reject",
                "l2_reject_reasons": ["unsupported_claim"],
            },
            id="overall-reasons-not-layer-concat",
        ),
    ],
)
def test_contradictory_layer_states_are_rejected(
    app_conn: psycopg.Connection[DictRow], attempt_kwargs: dict[str, Any]
) -> None:
    """층별 판정의 모순 상태는 DB CHECK 가 막는다 (NULL 스코프 포함)."""
    inquiry_id = _insert_inquiry(app_conn)
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_attempt(app_conn, inquiry_id, **attempt_kwargs)


def test_attempt_count_is_capped_at_two(app_conn: psycopg.Connection[DictRow]) -> None:
    """시도 기록은 최대 2건 (초안 + 재생성 1회) — DB 도 상한을 강제한다."""
    inquiry_id = _insert_inquiry(app_conn)
    with pytest.raises(psycopg.errors.CheckViolation):
        app_conn.execute(
            """INSERT INTO inquiry_attempts (inquiry_id, attempt_no, verdict, draft)
               VALUES (%s, 3, 'pass', '{}'::jsonb)""",
            (inquiry_id,),
        )


def test_escalated_inquiry_must_carry_a_reason(app_conn: psycopg.Connection[DictRow]) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_inquiry(app_conn, status=InquiryStatus.ESCALATED.value, answer=None)


def test_answered_inquiry_must_carry_an_answer(app_conn: psycopg.Connection[DictRow]) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_inquiry(app_conn, status=InquiryStatus.ANSWERED.value, answer=None)


def test_sql_evidence_without_a_snapshot_is_rejected(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """근거로 채택된 SQL 은 쿼리문과 결과 행이 반드시 함께 남아야 한다."""
    inquiry_id = _insert_inquiry(app_conn)
    with pytest.raises(psycopg.errors.CheckViolation):
        app_conn.execute(
            """INSERT INTO inquiry_evidence
                   (inquiry_id, evidence_id, source, sequence, content, evidence_text)
               VALUES (%s, %s, 'sql', 1, '요약', '원문')""",
            (inquiry_id, sql_evidence_id(inquiry_id=inquiry_id, sequence=1)),
        )


def test_inquiry_accepts_an_order_no_that_does_not_exist(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """`order_not_found` 인계 경로가 처리 기록에 남아야 하므로 FK 를 두지 않는다."""
    row = app_conn.execute(
        """INSERT INTO inquiries (order_no, content, status, escalation_reason, latency_ms)
           VALUES ('ORD-19990101-0001', '없는 주문 문의', 'escalated', 'order_not_found', 42)
           RETURNING order_no"""
    ).fetchone()
    assert row is not None
    assert row["order_no"] == "ORD-19990101-0001"
