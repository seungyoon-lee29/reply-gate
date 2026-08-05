"""사이클 간 계약이 docs/contracts.md "답변 계약" 대로 유지되는지 지키는 테스트."""

from __future__ import annotations

from reply_gate.contracts import (
    DRAFT_JSON_SCHEMA,
    Claim,
    Draft,
    EscalationReason,
    RejectReason,
    policy_evidence_id,
    sql_evidence_id,
)


def test_초안_스키마는_citation_최소개수를_강제하지_않는다() -> None:
    """강제하면 missing_citation 이 영원히 발화하지 않아 L1 사유 분리가 무너진다."""
    citation_ids = DRAFT_JSON_SCHEMA["properties"]["claims"]["items"]["properties"]["citation_ids"]

    assert "minItems" not in citation_ids
    assert DRAFT_JSON_SCHEMA["properties"]["claims"].get("minItems") is None


def test_근거_ID_형식() -> None:
    assert policy_evidence_id(document_slug="refund", article="3-2") == "policy:refund:3-2"
    assert sql_evidence_id(inquiry_id="inq_1", sequence=1) == "sql:inq_1:1"


def test_기각사유와_인계사유_집합이_스펙과_일치한다() -> None:
    assert {reason.value for reason in RejectReason} == {
        "schema_violation",
        "missing_citation",
        "invalid_citation",
        "pii_detected",
    }
    assert {reason.value for reason in EscalationReason} == {
        "no_evidence",
        "missing_order_ref",
        "order_not_found",
        "sql_failed",
        "llm_call_failed",
        "rejected_twice",
    }


def test_확정_답변은_claim_순서대로_이어붙인다() -> None:
    draft = Draft(
        claims=(
            Claim(text="환불은 수령 후 7일 이내에 가능합니다.", citation_ids=("policy:refund:1",)),
            Claim(
                text="주문 20240101-0001 은 배송 완료 상태입니다.", citation_ids=("sql:inq_1:1",)
            ),
        )
    )

    assert draft.answer_text.startswith("환불은 수령 후 7일 이내에 가능합니다.")
    assert draft.answer_text.endswith("배송 완료 상태입니다.")
