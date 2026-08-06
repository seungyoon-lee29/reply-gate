"""사이클 간 계약이 docs/contracts.md "답변 계약" 대로 유지되는지 지키는 테스트."""

from __future__ import annotations

import ast
import pathlib

from reply_gate import contracts, gate
from reply_gate.contracts import (
    COMBINED_REASON_ORDER,
    DRAFT_JSON_SCHEMA,
    Claim,
    ClaimJudgment,
    Draft,
    EscalationReason,
    EvidenceContradiction,
    JudgeResult,
    RejectReason,
    Verdict,
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


def test_기각사유와_인계사유_집합이_문서_계약과_일치한다() -> None:
    assert {reason.value for reason in RejectReason} == {
        # L1 — docs/business-rules.md "L1 게이트 판정 규칙" 의 4종이 전부다.
        "schema_violation",
        "missing_citation",
        "invalid_citation",
        "pii_detected",
        # L2 — claim 단위 의미 검증의 2종이 전부다.
        "unsupported_claim",
        "contradictory_evidence",
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


# ── 종합 사유 고정 순서 ──────────────────────────────────────────────────────


def test_종합_사유_순서는_L1_4종_먼저_L2_2종_뒤다() -> None:
    """시도 간 평탄화·집계가 결정론이려면 교차층 순서가 계약이어야 한다."""
    assert COMBINED_REASON_ORDER == (
        RejectReason.SCHEMA_VIOLATION,
        RejectReason.MISSING_CITATION,
        RejectReason.INVALID_CITATION,
        RejectReason.PII_DETECTED,
        RejectReason.UNSUPPORTED_CLAIM,
        RejectReason.CONTRADICTORY_EVIDENCE,
    )


def test_종합_사유_순서는_기각사유_전체를_빠짐없이_한_번씩_담는다() -> None:
    """사유가 늘면 이 상수도 같이 늘어야 한다 — 누락되면 집계에서 그 사유가 증발한다."""
    assert len(COMBINED_REASON_ORDER) == len(set(COMBINED_REASON_ORDER))
    assert set(COMBINED_REASON_ORDER) == set(RejectReason)


def test_gate_REASON_ORDER_는_종합_순서의_L1_접두와_일치한다() -> None:
    """L1 층 내부 순서의 정의는 gate.REASON_ORDER 하나뿐이다 — 종합 순서가 어긋나면
    같은 시도의 사유 목록이 리포트 단면마다 다른 순서로 나온다."""
    assert COMBINED_REASON_ORDER[: len(gate.REASON_ORDER)] == gate.REASON_ORDER


def test_contracts_는_gate_를_import_하지_않는다() -> None:
    """의존 방향은 gate → contracts 뿐이다. 역방향은 순환 import 다 — 구조로 못박는다."""
    source = pathlib.Path(str(contracts.__file__)).read_text(encoding="utf-8")
    banned_modules = {"reply_gate.gate", "gate"}  # 절대·상대 경로 둘 다
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            assert all(alias.name not in banned_modules for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module not in banned_modules
            if node.level > 0 or module == "reply_gate":
                assert all(alias.name != "gate" for alias in node.names)


# ── L2 판정 자료형 ──────────────────────────────────────────────────────────


def test_L2_판정_결과는_claim_판정과_모순쌍을_함께_담는다() -> None:
    judgment = ClaimJudgment(
        claim_text="환불은 수령 후 30일 이내에 가능합니다.",
        verdict=Verdict.REJECT,
        explanation="인용한 근거 어디에도 30일 기한이 없다.",
    )
    contradiction = EvidenceContradiction(
        evidence_id_a="policy:refund:1",
        evidence_id_b="policy:refund:9",
        explanation="1조는 7일, 9조는 14일을 말한다.",
    )
    result = JudgeResult(
        verdict=Verdict.REJECT,
        reject_reasons=(RejectReason.UNSUPPORTED_CLAIM, RejectReason.CONTRADICTORY_EVIDENCE),
        claim_judgments=(judgment,),
        contradictions=(contradiction,),
    )

    assert result.claim_judgments[0].verdict is Verdict.REJECT
    assert result.contradictions[0].evidence_id_a == "policy:refund:1"
    assert result.contradictions[0].evidence_id_b == "policy:refund:9"


def test_모순은_claim_판정과_별도의_근거쌍_단위_기록이다() -> None:
    """모순이 특정 claim 에 귀속되지 않을 수 있으므로, claim 판정 배열이 비어 있어도
    모순쌍만으로 기각을 기록할 수 있어야 한다."""
    result = JudgeResult(
        verdict=Verdict.REJECT,
        reject_reasons=(RejectReason.CONTRADICTORY_EVIDENCE,),
        contradictions=(
            EvidenceContradiction(
                evidence_id_a="policy:shipping:2",
                evidence_id_b="sql:inq_1:1",
                explanation="정책은 발송 전 취소 가능이라 하고, 주문 조회 결과는 이미 발송됨이다.",
            ),
        ),
    )

    assert result.claim_judgments == ()
    assert len(result.contradictions) == 1


def test_L2_통과_결과는_사유와_세부가_기본값으로_비어_있다() -> None:
    result = JudgeResult(verdict=Verdict.PASS)

    assert result.reject_reasons == ()
    assert result.claim_judgments == ()
    assert result.contradictions == ()


def test_L2_판정_자료형은_불변이다() -> None:
    """계약 자료형은 전부 frozen dataclass 다 — eq 만 있고 frozen 이 아니면 해시 불가다."""
    judgment = ClaimJudgment(claim_text="t", verdict=Verdict.PASS, explanation="")
    contradiction = EvidenceContradiction(evidence_id_a="a", evidence_id_b="b", explanation="")
    result = JudgeResult(verdict=Verdict.PASS)

    assert isinstance(hash(judgment), int)
    assert isinstance(hash(contradiction), int)
    assert isinstance(hash(result), int)
