"""사이클 간 계약 — 답변 계약, 근거 ID 체계, 판정·상태 enum.

이 모듈은 docs/contracts.md "답변 계약" 과 docs/business-rules.md 의
"L1 게이트 판정 규칙"·"상태 전이" 를 코드로 옮긴 것이다.
**L2(claim 단위 LLM judge)가 그대로 이어받는 계약**이므로, 편의로 구조를 바꾸지
않는다(바꾸려면 사용자 승인).

여러 모듈이 공유하는 정의만 둔다 — 초안 생성·게이트·근거 수집의 구현 세부는 각 모듈 소유다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "COMBINED_REASON_ORDER",
    "DRAFT_JSON_SCHEMA",
    "Claim",
    "ClaimJudgment",
    "Draft",
    "EscalationReason",
    "Evidence",
    "EvidenceContradiction",
    "EvidenceSource",
    "GateResult",
    "InquiryStatus",
    "IntentSource",
    "JudgeResult",
    "RejectReason",
    "Verdict",
    "policy_evidence_id",
    "sql_evidence_id",
]


class IntentSource(StrEnum):
    """의도 해석이 분류하는 필요 근거 소스."""

    POLICY = "policy"
    ORDER = "order"
    BOTH = "both"


class EvidenceSource(StrEnum):
    """근거의 출처 종류."""

    POLICY = "policy"
    SQL = "sql"


@dataclass(frozen=True)
class Evidence:
    """이번 문의에서 실제로 수집된 근거 1건.

    `content` 는 API 응답과 화면에 그대로 실리는 표시용 요약이고,
    `evidence_text` 는 PII allowlist 대조에 쓰는 **원문 전체**다(정책 청크 텍스트,
    SQL 결과 스냅샷 전문). 정책 근거에서는 보통 두 값이 같지만, SQL 근거에서는
    표시용 요약만으로 대조하면 근거에 있는 값을 놓쳐 오탐이 난다.
    """

    id: str
    source: EvidenceSource
    content: str
    evidence_text: str


def policy_evidence_id(*, document_slug: str, article: str) -> str:
    """정책 조항 청크의 안정 식별자: `policy:<문서 slug>:<조항 번호>`."""
    return f"policy:{document_slug}:{article}"


def sql_evidence_id(*, inquiry_id: str, sequence: int) -> str:
    """채택된 SQL 조회 1건의 식별자: `sql:<문의 ID>:<실행 순번>`.

    순번은 **채택된 쿼리에만** 매긴다 — 안전장치에 거부되거나 실행에 실패한 쿼리,
    그리고 존재성 선검사 쿼리는 근거가 아니므로 ID 를 받지 않는다.
    """
    return f"sql:{inquiry_id}:{sequence}"


@dataclass(frozen=True)
class Claim:
    """답변 문장 1개와 그것이 딛고 선 근거 ID 목록."""

    text: str
    citation_ids: tuple[str, ...]


@dataclass(frozen=True)
class Draft:
    """L1 스키마 검사를 통과한 초안."""

    claims: tuple[Claim, ...]

    @property
    def answer_text(self) -> str:
        """최종 사용자에게 보여줄 답변 — claims 의 text 를 순서대로 이은 것."""
        return " ".join(claim.text for claim in self.claims)


class RejectReason(StrEnum):
    """기각 사유 — L1 4종 + L2 2종이 전부다.

    L1 4종은 docs/business-rules.md "L1 게이트 판정 규칙" 의 enum 그대로이고 L1 의 전부다.
    L2 2종은 claim 단위 의미 검증의 전부다:

    * `unsupported_claim` — 어떤 claim 이 인용한 근거로 뒷받침되지 않는다.
    * `contradictory_evidence` — 수집 근거에 서로 모순되는 쌍이 있고, 초안이 그 모순을
      명시하지 않은 채 해당 근거를 딛고 답한다.
    """

    # L1 — 기계 검사 (gate.py)
    SCHEMA_VIOLATION = "schema_violation"
    MISSING_CITATION = "missing_citation"
    INVALID_CITATION = "invalid_citation"
    PII_DETECTED = "pii_detected"
    # L2 — claim 단위 의미 검증
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"


#: 종합 사유(= L1 사유와 L2 사유의 합집합)의 고정 순서 — L1 4종 먼저, L2 2종 뒤. 각 층 안 순서도
#: 고정이다. 한 시도 안에서 두 층 사유는 공존하지 않지만(L2 는 L1 pass 시에만 실행),
#: 시도 간 평탄화·집계가 결정론이려면 교차층 순서까지 계약이어야 한다.
#:
#: L1 접두 4종은 `gate.REASON_ORDER` 와 일치해야 한다 — contracts 는 gate 를 import 할 수
#: 없으므로(순환) 정합은 테스트가 고정한다(tests/test_contracts.py).
COMBINED_REASON_ORDER: tuple[RejectReason, ...] = (
    RejectReason.SCHEMA_VIOLATION,
    RejectReason.MISSING_CITATION,
    RejectReason.INVALID_CITATION,
    RejectReason.PII_DETECTED,
    RejectReason.UNSUPPORTED_CLAIM,
    RejectReason.CONTRADICTORY_EVIDENCE,
)


class Verdict(StrEnum):
    """판정값. 시도(attempt) 1건의 L1·L2 판정과 L2 의 claim 단위 판정이 공유한다."""

    PASS = "pass"
    REJECT = "reject"


@dataclass(frozen=True)
class GateResult:
    """L1 판정 결과. 기각 사유는 하나만 잡고 멈추지 않고 전부 수집한다."""

    verdict: Verdict
    reject_reasons: tuple[RejectReason, ...] = ()


@dataclass(frozen=True)
class ClaimJudgment:
    """claim 1개에 대한 L2 판정.

    `claim_text` 는 판정 대상 claim 의 text 참조다 — 답변 계약의 claim 은 별도 ID 가 없으므로
    text 로 어느 claim 인지 가리킨다. `explanation` 은 판정 사유 설명(한국어 산문)이다.
    """

    claim_text: str
    verdict: Verdict
    explanation: str


@dataclass(frozen=True)
class EvidenceContradiction:
    """수집 근거 중 서로 모순되는 근거쌍 1건.

    모순은 **근거쌍 단위**로 기록한다 — 특정 claim 에 귀속되지 않을 수 있으므로
    claim 판정 배열(`JudgeResult.claim_judgments`)과 별도다.
    """

    evidence_id_a: str
    evidence_id_b: str
    explanation: str


@dataclass(frozen=True)
class JudgeResult:
    """L2 판정 결과 — `GateResult` 와 같은 골격(verdict + 사유 목록)에 L2 세부를 더한다.

    `reject_reasons` 에는 L2 2종(`unsupported_claim`·`contradictory_evidence`)만 실린다.
    `claim_judgments` 는 claim 단위 판정 전부(통과한 claim 포함)이고, `contradictions` 는
    근거쌍 단위 모순 기록이다.
    """

    verdict: Verdict
    reject_reasons: tuple[RejectReason, ...] = ()
    claim_judgments: tuple[ClaimJudgment, ...] = ()
    contradictions: tuple[EvidenceContradiction, ...] = ()


class EscalationReason(StrEnum):
    """상담원 인계 사유. docs/business-rules.md "인계 사유 6종" 이 전부다."""

    NO_EVIDENCE = "no_evidence"
    MISSING_ORDER_REF = "missing_order_ref"
    ORDER_NOT_FOUND = "order_not_found"
    SQL_FAILED = "sql_failed"
    LLM_CALL_FAILED = "llm_call_failed"
    REJECTED_TWICE = "rejected_twice"


class InquiryStatus(StrEnum):
    """문의의 최종 상태 (종결 상태, 역방향 전이 없음)."""

    ANSWERED = "answered"
    ESCALATED = "escalated"


#: 초안 생성 LLM 의 구조화 출력 스키마 = L1 기계 검사의 대상.
#:
#: **`citation_ids` 에 `minItems` 를 넣지 않는다.** "citation_ids 1개 이상"의 위반 판정은
#: L1 `missing_citation` 이 전담한다(docs/business-rules.md "L1 게이트 판정 규칙").
#: 생성 측에서 최소 개수를 강제하면
#: missing_citation 이 영원히 발화하지 않아 사유 분리가 무너진다.
DRAFT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citation_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "citation_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}
