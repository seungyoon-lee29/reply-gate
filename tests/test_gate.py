"""L1 게이트 테스트 — spec "L1 게이트 검사 규칙" 을 코드로 고정한다.

게이트는 제품 정체성("근거 없는 답변을 스스로 기각한다")이 실제로 구현된 지점이므로,
사유 4종의 양성·음성, PII allowlist, 복수 사유 동시 수집, 결정론, LLM 호출 0회를
모두 테스트로 못박는다.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Sequence
from typing import Any

import pytest

from reply_gate import gate
from reply_gate.contracts import (
    Claim,
    Draft,
    Evidence,
    EvidenceSource,
    RejectReason,
    Verdict,
)
from reply_gate.gate import evaluate_draft, to_draft

POLICY_ID = "policy:refund:3-2"
SQL_ID = "sql:inq_1:1"
NORMAL_TEXT = "환불은 수령 후 7일 이내에 가능합니다."


def _evidence(
    evidence_id: str, text: str, *, source: EvidenceSource = EvidenceSource.POLICY
) -> Evidence:
    """근거 1건. 대조 대상인 `evidence_text` 를 명시적으로 채운다."""
    return Evidence(id=evidence_id, source=source, content=text, evidence_text=text)


def _evidences() -> list[Evidence]:
    return [
        _evidence(POLICY_ID, NORMAL_TEXT),
        _evidence(SQL_ID, "주문 20250101-0001 은 배송 완료 상태입니다.", source=EvidenceSource.SQL),
    ]


def _draft(
    *, text: str = NORMAL_TEXT, citation_ids: Sequence[str] = (POLICY_ID,)
) -> dict[str, Any]:
    return {"claims": [{"text": text, "citation_ids": list(citation_ids)}]}


# ── 통과 기준선 ──────────────────────────────────────────────────────────────


def test_근거를_갖춘_정상_초안은_통과한다() -> None:
    result = evaluate_draft(raw_draft=_draft(), evidences=_evidences())

    assert result.verdict is Verdict.PASS
    assert result.reject_reasons == ()


# ── schema_violation ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw_draft",
    [
        pytest.param({}, id="필수_키_누락"),
        pytest.param({"claims": "환불은 7일 이내 가능합니다"}, id="claims_가_리스트가_아님"),
        pytest.param({"claims": []}, id="claims_가_비어_있음"),
        pytest.param({"claims": [NORMAL_TEXT]}, id="claim_이_dict_가_아님"),
        pytest.param(
            {"claims": [{"text": 42, "citation_ids": [POLICY_ID]}]}, id="text_가_문자열이_아님"
        ),
        pytest.param({"claims": [{"citation_ids": [POLICY_ID]}]}, id="text_키_누락"),
        pytest.param(
            {"claims": [{"text": NORMAL_TEXT, "citation_ids": POLICY_ID}]},
            id="citation_ids_가_리스트가_아님",
        ),
        pytest.param({"claims": [{"text": NORMAL_TEXT}]}, id="citation_ids_키_누락"),
        pytest.param(
            {"claims": [{"text": NORMAL_TEXT, "citation_ids": [7]}]},
            id="citation_id_가_문자열이_아님",
        ),
        pytest.param("환불은 수령 후 7일 이내에 가능합니다.", id="초안_자체가_문자열"),
        pytest.param(None, id="초안_자체가_None"),
        pytest.param([{"text": NORMAL_TEXT, "citation_ids": [POLICY_ID]}], id="초안_자체가_리스트"),
    ],
)
def test_구조가_깨진_초안은_schema_violation_으로_기각된다(raw_draft: object) -> None:
    result = evaluate_draft(raw_draft=raw_draft, evidences=_evidences())

    assert result.verdict is Verdict.REJECT
    assert RejectReason.SCHEMA_VIOLATION in result.reject_reasons


def test_구조가_온전하면_schema_violation_이_붙지_않는다() -> None:
    """음성 케이스 — citation 이 비어 있어도 '구조'는 온전하다."""
    raw_draft = {"claims": [{"text": NORMAL_TEXT, "citation_ids": []}]}

    result = evaluate_draft(raw_draft=raw_draft, evidences=_evidences())

    assert RejectReason.SCHEMA_VIOLATION not in result.reject_reasons


def test_형식_불일치_원문_문자열도_예외를_던지지_않는다() -> None:
    """초안 생성은 형식 불일치 시 원문 문자열을 그대로 넘긴다 — 게이트가 삼켜야 한다."""
    result = evaluate_draft(raw_draft="죄송합니다, JSON 을 못 만들었습니다", evidences=[])

    assert result.verdict is Verdict.REJECT
    assert result.reject_reasons == (RejectReason.SCHEMA_VIOLATION,)


# ── missing_citation ────────────────────────────────────────────────────────


def test_citation_ids_가_빈_claim_은_missing_citation_이다() -> None:
    raw_draft = {"claims": [{"text": NORMAL_TEXT, "citation_ids": []}]}

    result = evaluate_draft(raw_draft=raw_draft, evidences=_evidences())

    assert result.verdict is Verdict.REJECT
    assert RejectReason.MISSING_CITATION in result.reject_reasons


def test_모든_claim_에_citation_이_있으면_missing_citation_이_아니다() -> None:
    raw_draft = {
        "claims": [
            {"text": NORMAL_TEXT, "citation_ids": [POLICY_ID]},
            {"text": "주문은 배송 완료 상태입니다.", "citation_ids": [SQL_ID]},
        ]
    }

    result = evaluate_draft(raw_draft=raw_draft, evidences=_evidences())

    assert result.verdict is Verdict.PASS
    assert RejectReason.MISSING_CITATION not in result.reject_reasons


def test_회귀_스키마검사에_citation_최소개수를_넣으면_missing_citation_이_죽는다() -> None:
    """`citation_ids: []` 의 유일한 사유는 missing_citation 이어야 한다.

    스키마 검사기에 minItems 성격의 제약이 들어가면 schema_violation 이 함께 붙어
    이 단언이 깨진다 — 사유 분리가 무너졌다는 신호다(spec L1 절 각주).
    """
    raw_draft = {"claims": [{"text": NORMAL_TEXT, "citation_ids": []}]}

    result = evaluate_draft(raw_draft=raw_draft, evidences=_evidences())

    assert result.reject_reasons == (RejectReason.MISSING_CITATION,)


# ── invalid_citation ────────────────────────────────────────────────────────


def test_수집되지_않은_근거_ID_는_invalid_citation_이다() -> None:
    raw_draft = _draft(citation_ids=("policy:refund:9-9",))

    result = evaluate_draft(raw_draft=raw_draft, evidences=_evidences())

    assert result.verdict is Verdict.REJECT
    assert result.reject_reasons == (RejectReason.INVALID_CITATION,)


def test_수집된_ID_만_참조하면_invalid_citation_이_아니다() -> None:
    raw_draft = _draft(citation_ids=(POLICY_ID, SQL_ID))

    result = evaluate_draft(raw_draft=raw_draft, evidences=_evidences())

    assert result.verdict is Verdict.PASS
    assert RejectReason.INVALID_CITATION not in result.reject_reasons


def test_근거가_하나도_없으면_모든_참조가_invalid_citation_이다() -> None:
    result = evaluate_draft(raw_draft=_draft(), evidences=[])

    assert result.reject_reasons == (RejectReason.INVALID_CITATION,)


def test_missing_citation_과_invalid_citation_은_동시에_수집된다() -> None:
    raw_draft = {
        "claims": [
            {"text": NORMAL_TEXT, "citation_ids": []},
            {"text": "주문은 배송 완료 상태입니다.", "citation_ids": ["sql:inq_9:7"]},
        ]
    }

    result = evaluate_draft(raw_draft=raw_draft, evidences=_evidences())

    assert result.reject_reasons == (
        RejectReason.MISSING_CITATION,
        RejectReason.INVALID_CITATION,
    )


# ── pii_detected ────────────────────────────────────────────────────────────


def test_지어낸_전화번호는_pii_detected_로_기각된다() -> None:
    evidences = [_evidence(POLICY_ID, "고객센터 운영 시간은 평일 09시부터 18시까지입니다.")]
    raw_draft = _draft(text="담당자 010-9999-8888 로 연락 주세요.")

    result = evaluate_draft(raw_draft=raw_draft, evidences=evidences)

    assert result.verdict is Verdict.REJECT
    assert RejectReason.PII_DETECTED in result.reject_reasons


def test_근거에_있는_번호는_표기가_달라도_통과한다() -> None:
    """정규화 후 대조 — 근거 `010-1234-5678`, 초안 `01012345678`."""
    evidences = [
        _evidence(SQL_ID, "수령인 연락처: 010-1234-5678", source=EvidenceSource.SQL),
    ]
    raw_draft = _draft(text="등록된 연락처는 01012345678 입니다.", citation_ids=(SQL_ID,))

    result = evaluate_draft(raw_draft=raw_draft, evidences=evidences)

    assert result.verdict is Verdict.PASS
    assert RejectReason.PII_DETECTED not in result.reject_reasons


def test_근거에_있는_이메일은_대문자로_에코해도_통과한다() -> None:
    evidences = [
        _evidence(SQL_ID, '{"email": "hong@example.com"}', source=EvidenceSource.SQL),
    ]
    raw_draft = _draft(text="확인 메일을 HONG@Example.COM 으로 보냈습니다.", citation_ids=(SQL_ID,))

    result = evaluate_draft(raw_draft=raw_draft, evidences=evidences)

    assert result.verdict is Verdict.PASS


def test_다른_고객의_연락처는_기각된다() -> None:
    evidences = [
        _evidence(SQL_ID, "수령인 연락처: 010-1111-2222", source=EvidenceSource.SQL),
    ]
    raw_draft = _draft(text="연락처 010-3333-4444 로 안내드렸습니다.", citation_ids=(SQL_ID,))

    result = evaluate_draft(raw_draft=raw_draft, evidences=evidences)

    assert RejectReason.PII_DETECTED in result.reject_reasons


def test_근거에_없는_주민등록번호는_기각되고_있는_값은_통과한다() -> None:
    evidences = [
        _evidence(SQL_ID, "본인확인 기록: 900101-1234567", source=EvidenceSource.SQL),
    ]

    echoed = evaluate_draft(
        raw_draft=_draft(text="등록된 번호는 9001011234567 입니다.", citation_ids=(SQL_ID,)),
        evidences=evidences,
    )
    fabricated = evaluate_draft(
        raw_draft=_draft(text="등록된 번호는 900101-2345678 입니다.", citation_ids=(SQL_ID,)),
        evidences=evidences,
    )

    assert echoed.verdict is Verdict.PASS
    assert RejectReason.PII_DETECTED in fabricated.reject_reasons


def test_PII_대조는_표시용_content_가_아니라_evidence_text_를_쓴다() -> None:
    """SQL 근거의 `content` 는 요약이라, 그것으로 대조하면 정상 에코가 오탐된다."""
    evidences = [
        Evidence(
            id=SQL_ID,
            source=EvidenceSource.SQL,
            content="주문 1건 조회됨",
            evidence_text='{"order_id": "20250101-0001", "phone": "010-1234-5678"}',
        )
    ]
    raw_draft = _draft(text="등록된 연락처는 010-1234-5678 입니다.", citation_ids=(SQL_ID,))

    result = evaluate_draft(raw_draft=raw_draft, evidences=evidences)

    assert result.verdict is Verdict.PASS


def test_근거의_긴_숫자에_우연히_포함된_번호는_통과시키지_않는다() -> None:
    """오탐 방지의 핵심.

    정규화한 숫자열을 근거 텍스트에 '부분 문자열 포함'으로 대조하면,
    근거의 긴 주문번호(9901012345678123) 안에 초안의 전화번호(01012345678)가
    우연히 들어 있어 통과해 버린다. 게이트는 패턴으로 뽑은 값끼리 **완전 일치**로
    비교하므로 이 케이스를 기각해야 한다.
    """
    evidences = [
        _evidence(
            SQL_ID, "주문번호 9901012345678123 로 접수되었습니다.", source=EvidenceSource.SQL
        ),
    ]
    raw_draft = _draft(text="연락처 010-1234-5678 로 안내드렸습니다.", citation_ids=(SQL_ID,))

    result = evaluate_draft(raw_draft=raw_draft, evidences=evidences)

    assert RejectReason.PII_DETECTED in result.reject_reasons


def test_주문번호_같은_긴_숫자는_PII_로_오탐하지_않는다() -> None:
    evidences = [_evidence(SQL_ID, "주문 상태 조회 결과", source=EvidenceSource.SQL)]
    raw_draft = _draft(
        text="주문번호 9901012345678123 은 배송 완료 상태입니다.", citation_ids=(SQL_ID,)
    )

    result = evaluate_draft(raw_draft=raw_draft, evidences=evidences)

    assert result.verdict is Verdict.PASS


def test_비패턴형_개인정보는_L1_검사대상이_아니다() -> None:
    """이름·주소는 정규식으로 잡을 수 없다 — L2 claim 단위 대조로 이월한다.

    한계를 코드로 고정하는 테스트다. 여기서 기각하도록 '개선'하면 안 된다.
    """
    evidences = [_evidence(SQL_ID, "주문 상태 조회 결과", source=EvidenceSource.SQL)]
    raw_draft = _draft(
        text="수령인 홍길동 님, 서울시 강남구 테헤란로 123 으로 발송되었습니다.",
        citation_ids=(SQL_ID,),
    )

    result = evaluate_draft(raw_draft=raw_draft, evidences=evidences)

    assert result.verdict is Verdict.PASS
    assert RejectReason.PII_DETECTED not in result.reject_reasons


def test_PII_검사는_스키마가_깨져도_텍스트만_있으면_수행한다() -> None:
    result = evaluate_draft(raw_draft="연락처는 010-9999-8888 입니다", evidences=[])

    assert result.reject_reasons == (
        RejectReason.SCHEMA_VIOLATION,
        RejectReason.PII_DETECTED,
    )


def test_PII_패턴은_호출자가_바꿔_넣을_수_있다() -> None:
    """패턴 집합은 '조정 가능 기본값' 이다."""
    evidences = [_evidence(POLICY_ID, "고객센터 운영 시간 안내")]
    raw_draft = _draft(text="담당자 010-9999-8888 로 연락 주세요.")

    without_patterns = evaluate_draft(raw_draft=raw_draft, evidences=evidences, pii_patterns=())

    assert without_patterns.verdict is Verdict.PASS
    assert gate.DEFAULT_PII_PATTERNS  # 기본값 하나로 동작한다


# ── 복수 사유 · 결정론 ───────────────────────────────────────────────────────


def test_기각_사유를_하나만_잡고_멈추지_않고_전부_수집한다() -> None:
    raw_draft = {
        "claims": [
            {"text": "담당자 010-9999-8888 로 연락 주세요.", "citation_ids": []},
            {"text": 42, "citation_ids": ["policy:없는문서:1"]},
        ]
    }

    result = evaluate_draft(raw_draft=raw_draft, evidences=_evidences())

    assert result.verdict is Verdict.REJECT
    assert result.reject_reasons == (
        RejectReason.SCHEMA_VIOLATION,
        RejectReason.MISSING_CITATION,
        RejectReason.INVALID_CITATION,
        RejectReason.PII_DETECTED,
    )


def test_같은_입력은_항상_같은_판정과_같은_사유_순서를_낸다() -> None:
    raw_draft = {
        "claims": [
            {"text": "담당자 010-9999-8888 로 연락 주세요.", "citation_ids": []},
            {"text": 42, "citation_ids": ["policy:없는문서:1"]},
        ]
    }
    evidences = _evidences()

    results = [evaluate_draft(raw_draft=raw_draft, evidences=evidences) for _ in range(20)]

    assert len({(r.verdict, r.reject_reasons) for r in results}) == 1


def test_같은_사유는_중복되지_않는다() -> None:
    raw_draft = {
        "claims": [
            {"text": NORMAL_TEXT, "citation_ids": []},
            {"text": "주문은 배송 완료 상태입니다.", "citation_ids": []},
        ]
    }

    result = evaluate_draft(raw_draft=raw_draft, evidences=_evidences())

    assert result.reject_reasons == (RejectReason.MISSING_CITATION,)


# ── LLM 호출 0회 보증 ────────────────────────────────────────────────────────


def test_게이트는_LLM_네트워크_시간_난수_모듈을_import_하지_않는다() -> None:
    """L1 은 LLM 호출 0회이고 100% 재현 가능해야 한다 — 구조로 못박는다."""
    source = pathlib.Path(gate.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])

    forbidden = {
        "anthropic",
        "openai",
        "httpx",
        "requests",
        "urllib",
        "http",
        "socket",
        "random",
        "secrets",
        "time",
        "datetime",
        "os",
        "uuid",
    }
    assert not (imported & forbidden)


# ── Draft 변환 ──────────────────────────────────────────────────────────────


def test_to_draft_는_통과한_초안을_Draft_로_바꾼다() -> None:
    raw_draft = {
        "claims": [
            {"text": NORMAL_TEXT, "citation_ids": [POLICY_ID]},
            {"text": "주문은 배송 완료 상태입니다.", "citation_ids": [SQL_ID]},
        ]
    }

    draft = to_draft(raw_draft)

    assert draft == Draft(
        claims=(
            Claim(text=NORMAL_TEXT, citation_ids=(POLICY_ID,)),
            Claim(text="주문은 배송 완료 상태입니다.", citation_ids=(SQL_ID,)),
        )
    )


def test_to_draft_는_citation_이_비어도_변환한다() -> None:
    """missing_citation 은 판정 사유일 뿐 구조 오류가 아니다."""
    draft = to_draft({"claims": [{"text": NORMAL_TEXT, "citation_ids": []}]})

    assert draft.claims == (Claim(text=NORMAL_TEXT, citation_ids=()),)


@pytest.mark.parametrize(
    "raw_draft",
    [
        pytest.param("원문 문자열", id="문자열"),
        pytest.param(None, id="None"),
        pytest.param({"claims": []}, id="빈_claims"),
        pytest.param({"claims": [{"text": 1, "citation_ids": []}]}, id="text_타입_오류"),
    ],
)
def test_to_draft_는_구조가_깨진_초안을_거부한다(raw_draft: object) -> None:
    with pytest.raises(ValueError):
        to_draft(raw_draft)
