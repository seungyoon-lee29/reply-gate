"""L1 게이트 테스트 — docs/business-rules.md "L1 게이트 판정 규칙" 을 코드로 고정한다.

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


@pytest.mark.parametrize(
    "raw_draft",
    [
        pytest.param(
            {"claims": [{"text": "", "citation_ids": ["policy:x:1"]}]},
            id="빈_문자열",
        ),
        pytest.param(
            {"claims": [{"text": "   ", "citation_ids": ["policy:x:1"]}]},
            id="공백뿐인_문자열",
        ),
    ],
)
def test_빈_claim_text는_schema_violation_으로_기각된다(raw_draft: object) -> None:
    evidence = _evidence("policy:x:1", "답변 근거")

    result = evaluate_draft(raw_draft=raw_draft, evidences=[evidence])

    assert result.reject_reasons == (RejectReason.SCHEMA_VIOLATION,)


def test_공백이_아닌_claim_text는_계속_통과한다() -> None:
    """양성 대조 — 비어 있지 않은 문장까지 구조 위반으로 접으면 안 된다."""
    evidence = _evidence("policy:x:1", "답변 근거")
    raw_draft = {"claims": [{"text": "답변", "citation_ids": ["policy:x:1"]}]}

    result = evaluate_draft(raw_draft=raw_draft, evidences=[evidence])

    assert result.verdict is Verdict.PASS


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
    이 단언이 깨진다 — 사유 분리가 무너졌다는 신호다(docs/business-rules.md "L1 게이트 판정 규칙").
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


@pytest.mark.parametrize(
    "draft_text",
    [
        "연락처는 010-1234-5678입니다.",
        "주민등록번호는 900101-1234567입니다.",
        "문의는 help@shop.co.kr 로 보내세요.",
    ],
)
def test_기존_PII_표준형도_탐지하고_근거에_있으면_통과한다(draft_text: str) -> None:
    no_pii_evidence = _evidence(POLICY_ID, "고객센터 운영 시간 안내")
    matching_evidence = _evidence(SQL_ID, draft_text, source=EvidenceSource.SQL)

    rejected = evaluate_draft(raw_draft=_draft(text=draft_text), evidences=[no_pii_evidence])
    echoed = evaluate_draft(
        raw_draft=_draft(text=draft_text, citation_ids=(SQL_ID,)), evidences=[matching_evidence]
    )

    assert rejected.reject_reasons == (RejectReason.PII_DETECTED,)
    assert echoed.verdict is Verdict.PASS


@pytest.mark.parametrize(
    ("draft_text", "evidence_text"),
    [
        pytest.param(
            "연락처는 +82 10-1234-5678입니다.",
            "등록 연락처는 010-1234-5678입니다.",
            id="국가번호_휴대전화",
        ),
        pytest.param(
            "주민등록번호는 900101 1234567입니다.",
            "본인확인 값은 900101-1234567입니다.",
            id="공백_주민등록번호",
        ),
        pytest.param(
            "문의는 support@쇼핑몰.kr 로 보내세요.",
            "문의 이메일은 support@쇼핑몰.kr 입니다.",
            id="국제화_도메인_이메일",
        ),
        # 아래 셋은 사람 눈에는 같은 값인데 ASCII 정규식만으로는 비껴간다. 눈으로 구분되지
        # 않는 문자라 **이스케이프로 적는다** — 리터럴로 쓰면 읽는 사람이 무엇을 시험하는지
        # 알 수 없고, 편집 중에 조용히 반각으로 바뀌어도 아무도 모른다.
        # 근거 쪽 표기를 일부러 반각으로 둬, 접기가 **양쪽에** 걸려야만 양성 대조가 통과한다.
        pytest.param(
            "연락처는 \uff10\uff11\uff10-\uff11\uff12\uff13\uff14-\uff15\uff16\uff17\uff18입니다.",
            "등록 연락처는 010-1234-5678입니다.",
            id="전각_숫자_휴대전화",
        ),
        pytest.param(
            "연락처는 010-1234\u200b-5678입니다.",
            "등록 연락처는 010-1234-5678입니다.",
            id="폭없는_서식문자_삽입",
        ),
        pytest.param(
            "문의는 help\uff20shop.co.kr 로 보내세요.",
            "문의 이메일은 help@shop.co.kr 입니다.",
            id="전각_골뱅이_이메일",
        ),
    ],
)
def test_흔한_PII_표기_변형은_탐지하고_근거에_있는_값만_통과시킨다(
    draft_text: str, evidence_text: str
) -> None:
    no_pii_evidence = _evidence(POLICY_ID, "고객센터 운영 시간 안내")
    matching_evidence = _evidence(SQL_ID, evidence_text, source=EvidenceSource.SQL)

    rejected = evaluate_draft(raw_draft=_draft(text=draft_text), evidences=[no_pii_evidence])
    echoed = evaluate_draft(
        raw_draft=_draft(text=draft_text, citation_ids=(SQL_ID,)), evidences=[matching_evidence]
    )

    assert rejected.reject_reasons == (RejectReason.PII_DETECTED,)
    assert echoed.verdict is Verdict.PASS


def test_국가번호와_괄호를_정규화해도_서로_다른_번호는_같은_값으로_접지_않는다() -> None:
    evidences = [
        _evidence(
            SQL_ID,
            "등록 연락처는 0082 (10) 1234 5679입니다.",
            source=EvidenceSource.SQL,
        )
    ]
    raw_draft = _draft(text="연락처는 +82 10-1234-5678입니다.", citation_ids=(SQL_ID,))

    result = evaluate_draft(raw_draft=raw_draft, evidences=evidences)

    assert result.reject_reasons == (RejectReason.PII_DETECTED,)


def test_더하기와_0082_국가번호_표기는_같은_번호로_정규화해_통과한다() -> None:
    evidences = [
        _evidence(
            SQL_ID,
            "등록 연락처는 0082 (10) 1234 5678입니다.",
            source=EvidenceSource.SQL,
        )
    ]
    raw_draft = _draft(text="연락처는 +82 10-1234-5678입니다.", citation_ids=(SQL_ID,))

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


def _gate_imports(source: str) -> tuple[set[str], set[str]]:
    """`(최상위 패키지 이름, 이 패키지 안에서 부른 모듈 전체 경로)`.

    **최상위 이름만 모으면 안 된다.** `from reply_gate.llm import JudgeClient` 는 최상위가
    `reply_gate` 라 금지 집합에 걸리지 않는데, `llm.py` 는 최상위에서 `anthropic`·`openai`
    를 부른다 — 게이트가 그 한 줄로 LLM 에 닿는다. 그래서 자기 패키지 안쪽은 **전체 경로**
    로 따로 모아 불변식 1(잎 노드)로 검사한다. `ast.walk` 라 함수 안의 지연 import 도 본다.
    """
    top: set[str] = set()
    internal: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top.add(alias.name.split(".")[0])
                if alias.name.split(".")[0] == _PACKAGE:
                    internal.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            top.add(node.module.split(".")[0])
            if node.module.split(".")[0] == _PACKAGE:
                internal.add(node.module)
    return top, internal


_PACKAGE = "reply_gate"
#: 불변식 1 — `gate.py` 는 잎 노드다. 이 패키지에서 부를 수 있는 것은 계약 하나뿐이다.
_ALLOWED_INTERNAL_IMPORTS = {"reply_gate.contracts"}


def test_게이트는_자기_패키지에서_계약_외의_모듈을_import_하지_않는다() -> None:
    """불변식 1(잎 노드) — 여기가 열리면 하드 게이트 1 이 한 줄로 깨진다.

    `from reply_gate.llm import …` 한 줄이면 게이트가 LLM SDK 에 닿는데, 최상위 이름만
    보는 검사는 그것을 `reply_gate` 로 읽어 통과시킨다. 실제로 그 구멍이 있었다.
    """
    _top, internal = _gate_imports(pathlib.Path(gate.__file__).read_text(encoding="utf-8"))
    assert internal, "검사 대상이 비면 이 가드는 아무것도 지키지 않는다"
    assert internal <= _ALLOWED_INTERNAL_IMPORTS


def test_잎_노드_검사가_패키지_내부_우회를_실제로_잡는다() -> None:
    """음성 대조 — 검사기가 무언가를 잡는다는 것을 같은 파일이 증명한다."""
    _top, internal = _gate_imports(
        "from reply_gate.contracts import Claim\n"
        "def f():\n"
        "    from reply_gate.llm import JudgeClient\n"
        "    return JudgeClient\n"
    )
    assert internal == {"reply_gate.contracts", "reply_gate.llm"}
    assert not (internal <= _ALLOWED_INTERNAL_IMPORTS)


def test_게이트는_LLM_네트워크_시간_난수_모듈을_import_하지_않는다() -> None:
    """L1 은 LLM 호출 0회이고 100% 재현 가능해야 한다 — 구조로 못박는다."""
    imported, _internal = _gate_imports(pathlib.Path(gate.__file__).read_text(encoding="utf-8"))

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


# ── 미끼 조항이 실제로 기각을 유발하는지 (docs/business-rules.md "평가용 데이터의 지위" 절) ──────────────────


def _decoy_evidence() -> Evidence:
    """고객센터 안내이면서 전화번호를 일부러 비워 둔 미끼 조항."""
    text = (
        "고객센터는 평일 09:00부터 18:00까지 운영하며 점심시간은 12:00부터 13:00까지입니다. "
        "상담이 필요하시면 운영시간 내에 고객센터로 연락해 주십시오."
    )
    return Evidence(
        id="policy:support:4-1",
        source=EvidenceSource.POLICY,
        content=text,
        evidence_text=text,
    )


def test_미끼_조항에_없는_대표번호를_지어내면_기각된다() -> None:
    draft = {
        "claims": [
            {
                "text": "고객센터 전화번호는 1588-1234 입니다.",
                "citation_ids": ["policy:support:4-1"],
            }
        ]
    }

    result = evaluate_draft(raw_draft=draft, evidences=[_decoy_evidence()])

    assert result.verdict is Verdict.REJECT
    assert RejectReason.PII_DETECTED in result.reject_reasons


def test_미끼_조항에_없는_이메일을_지어내면_기각된다() -> None:
    draft = {
        "claims": [
            {
                "text": "서류는 help@example.com 으로 보내주세요.",
                "citation_ids": ["policy:support:4-1"],
            }
        ]
    }

    result = evaluate_draft(raw_draft=draft, evidences=[_decoy_evidence()])

    assert RejectReason.PII_DETECTED in result.reject_reasons


def test_미끼를_피해_운영시간만_답하면_통과한다() -> None:
    """기각이 목적이 아니다 — 근거 안에 머무른 답변은 그대로 통과해야 한다."""
    draft = {
        "claims": [
            {
                "text": "고객센터는 평일 09:00부터 18:00까지 운영합니다.",
                "citation_ids": ["policy:support:4-1"],
            }
        ]
    }

    result = evaluate_draft(raw_draft=draft, evidences=[_decoy_evidence()])

    assert result.verdict is Verdict.PASS


def test_근거에_있는_대표번호_에코는_통과한다() -> None:
    text = "고객센터 대표번호는 1588-0000 입니다."
    evidence = Evidence(
        id="policy:support:4-9",
        source=EvidenceSource.POLICY,
        content=text,
        evidence_text=text,
    )
    draft = {
        "claims": [
            {"text": "고객센터는 15880000 번으로 연락하시면 됩니다.", "citation_ids": [evidence.id]}
        ]
    }

    result = evaluate_draft(raw_draft=draft, evidences=[evidence])

    assert result.verdict is Verdict.PASS


# ── 근거 ID 는 PII 검사 대상이 아니다 (오탐률 = 헤드라인 지표) ────────────────


def test_근거_ID_의_숫자는_PII_로_오인하지_않는다() -> None:
    """SQL 근거 ID 는 `sql:<문의 UUID>:<순번>` 이라 UUID 16진 숫자가 전화번호 패턴에 걸린다.

    근거 ID 는 답변 문장이 아니라 식별자이고 evidence_text 에도 없어 allowlist 에 오르지
    않으므로, 검사 대상에 두면 PII 가 전혀 없는 정상 초안이 기각된다.
    """
    cid = "sql:172219ff-f0ef-4a3d-9e8d-085527370ac7:1"
    evidence = Evidence(
        id=cid,
        source=EvidenceSource.SQL,
        content="주문 조회 결과",
        evidence_text="order_no=ORD-20260202-0001 status=배송중",
    )
    draft = {"claims": [{"text": "주문은 배송중입니다.", "citation_ids": [cid]}]}

    result = evaluate_draft(raw_draft=draft, evidences=[evidence])

    assert result.verdict is Verdict.PASS


def test_무작위_문의ID_로_만든_근거ID_는_한_건도_오탐되지_않는다() -> None:
    """0.84% 오탐은 지표를 오염시킨다 — 회귀하면 이 테스트가 잡는다."""
    import uuid

    for index in range(2000):
        inquiry_id = str(uuid.UUID(int=index * 2654435761 % (1 << 128)))
        cid = f"sql:{inquiry_id}:1"
        evidence = Evidence(
            id=cid, source=EvidenceSource.SQL, content="c", evidence_text="주문 상태: 배송중"
        )
        draft = {"claims": [{"text": "주문은 배송중입니다.", "citation_ids": [cid]}]}
        result = evaluate_draft(raw_draft=draft, evidences=[evidence])
        assert result.verdict is Verdict.PASS, f"{cid} 가 오탐됐다: {result.reject_reasons}"


def test_답변_문장의_PII_는_여전히_잡는다() -> None:
    """근거 ID 를 제외한 것이 검사를 느슨하게 만들지 않았다는 양성 대조."""
    cid = "sql:172219ff-f0ef-4a3d-9e8d-085527370ac7:1"
    evidence = Evidence(
        id=cid, source=EvidenceSource.SQL, content="c", evidence_text="주문 상태: 배송중"
    )
    draft = {"claims": [{"text": "고객센터 010-9999-8888 로 연락주세요.", "citation_ids": [cid]}]}

    result = evaluate_draft(raw_draft=draft, evidences=[evidence])

    assert RejectReason.PII_DETECTED in result.reject_reasons


def test_형식이_깨진_초안의_PII_는_계속_검사한다() -> None:
    """원문 문자열로 넘어온 산출에서도 PII 는 새어 나갈 수 있다."""
    evidence = Evidence(
        id="policy:support:4-1", source=EvidenceSource.POLICY, content="c", evidence_text="운영시간"
    )

    result = evaluate_draft(raw_draft="고객센터는 1588-1234 입니다", evidences=[evidence])

    assert RejectReason.SCHEMA_VIOLATION in result.reject_reasons


# ── 답변에 나가지 않는 키는 PII 검사 대상이 아니다 (불변식 7) ─────────────────


@pytest.mark.parametrize(
    ("case", "raw_draft"),
    [
        (
            "최상위 추가 키",
            {"claims": [{"text": NORMAL_TEXT, "citation_ids": [POLICY_ID]}], "debug": "1588-1234"},
        ),
        (
            "claim 안의 추가 키",
            {
                "claims": [
                    {"text": NORMAL_TEXT, "citation_ids": [POLICY_ID], "note": "010-9999-8888"}
                ]
            },
        ),
        (
            "중첩된 추가 키",
            {
                "claims": [{"text": NORMAL_TEXT, "citation_ids": [POLICY_ID]}],
                "trace": {"prompt": {"raw": "담당자 010-9999-8888"}},
            },
        ),
    ],
)
def test_답변에_나가지_않는_키의_PII_로_기각하지_않는다(case: str, raw_draft: object) -> None:
    """`to_draft` 가 버리는 키는 답변에 실리지 않으므로 검사 대상이 아니다.

    `src/reply_gate/AGENTS.md` 불변식 7 — "L1 의 PII 검사 대상은 답변 텍스트뿐이다".
    초안 JSON 은 LLM 산출이라 이런 키가 언제든 늘어날 수 있고, 헤드라인 지표가
    "정상 초안 오탐률"이라 여기서 기각하면 지표가 직접 오염된다.
    """
    result = evaluate_draft(raw_draft=raw_draft, evidences=_evidences())

    assert result.verdict is Verdict.PASS, f"{case}: {result.reject_reasons}"
    assert to_draft(raw_draft).answer_text == NORMAL_TEXT  # 답변에 그 값이 없다는 확인


def test_검사에서_뺀_키가_답변에_실리면_그때는_기각된다() -> None:
    """양성 대조 — 제외 기준은 '키 이름' 이 아니라 '답변에 나가는가' 다."""
    raw_draft = {
        "claims": [{"text": f"{NORMAL_TEXT} 담당자 010-9999-8888", "citation_ids": [POLICY_ID]}],
        "debug": "무해한 값",
    }

    result = evaluate_draft(raw_draft=raw_draft, evidences=_evidences())

    assert RejectReason.PII_DETECTED in result.reject_reasons


def test_깨진_claim_옆의_멀쩡한_claim_은_계속_검사한다() -> None:
    """구조가 깨져도 답변 후보 텍스트가 있으면 그것을 검사한다."""
    raw_draft = {
        "claims": [
            {"text": "담당자 010-9999-8888 로 연락주세요.", "citation_ids": [POLICY_ID]},
            {"text": NORMAL_TEXT, "citation_ids": "리스트가 아니다"},
        ]
    }

    result = evaluate_draft(raw_draft=raw_draft, evidences=_evidences())

    assert result.reject_reasons == (
        RejectReason.SCHEMA_VIOLATION,
        RejectReason.PII_DETECTED,
    )
    assert RejectReason.PII_DETECTED in result.reject_reasons
