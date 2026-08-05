"""초안 생성 모듈 단위 테스트 (LLM 호출은 전부 목).

이 모듈은 게이트가 아니다 — 계약 위반 산출을 여기서 고치거나 걸러내지 않고
원시 산출 그대로 돌려주는지가 검증 대상이다(판정은 L1 몫).
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from reply_gate.contracts import DRAFT_JSON_SCHEMA, Evidence, EvidenceSource, RejectReason
from reply_gate.draft import (
    DRAFT_STAGE,
    DRAFT_SYSTEM_PROMPT,
    DraftGenerator,
    build_draft_user_prompt,
)
from reply_gate.llm import GenerationClient, JsonCompletion, LLMCallError, LLMFormatError

INQUIRY = "지난주에 주문한 운동화가 아직 안 왔어요. 언제쯤 받을 수 있나요?"

EVIDENCE: tuple[Evidence, ...] = (
    Evidence(
        id="policy:shipping-policy:3",
        source=EvidenceSource.POLICY,
        content="제3조(배송기간) 주문일로부터 영업일 기준 3일 이내에 출고한다.",
        evidence_text="제3조(배송기간) 주문일로부터 영업일 기준 3일 이내에 출고한다.",
    ),
    Evidence(
        id="sql:INQ-1:1",
        source=EvidenceSource.SQL,
        content="order_no=ORD-20260101-0001, status=배송중, shipped_at=2026-01-02",
        evidence_text="order_no=ORD-20260101-0001, status=배송중, shipped_at=2026-01-02",
    ),
)

VALID_DRAFT: dict[str, Any] = {
    "claims": [
        {
            "text": "주문하신 상품은 2026-01-02에 출고되어 현재 배송중입니다.",
            "citation_ids": ["sql:INQ-1:1"],
        }
    ]
}


class _RecordingClient:
    """`GenerationClient.complete_json` 대역 — 호출 인자를 기록하고 정해진 결과를 돌려준다."""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self._outcomes[min(len(self.calls) - 1, len(self._outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _generator(outcomes: list[Any]) -> tuple[DraftGenerator, _RecordingClient]:
    recorder = _RecordingClient(outcomes)
    generator = DraftGenerator(client=cast(GenerationClient, recorder))
    return generator, recorder


# ── 프롬프트 ────────────────────────────────────────────────────────────────


def test_프롬프트에_모든_근거의_id와_내용이_들어간다() -> None:
    prompt = build_draft_user_prompt(inquiry=INQUIRY, evidence=EVIDENCE)

    assert INQUIRY in prompt
    for item in EVIDENCE:
        assert item.id in prompt
        assert item.content in prompt


def test_시스템_프롬프트는_근거_밖_지식_사용을_금지한다() -> None:
    # 근거 밖 지식 금지 / 근거에 없는 값 날조 금지 / 문장별 근거 ID / 한국어 CS 톤.
    assert "제공된 근거 안의 정보만" in DRAFT_SYSTEM_PROMPT
    assert "지어내지 않는다" in DRAFT_SYSTEM_PROMPT
    assert "citation_ids" in DRAFT_SYSTEM_PROMPT
    assert "존댓말" in DRAFT_SYSTEM_PROMPT


def test_재생성_프롬프트는_기각사유_전부와_같은_근거를_싣는다() -> None:
    reasons = (
        RejectReason.MISSING_CITATION,
        RejectReason.INVALID_CITATION,
        RejectReason.PII_DETECTED,
        RejectReason.SCHEMA_VIOLATION,
    )

    prompt = build_draft_user_prompt(inquiry=INQUIRY, evidence=EVIDENCE, reject_reasons=reasons)

    for reason in reasons:
        assert reason.value in prompt
    # 근거는 재수집하지 않는다 — 최초 생성과 같은 근거가 그대로 실린다.
    for item in EVIDENCE:
        assert item.id in prompt
        assert item.content in prompt


def test_최초_프롬프트에는_기각사유_절이_없다() -> None:
    prompt = build_draft_user_prompt(inquiry=INQUIRY, evidence=EVIDENCE)

    for reason in RejectReason:
        assert reason.value not in prompt


# ── 생성 ────────────────────────────────────────────────────────────────────


def test_정상_응답은_원시_산출과_토큰을_그대로_돌려준다() -> None:
    generator, recorder = _generator(
        [JsonCompletion(data=VALID_DRAFT, input_tokens=120, output_tokens=45)]
    )

    result = generator.generate(inquiry=INQUIRY, evidence=EVIDENCE)

    assert result.raw == VALID_DRAFT
    assert (result.input_tokens, result.output_tokens) == (120, 45)
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["stage"] == DRAFT_STAGE


def test_계약을_어긴_산출도_고치지_않고_그대로_돌려준다() -> None:
    """생성 측에서 걸러내면 L1 지표가 무의미해진다 — 판정은 L1 몫이다."""
    broken: dict[str, Any] = {"claims": [{"text": "근거 없는 문장", "citation_ids": []}]}
    generator, _ = _generator([JsonCompletion(data=broken, input_tokens=1, output_tokens=1)])

    assert generator.generate(inquiry=INQUIRY, evidence=EVIDENCE).raw == broken


def test_형식_불일치는_재시도없이_원문을_원시산출로_돌려준다() -> None:
    error = LLMFormatError(
        stage=DRAFT_STAGE,
        detail="JSON 파싱 실패",
        raw_text="이건 JSON 이 아니다",
        input_tokens=99,
        output_tokens=3,
    )
    generator, recorder = _generator([error])

    result = generator.generate(inquiry=INQUIRY, evidence=EVIDENCE)

    assert result.raw == "이건 JSON 이 아니다"
    assert (result.input_tokens, result.output_tokens) == (99, 3)
    # 초안 생성은 형식 불일치를 재시도하지 않는다 (docs/standards.md "재시도 상한").
    assert len(recorder.calls) == 1


def test_전송오류는_그대로_위로_전파된다() -> None:
    error = LLMCallError(stage=DRAFT_STAGE, reason="transport_error", attempts=2)
    generator, recorder = _generator([error])

    with pytest.raises(LLMCallError) as excinfo:
        generator.generate(inquiry=INQUIRY, evidence=EVIDENCE)

    assert excinfo.value.stage == DRAFT_STAGE
    assert len(recorder.calls) == 1


def test_구조화_출력에_답변계약_스키마를_그대로_넘긴다() -> None:
    generator, recorder = _generator(
        [JsonCompletion(data=VALID_DRAFT, input_tokens=0, output_tokens=0)]
    )

    generator.generate(inquiry=INQUIRY, evidence=EVIDENCE)

    # 계약이 바뀌지 않았다는 보증 — 스키마를 재조립하지 않고 공유 정의를 그대로 쓴다.
    assert recorder.calls[0]["schema"] is DRAFT_JSON_SCHEMA


def test_effort_는_medium_기본이고_호출자가_바꿀_수_있다() -> None:
    generator, recorder = _generator(
        [JsonCompletion(data=VALID_DRAFT, input_tokens=0, output_tokens=0)]
    )
    generator.generate(inquiry=INQUIRY, evidence=EVIDENCE)
    # reasoning effort 는 기본적으로 보내지 않는다 — 모델 등급이 조정 가능이라
    # 지원하지 않는 계열에 보내면 요청 자체가 거부된다.
    assert recorder.calls[0]["effort"] is None

    high_recorder = _RecordingClient(
        [JsonCompletion(data=VALID_DRAFT, input_tokens=0, output_tokens=0)]
    )
    high = DraftGenerator(client=cast(GenerationClient, high_recorder), effort="high")
    high.generate(inquiry=INQUIRY, evidence=EVIDENCE)
    assert high_recorder.calls[0]["effort"] == "high"


def test_근거가_비어도_모듈이_깨지지_않는다() -> None:
    """근거 0건 인계 판단은 상위 루프 몫 — 생성기는 방어적으로 동작만 한다."""
    prompt = build_draft_user_prompt(inquiry=INQUIRY, evidence=())
    assert INQUIRY in prompt

    generator, recorder = _generator(
        [JsonCompletion(data={"claims": []}, input_tokens=5, output_tokens=2)]
    )

    result = generator.generate(inquiry=INQUIRY, evidence=())

    assert result.raw == {"claims": []}
    assert len(recorder.calls) == 1
