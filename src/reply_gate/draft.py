"""초안 생성 — 수집된 근거만 컨텍스트로 답변 계약 JSON 을 만든다 (spec 파이프라인 4단계).

**이 모듈은 게이트가 아니다.** 생성만 하고 검증하지 않는다 — 산출이 답변 계약을 어겨도
여기서 고치거나 걸러내지 않는다. 그건 L1 의 일이고, 여기서 미리 막으면 게이트 지표가
무의미해진다. 구조화 출력으로 스키마를 강제하되(생성 측 수단), L1 검사를 대체하지 않는다.

재생성도 이 모듈이 담당한다: L1 기각 사유 목록을 피드백으로 붙여 **같은 근거로** 다시 생성한다
(근거 재수집 없음). 재생성 1회 상한을 강제하는 것은 상위 루프(코드)다.

실패 정책은 spec "LLM 호출 공통 실패 정책"을 그대로 따른다.
- 전송 오류: `llm.AnthropicClient` 가 이미 1회 재시도했다 — `LLMCallError` 는 그대로 위로
  던져 호출자가 `llm_call_failed` 인계로 매핑하게 한다.
- 구조화 출력 형식 불일치: **재시도하지 않는다.** 모델 응답 원문을 원시 산출로 담아 돌려주고
  L1 이 `schema_violation` 으로 판정한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from reply_gate.contracts import DRAFT_JSON_SCHEMA, Evidence, RejectReason
from reply_gate.llm import AnthropicClient, Effort, LLMFormatError

__all__ = [
    "DRAFT_STAGE",
    "DRAFT_SYSTEM_PROMPT",
    "DraftGeneration",
    "DraftGenerator",
    "build_draft_user_prompt",
]

#: 처리 기록·실패 사유에 남는 단계 이름 (spec: 실패한 단계를 처리 기록에 남긴다).
DRAFT_STAGE = "draft"

#: 초안 생성 시스템 프롬프트.
#:
#: 이 프롬프트의 품질이 기각 분포와 데모 품질을 좌우하는 확률적 핵심이다 — 골든셋 실측을
#: 보고 조정한다. 단, **미끼 조항을 무력화하는 우회 장치는 넣지 않는다**: 근거에 없는 값을
#: 모델이 일반 지식으로 채우고 L1 `pii_detected` 에 걸리는 것이 데모의 핵심이므로,
#: "근거에 없는 값을 지어내지 말라"는 일반 지시까지가 여기서 할 일이다.
DRAFT_SYSTEM_PROMPT = """\
너는 한국어 이커머스 고객센터 상담원이다. 아래 규칙을 지켜 고객 문의에 답한다.

1. **제공된 근거 안의 정보만 쓴다.** 근거 목록에 없는 내용은 답변에 넣지 않는다. 일반 상식으로
   알고 있거나 다른 쇼핑몰에서 흔한 규정이라도, 이번 문의의 근거에 없으면 쓰지 않는다.
2. **근거에 없는 값을 지어내지 않는다.** 전화번호·이메일·주소·금액·기간·날짜·주문 상태 같은
   구체적인 값은 근거에 적힌 것만 그대로 쓴다. 근거에 그 값이 없으면 채워 넣지 말고, 안내가
   어렵다고 쓰거나 그 문장을 아예 빼라.
3. **문장마다 근거 ID 를 붙인다.** 답변을 문장(claim) 단위로 쪼개고, 각 문장을 뒷받침하는 근거의
   ID 를 `citation_ids` 에 모두 적는다. ID 는 근거 목록에 있는 문자열을 글자 그대로 쓴다.
4. **한국어 CS 상담 톤.** 정중한 존댓말로, 고객이 바로 이해할 수 있게 간결히 쓴다. claims 의
   text 를 순서대로 이으면 그대로 하나의 답변이 되어야 한다.

산출은 다음 형태의 JSON 이다:
{"claims": [{"text": "<답변 문장 1개>", "citation_ids": ["<근거 ID>", "..."]}]}
"""

#: 근거가 0건일 때 근거 절에 넣는 표시. 근거 0건 인계 판단은 상위 루프 몫이지만,
#: 프롬프트가 빈 절로 무너지지 않게 방어적으로 채운다.
NO_EVIDENCE_MARKER = "(이번 문의에서 수집된 근거가 없다.)"

#: 기각 사유 코드에 붙는 재생성 지침. 사유 코드 자체가 피드백의 본체이고, 이 문구는
#: 모델이 무엇을 고쳐야 하는지 알려주는 L1 규칙의 재진술이다.
_REJECT_FEEDBACK: dict[RejectReason, str] = {
    RejectReason.SCHEMA_VIOLATION: "산출이 답변 계약 JSON 형식에 맞지 않았다. 위 형태 그대로 낸다.",
    RejectReason.MISSING_CITATION: (
        "citation_ids 가 빈 문장이 있었다. 모든 문장에 근거 ID 를 1개 이상 붙인다."
    ),
    RejectReason.INVALID_CITATION: (
        "근거 목록에 없는 ID 를 적었다. 아래 근거 목록의 ID 만 글자 그대로 쓴다."
    ),
    RejectReason.PII_DETECTED: (
        "근거에 없는 전화번호·이메일 같은 값이 답변에 있었다. 근거에 적힌 값만 쓴다."
    ),
}


def _format_evidence(evidence: Sequence[Evidence]) -> str:
    """근거 목록을 프롬프트 블록으로. 표시용 `content` 만 싣는다(대조용 원문은 L1 몫)."""
    if not evidence:
        return NO_EVIDENCE_MARKER
    return "\n\n".join(
        f"- 근거 ID: {item.id}\n  출처: {item.source.value}\n  내용: {item.content}"
        for item in evidence
    )


def _format_reject_reasons(reasons: Sequence[RejectReason]) -> str:
    """기각 사유를 **전부** 나열한다 — 하나만 골라 넘기면 재생성이 같은 실수를 반복한다."""
    lines = []
    for reason in reasons:
        guidance = _REJECT_FEEDBACK.get(reason)
        lines.append(f"- {reason.value}: {guidance}" if guidance else f"- {reason.value}")
    return "\n".join(lines)


def build_draft_user_prompt(
    *,
    inquiry: str,
    evidence: Sequence[Evidence],
    reject_reasons: Sequence[RejectReason] = (),
) -> str:
    """문의 + 근거(+ 재생성이면 기각 사유)를 사용자 프롬프트로 조립한다.

    `reject_reasons` 가 비어 있으면 최초 생성, 비어 있지 않으면 재생성이다. 두 경우 모두
    **같은 근거 집합**을 그대로 싣는다 — 근거 재수집은 이 모듈의 일이 아니다.
    """
    sections = [
        f"[문의]\n{inquiry}",
        f"[근거]\n{_format_evidence(evidence)}",
    ]
    if reject_reasons:
        sections.append(
            "[직전 초안이 기각된 사유]\n"
            f"{_format_reject_reasons(reject_reasons)}\n\n"
            "위 근거를 그대로 다시 써서 초안을 새로 작성한다. 근거를 새로 찾거나 추가하지 않는다."
        )
    return "\n\n".join(sections)


@dataclass(frozen=True)
class DraftGeneration:
    """초안 생성 1회의 산출 + 토큰 사용량.

    `raw` 는 **정규화하지 않은 원시 산출**이다 — L1 이 그대로 받아 판정한다. 구조화 출력이
    형식에 맞았으면 파싱된 값이, 형식 불일치였으면 모델 응답 원문 문자열이 들어간다
    (후자는 L1 이 `schema_violation` 으로 잡는다).
    """

    raw: Any
    input_tokens: int
    output_tokens: int


class DraftGenerator:
    """근거 목록만 컨텍스트로 답변 계약 JSON 을 생성한다. 근거는 스스로 수집하지 않는다."""

    def __init__(self, *, client: AnthropicClient, effort: Effort = "medium") -> None:
        self._client = client
        self._effort = effort

    def generate(
        self,
        *,
        inquiry: str,
        evidence: Sequence[Evidence],
        reject_reasons: Sequence[RejectReason] = (),
    ) -> DraftGeneration:
        """초안을 1회 생성한다. `reject_reasons` 를 주면 같은 근거로 재생성한다.

        전송 오류(`LLMCallError`)는 그대로 위로 던진다 — 호출자가 `llm_call_failed` 인계로
        매핑한다. 형식 불일치는 재시도하지 않고 원문을 원시 산출로 담아 돌려준다.
        """
        user = build_draft_user_prompt(
            inquiry=inquiry, evidence=evidence, reject_reasons=reject_reasons
        )
        try:
            completion = self._client.complete_json(
                stage=DRAFT_STAGE,
                system=DRAFT_SYSTEM_PROMPT,
                user=user,
                schema=DRAFT_JSON_SCHEMA,
                effort=self._effort,
            )
        except LLMFormatError as exc:
            return DraftGeneration(
                raw=exc.raw_text,
                input_tokens=exc.input_tokens,
                output_tokens=exc.output_tokens,
            )
        return DraftGeneration(
            raw=completion.data,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )
