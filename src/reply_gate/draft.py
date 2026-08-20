"""초안 생성 — 수집된 근거만 컨텍스트로 답변 계약 JSON 을 만든다
(docs/architecture.md "대표 흐름" 5단계).

**이 모듈은 게이트가 아니다.** 생성만 하고 검증하지 않는다 — 산출이 답변 계약을 어겨도
여기서 고치거나 걸러내지 않는다. 그건 L1 의 일이고, 여기서 미리 막으면 게이트 지표가
무의미해진다. 구조화 출력으로 스키마를 강제하되(생성 측 수단), L1 검사를 대체하지 않는다.

재생성도 이 모듈이 담당한다: 기각 사유 목록(L1 4종 + L2 2종)과 **L2 판정 상세**(어느
문장이 왜 뒷받침되지 않았는지, 어느 근거쌍이 모순인지)를 피드백으로 붙여 **같은 근거로**
다시 생성한다(근거 재수집 없음). 재생성 1회 상한을 강제하는 것은 상위 루프(코드)다.
피드백 형태는 SQL 검증기 선례를 따른다: **사유 코드 + 무엇을 어떻게 고칠지**.

실패 정책은 docs/standards.md "재시도 상한"을 그대로 따른다.
- 전송 오류: `llm.OpenAIGenerationClient` 가 이미 1회 재시도했다 — `LLMCallError` 는 그대로 위로
  던져 호출자가 `llm_call_failed` 인계로 매핑하게 한다.
- 구조화 출력 형식 불일치: **재시도하지 않는다.** 모델 응답 원문을 원시 산출로 담아 돌려주고
  L1 이 `schema_violation` 으로 판정한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from reply_gate.contracts import (
    DRAFT_JSON_SCHEMA,
    Evidence,
    JudgeResult,
    RejectReason,
    Verdict,
)
from reply_gate.llm import GenerationClient, LLMFormatError

__all__ = [
    "DRAFT_STAGE",
    "DRAFT_SYSTEM_PROMPT",
    "DraftGeneration",
    "DraftGenerator",
    "build_draft_user_prompt",
]

#: 처리 기록·실패 사유에 남는 단계 이름
#: (docs/business-rules.md "인계 사유 6종" — `llm_call_failed` 는 실패한 단계 이름을 기록한다).
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
    RejectReason.UNSUPPORTED_CLAIM: (
        "인용한 근거가 그 문장을 뒷받침하지 않았다. 각 문장을 그 주제를 실제로 다루는 근거로 "
        "다시 딛고, 근거가 말하지 않는 내용은 단정하지 않는다."
    ),
    RejectReason.CONTRADICTORY_EVIDENCE: (
        "근거끼리 서로 어긋나는데 초안이 그것을 밝히지 않았다. **모순을 명시하고 두 기준을 "
        "모두 안내한다** — 한쪽만 골라 답하지 않는다."
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


def _format_judge_detail(result: JudgeResult) -> str:
    """L2 판정 상세 — **기각된 claim** 과 모순 근거쌍만 싣는다.

    통과한 claim 의 설명까지 실으면 프롬프트가 길어지기만 하고 고칠 지점이 흐려진다.
    """
    lines: list[str] = []
    rejected = [
        judgment for judgment in result.claim_judgments if judgment.verdict is Verdict.REJECT
    ]
    if rejected:
        lines.append("- 인용 근거가 뒷받침하지 않는다고 판정된 문장:")
        lines.extend(
            f'  - "{judgment.claim_text}" → {judgment.explanation}' for judgment in rejected
        )
    if result.contradictions:
        lines.append("- 서로 어긋나는 근거쌍:")
        lines.extend(
            f"  - {item.evidence_id_a} ↔ {item.evidence_id_b}: {item.explanation}"
            for item in result.contradictions
        )
    return "\n".join(lines)


def build_draft_user_prompt(
    *,
    inquiry: str,
    evidence: Sequence[Evidence],
    reject_reasons: Sequence[RejectReason] = (),
    judge_result: JudgeResult | None = None,
) -> str:
    """문의 + 근거(+ 재생성이면 기각 사유와 L2 판정 상세)를 사용자 프롬프트로 조립한다.

    `reject_reasons` 가 비어 있으면 최초 생성, 비어 있지 않으면 재생성이다. 두 경우 모두
    **같은 근거 집합**을 그대로 싣는다 — 근거 재수집은 이 모듈의 일이 아니다.

    `judge_result` 는 직전 시도의 L2 판정이다(L2 가 실행되지 않았으면 `None`). 사유 코드는
    무엇이 틀렸는지만 말하므로, claim 단위 상세가 있어야 재생성이 **어느 문장을** 고칠지
    안다.
    """
    sections = [
        f"[문의]\n{inquiry}",
        f"[근거]\n{_format_evidence(evidence)}",
    ]
    if reject_reasons or judge_result is not None:
        block = ["[직전 초안이 기각된 사유]", _format_reject_reasons(reject_reasons)]
        detail = "" if judge_result is None else _format_judge_detail(judge_result)
        if detail:
            block.extend(["", "[판정 세부 — 어느 문장이 왜]", detail])
        block.extend(
            [
                "",
                "위 근거를 그대로 다시 써서 초안을 새로 작성한다. "
                "근거를 새로 찾거나 추가하지 않는다.",
            ]
        )
        sections.append("\n".join(block))
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
    #: 이 호출에 흐른 벽시계(ms) — **초안 생성 구간의 시간**이다. 형식이 어긋나 원문을
    #: 그대로 넘기는 산출도 시간을 썼으므로 0 으로 접지 않는다. 전송 오류로 죽은 호출의
    #: 경과는 `LLMCallError.elapsed_ms` 가 들고 위로 올라간다.
    elapsed_ms: float = 0.0
    #: 이 호출의 캐시 계열 토큰 — **생성 계열의 칸으로 간다.** 입력 토큰에 접지 않는다:
    #: 단가가 다른 값을 한 칸에 넣으면 달러 환산이 그 차이를 볼 수 없다. 캐시 계열을 싣지
    #: 않은 응답에서는 0 이 아니라 `None`(미측정)이다.
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class DraftGenerator:
    """근거 목록만 컨텍스트로 답변 계약 JSON 을 생성한다. 근거는 스스로 수집하지 않는다."""

    def __init__(self, *, client: GenerationClient, effort: str | None = None) -> None:
        self._client = client
        self._effort = effort

    def generate(
        self,
        *,
        inquiry: str,
        evidence: Sequence[Evidence],
        reject_reasons: Sequence[RejectReason] = (),
        judge_result: JudgeResult | None = None,
    ) -> DraftGeneration:
        """초안을 1회 생성한다. 기각 사유·L2 상세를 주면 같은 근거로 재생성한다.

        전송 오류(`LLMCallError`)는 그대로 위로 던진다 — 호출자가 `llm_call_failed` 인계로
        매핑한다. 형식 불일치는 재시도하지 않고 원문을 원시 산출로 담아 돌려준다.
        """
        user = build_draft_user_prompt(
            inquiry=inquiry,
            evidence=evidence,
            reject_reasons=reject_reasons,
            judge_result=judge_result,
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
                elapsed_ms=exc.elapsed_ms,
                cache_creation_input_tokens=exc.cache_creation_input_tokens,
                cache_read_input_tokens=exc.cache_read_input_tokens,
            )
        return DraftGeneration(
            raw=completion.data,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            elapsed_ms=completion.elapsed_ms,
            cache_creation_input_tokens=completion.cache_creation_input_tokens,
            cache_read_input_tokens=completion.cache_read_input_tokens,
        )
