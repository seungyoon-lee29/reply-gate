"""L2 판정 — L1 을 통과한 초안을 claim 단위로 근거와 대조한다[LLM].

**이 모듈은 판정만 한다.** 파이프라인 제어·저장·재생성·L2 스위치 판단은 하지 않고,
"초안 1개 + 근거"를 받아 **시도당 1회 배치 판정**하는 클래스만 제공한다 — 재생성 초안을
다시 판정해 문의당 최대 2회가 되는 것은 호출 횟수를 통제하는 파이프라인의 몫이다.
배치인 이유는 기능 근거다: 모순 감지가 claim·근거 교차 시야를 요구하므로 claim 별
개별 호출로 쪼갤 수 없다 (docs/business-rules.md "L2 판정 규칙").

입력 조립은 두 절로 나뉜다 (docs/business-rules.md "뒷받침 판정 — claim 단위"·
"모순 판정 — 근거쌍 단위"):

* **뒷받침 판정용** — 각 claim 의 text + 그 claim 이 **인용한** 근거의 `evidence_text`.
* **모순 감지용** — 이번 문의의 **수집 근거 전체**(인용 여부 무관)의 `evidence_text`.
  상충 쌍이 한쪽만 인용됐거나 아예 인용되지 않아도 검출돼야 하기 때문이다.

판정 사유는 L2 2종(`unsupported_claim`·`contradictory_evidence`)이 전부다.
`unsupported_claim` 은 **근거-주장 정합만** 본다 — 문의-답변 관련성은 판정 범위 밖이다.

실패 정책 (docs/standards.md "재시도 상한" 의 "의도 해석" 패턴):

* **형식 불일치**는 직전 산출과 사유를 피드백으로 실어 **코드가 1회만** 재시도하고,
  재실패하면 누적 토큰을 실은 `LLMFormatError` 를 위로 던진다 — 호출자가
  `llm_call_failed` 로 매핑한다.
* **전송 오류**(`LLMCallError`)는 래퍼가 이미 1회 재시도했다 — 여기서 또 재시도하지 않고
  위로 전파하되, **그때까지 누적된 토큰을 예외에 실어** 보낸다. 1회차가 200 으로 돌아와
  과금된 뒤 2회차가 전송 오류로 죽는 조합에서 누적분을 버리면 파이프라인이 그 실비용을
  되찾을 방법이 없다 — 처리 기록·API 응답이 판정 토큰을 싣게 되는 순간(뒤 태스크 몫)
  그대로 0 으로 굳는다.

판정 호출의 토큰(입력/출력)은 `JudgeOutcome` 에 그대로 노출한다 — 파이프라인이 생성
토큰과 **분리 집계**해야 하므로(docs/contracts.md "토큰 집계 경계") 생성 합산에 섞지 않는다.
**캐시 계열(write/read)도 같은 자격으로 분리해 노출한다** — 단가가 다르고(write 는 비싸고
read 는 싸다), 켜짐 조건의 `input_tokens` 는 적중분을 뺀 값이라 합치면 적중이 "입력 토큰
감소"로 위장한다. 재지 않은 실행에서는 0 이 아니라 `None`(미측정)이다.

**판정 호출에 흐른 벽시계(`elapsed_ms`)도 같은 자격으로 노출한다** — 호출자가 판정 구간의
시간으로 쓴다. 형식 재시도로 버려진 시도의 시간까지 합산하며, 실패로 끝나면 두 예외가 그
값을 그대로 들고 올라간다. 프롬프트·모델·캐싱 어느 것도 이 때문에 바뀌지 않는다.

해석하지 못한 산출은 통과가 아니라 거부다(fail-closed): 사유 2종 밖 값, 판정값 밖 값,
수집 근거에 없는 ID 의 모순 쌍, verdict·사유·세부 배열이 서로 어긋나는 산출은 전부
형식 불일치로 취급한다 — 어긋난 판정 기록이 그대로 남으면 헤드라인 지표가 오염된다.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from reply_gate.contracts import (
    Claim,
    ClaimJudgment,
    Draft,
    Evidence,
    EvidenceContradiction,
    JudgeResult,
    RejectReason,
    Verdict,
)
from reply_gate.llm import (
    GenerationClient,
    JsonCompletion,
    LLMCallError,
    LLMFormatError,
    accumulate_optional_tokens,
)

__all__ = [
    "JUDGE_JSON_SCHEMA",
    "JUDGE_MAX_ATTEMPTS",
    "JUDGE_STAGE",
    "JUDGE_SYSTEM_PROMPT",
    "L2_REJECT_REASONS",
    "Judge",
    "JudgeOutcome",
    "build_judge_user_prompt",
]

#: 판정 모듈이 LLM 래퍼에 넘기는 단계 이름 — `LLMCallError`/`LLMFormatError` 의 `stage`
#: 로만 남는다. **처리 기록의 failed_stage 는 이 값이 아니다**: 파이프라인이 판정 실패를
#: `llm_call_failed` 로 매핑할 때 층을 가리키는 `pipeline.L2_JUDGE_STAGE`(`"l2_judge"`)
#: 를 적는다 (docs/standards.md "재시도 상한" 의 L2 두 행).
JUDGE_STAGE: Final = "judge"

#: 판정: 최초 호출 + 형식 불일치 재시도 1회 (docs/standards.md "재시도 상한").
JUDGE_MAX_ATTEMPTS: Final = 2

#: L2 기각 사유 전부 — `contracts.COMBINED_REASON_ORDER` 의 L2 구간과 같은 순서다.
#: 파싱은 이 밖의 값을 거부하고, 결과의 사유 순서는 이 튜플이 정한다.
L2_REJECT_REASONS: Final[tuple[RejectReason, ...]] = (
    RejectReason.UNSUPPORTED_CLAIM,
    RejectReason.CONTRADICTORY_EVIDENCE,
)

#: 판정의 구조화 출력 스키마 = `contracts.JudgeResult` 의 JSON 대응.
#: enum 은 API 측 강제 수단일 뿐이다 — 내용 검증은 `_parse_judge_result` 가
#: fail-closed 로 다시 한다(구조화 출력을 우회한 산출도 같은 검증을 받는다).
JUDGE_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "claim_judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_text": {"type": "string"},
                    "verdict": {"type": "string", "enum": [verdict.value for verdict in Verdict]},
                    "explanation": {"type": "string"},
                },
                "required": ["claim_text", "verdict", "explanation"],
                "additionalProperties": False,
            },
        },
        "contradictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "evidence_id_a": {"type": "string"},
                    "evidence_id_b": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["evidence_id_a", "evidence_id_b", "explanation"],
                "additionalProperties": False,
            },
        },
        "verdict": {"type": "string", "enum": [verdict.value for verdict in Verdict]},
        "reject_reasons": {
            "type": "array",
            "items": {"type": "string", "enum": [reason.value for reason in L2_REJECT_REASONS]},
        },
    },
    "required": ["claim_judgments", "contradictions", "verdict", "reject_reasons"],
    "additionalProperties": False,
}

#: 판정 시스템 프롬프트 — docs/business-rules.md "뒷받침 판정 — claim 단위"·
#: "모순 판정 — 근거쌍 단위" 의 의미 정책을 지시로 옮긴 것.
#: 이 정책이 제품 서사의 심장이다: 주제 인접 인용 기각 / 정면 조항의 "없다" 통과 /
#: 모순 비명시 기각 / 모순 명시 + 두 기준 안내 통과.
JUDGE_SYSTEM_PROMPT: Final = """\
너는 한국어 이커머스 CS 답변 초안의 검증자다. 초안을 고쳐 쓰지 않는다 — 두 가지만 판정한다:
(1) 각 claim 이 자신이 **인용한 근거**로 뒷받침되는가, (2) **수집 근거 전체**에 서로 모순되는
근거쌍이 있는가. 문의와 답변의 관련성, 답변의 품질·톤, 외부 지식과의 사실 대조는 전부
판정 범위 밖이다 — 근거와 claim 사이의 정합만 본다.

[뒷받침 판정 — claim 단위]
- claim 마다, 그 claim 이 인용한 근거의 원문만 보고 pass/reject 를 정한다.
- 근거의 내용을 그대로 옮기거나 왜곡 없이 요약·환언한 claim 은 pass 다.
- **주제 인접 인용은 기각한다.** 인용한 조항이 claim 의 주제를 실제로 다루지 않으면 그 claim 은
  reject 다. 예: 해외 배송을 묻는 문의에 국내 배송 조항만 인용해 "안내가 어렵다"고 쓴 claim —
  국내 배송 조항은 해외 배송이라는 주제를 다루지 않으므로 뒷받침이 아니다.
- **정면 조항의 "없다" 안내는 통과시킨다.** 인용한 조항이 claim 의 주제를 정면으로 다루면서
  해당 값만 비어 있는 경우, "기재되어 있지 않다"는 claim 은 그 조항이 뒷받침한다(pass).
- claim 판정은 뒷받침 여부만 본다 — 근거 사이의 모순은 claim 판정에 반영하지 말고 아래
  모순 감지에만 기록한다.

[모순 감지 — 근거쌍 단위]
- 수집 근거 전체에서, 초안이 인용했는지와 무관하게, 같은 사안에 대해 양립할 수 없는 내용을
  말하는 근거쌍을 전부 찾아 contradictions 에 기록한다.
- 초안이 그 모순을 명시하고 **두 기준을 모두 안내**했다면 그 모순은 기각 사유가 아니다 —
  contradictions 에는 기록하되 reject_reasons 에 contradictory_evidence 를 넣지 않는다.
- 초안이 모순을 명시하지 않은 채 모순 쌍의 어느 한쪽이라도 딛고(인용하고) 답하면
  contradictory_evidence 로 기각한다.

[산출 규칙]
- reject_reasons 에는 unsupported_claim 과 contradictory_evidence 두 값만 올 수 있다.
  - unsupported_claim: reject 인 claim 이 하나라도 있으면 넣고, 없으면 넣지 않는다.
  - contradictory_evidence: 위 기각 조건에 해당하는 모순 쌍이 있으면 넣는다.
- verdict 는 reject_reasons 가 비어 있으면 "pass", 하나라도 있으면 "reject" 다.
- claim_judgments 에는 초안의 claim **전부**를(통과한 claim 포함) 순서대로 넣고, claim_text 는
  초안의 문장을 글자 그대로 쓴다. explanation 은 한국어로 짧게 쓴다.
  **개수가 claim 목록과 정확히 같아야 한다** — 같은 문장이 claim 목록에 두 번 나오면 판정도
  두 번 낸다. 인용 근거가 claim 마다 다를 수 있어 판정도 달라질 수 있다.
- contradictions 의 evidence_id 는 수집 근거 목록의 ID 를 글자 그대로 쓴다.

산출은 다음 형태의 JSON 이다:
{"claim_judgments": [{"claim_text": "...", "verdict": "pass", "explanation": "..."}],
 "contradictions": [{"evidence_id_a": "...", "evidence_id_b": "...", "explanation": "..."}],
 "verdict": "pass", "reject_reasons": []}
"""


# ── 프롬프트 조립 ───────────────────────────────────────────────────────────


def _format_claim_block(index: int, claim: Claim, evidence_by_id: Mapping[str, Evidence]) -> str:
    """claim 1개와 그 claim 이 **인용한** 근거의 원문. 미인용 근거는 싣지 않는다."""
    lines = [f"claim {index}: {claim.text}", "  인용 근거:"]
    if not claim.citation_ids:
        # L1(missing_citation)을 통과한 초안이라 정상 경로에서는 오지 않는다 — 방어적 표기.
        lines.append("  - (인용 근거 없음)")
    for citation_id in claim.citation_ids:
        cited = evidence_by_id.get(citation_id)
        # L1(invalid_citation)을 통과한 초안이라 정상 경로에서는 전부 해석된다 — 방어적 표기.
        text = cited.evidence_text if cited is not None else "(수집 근거 목록에 없는 ID)"
        lines.append(f"  - {citation_id}: {text}")
    return "\n".join(lines)


def build_judge_user_prompt(
    *,
    draft: Draft,
    evidence: Sequence[Evidence],
    previous_output: str | None = None,
    previous_error: str | None = None,
) -> str:
    """판정 프롬프트 — 뒷받침 절(인용 근거만)과 모순 절(수집 근거 전체)을 조립한다.

    재시도면 직전 산출과 실패 사유를 피드백으로 싣는다(의도 해석 패턴).
    """
    evidence_by_id = {item.id: item for item in evidence}
    claim_blocks = [
        _format_claim_block(index, claim, evidence_by_id)
        for index, claim in enumerate(draft.claims, start=1)
    ]
    evidence_blocks = [
        f"- {item.id} ({item.source.value})\n{item.evidence_text}" for item in evidence
    ]
    sections = [
        "[초안 claim 목록 — claim 별 인용 근거 (뒷받침 판정 대상)]\n"
        + ("\n\n".join(claim_blocks) if claim_blocks else "(초안에 claim 이 없다.)"),
        "[수집 근거 전체 — 모순 감지 대상 (인용 여부 무관)]\n"
        + ("\n\n".join(evidence_blocks) if evidence_blocks else "(수집된 근거가 없다.)"),
    ]
    if previous_error is not None:
        lines = ["[직전 산출이 형식에 맞지 않았다]", f"- 사유: {previous_error}"]
        if previous_output:
            lines.append(f"- 직전 산출: {previous_output}")
        lines.append("판정은 다시 하되, 산출은 지정된 JSON 형식 그대로만 낸다.")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


# ── 산출 파싱 (fail-closed) ─────────────────────────────────────────────────


class _ParseError(ValueError):
    """판정 산출이 `JudgeResult` 형식·정합성에 맞지 않음 — 재시도 피드백에 실린다."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _require_str(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _ParseError(f"{field} 는 비어 있지 않은 문자열이어야 한다 (받은 값: {value!r})")
    return value


def _parse_verdict(value: object, *, field: str) -> Verdict:
    if isinstance(value, str):
        try:
            return Verdict(value)
        except ValueError:
            pass
    raise _ParseError(f"{field} 는 pass·reject 중 하나여야 한다 (받은 값: {value!r})")


def _require_items(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise _ParseError(f"{field} 는 배열이어야 한다 (받은 값: {value!r})")
    return value


def _parse_claim_judgments(value: object, *, draft: Draft) -> tuple[ClaimJudgment, ...]:
    """claim 판정 배열 — 초안의 claim 전부와 1:1 로 대응해야 한다(통과한 claim 포함).

    **대조는 집합이 아니라 다중집합(개수 포함)으로 한다.** 초안 claim 의 text 가 유일하다는
    보장이 없기 때문이다 — 초안은 LLM 산출이고 같은 문장이 두 번 들어올 수 있으며, 그때
    두 claim 의 `citation_ids` 는 다를 수 있다(한쪽은 뒷받침되고 한쪽은 아닌 경우가 실제
    시나리오다). 집합으로 대조하면 두 방향으로 다 틀렸다:

    * 판정 1건이 claim 2건을 덮어도 **통과** — 판정받지 못한 claim 이 답변에 실린다.
    * 판정 2건을 정직하게 낸 **옳은 산출이 "중복"으로 거부** — 모델이 맞게 답할 길이 없었다.

    개수까지 맞추면 둘 다 닫힌다. 짝짓기는 여전히 `claim_text` 로 한다 — 배열 위치를 계약으로
    올리지 않는다(docs/contracts.md "층별 판정 키").
    """
    judgments: list[ClaimJudgment] = []
    for position, item in enumerate(_require_items(value, field="claim_judgments"), start=1):
        if not isinstance(item, Mapping):
            raise _ParseError(f"claim_judgments[{position}] 는 객체여야 한다 (받은 값: {item!r})")
        judgments.append(
            ClaimJudgment(
                claim_text=_require_str(
                    item.get("claim_text"), field=f"claim_judgments[{position}].claim_text"
                ),
                verdict=_parse_verdict(
                    item.get("verdict"), field=f"claim_judgments[{position}].verdict"
                ),
                explanation=_require_str(
                    item.get("explanation"), field=f"claim_judgments[{position}].explanation"
                ),
            )
        )

    judged = Counter(judgment.claim_text for judgment in judgments)
    expected = Counter(claim.text for claim in draft.claims)
    if judged != expected:
        # `Counter` 뺄셈은 개수 차이만 남긴다 — 같은 문장이 초안에 2건인데 판정이 1건이면
        # 그 문장이 `missing` 에 1개 남는다(집합 대조에서는 아무것도 남지 않던 자리다).
        missing = sorted((expected - judged).elements())
        unknown = sorted((judged - expected).elements())
        raise _ParseError(
            "claim_judgments 가 초안의 claim 전부와 1:1 로 대응하지 않는다 "
            f"(판정 누락: {missing!r}, 초안에 없거나 개수를 넘긴 판정: {unknown!r})"
        )
    return tuple(judgments)


def _parse_contradictions(
    value: object, *, evidence_ids: frozenset[str]
) -> tuple[EvidenceContradiction, ...]:
    """모순 배열 — 근거쌍 단위이고, ID 는 수집 근거 목록 안에서만 유효하다."""
    contradictions: list[EvidenceContradiction] = []
    for position, item in enumerate(_require_items(value, field="contradictions"), start=1):
        if not isinstance(item, Mapping):
            raise _ParseError(f"contradictions[{position}] 는 객체여야 한다 (받은 값: {item!r})")
        id_a = _require_str(
            item.get("evidence_id_a"), field=f"contradictions[{position}].evidence_id_a"
        )
        id_b = _require_str(
            item.get("evidence_id_b"), field=f"contradictions[{position}].evidence_id_b"
        )
        for label, evidence_id in (("evidence_id_a", id_a), ("evidence_id_b", id_b)):
            if evidence_id not in evidence_ids:
                raise _ParseError(
                    f"contradictions[{position}].{label} 가 수집 근거 목록에 없다: {evidence_id!r}"
                )
        if id_a == id_b:
            raise _ParseError(f"contradictions[{position}] 는 같은 근거끼리의 쌍이다: {id_a!r}")
        contradictions.append(
            EvidenceContradiction(
                evidence_id_a=id_a,
                evidence_id_b=id_b,
                explanation=_require_str(
                    item.get("explanation"), field=f"contradictions[{position}].explanation"
                ),
            )
        )
    return tuple(contradictions)


def _parse_reject_reasons(value: object) -> tuple[RejectReason, ...]:
    """사유 배열 — L2 2종 밖의 값은 거부하고, 순서는 `L2_REJECT_REASONS` 로 정규화한다."""
    allowed = {reason.value: reason for reason in L2_REJECT_REASONS}
    provided: set[RejectReason] = set()
    for item in _require_items(value, field="reject_reasons"):
        if not isinstance(item, str) or item not in allowed:
            raise _ParseError(
                "reject_reasons 에 L2 사유 2종(unsupported_claim·contradictory_evidence) "
                f"밖의 값이 있다: {item!r}"
            )
        provided.add(allowed[item])
    return tuple(reason for reason in L2_REJECT_REASONS if reason in provided)


def _parse_judge_result(data: object, *, draft: Draft, evidence: Sequence[Evidence]) -> JudgeResult:
    """구조화 산출 → `JudgeResult`. 형식·정합성 위반은 전부 `_ParseError` (fail-closed)."""
    if not isinstance(data, Mapping):
        raise _ParseError(f"산출 최상위는 JSON 객체여야 한다 (받은 값: {data!r})")
    missing_keys = sorted(
        {"claim_judgments", "contradictions", "verdict", "reject_reasons"} - set(data.keys())
    )
    if missing_keys:
        raise _ParseError(f"필수 키가 없다: {missing_keys!r}")

    claim_judgments = _parse_claim_judgments(data["claim_judgments"], draft=draft)
    contradictions = _parse_contradictions(
        data["contradictions"], evidence_ids=frozenset(item.id for item in evidence)
    )
    verdict = _parse_verdict(data["verdict"], field="verdict")
    reject_reasons = _parse_reject_reasons(data["reject_reasons"])

    # 정합성 — verdict·사유·세부 배열이 서로 어긋난 판정은 쓸 수 없다.
    if (verdict is Verdict.REJECT) != bool(reject_reasons):
        raise _ParseError(
            f"verdict({verdict.value})와 reject_reasons({[r.value for r in reject_reasons]!r})가 "
            "어긋난다 — 사유가 있으면 reject, 없으면 pass 여야 한다"
        )
    any_rejected_claim = any(judgment.verdict is Verdict.REJECT for judgment in claim_judgments)
    if (RejectReason.UNSUPPORTED_CLAIM in reject_reasons) != any_rejected_claim:
        raise _ParseError(
            "unsupported_claim 사유와 claim 판정이 어긋난다 — reject 인 claim 이 있을 때만, "
            "그리고 있으면 반드시 unsupported_claim 을 넣는다"
        )
    if RejectReason.CONTRADICTORY_EVIDENCE in reject_reasons and not contradictions:
        raise _ParseError("contradictory_evidence 사유가 있는데 모순 근거쌍 기록이 없다")

    return JudgeResult(
        verdict=verdict,
        reject_reasons=reject_reasons,
        claim_judgments=claim_judgments,
        contradictions=contradictions,
    )


# ── 판정기 ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JudgeOutcome:
    """판정 1회분(형식 불일치 재시도 포함)의 결과 + 판정 토큰.

    토큰은 이 판정 호출에서 쓴 전부(재시도 포함 합산)다 — 파이프라인은 이 값을 생성
    토큰과 **분리 집계**한다(docs/contracts.md "토큰 집계 경계"). 생성 합산에 섞으면 안 된다.
    """

    result: JudgeResult
    input_tokens: int
    output_tokens: int
    attempts: int
    #: 이 판정 호출에 흐른 벽시계(ms) — **형식 재시도와 전송 재시도를 포함한 합산**이다.
    #: 토큰과 같은 규칙으로 누적한다: 버려진 시도도 그만큼 시간을 썼다. 실패로 끝나면
    #: 예외(`LLMFormatError`/`LLMCallError`)의 `elapsed_ms` 가 같은 값을 들고 올라간다.
    elapsed_ms: float = 0.0
    #: 캐시에 **쓴**/캐시에서 **읽은** 토큰. 단가가 다르므로 뭉뚱그리지 않는다.
    #: 캐싱을 쓰지 않는 경로에서는 0 이 아니라 `None`(해당 없음/미측정)이다.
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class Judge:
    """초안 1개 + 근거를 받아 1회 배치 판정한다. 파이프라인 제어는 하지 않는다.

    판정 모델은 주입된 `client` 가 정하고, `effort`·`max_output_tokens` 는 인자로 받는다
    (하드코딩 금지 — 기본값 배선은 설정을 아는 쪽의 몫이다).
    """

    def __init__(
        self,
        *,
        client: GenerationClient,
        effort: str | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self._client = client
        self._effort = effort
        self._max_output_tokens = max_output_tokens

    def _complete(self, user: str) -> JsonCompletion:
        """1회 호출. `max_output_tokens` 는 지정했을 때만 보낸다 — 미지정이면 래퍼 기본값."""
        if self._max_output_tokens is None:
            return self._client.complete_json(
                stage=JUDGE_STAGE,
                system=JUDGE_SYSTEM_PROMPT,
                user=user,
                schema=JUDGE_JSON_SCHEMA,
                schema_name="judge",
                effort=self._effort,
            )
        return self._client.complete_json(
            stage=JUDGE_STAGE,
            system=JUDGE_SYSTEM_PROMPT,
            user=user,
            schema=JUDGE_JSON_SCHEMA,
            schema_name="judge",
            effort=self._effort,
            max_output_tokens=self._max_output_tokens,
        )

    def judge(self, *, draft: Draft, evidence: Sequence[Evidence]) -> JudgeOutcome:
        """초안 1개를 수집 근거 전체와 대조해 판정한다.

        형식 불일치는 **코드가 1회만** 재시도하고, 재실패하면 누적 토큰을 실은
        `LLMFormatError` 를 던진다. 전송 오류(`LLMCallError`)는 재시도하지 않고 전파한다 —
        래퍼가 이미 재시도했으므로 여기서 또 재시도하면 상한이 깨진다. 다만 **누적 토큰은
        예외에 실어** 보낸다: 두 실패 모두 실비용이므로 호출자가 같은 방식으로 집계한다.
        """
        input_tokens = 0
        output_tokens = 0
        # 경과도 토큰과 같은 자격으로 누적한다 — 형식이 어긋나 버려진 시도의 시간도
        # 이 구간이 쓴 시간이다. 여기서 마지막 시도만 재면 판정 구간이 실제보다 짧아진다.
        elapsed_ms = 0.0
        # 캐시 계열은 **0 에서 시작하지 않는다** — 한 번도 보고되지 않으면 미측정이다.
        cache_creation: int | None = None
        cache_read: int | None = None
        # 실제로 나간 전송 수. **토큰과 같은 이유로 버리지 않는다** — 앞선 형식 실패의
        # 비용을 세면서 그 시도가 있었다는 사실을 세지 않으면 기록과 실제가 갈린다.
        sent = 0
        error: str | None = None
        previous_output: str | None = None

        for attempt in range(1, JUDGE_MAX_ATTEMPTS + 1):
            user = build_judge_user_prompt(
                draft=draft,
                evidence=evidence,
                previous_output=previous_output,
                previous_error=error,
            )
            try:
                completion = self._complete(user)
            except LLMFormatError as exc:
                input_tokens += exc.input_tokens
                output_tokens += exc.output_tokens
                elapsed_ms += exc.elapsed_ms
                cache_creation = accumulate_optional_tokens(
                    cache_creation, exc.cache_creation_input_tokens
                )
                cache_read = accumulate_optional_tokens(cache_read, exc.cache_read_input_tokens)
                sent += exc.transport_attempts
                error = exc.detail
                previous_output = exc.raw_text or None
                continue
            except LLMCallError as exc:
                # 재시도는 하지 않되 **앞선 시도에서 이미 과금된 토큰**을 버리지 않는다.
                # 새 예외로 다시 던지는 것은 대역이 같은 예외 객체를 재사용해도 누적이
                # 이중으로 실리지 않게 하기 위해서다(원인은 `from exc` 로 잇는다).
                raise LLMCallError(
                    stage=exc.stage,
                    reason=exc.reason,
                    attempts=sent + exc.attempts,
                    cause=exc.cause,
                    input_tokens=input_tokens + exc.input_tokens,
                    output_tokens=output_tokens + exc.output_tokens,
                    cache_creation_input_tokens=accumulate_optional_tokens(
                        cache_creation, exc.cache_creation_input_tokens
                    ),
                    cache_read_input_tokens=accumulate_optional_tokens(
                        cache_read, exc.cache_read_input_tokens
                    ),
                    elapsed_ms=elapsed_ms + exc.elapsed_ms,
                ) from exc

            input_tokens += completion.input_tokens
            output_tokens += completion.output_tokens
            elapsed_ms += completion.elapsed_ms
            cache_creation = accumulate_optional_tokens(
                cache_creation, completion.cache_creation_input_tokens
            )
            cache_read = accumulate_optional_tokens(cache_read, completion.cache_read_input_tokens)
            # `_ParseError` 로 continue 하는 경로에서도 이 전송은 이미 나갔다.
            sent += completion.transport_attempts
            try:
                result = _parse_judge_result(completion.data, draft=draft, evidence=evidence)
            except _ParseError as exc:
                error = exc.detail
                previous_output = repr(completion.data)
                continue
            return JudgeOutcome(
                result=result,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                attempts=attempt,
                elapsed_ms=elapsed_ms,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
            )

        raise LLMFormatError(
            stage=JUDGE_STAGE,
            detail=error or "판정 산출이 형식에 맞지 않았다",
            raw_text=previous_output or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            transport_attempts=sent,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            elapsed_ms=elapsed_ms,
        )
