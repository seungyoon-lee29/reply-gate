"""L2 판정 모듈 단위 테스트 (LLM 호출은 전부 목).

의미 판정 자체는 모델 몫이므로 여기서 검증하는 것은 네 가지다:
(1) 입력 조립 — 뒷받침 절에는 claim 별 **인용한** 근거만, 모순 절에는 **수집 근거 전체**가 실리는지,
(2) 구조화 산출 파싱 — `JudgeResult` 변환과 fail-closed 검증(사유 2종 밖 값·정합성 위반 거부),
(3) 재시도·전파 정책 — 형식 불일치 1회 재시도, 재실패 예외, 전송 오류 전파,
(4) 판정 토큰이 결과에 그대로 노출되는지(생성 합산과 분리 집계하는 파이프라인의 전제).
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any, cast

import pytest

from reply_gate.contracts import (
    Claim,
    Draft,
    Evidence,
    EvidenceSource,
    RejectReason,
    Verdict,
)
from reply_gate.judge import (
    JUDGE_JSON_SCHEMA,
    JUDGE_STAGE,
    JUDGE_SYSTEM_PROMPT,
    L2_REJECT_REASONS,
    Judge,
    build_judge_user_prompt,
)
from reply_gate.llm import GenerationClient, JsonCompletion, LLMCallError, LLMFormatError

# ── 픽스처 ──────────────────────────────────────────────────────────────────

DOMESTIC = Evidence(
    id="policy:shipping-policy:3",
    source=EvidenceSource.POLICY,
    content="[배송 정책 제3조 국내 배송] 국내 배송은 출고일로부터 영업일 기준 3일 이내에 도착한다.",
    evidence_text="[배송 정책 제3조 국내 배송] 국내 배송은 출고일로부터 영업일 기준 3일 이내에 도착한다.",
)

OVERSEAS_ABSENT = Evidence(
    id="policy:shipping-policy:7",
    source=EvidenceSource.POLICY,
    content="[배송 정책 제7조 해외 배송] 해외 배송 요금은 본 정책 문서에 기재하지 않는다.",
    evidence_text="[배송 정책 제7조 해외 배송] 해외 배송 요금은 본 정책 문서에 기재하지 않는다.",
)

REFUND_7 = Evidence(
    id="policy:refund-policy:2",
    source=EvidenceSource.POLICY,
    content="[환불 정책 제2조 환불 기간] 환불 신청은 상품 수령일로부터 7일 이내에 할 수 있다.",
    evidence_text="[환불 정책 제2조 환불 기간] 환불 신청은 상품 수령일로부터 7일 이내에 할 수 있다.",
)

REFUND_30 = Evidence(
    id="policy:faq:12",
    source=EvidenceSource.POLICY,
    content="[FAQ 12] 단순 변심 환불은 상품 수령일로부터 30일까지 접수된다.",
    evidence_text="[FAQ 12] 단순 변심 환불은 상품 수령일로부터 30일까지 접수된다.",
)

SQL_STATUS = Evidence(
    id="sql:INQ-1:1",
    source=EvidenceSource.SQL,
    content="실행 쿼리: SELECT ...\n결과 1건\n1) order_no=ORD-20260101-0001, status=배송중",
    evidence_text="실행 쿼리: SELECT ...\n결과 1건\n1) order_no=ORD-20260101-0001, status=배송중",
)


def _draft(*claims: tuple[str, tuple[str, ...]]) -> Draft:
    return Draft(claims=tuple(Claim(text=text, citation_ids=ids) for text, ids in claims))


def _completion(data: Any, input_tokens: int = 0, output_tokens: int = 0) -> JsonCompletion:
    return JsonCompletion(data=data, input_tokens=input_tokens, output_tokens=output_tokens)


def _all_pass_payload(draft: Draft) -> dict[str, Any]:
    """초안의 모든 claim 을 통과시키는 정합적인 판정 산출."""
    return {
        "claim_judgments": [
            {"claim_text": claim.text, "verdict": "pass", "explanation": "인용 근거가 뒷받침한다."}
            for claim in draft.claims
        ],
        "contradictions": [],
        "verdict": "pass",
        "reject_reasons": [],
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


def _judge(outcomes: list[Any], **kwargs: Any) -> tuple[Judge, _RecordingClient]:
    recorder = _RecordingClient(outcomes)
    return Judge(client=cast(GenerationClient, recorder), **kwargs), recorder


# ── 의미 정책 4분면 — 파싱·판정 (목이 구조화 응답을 돌려주는 형태) ──────────


def test_주제_인접_인용은_unsupported_claim_기각으로_파싱된다() -> None:
    """해외 배송 문의에 국내 배송 조항을 인용한 claim — 주제를 다루지 않으므로 기각."""
    draft = _draft(("해외 배송 소요 기간은 안내가 어렵습니다.", (DOMESTIC.id,)))
    payload = {
        "claim_judgments": [
            {
                "claim_text": draft.claims[0].text,
                "verdict": "reject",
                "explanation": "국내 배송 조항은 해외 배송이라는 주제를 다루지 않는다.",
            }
        ],
        "contradictions": [],
        "verdict": "reject",
        "reject_reasons": ["unsupported_claim"],
    }
    judge, _ = _judge([_completion(payload)])

    outcome = judge.judge(draft=draft, evidence=(DOMESTIC,))

    assert outcome.result.verdict is Verdict.REJECT
    assert outcome.result.reject_reasons == (RejectReason.UNSUPPORTED_CLAIM,)
    assert len(outcome.result.claim_judgments) == 1
    judgment = outcome.result.claim_judgments[0]
    assert judgment.claim_text == draft.claims[0].text
    assert judgment.verdict is Verdict.REJECT
    assert "해외 배송" in judgment.explanation


def test_정면_조항의_없다_안내는_통과로_파싱된다() -> None:
    """주제를 정면으로 다루며 값만 비어 있는 조항 — '기재되어 있지 않다' claim 은 통과."""
    draft = _draft(("해외 배송 요금은 정책에 기재되어 있지 않습니다.", (OVERSEAS_ABSENT.id,)))
    judge, _ = _judge([_completion(_all_pass_payload(draft))])

    outcome = judge.judge(draft=draft, evidence=(OVERSEAS_ABSENT,))

    assert outcome.result.verdict is Verdict.PASS
    assert outcome.result.reject_reasons == ()
    assert outcome.result.claim_judgments[0].verdict is Verdict.PASS


def test_모순_비명시_기각은_근거쌍과_사유로_파싱된다() -> None:
    """상충하는 두 조항 중 한쪽만 딛고 답한 초안 — contradictory_evidence 기각."""
    draft = _draft(("환불 신청은 상품 수령일로부터 7일 이내에 가능합니다.", (REFUND_7.id,)))
    payload = _all_pass_payload(draft)
    payload["contradictions"] = [
        {
            "evidence_id_a": REFUND_7.id,
            "evidence_id_b": REFUND_30.id,
            "explanation": "환불 가능 기간이 7일과 30일로 상충한다.",
        }
    ]
    payload["verdict"] = "reject"
    payload["reject_reasons"] = ["contradictory_evidence"]
    judge, _ = _judge([_completion(payload)])

    outcome = judge.judge(draft=draft, evidence=(REFUND_7, REFUND_30))

    assert outcome.result.verdict is Verdict.REJECT
    assert outcome.result.reject_reasons == (RejectReason.CONTRADICTORY_EVIDENCE,)
    assert len(outcome.result.contradictions) == 1
    contradiction = outcome.result.contradictions[0]
    assert {contradiction.evidence_id_a, contradiction.evidence_id_b} == {REFUND_7.id, REFUND_30.id}
    assert "상충" in contradiction.explanation


def test_모순을_명시하고_두_기준을_안내한_초안은_통과한다() -> None:
    """모순 쌍은 contradictions 에 기록되지만 기각 사유는 아니다 — verdict pass 가 유효하다."""
    draft = _draft(
        (
            "환불 기간은 문서에 따라 7일과 30일로 다르게 안내되어 있어, 두 기준을 함께 안내드립니다.",
            (REFUND_7.id, REFUND_30.id),
        )
    )
    payload = _all_pass_payload(draft)
    payload["contradictions"] = [
        {
            "evidence_id_a": REFUND_7.id,
            "evidence_id_b": REFUND_30.id,
            "explanation": "환불 가능 기간이 상충하지만 초안이 모순을 명시하고 두 기준을 모두 안내했다.",
        }
    ]
    judge, _ = _judge([_completion(payload)])

    outcome = judge.judge(draft=draft, evidence=(REFUND_7, REFUND_30))

    assert outcome.result.verdict is Verdict.PASS
    assert outcome.result.reject_reasons == ()
    # 모순 기록은 남는다 — 통과와 모순 기록은 양립한다
    # (docs/business-rules.md "모순 판정 — 근거쌍 단위").
    assert len(outcome.result.contradictions) == 1


# ── 양성 대조 ───────────────────────────────────────────────────────────────


def test_양성_대조_정상_초안은_전부_통과한다() -> None:
    """전부 기각하는 판정기는 판정기가 아니다 — 근거를 그대로 옮긴 정상 초안의 통과 경로."""
    draft = _draft(
        ("주문하신 상품은 현재 배송중입니다.", (SQL_STATUS.id,)),
        ("국내 배송은 출고일로부터 영업일 기준 3일 이내에 도착합니다.", (DOMESTIC.id,)),
    )
    judge, recorder = _judge(
        [_completion(_all_pass_payload(draft), input_tokens=200, output_tokens=80)]
    )

    outcome = judge.judge(draft=draft, evidence=(SQL_STATUS, DOMESTIC))

    assert outcome.result.verdict is Verdict.PASS
    assert outcome.result.reject_reasons == ()
    assert outcome.result.contradictions == ()
    assert [j.verdict for j in outcome.result.claim_judgments] == [Verdict.PASS, Verdict.PASS]
    # claim 이 여럿이어도 판정 호출은 시도당 1회 배치다
    # (docs/business-rules.md "L2 판정 규칙").
    assert len(recorder.calls) == 1


# ── 입력 조립 ───────────────────────────────────────────────────────────────


def test_뒷받침_절에는_claim_이_인용한_근거만_실린다() -> None:
    draft = _draft(("환불 신청은 상품 수령일로부터 7일 이내에 가능합니다.", (REFUND_7.id,)))

    prompt = build_judge_user_prompt(draft=draft, evidence=(REFUND_7, REFUND_30, SQL_STATUS))

    support_section, _, contradiction_section = prompt.partition("[수집 근거 전체")
    assert draft.claims[0].text in support_section
    assert REFUND_7.evidence_text in support_section
    # 인용하지 않은 근거는 뒷받침 절에 실리지 않는다.
    assert REFUND_30.evidence_text not in support_section
    assert SQL_STATUS.evidence_text not in support_section
    # 모순 절에는 인용 여부와 무관하게 수집 근거 전체가 실린다.
    for item in (REFUND_7, REFUND_30, SQL_STATUS):
        assert item.id in contradiction_section
        assert item.evidence_text in contradiction_section


def test_미인용_근거만으로_이뤄진_모순도_검출된다() -> None:
    """모순 입력이 수집 근거 전체라야 미인용 쌍의 모순이 잡힌다.

    docs/business-rules.md "모순 판정 — 근거쌍 단위".
    """
    draft = _draft(("주문하신 상품은 현재 배송중입니다.", (SQL_STATUS.id,)))
    payload = _all_pass_payload(draft)
    payload["contradictions"] = [
        {
            "evidence_id_a": REFUND_7.id,
            "evidence_id_b": REFUND_30.id,
            "explanation": "환불 가능 기간이 7일과 30일로 상충한다.",
        }
    ]
    judge, recorder = _judge([_completion(payload)])

    outcome = judge.judge(draft=draft, evidence=(SQL_STATUS, REFUND_7, REFUND_30))

    # 초안이 인용하지 않은 근거쌍의 모순이 결과에 실린다.
    contradiction = outcome.result.contradictions[0]
    assert {contradiction.evidence_id_a, contradiction.evidence_id_b} == {REFUND_7.id, REFUND_30.id}
    # 그 전제: 미인용 근거의 원문이 판정 프롬프트에 들어갔다.
    user_prompt = recorder.calls[0]["user"]
    assert REFUND_7.evidence_text in user_prompt
    assert REFUND_30.evidence_text in user_prompt


# ── 재시도·전파 정책 ────────────────────────────────────────────────────────


def test_형식_불일치는_피드백을_실어_1회_재시도한다() -> None:
    draft = _draft(("해외 배송 요금은 정책에 기재되어 있지 않습니다.", (OVERSEAS_ABSENT.id,)))
    error = LLMFormatError(
        stage=JUDGE_STAGE,
        detail="JSON 파싱 실패",
        raw_text="이건 JSON 이 아니다",
        input_tokens=30,
        output_tokens=5,
    )
    judge, recorder = _judge(
        [error, _completion(_all_pass_payload(draft), input_tokens=40, output_tokens=10)]
    )

    outcome = judge.judge(draft=draft, evidence=(OVERSEAS_ABSENT,))

    assert outcome.result.verdict is Verdict.PASS
    assert outcome.attempts == 2
    assert len(recorder.calls) == 2
    # 재시도 프롬프트에 직전 실패의 사유와 산출이 피드백으로 실린다 (의도 해석 패턴).
    retry_prompt = recorder.calls[1]["user"]
    assert "형식에 맞지 않았다" in retry_prompt
    assert "JSON 파싱 실패" in retry_prompt
    assert "이건 JSON 이 아니다" in retry_prompt
    # 두 시도의 토큰이 합산되어 노출된다.
    assert (outcome.input_tokens, outcome.output_tokens) == (70, 15)


def test_형식_재실패는_토큰을_실어_예외를_위로_던진다() -> None:
    draft = _draft(("해외 배송 요금은 정책에 기재되어 있지 않습니다.", (OVERSEAS_ABSENT.id,)))
    error = LLMFormatError(
        stage=JUDGE_STAGE, detail="빈 응답", raw_text="", input_tokens=25, output_tokens=0
    )
    judge, recorder = _judge([error, error])

    with pytest.raises(LLMFormatError) as excinfo:
        judge.judge(draft=draft, evidence=(OVERSEAS_ABSENT,))

    # 최초 + 재시도 1회 = 2회에서 멈춘다. 예외 매핑(llm_call_failed)은 호출자 몫이다.
    assert len(recorder.calls) == 2
    assert excinfo.value.stage == JUDGE_STAGE
    assert (excinfo.value.input_tokens, excinfo.value.output_tokens) == (50, 0)


def test_의미상_형식_오류도_같은_재시도를_탄다() -> None:
    """JSON 으로는 유효하지만 판정 형식이 아닌 산출 — 같은 1회 재시도 경로다."""
    draft = _draft(("해외 배송 요금은 정책에 기재되어 있지 않습니다.", (OVERSEAS_ABSENT.id,)))
    judge, recorder = _judge(
        [
            _completion({"claims": []}, input_tokens=10, output_tokens=2),
            _completion(_all_pass_payload(draft), input_tokens=20, output_tokens=8),
        ]
    )

    outcome = judge.judge(draft=draft, evidence=(OVERSEAS_ABSENT,))

    assert outcome.result.verdict is Verdict.PASS
    assert outcome.attempts == 2
    assert len(recorder.calls) == 2
    assert "형식에 맞지 않았다" in recorder.calls[1]["user"]
    assert (outcome.input_tokens, outcome.output_tokens) == (30, 10)


def test_전송_오류는_재시도_없이_그대로_전파된다() -> None:
    """전송 오류는 래퍼가 이미 1회 재시도했다 — 이 모듈이 또 재시도하면 상한이 깨진다."""
    draft = _draft(("해외 배송 요금은 정책에 기재되어 있지 않습니다.", (OVERSEAS_ABSENT.id,)))
    judge, recorder = _judge(
        [LLMCallError(stage=JUDGE_STAGE, reason="transport_error", attempts=2)]
    )

    with pytest.raises(LLMCallError) as excinfo:
        judge.judge(draft=draft, evidence=(OVERSEAS_ABSENT,))

    assert excinfo.value.stage == JUDGE_STAGE
    assert len(recorder.calls) == 1
    # 앞선 시도가 없었으므로 실을 토큰도 없다 (양성 대조 — 아래 회귀 테스트의 짝).
    assert (excinfo.value.input_tokens, excinfo.value.output_tokens) == (0, 0)


def test_형식_실패_뒤_전송_오류는_누적_토큰을_실어_전파한다() -> None:
    """1회차가 200 으로 과금된 뒤 2회차가 전송 오류로 죽는 조합 — 누적분을 버리지 않는다.

    규칙은 "실행됐으나 실패한 호출의 토큰도 그대로 집계한다"(docs/contracts.md
    "토큰 집계 경계")이다.
    여기서 버리면 실제로 과금된 판정 호출의 값이 파이프라인 밖으로 나가지 못해, 처리
    기록·API 응답이 판정 토큰을 싣는 뒤 태스크 시점에 실비용이 0 으로 굳는다.
    """
    draft = _draft(("해외 배송 요금은 정책에 기재되어 있지 않습니다.", (OVERSEAS_ABSENT.id,)))
    format_error = LLMFormatError(
        stage=JUDGE_STAGE,
        detail="JSON 파싱 실패",
        raw_text="이건 JSON 이 아니다",
        input_tokens=30,
        output_tokens=5,
    )
    transport_error = LLMCallError(stage=JUDGE_STAGE, reason="transport_error", attempts=2)
    judge, recorder = _judge([format_error, transport_error])

    with pytest.raises(LLMCallError) as excinfo:
        judge.judge(draft=draft, evidence=(OVERSEAS_ABSENT,))

    assert len(recorder.calls) == 2
    assert excinfo.value.stage == JUDGE_STAGE
    assert excinfo.value.reason == "transport_error"
    assert (excinfo.value.input_tokens, excinfo.value.output_tokens) == (30, 5)
    # 원본 예외는 원인으로 남는다 — 새로 던지는 것은 토큰을 싣기 위해서다.
    assert excinfo.value.__cause__ is transport_error


# ── fail-closed 검증 — 사유 2종 밖 값·정합성 위반 거부 ──────────────────────


def test_L2_사유_2종_밖_값은_거부된다() -> None:
    """L1 사유든 지어낸 값이든 — L2 의 사유는 unsupported_claim·contradictory_evidence 가 전부다."""
    draft = _draft(("해외 배송 소요 기간은 안내가 어렵습니다.", (DOMESTIC.id,)))
    payload = {
        "claim_judgments": [
            {
                "claim_text": draft.claims[0].text,
                "verdict": "reject",
                "explanation": "뒷받침되지 않는다.",
            }
        ],
        "contradictions": [],
        "verdict": "reject",
        "reject_reasons": ["pii_detected"],
    }
    judge, recorder = _judge([_completion(payload)])

    with pytest.raises(LLMFormatError) as excinfo:
        judge.judge(draft=draft, evidence=(DOMESTIC,))

    assert len(recorder.calls) == 2
    assert "pii_detected" in excinfo.value.detail


def _base_reject_payload(draft: Draft) -> dict[str, Any]:
    """정합적인 unsupported_claim 기각 산출 — 각 케이스가 이것을 한 군데씩 망가뜨린다."""
    return {
        "claim_judgments": [
            {
                "claim_text": draft.claims[0].text,
                "verdict": "reject",
                "explanation": "뒷받침되지 않는다.",
            }
        ],
        "contradictions": [],
        "verdict": "reject",
        "reject_reasons": ["unsupported_claim"],
    }


def _break_verdict_pass_with_reasons(payload: dict[str, Any]) -> None:
    payload["verdict"] = "pass"


def _break_verdict_reject_without_reasons(payload: dict[str, Any]) -> None:
    payload["claim_judgments"][0]["verdict"] = "pass"
    payload["reject_reasons"] = []


def _break_unsupported_without_rejected_claim(payload: dict[str, Any]) -> None:
    payload["claim_judgments"][0]["verdict"] = "pass"


def _break_rejected_claim_without_unsupported(payload: dict[str, Any]) -> None:
    payload["contradictions"] = [
        {"evidence_id_a": REFUND_7.id, "evidence_id_b": REFUND_30.id, "explanation": "상충."}
    ]
    payload["reject_reasons"] = ["contradictory_evidence"]


def _break_contradictory_without_pairs(payload: dict[str, Any]) -> None:
    payload["reject_reasons"] = ["unsupported_claim", "contradictory_evidence"]


def _break_unknown_evidence_id(payload: dict[str, Any]) -> None:
    payload["contradictions"] = [
        {"evidence_id_a": REFUND_7.id, "evidence_id_b": "policy:ghost:1", "explanation": "상충."}
    ]
    payload["reject_reasons"] = ["unsupported_claim", "contradictory_evidence"]


def _break_self_contradiction(payload: dict[str, Any]) -> None:
    payload["contradictions"] = [
        {"evidence_id_a": REFUND_7.id, "evidence_id_b": REFUND_7.id, "explanation": "자기 자신."}
    ]
    payload["reject_reasons"] = ["unsupported_claim", "contradictory_evidence"]


def _break_missing_claim_judgment(payload: dict[str, Any]) -> None:
    payload["claim_judgments"] = []
    payload["reject_reasons"] = []
    payload["verdict"] = "pass"


def _break_unknown_claim_text(payload: dict[str, Any]) -> None:
    payload["claim_judgments"].append(
        {"claim_text": "초안에 없는 문장이다.", "verdict": "pass", "explanation": "?"}
    )


def _break_duplicate_claim_judgment(payload: dict[str, Any]) -> None:
    """초안에 1건뿐인 claim 을 두 번 판정한 산출 — 개수가 초안을 넘는다.

    짝짓기는 `claim_text` 로 한다(docs/contracts.md "층별 판정 키"). 초안에 없는 개수만큼
    실리면 어느 판정이 그 문장의 것인지 정할 수 없고, 두 판정이 엇갈리면(pass/reject)
    "어느 문장이 왜 기각됐는지"가 화면과 재생성 피드백에서 갈린다.
    """
    duplicate = dict(payload["claim_judgments"][0])
    duplicate["verdict"] = "pass"
    duplicate["explanation"] = "같은 문장에 대한 두 번째 판정."
    payload["claim_judgments"].append(duplicate)


def _break_unknown_verdict_value(payload: dict[str, Any]) -> None:
    payload["claim_judgments"][0]["verdict"] = "maybe"


def _break_missing_required_key(payload: dict[str, Any]) -> None:
    del payload["verdict"]


@pytest.mark.parametrize(
    "mutate",
    [
        _break_verdict_pass_with_reasons,
        _break_verdict_reject_without_reasons,
        _break_unsupported_without_rejected_claim,
        _break_rejected_claim_without_unsupported,
        _break_contradictory_without_pairs,
        _break_unknown_evidence_id,
        _break_self_contradiction,
        _break_missing_claim_judgment,
        _break_unknown_claim_text,
        _break_duplicate_claim_judgment,
        _break_unknown_verdict_value,
        _break_missing_required_key,
    ],
    ids=lambda fn: fn.__name__.removeprefix("_break_"),
)
def test_정합성이_깨진_판정_산출은_거부된다(mutate: Any) -> None:
    """해석하지 못한 산출은 통과가 아니라 거부다 (fail-closed) — 재시도까지 실패하면 예외."""
    draft = _draft(("환불 신청은 상품 수령일로부터 7일 이내에 가능합니다.", (REFUND_7.id,)))
    payload = _base_reject_payload(draft)
    mutate(payload)
    judge, recorder = _judge([_completion(payload)])

    with pytest.raises(LLMFormatError):
        judge.judge(draft=draft, evidence=(REFUND_7, REFUND_30))

    assert len(recorder.calls) == 2


# ── 같은 text 의 claim 이 둘일 때 (초안은 LLM 산출이라 text 가 유일하지 않다) ──


def _repeated_claim_draft() -> Draft:
    """같은 문장을 두 번 담은 초안 — **인용 근거가 다르다.**

    한쪽은 실제로 뒷받침되고 다른 쪽은 무관한 조항을 딛는다. 두 claim 의 판정이 갈릴 수
    있으므로 하나로 접으면 판정받지 못한 문장이 답변에 실린다.
    """
    text = "환불 신청은 상품 수령일로부터 7일 이내에 가능합니다."
    return _draft((text, (REFUND_7.id,)), (text, (REFUND_30.id,)))


def test_같은_text_의_claim_이_둘이면_판정도_둘이어야_한다() -> None:
    """판정 1건이 claim 2건을 덮으면 통과시키지 않는다 — 커버리지가 조용히 절반이 된다."""
    draft = _repeated_claim_draft()
    covered_once = {
        "claim_judgments": [
            {
                "claim_text": draft.claims[0].text,
                "verdict": "pass",
                "explanation": "근거가 뒷받침한다.",
            }
        ],
        "contradictions": [],
        "verdict": "pass",
        "reject_reasons": [],
    }
    judge, recorder = _judge([_completion(covered_once)])

    with pytest.raises(LLMFormatError) as caught:
        judge.judge(draft=draft, evidence=(REFUND_7, REFUND_30))

    # 재시도 피드백이 무엇이 빠졌는지 말해야 모델이 고칠 수 있다.
    assert "판정 누락" in str(caught.value)
    assert len(recorder.calls) == 2


def test_같은_text_의_claim_둘에_판정_둘을_내면_수용된다() -> None:
    """옳은 산출이 '중복' 으로 거부되면 모델이 맞게 답할 길이 없다 — 반대 방향 고정."""
    draft = _repeated_claim_draft()
    text = draft.claims[0].text
    payload = {
        "claim_judgments": [
            {"claim_text": text, "verdict": "pass", "explanation": "환불 기간 조항이 뒷받침한다."},
            {"claim_text": text, "verdict": "reject", "explanation": "FAQ 12 는 30일이라 다르다."},
        ],
        "contradictions": [],
        "verdict": "reject",
        "reject_reasons": ["unsupported_claim"],
    }
    judge, recorder = _judge([_completion(payload)])

    outcome = judge.judge(draft=draft, evidence=(REFUND_7, REFUND_30))

    assert len(outcome.result.claim_judgments) == len(draft.claims)
    assert [item.verdict for item in outcome.result.claim_judgments] == [
        Verdict.PASS,
        Verdict.REJECT,
    ]
    assert len(recorder.calls) == 1  # 재시도 없이 한 번에 받았다


def test_판정_개수가_초안을_넘으면_거부된다() -> None:
    """같은 text 라도 초안 개수보다 많으면 짝지을 수 없다 — 위쪽 경계."""
    draft = _repeated_claim_draft()
    text = draft.claims[0].text
    payload = {
        "claim_judgments": [
            {"claim_text": text, "verdict": "pass", "explanation": f"{index}번째."}
            for index in "가나다"
        ],
        "contradictions": [],
        "verdict": "pass",
        "reject_reasons": [],
    }
    judge, _ = _judge([_completion(payload)])

    with pytest.raises(LLMFormatError) as caught:
        judge.judge(draft=draft, evidence=(REFUND_7, REFUND_30))

    assert "개수를 넘긴 판정" in str(caught.value)


def test_프롬프트가_같은_문장의_반복을_어떻게_다룰지_알려준다() -> None:
    """검사만 조이고 지시를 안 주면 모델은 계속 1건만 낸다 — 재시도 2회를 태우고 인계된다."""
    draft = _repeated_claim_draft()
    judge, recorder = _judge([_completion(_all_pass_payload(draft))])

    judge.judge(draft=draft, evidence=(REFUND_7, REFUND_30))

    system_prompt = recorder.calls[0]["system"]
    assert "개수가 claim 목록과 정확히 같아야 한다" in system_prompt
    assert "두 번 나오면 판정도" in system_prompt


# ── 토큰 노출·호출 인자 ─────────────────────────────────────────────────────


def test_판정_토큰이_결과에_그대로_노출된다() -> None:
    """파이프라인이 생성 토큰과 분리 집계하는 전제 (docs/contracts.md "토큰 집계 경계")."""
    draft = _draft(("해외 배송 요금은 정책에 기재되어 있지 않습니다.", (OVERSEAS_ABSENT.id,)))
    judge, _ = _judge([_completion(_all_pass_payload(draft), input_tokens=321, output_tokens=87)])

    outcome = judge.judge(draft=draft, evidence=(OVERSEAS_ABSENT,))

    assert (outcome.input_tokens, outcome.output_tokens) == (321, 87)
    assert outcome.attempts == 1


def test_호출은_judge_스테이지와_판정_스키마를_쓴다() -> None:
    draft = _draft(("해외 배송 요금은 정책에 기재되어 있지 않습니다.", (OVERSEAS_ABSENT.id,)))
    judge, recorder = _judge([_completion(_all_pass_payload(draft))])

    judge.judge(draft=draft, evidence=(OVERSEAS_ABSENT,))

    call = recorder.calls[0]
    assert call["stage"] == JUDGE_STAGE
    assert call["system"] == JUDGE_SYSTEM_PROMPT
    assert call["schema"] is JUDGE_JSON_SCHEMA


def test_effort_와_max_output_tokens_는_지정했을때만_전달된다() -> None:
    draft = _draft(("해외 배송 요금은 정책에 기재되어 있지 않습니다.", (OVERSEAS_ABSENT.id,)))
    default_judge, default_recorder = _judge([_completion(_all_pass_payload(draft))])
    default_judge.judge(draft=draft, evidence=(OVERSEAS_ABSENT,))
    assert default_recorder.calls[0]["effort"] is None
    assert "max_output_tokens" not in default_recorder.calls[0]

    tuned_judge, tuned_recorder = _judge(
        [_completion(_all_pass_payload(draft))], effort="low", max_output_tokens=16000
    )
    tuned_judge.judge(draft=draft, evidence=(OVERSEAS_ABSENT,))
    assert tuned_recorder.calls[0]["effort"] == "low"
    assert tuned_recorder.calls[0]["max_output_tokens"] == 16000


# ── 계약·구조 고정 ──────────────────────────────────────────────────────────


def test_판정_스키마의_사유_enum_은_L2_2종뿐이다() -> None:
    reasons_schema = JUDGE_JSON_SCHEMA["properties"]["reject_reasons"]["items"]
    assert reasons_schema["enum"] == ["unsupported_claim", "contradictory_evidence"]
    assert [reason.value for reason in L2_REJECT_REASONS] == [
        "unsupported_claim",
        "contradictory_evidence",
    ]
    verdict_schema = JUDGE_JSON_SCHEMA["properties"]["verdict"]
    assert verdict_schema["enum"] == [verdict.value for verdict in Verdict]
    assert JUDGE_JSON_SCHEMA["additionalProperties"] is False


def test_시스템_프롬프트는_의미_정책_4분면을_담는다() -> None:
    # 주제 인접 인용 기각 / 정면 조항의 "없다" 통과 / 모순 명시 + 두 기준 안내 통과 /
    # 비명시 기각 — docs/business-rules.md "뒷받침 판정 — claim 단위"·"모순 판정 —
    # 근거쌍 단위" 의 의미 정책이 지시로 옮겨져 있어야 한다.
    assert "주제 인접 인용은 기각" in JUDGE_SYSTEM_PROMPT
    assert "정면으로 다루" in JUDGE_SYSTEM_PROMPT
    assert "기재되어 있지 않다" in JUDGE_SYSTEM_PROMPT
    assert "두 기준을 모두 안내" in JUDGE_SYSTEM_PROMPT
    assert "unsupported_claim" in JUDGE_SYSTEM_PROMPT
    assert "contradictory_evidence" in JUDGE_SYSTEM_PROMPT
    # 판정 범위 제한 — 문의-답변 관련성은 L2 의 일이 아니다.
    assert "판정 범위 밖" in JUDGE_SYSTEM_PROMPT


def test_judge_는_gate_를_import_하지_않는다() -> None:
    """L2 는 L1 과 독립이다 — 경계 위반은 구조 테스트로 못박는다."""
    import reply_gate.judge as judge_module

    assert judge_module.__file__ is not None
    source = pathlib.Path(judge_module.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            assert all(alias.name != "reply_gate.gate" for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            assert "gate" not in node.module.split(".")
