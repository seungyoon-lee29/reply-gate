"""판정 프롬프트 캐싱 — 캐시 계열 토큰 수집 배선과 스위치 (외부 호출 없음, 목 사용).

**이 파일이 지키는 것은 절감 주장의 전제다.** 캐싱을 켜면 Anthropic 의 `usage.input_tokens`
는 캐시 적중분을 **제외한** 값이 된다. 캐시 계열 두 필드를 읽지 않으면 캐시 적중이 "판정
입력 토큰이 줄었다"로 보이고, 그건 절감이 아니라 **조용한 토큰 은폐**다. 그래서 여기서
고정하는 것은 "수집하는 코드가 있다"가 아니라 **"캐시 필드가 있는 응답의 값이 평가 리포트
까지 간다"** 이고, 반대로 **필드가 없는 응답은 0 이 아니라 미측정으로 남는다**는 것이다
(미실행·미측정을 0 으로 채우지 않는 저장소 규칙 — `scripts/AGENTS.md` 불변식 5).

캐싱은 **호출 구성**이지 지침 변경이 아니다 — 프롬프트 문면이 그대로인 것도 여기서 본다.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import anthropic
import pytest
from scripts import evaluate

import reply_gate.pipeline as pipeline_module
from reply_gate.config import Settings
from reply_gate.contracts import (
    Claim,
    ClaimJudgment,
    Draft,
    Evidence,
    EvidenceSource,
    JudgeResult,
    RejectReason,
    Verdict,
)
from reply_gate.evaluation import (
    JudgeAccuracy,
    RunConditions,
    SkippedMeasurement,
    build_report,
    load_judge_fixtures,
    load_l1_fixtures,
    measure_gate_accuracy,
    measure_judge_accuracy,
    render_markdown,
    report_to_json,
)
from reply_gate.judge import JUDGE_SYSTEM_PROMPT, Judge, JudgeOutcome
from reply_gate.llm import (
    AnthropicGenerationClient,
    LLMCallError,
    LLMFormatError,
    OpenAIGenerationClient,
    accumulate_optional_tokens,
)
from reply_gate.pipeline import build_judge

_ROOT = pathlib.Path(__file__).resolve().parents[1]

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}},
    "required": ["verdict"],
    "additionalProperties": False,
}


# ── 대역 ────────────────────────────────────────────────────────────────────


class _RecordingCalls:
    """SDK 엔드포인트 대역 — 호출 인자를 모으고 정해진 결과를 순서대로 돌려준다."""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self._outcomes[min(len(self.calls) - 1, len(self._outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _usage(
    *,
    input_tokens: int = 11,
    output_tokens: int = 7,
    cache_creation: int | None = None,
    cache_read: int | None = None,
) -> SimpleNamespace:
    """usage 대역. **캐시 계열 키는 `None` 을 주면 아예 없는 응답**이 된다.

    캐싱을 켜지 않은 provider·옛 응답에는 그 필드가 없다 — 그때 0 으로 접으면 "캐시가
    0 토큰 적중했다"와 "캐시 얘기를 한 적이 없다"가 구분되지 않는다.
    """
    fields: dict[str, Any] = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if cache_creation is not None:
        fields["cache_creation_input_tokens"] = cache_creation
    if cache_read is not None:
        fields["cache_read_input_tokens"] = cache_read
    return SimpleNamespace(**fields)


def _anthropic_response(
    text: str,
    *,
    stop_reason: str = "end_turn",
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    content: list[SimpleNamespace] = []
    if stop_reason != "refusal":
        content = [
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ]
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=_usage() if usage is None else usage,
    )


def _anthropic_client(
    outcomes: list[Any], *, prompt_caching: bool = False
) -> tuple[AnthropicGenerationClient, _RecordingCalls]:
    messages = _RecordingCalls(outcomes)
    fake_sdk = cast(anthropic.Anthropic, SimpleNamespace(messages=messages))
    client = AnthropicGenerationClient(
        api_key="test",
        model="claude-sonnet-5",
        client=fake_sdk,
        prompt_caching=prompt_caching,
    )
    return client, messages


def _openai_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        output=[SimpleNamespace(content=[SimpleNamespace(type="output_text", text=text)])],
        output_text=text,
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )


# ── 계약 A — 호출 구성: 고정 프리픽스에만 캐싱을 건다 ────────────────────────


def test_캐싱_꺼짐은_고정_프리픽스를_문자열_그대로_보낸다() -> None:
    """음성 대조 — 스위치가 꺼진 실행의 요청 모양이 바뀌면 기준선 실측과 조건이 갈린다."""
    client, messages = _anthropic_client([_anthropic_response(json.dumps({"verdict": "pass"}))])

    client.complete_json(stage="judge", system="s", user="u", schema=SCHEMA)

    call = messages.calls[0]
    assert call["system"] == "s"
    assert "cache_control" not in json.dumps(call, default=str)


def test_캐싱_켜짐은_고정_프리픽스에만_cache_control_을_붙인다() -> None:
    """캐싱은 **호출 구성**이다 — 문면은 한 글자도 바뀌지 않는다."""
    client, messages = _anthropic_client(
        [_anthropic_response(json.dumps({"verdict": "pass"}))], prompt_caching=True
    )

    client.complete_json(stage="judge", system=JUDGE_SYSTEM_PROMPT, user="u", schema=SCHEMA)

    call = messages.calls[0]
    assert call["system"] == [
        {
            "type": "text",
            "text": JUDGE_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    # 질의(user)는 픽스처마다 달라지는 부분이다 — 여기에 브레이크포인트를 걸면 매 호출이
    # 새 프리픽스가 되어 적중이 0 이 된다.
    assert call["messages"] == [{"role": "user", "content": "u"}]


# ── 계약 B — 캐시 계열 토큰 수집 (성공·실패 경로 모두) ──────────────────────


def test_캐시_계열_토큰을_수집해_산출에_싣는다() -> None:
    client, _ = _anthropic_client(
        [
            _anthropic_response(
                json.dumps({"verdict": "pass"}),
                usage=_usage(input_tokens=40, cache_creation=1450, cache_read=0),
            )
        ],
        prompt_caching=True,
    )

    result = client.complete_json(stage="judge", system="s", user="u", schema=SCHEMA)

    assert result.input_tokens == 40
    assert result.cache_creation_input_tokens == 1450
    assert result.cache_read_input_tokens == 0


def test_캐시_적중은_판정_입력_토큰_감소로_뭉뚱그려지지_않는다() -> None:
    """적중 회차의 `input_tokens` 는 비캐시 입력뿐이다 — 총 입력은 세 값의 합이다."""
    client, _ = _anthropic_client(
        [
            _anthropic_response(
                json.dumps({"verdict": "pass"}),
                usage=_usage(input_tokens=40, cache_creation=0, cache_read=1450),
            )
        ],
        prompt_caching=True,
    )

    result = client.complete_json(stage="judge", system="s", user="u", schema=SCHEMA)

    assert result.input_tokens == 40
    assert result.cache_read_input_tokens == 1450
    assert (
        result.input_tokens
        + (result.cache_creation_input_tokens or 0)
        + (result.cache_read_input_tokens or 0)
        == 1490
    )


def test_캐시_필드가_없는_응답은_0_이_아니라_미측정이다() -> None:
    """0 으로 채우면 "적중 0" 과 "캐시를 켠 적이 없다" 가 구분되지 않는다."""
    client, _ = _anthropic_client([_anthropic_response(json.dumps({"verdict": "pass"}))])

    result = client.complete_json(stage="judge", system="s", user="u", schema=SCHEMA)

    assert result.cache_creation_input_tokens is None
    assert result.cache_read_input_tokens is None


def test_OpenAI_래퍼의_캐시_계열은_해당_없음이다() -> None:
    """양성 대조가 아니라 경계 확인 — 생성 계열은 이 축을 재지 않는다."""
    calls = _RecordingCalls([_openai_response(json.dumps({"verdict": "pass"}))])
    fake_sdk = cast(Any, SimpleNamespace(responses=calls))
    client = OpenAIGenerationClient(api_key="test", model="gpt-5.6-terra", client=fake_sdk)

    result = client.complete_json(stage="draft", system="s", user="u", schema=SCHEMA)

    assert result.cache_creation_input_tokens is None
    assert result.cache_read_input_tokens is None


def test_거절_실패에도_캐시_계열_토큰이_실린다() -> None:
    """200 으로 온 응답이라 캐시 write 도 이미 과금됐다 — 실패에 실어 올린다."""
    client, _ = _anthropic_client(
        [
            _anthropic_response(
                "",
                stop_reason="refusal",
                usage=_usage(input_tokens=40, cache_creation=1450, cache_read=0),
            )
        ],
        prompt_caching=True,
    )

    with pytest.raises(LLMCallError) as excinfo:
        client.complete_json(stage="judge", system="s", user="u", schema=SCHEMA)

    assert excinfo.value.cache_creation_input_tokens == 1450
    assert excinfo.value.cache_read_input_tokens == 0


def test_형식_오류에도_캐시_계열_토큰이_실린다() -> None:
    client, _ = _anthropic_client(
        [
            _anthropic_response(
                "JSON 이 아니다",
                usage=_usage(input_tokens=40, cache_creation=0, cache_read=1450),
            )
        ],
        prompt_caching=True,
    )

    with pytest.raises(LLMFormatError) as excinfo:
        client.complete_json(stage="judge", system="s", user="u", schema=SCHEMA)

    assert excinfo.value.cache_read_input_tokens == 1450
    assert excinfo.value.cache_creation_input_tokens == 0


def test_누적기는_미측정을_0_으로_접지_않는다() -> None:
    assert accumulate_optional_tokens(None, None) is None
    assert accumulate_optional_tokens(None, 0) == 0
    assert accumulate_optional_tokens(3, None) == 3
    assert accumulate_optional_tokens(3, 4) == 7


# ── 계약 C — 판정기가 캐시 계열을 분리한 채 누적한다 ────────────────────────

_EVIDENCE = (
    Evidence(
        id="policy:refund:1-1",
        source=EvidenceSource.POLICY,
        content="환불은 수령일로부터 7일 이내 신청한다.",
        evidence_text="환불은 수령일로부터 7일 이내 신청한다.",
    ),
)
_DRAFT = Draft(claims=(Claim(text="환불은 7일 이내입니다.", citation_ids=("policy:refund:1-1",)),))


def _judge_payload(*, verdict: str, reasons: list[str], claim_verdict: str) -> str:
    return json.dumps(
        {
            "claim_judgments": [
                {
                    "claim_text": _DRAFT.claims[0].text,
                    "verdict": claim_verdict,
                    "explanation": "설명",
                }
            ],
            "contradictions": [],
            "verdict": verdict,
            "reject_reasons": reasons,
        },
        ensure_ascii=False,
    )


def test_판정_결과에_캐시_계열_토큰이_분리되어_실린다() -> None:
    """형식 재시도를 낀 경로 — 앞선 시도의 캐시 write 도 실비용이므로 누적된다."""
    client, _ = _anthropic_client(
        [
            _anthropic_response(
                "JSON 이 아니다", usage=_usage(input_tokens=40, cache_creation=1450, cache_read=0)
            ),
            _anthropic_response(
                _judge_payload(verdict="pass", reasons=[], claim_verdict="pass"),
                usage=_usage(input_tokens=45, cache_creation=0, cache_read=1450),
            ),
        ],
        prompt_caching=True,
    )
    judge = Judge(client=client)

    outcome = judge.judge(draft=_DRAFT, evidence=_EVIDENCE)

    assert outcome.input_tokens == 85
    assert outcome.cache_creation_input_tokens == 1450
    assert outcome.cache_read_input_tokens == 1450


def test_캐시_계열을_보고하지_않는_판정은_미측정으로_남는다() -> None:
    client, _ = _anthropic_client(
        [_anthropic_response(_judge_payload(verdict="pass", reasons=[], claim_verdict="pass"))]
    )
    judge = Judge(client=client)

    outcome = judge.judge(draft=_DRAFT, evidence=_EVIDENCE)

    assert outcome.cache_creation_input_tokens is None
    assert outcome.cache_read_input_tokens is None


# ── 계약 D — fail-closed 와 사유 2종이 캐싱 양쪽에서 같다 ───────────────────


@pytest.mark.parametrize("prompt_caching", [False, True])
def test_사유_2종_판정은_캐싱_양쪽에서_같다(prompt_caching: bool) -> None:
    client, _ = _anthropic_client(
        [
            _anthropic_response(
                _judge_payload(
                    verdict="reject", reasons=["unsupported_claim"], claim_verdict="reject"
                )
            )
        ],
        prompt_caching=prompt_caching,
    )

    outcome = Judge(client=client).judge(draft=_DRAFT, evidence=_EVIDENCE)

    assert outcome.result.verdict is Verdict.REJECT
    assert outcome.result.reject_reasons == (RejectReason.UNSUPPORTED_CLAIM,)


@pytest.mark.parametrize("prompt_caching", [False, True])
def test_해석되지_않는_산출은_캐싱_양쪽에서_거부된다(prompt_caching: bool) -> None:
    """fail-closed — 사유 2종 밖의 값은 통과가 아니라 형식 불일치다."""
    bad = json.dumps(
        {
            "claim_judgments": [
                {"claim_text": _DRAFT.claims[0].text, "verdict": "reject", "explanation": "x"}
            ],
            "contradictions": [],
            "verdict": "reject",
            "reject_reasons": ["pii_detected"],
        },
        ensure_ascii=False,
    )
    client, messages = _anthropic_client([_anthropic_response(bad)], prompt_caching=prompt_caching)

    with pytest.raises(LLMFormatError):
        Judge(client=client).judge(draft=_DRAFT, evidence=_EVIDENCE)

    # 형식 재시도 1회까지 — 캐싱 여부가 재시도 상한을 바꾸지 않는다.
    assert len(messages.calls) == 2


@pytest.mark.parametrize("prompt_caching", [False, True])
def test_수집_근거_밖의_모순_쌍은_캐싱_양쪽에서_거부된다(prompt_caching: bool) -> None:
    bad = json.dumps(
        {
            "claim_judgments": [
                {"claim_text": _DRAFT.claims[0].text, "verdict": "pass", "explanation": "x"}
            ],
            "contradictions": [
                {
                    "evidence_id_a": "policy:refund:1-1",
                    "evidence_id_b": "policy:없는:9-9",
                    "explanation": "x",
                }
            ],
            "verdict": "pass",
            "reject_reasons": [],
        },
        ensure_ascii=False,
    )
    client, _ = _anthropic_client([_anthropic_response(bad)], prompt_caching=prompt_caching)

    with pytest.raises(LLMFormatError):
        Judge(client=client).judge(draft=_DRAFT, evidence=_EVIDENCE)


# ── 계약 E — 스위치 배선 (기본 꺼짐 → 실행 조건 지문까지) ───────────────────


def test_설정_기본값은_캐싱_꺼짐이다() -> None:
    """실측이 정당화하지 않는 기본값을 남기지 않는다 — 채택 전까지 꺼짐이다."""
    assert Settings().judge_prompt_caching_enabled is False


@pytest.mark.parametrize("enabled", [False, True])
def test_판정_클라이언트가_설정_스위치를_그대로_받는다(enabled: bool) -> None:
    judge = build_judge(
        Settings(
            l2_enabled=True,
            anthropic_api_key="키가-아닌-테스트값",
            judge_prompt_caching_enabled=enabled,
        )
    )
    resolved = cast(Any, judge._client)._resolve()

    assert isinstance(resolved, AnthropicGenerationClient)
    assert resolved.prompt_caching is enabled


def _fingerprint(settings: Settings) -> dict[str, str]:
    args = argparse.Namespace(
        golden_set=_ROOT / "data" / "golden_set.jsonl",
        judge_fixtures=_ROOT / "data" / "judge_fixtures.jsonl",
        stub_llm=False,
    )
    return evaluate._condition_fingerprint(args=args, settings=settings, run_settings=settings)


@pytest.mark.parametrize(("enabled", "expected"), [(False, "off"), (True, "on")])
def test_실행_조건_지문이_캐싱_스위치를_읽는다(enabled: bool, expected: str) -> None:
    """지문이 하드코딩이면 캐싱을 켜도 산출물이 거짓말을 한다."""
    values = _fingerprint(Settings(judge_prompt_caching_enabled=enabled))

    assert values["judge_prompt_caching"] == expected


def test_캐싱_스위치는_판정_프롬프트_문면을_바꾸지_않는다() -> None:
    """캐싱 적용은 호출 구성이다 — 지문의 프롬프트 판이 따라 움직이면 안 된다."""
    off = _fingerprint(Settings(judge_prompt_caching_enabled=False))
    on = _fingerprint(Settings(judge_prompt_caching_enabled=True))

    assert off["judge_prompt_version"] == on["judge_prompt_version"]


# ── 계약 F — 캐시 계열이 평가 리포트까지 간다 ───────────────────────────────


class _CacheReportingJudge:
    """`Judging` 대역 — 판정 정확도가 아니라 **캐시 계열 배선**만 보기 위한 것."""

    def __init__(self, *, cache_creation: int | None, cache_read: int | None) -> None:
        self._cache_creation = cache_creation
        self._cache_read = cache_read

    def judge(self, *, draft: Draft, evidence: Sequence[Evidence]) -> JudgeOutcome:
        del evidence
        return JudgeOutcome(
            result=JudgeResult(
                verdict=Verdict.PASS,
                claim_judgments=tuple(
                    ClaimJudgment(claim_text=claim.text, verdict=Verdict.PASS, explanation="통과")
                    for claim in draft.claims
                ),
            ),
            input_tokens=40,
            output_tokens=20,
            attempts=1,
            cache_creation_input_tokens=self._cache_creation,
            cache_read_input_tokens=self._cache_read,
        )


def _conditions() -> RunConditions:
    return RunConditions(
        started_at="2026-08-19T00:00:00+00:00",
        generation="미실행",
        embedding="미실행",
        judge="Anthropic `claude-sonnet-5` (effort=기본값)",
        similarity_threshold=0.3,
        top_k=5,
        embedding_dimensions=1536,
        retrieval_strategy="vector+rewrite",
        l1_fixture_count=27,
        golden_case_count=30,
        judge_fixture_count=11,
        l1_fixtures_path="data/l1_fixtures.jsonl",
        golden_set_path="data/golden_set.jsonl",
        judge_fixtures_path="data/judge_fixtures.jsonl",
        api_key_present=False,
        judge_api_key_present=True,
        l2_enabled=True,
        measurement2_is_real=False,
        measurement3_is_real=True,
        billed=True,
        measurement_scope="measurement_1_3_only",
    )


def _accuracy(*, cache_creation: int | None, cache_read: int | None) -> JudgeAccuracy:
    return measure_judge_accuracy(
        fixtures=load_judge_fixtures(),
        judge=_CacheReportingJudge(cache_creation=cache_creation, cache_read=cache_read),
    )


def test_캐시_계열_토큰이_평가_리포트까지_간다() -> None:
    accuracy = _accuracy(cache_creation=100, cache_read=1000)
    fixtures = load_judge_fixtures()

    assert accuracy.cache_creation_tokens_total == 100 * len(fixtures)
    assert accuracy.cache_read_tokens_total == 1000 * len(fixtures)

    report = build_report(
        conditions=_conditions(),
        gate_accuracy=measure_gate_accuracy(load_l1_fixtures()),
        pipeline=SkippedMeasurement(reason="측정 2 를 고르지 않았다"),
        judge_accuracy=accuracy,
    )
    payload = report_to_json(report)
    tokens = payload["measurement_3_l2_judge_accuracy"]["tokens"]

    assert tokens["cache_creation_total"] == 100 * len(fixtures)
    assert tokens["cache_read_total"] == 1000 * len(fixtures)
    # 판정 입력 토큰의 정의는 그대로다 — 캐시 계열은 별도 칸이다.
    assert tokens["input_total"] == 40 * len(fixtures)

    outcome = payload["measurement_3_l2_judge_accuracy"]["outcomes"][0]
    assert outcome["cache_creation_input_tokens"] == 100
    assert outcome["cache_read_input_tokens"] == 1000

    markdown = render_markdown(report)
    assert "캐시" in markdown


def test_캐시를_보고하지_않은_실행은_리포트에서_0_이_아니라_미측정이다() -> None:
    accuracy = _accuracy(cache_creation=None, cache_read=None)

    assert accuracy.cache_creation_tokens_total is None
    assert accuracy.cache_read_tokens_total is None

    report = build_report(
        conditions=_conditions(),
        gate_accuracy=measure_gate_accuracy(load_l1_fixtures()),
        pipeline=SkippedMeasurement(reason="측정 2 를 고르지 않았다"),
        judge_accuracy=accuracy,
    )
    tokens = report_to_json(report)["measurement_3_l2_judge_accuracy"]["tokens"]

    assert tokens["cache_creation_total"] is None
    assert tokens["cache_read_total"] is None


def test_판정_실패한_픽스처의_캐시_계열도_버려지지_않는다() -> None:
    """실행됐으나 실패한 호출의 토큰도 실비용이다 — 캐시 계열도 같은 자격이다."""

    class _FailingJudge:
        def judge(self, *, draft: Draft, evidence: Sequence[Evidence]) -> JudgeOutcome:
            del draft, evidence
            raise LLMCallError(
                stage="judge",
                reason="refusal",
                attempts=1,
                input_tokens=40,
                output_tokens=0,
                cache_creation_input_tokens=1450,
                cache_read_input_tokens=0,
            )

    fixtures = load_judge_fixtures()
    accuracy = measure_judge_accuracy(fixtures=fixtures, judge=_FailingJudge())

    assert accuracy.error_total == len(fixtures)
    assert accuracy.cache_creation_tokens_total == 1450 * len(fixtures)
    assert accuracy.cache_read_tokens_total == 0


def test_런타임_집계가_없는_채로_캐싱을_기본으로_켤_수_없다() -> None:
    """캐시 계열 집계는 지금 **측정 3 경로에만** 있다.

    런타임(파이프라인 → 처리 기록)의 판정 토큰은 `judge_input_tokens` 하나뿐이라, 캐싱을
    기본으로 켜면 그 값이 조용히 "비캐시 입력"으로 바뀌고 캐시 계열은 아무 데도 안 남는다.
    측정 3 에서 막은 은폐가 런타임에서 그대로 재발한다는 뜻이다.

    지금은 스위치가 꺼져 있어 잠재 상태다. 이 검사는 **기본값을 켜는 순간** 소리가 나게
    한다 — 런타임 집계를 먼저 배선하지 않으면 실패한다. 런타임을 배선하면 이 검사는
    저절로 조용해진다(전제가 사라진다).
    """
    파이프라인 = pathlib.Path(str(pipeline_module.__file__)).read_text(encoding="utf-8")
    런타임_집계가_있다 = (
        "cache_read_input_tokens" in 파이프라인 or "cache_creation_input_tokens" in 파이프라인
    )
    if 런타임_집계가_있다:
        return

    # `Settings` 를 assert 식 안에 그대로 두면 실패 출력이 설정 객체를 통째로 repr 하고,
    # 거기에 API 키가 평문으로 실린다. **bool 로 먼저 묶는다.**
    캐싱이_기본으로_켜져_있다 = bool(Settings().judge_prompt_caching_enabled)
    assert not 캐싱이_기본으로_켜져_있다, (
        "런타임 경로에 캐시 계열 토큰 집계가 없는 채로 판정 프롬프트 캐싱을 기본으로 켤 수 "
        "없다 — `judge_input_tokens` 가 캐시 적중분을 제외한 값으로 조용히 바뀐다. "
        "켜려면 파이프라인·처리 기록의 판정 토큰 집계를 먼저 배선해라."
    )
