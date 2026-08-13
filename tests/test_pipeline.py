"""파이프라인 루프 테스트 — 접수 → 근거 수집 → 초안 → L1 → (L2) → 종결.

생성 LLM·판정 LLM 은 전부 목이다(실제 API 키를 쓰지 않는다). 임베딩은
`LexicalEmbeddingClient` 결정론 대역이다.

두 층으로 나눠 검증한다.

* **DB 없는 층** — 근거 수집을 대역으로 갈아 끼워 루프 계약만 본다: 재생성 1회 상한,
  근거 재수집 금지, 기각 사유 전량 피드백, 형식 불일치 → `schema_violation` 루프,
  초안 전 인계의 근거 보존, 그리고 L2 배선(스위치·기각 후 재생성·실패 정책·토큰 분리).
* **DB 층(`db` 마커)** — 실제 `EvidenceCollector` + 시딩된 Postgres 로 인계 사유 6종을
  끝까지 재현한다. 이 층은 **L2 꺼짐**으로 돌린다(판정은 확률 층이라 대역이 필요하고,
  인계 사유 6종 재현은 L2 와 무관하다).

파이프라인은 **DB 에 쓰지 않는다** (처리 기록 저장은 `records.py` 몫). 따라서 이 파일의
DB 테스트는 정책 인덱싱 픽스처(트랜잭션 롤백) 말고는 남기는 행이 없다.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

import psycopg
import pytest
from psycopg.rows import DictRow

from reply_gate.config import Settings
from reply_gate.contracts import (
    ClaimJudgment,
    EscalationReason,
    Evidence,
    EvidenceContradiction,
    EvidenceSource,
    InquiryStatus,
    IntentSource,
    JudgeResult,
    RejectReason,
    Verdict,
)
from reply_gate.draft import DRAFT_STAGE, DraftGenerator
from reply_gate.evidence import (
    INTENT_STAGE,
    SQL_GENERATION_STAGE,
    EvidenceCollection,
)
from reply_gate.judge import JUDGE_STAGE, Judge, JudgeOutcome
from reply_gate.llm import (
    AnthropicGenerationClient,
    EmbeddingClient,
    GenerationClient,
    JsonCompletion,
    LLMCallError,
    LLMFormatError,
    OpenAIGenerationClient,
)
from reply_gate.pipeline import (
    L2_JUDGE_STAGE,
    MAX_DRAFT_ATTEMPTS,
    AcceptedInquiry,
    DraftGenerating,
    EvidenceCollecting,
    InquiryPipeline,
    Judging,
    MissingCredentialsError,
    PipelineWiringError,
    ReceiptError,
    accept_inquiry,
    build_pipeline,
    new_inquiry_id,
)
from reply_gate.pipeline import _LazyJudgeClient as LazyJudgeClient
from reply_gate.policy_index import index_policy_documents, load_policy_documents
from reply_gate.query_rewrite import QUERY_REWRITE_STAGE
from reply_gate.testing import LexicalEmbeddingClient

INQUIRY = "환불은 언제까지 신청할 수 있나요?"
MISSING_ORDER_NO = "ORD-20991231-9999"
POLICY_EVIDENCE = Evidence(
    id="policy:refund:2.1",
    source=EvidenceSource.POLICY,
    content="환불은 수령 후 7일 이내에 신청할 수 있다.",
    evidence_text="환불은 수령 후 7일 이내에 신청할 수 있다.",
)

_NO_CONN = cast(psycopg.Connection[DictRow], None)


# ── 목 / 대역 ───────────────────────────────────────────────────────────────


class ScriptedGenerationClient:
    """`GenerationClient` 대역 — 단계별로 미리 정한 결과를 순서대로 돌려준다.

    큐에 담긴 항목이 예외면 던지고, 호출 가능 객체면 호출 인자를 넘겨 실행한다
    (근거 ID 를 프롬프트에서 읽어 인용하는 초안 대역에 쓴다).

    **재작성 단계만 예외**다: 대본에 아예 없으면 "원문 그대로"를 돌려준다 — 재작성은 정책
    경로의 기본 호출이라 대본에 넣기를 강제하면 관계없는 테스트가 전부 재작성 대본을 이고
    다니게 된다. 원문 그대로는 픽스처 계약이 허용하는 산출이고, 그때 수집기는 검색을 한 번만
    돈다. **대본이 지정했으면 그것이 이기고, 큐가 마르면 그때는 오류다.**
    """

    def __init__(self, script: Mapping[str, Sequence[Any]]) -> None:
        self._script = {stage: list(outcomes) for stage, outcomes in script.items()}
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        stage = kwargs["stage"]
        if stage == QUERY_REWRITE_STAGE and stage not in self._script:
            return echo_rewrite(kwargs["user"])
        queue = self._script.get(stage)
        if not queue:
            raise AssertionError(f"대본에 없는 호출이다: stage={stage!r}")
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            return outcome(**kwargs)
        return outcome

    def calls_for(self, stage: str) -> list[dict[str, Any]]:
        return [call for call in self.calls if call["stage"] == stage]


def echo_rewrite(user_prompt: str) -> JsonCompletion:
    """재작성 대역의 기본 산출 — 프롬프트에 실린 문의 원문을 그대로 되돌려준다."""
    return JsonCompletion(
        data={"rewritten": user_prompt.removeprefix("[문의]\n")},
        input_tokens=7,
        output_tokens=3,
    )


def rewrite_completion(
    text: str, *, input_tokens: int = 7, output_tokens: int = 3
) -> JsonCompletion:
    return JsonCompletion(
        data={"rewritten": text}, input_tokens=input_tokens, output_tokens=output_tokens
    )


class StubCollector:
    """`EvidenceCollecting` 대역 — 정해진 수집 결과를 돌려주고 호출을 기록한다."""

    def __init__(self, collection: EvidenceCollection) -> None:
        self._collection = collection
        self.calls: list[dict[str, Any]] = []

    def collect(self, **kwargs: Any) -> EvidenceCollection:
        self.calls.append(kwargs)
        return self._collection


class BrokenCollector:
    """정책 검색이 DB 오류로 무너지는 상황."""

    def collect(self, **kwargs: Any) -> EvidenceCollection:
        del kwargs
        raise psycopg.OperationalError("policy_chunks 를 읽을 수 없다")


class ScriptedJudge:
    """`Judging` 대역 — 시도 순서대로 미리 정한 판정을 돌려준다(예외면 던진다).

    T9 가 `testing.py` 에 결정론 판정 대역을 넣기 전까지, 판정 목은 이 파일 안에만 둔다.
    """

    def __init__(self, outcomes: Sequence[Any]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def judge(self, **kwargs: Any) -> JudgeOutcome:
        self.calls.append(kwargs)
        if not self._outcomes:
            raise AssertionError("대본에 없는 판정 호출이다")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return cast(JudgeOutcome, outcome)


def judge_pass(*, input_tokens: int = 40, output_tokens: int = 9) -> JudgeOutcome:
    return JudgeOutcome(
        result=JudgeResult(verdict=Verdict.PASS),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        attempts=1,
    )


def judge_reject(
    *,
    claim_text: str,
    explanation: str = "인용한 근거가 이 문장의 주제를 다루지 않는다.",
    reasons: tuple[RejectReason, ...] = (RejectReason.UNSUPPORTED_CLAIM,),
    contradictions: tuple[EvidenceContradiction, ...] = (),
    input_tokens: int = 40,
    output_tokens: int = 9,
) -> JudgeOutcome:
    return JudgeOutcome(
        result=JudgeResult(
            verdict=Verdict.REJECT,
            reject_reasons=reasons,
            claim_judgments=(
                ClaimJudgment(
                    claim_text=claim_text, verdict=Verdict.REJECT, explanation=explanation
                ),
            ),
            contradictions=contradictions,
        ),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        attempts=1,
    )


def scripted_client(script: Mapping[str, Sequence[Any]]) -> ScriptedGenerationClient:
    return ScriptedGenerationClient(script)


def intent_completion(
    value: str, *, input_tokens: int = 11, output_tokens: int = 3
) -> JsonCompletion:
    return JsonCompletion(
        data={"source": value}, input_tokens=input_tokens, output_tokens=output_tokens
    )


def sql_completion(text: str, *, input_tokens: int = 30, output_tokens: int = 12) -> JsonCompletion:
    return JsonCompletion(
        data={"sql": text}, input_tokens=input_tokens, output_tokens=output_tokens
    )


def draft_completion(raw: Any, *, input_tokens: int = 5, output_tokens: int = 2) -> JsonCompletion:
    return JsonCompletion(data=raw, input_tokens=input_tokens, output_tokens=output_tokens)


def citing_draft(*, text: str = "안내드립니다.", limit: int = 1) -> Any:
    """프롬프트에 실린 근거 ID 를 그대로 인용하는 초안 대역 (L1 통과용)."""

    def _outcome(**kwargs: Any) -> JsonCompletion:
        ids = re.findall(r"근거 ID: (\S+)", kwargs["user"])
        assert ids, "근거가 실리지 않은 프롬프트다"
        return draft_completion({"claims": [{"text": text, "citation_ids": ids[:limit]}]})

    return _outcome


def collection(
    *,
    evidence: Sequence[Evidence] = (),
    escalation: EscalationReason | None = None,
    failed_stage: str | None = None,
    intent: IntentSource | None = IntentSource.POLICY,
    input_tokens: int = 11,
    output_tokens: int = 3,
    embedding_tokens: int = 7,
) -> EvidenceCollection:
    return EvidenceCollection(
        intent=intent,
        evidence=tuple(evidence),
        escalation_reason=escalation,
        failed_stage=failed_stage,
        sql_snapshots=(),
        sql_failures=(),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        embedding_tokens=embedding_tokens,
    )


def pipeline_with(
    *,
    collector: Any,
    client: ScriptedGenerationClient | None = None,
    judge: Any = None,
    l2_enabled: bool = False,
) -> InquiryPipeline:
    """루프 계약 검증용 조립.

    **스위치는 기본 꺼짐**이다 — 사이클 1 동작(L1 pass → answered)을 검증하는 케이스가
    다수이므로 여기서 명시적으로 끈다. L2 케이스는 판정 대역과 함께 `l2_enabled=True` 를
    명시한다(스위치 켜짐 + 판정자 미배선은 조립 시점 오류라 조용히 꺼지지 않는다).
    """
    generator = DraftGenerator(client=cast(GenerationClient, client)) if client else None
    return InquiryPipeline(
        collector=cast(EvidenceCollecting, collector),
        drafter=cast(DraftGenerating, generator),
        judge=cast(Judging, judge) if judge is not None else None,
        l2_enabled=l2_enabled,
    )


def run(pipeline: InquiryPipeline, *, content: str = INQUIRY, order_no: str | None = None) -> Any:
    return pipeline.run(
        inquiry_id=new_inquiry_id(),
        content=content,
        order_no=order_no,
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )


# ── 접수 [코드] ─────────────────────────────────────────────────────────────


def test_접수는_내용이_비면_거부한다() -> None:
    with pytest.raises(ReceiptError):
        accept_inquiry(content="   ")


def test_접수는_주문번호_형식을_코드가_검증한다() -> None:
    with pytest.raises(ReceiptError):
        accept_inquiry(content=INQUIRY, order_no="주문번호-없음")


def test_접수는_주문번호를_정규화해_보관한다() -> None:
    accepted = accept_inquiry(content=f"  {INQUIRY} ", order_no=" ord-20260315-0001 ")

    assert accepted == AcceptedInquiry(content=INQUIRY, order_no="ORD-20260315-0001")


def test_접수는_빈_주문번호를_미입력으로_본다() -> None:
    """웹 폼은 빈 문자열을 보낸다 — 미입력과 같게 다룬다(형식 오류가 아니다)."""
    assert accept_inquiry(content=INQUIRY, order_no="  ").order_no is None


def test_문의_ID_는_소문자_하이픈_UUID_다() -> None:
    """SQL 근거 ID 의 CHECK 제약이 Postgres 의 uuid 텍스트 표현과 글자 단위로 맞아야 한다."""
    assert re.fullmatch(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", new_inquiry_id())


# ── 루프 계약 (DB 불필요) ───────────────────────────────────────────────────


def test_첫_초안이_통과하면_시도_1건으로_확정된다() -> None:
    client = scripted_client({DRAFT_STAGE: [citing_draft(text="7일 이내 신청 가능합니다.")]})
    pipeline = pipeline_with(
        collector=StubCollector(collection(evidence=[POLICY_EVIDENCE])), client=client
    )

    processed = run(pipeline)

    assert processed.status is InquiryStatus.ANSWERED
    assert processed.answer == "7일 이내 신청 가능합니다."
    assert processed.escalation_reason is None
    assert [attempt.verdict for attempt in processed.attempts] == [Verdict.PASS]
    assert processed.claims[0].citation_ids == (POLICY_EVIDENCE.id,)


def test_기각_후_재생성이_통과하면_확정된다() -> None:
    rejected = draft_completion({"claims": [{"text": "환불 가능합니다.", "citation_ids": []}]})
    client = scripted_client({DRAFT_STAGE: [rejected, citing_draft()]})
    pipeline = pipeline_with(
        collector=StubCollector(collection(evidence=[POLICY_EVIDENCE])), client=client
    )

    processed = run(pipeline)

    assert processed.status is InquiryStatus.ANSWERED
    assert [(a.attempt_no, a.verdict) for a in processed.attempts] == [
        (1, Verdict.REJECT),
        (2, Verdict.PASS),
    ]
    assert processed.attempts[0].reject_reasons == (RejectReason.MISSING_CITATION,)
    assert processed.attempts[1].reject_reasons == ()


def test_두_번_모두_기각이면_rejected_twice_로_인계한다() -> None:
    rejected = draft_completion({"claims": [{"text": "환불 가능합니다.", "citation_ids": []}]})
    client = scripted_client({DRAFT_STAGE: [rejected, rejected]})
    pipeline = pipeline_with(
        collector=StubCollector(collection(evidence=[POLICY_EVIDENCE])), client=client
    )

    processed = run(pipeline)

    assert processed.status is InquiryStatus.ESCALATED
    assert processed.escalation_reason is EscalationReason.REJECTED_TWICE
    assert processed.answer is None
    assert processed.claims == ()
    assert [a.verdict for a in processed.attempts] == [Verdict.REJECT, Verdict.REJECT]


def test_재생성_상한은_코드가_강제한다_초안_생성은_2회를_넘지_않는다() -> None:
    """기각이 계속돼도 초안 생성 호출은 최대 2회 (최초 + 재생성 1회)."""
    rejected = draft_completion({"claims": [{"text": "환불 가능합니다.", "citation_ids": []}]})
    client = scripted_client({DRAFT_STAGE: [rejected] * 5})
    pipeline = pipeline_with(
        collector=StubCollector(collection(evidence=[POLICY_EVIDENCE])), client=client
    )

    processed = run(pipeline)

    assert MAX_DRAFT_ATTEMPTS == 2
    assert len(client.calls_for(DRAFT_STAGE)) == MAX_DRAFT_ATTEMPTS
    assert len(processed.attempts) == MAX_DRAFT_ATTEMPTS


def test_재생성은_근거를_다시_수집하지_않는다() -> None:
    rejected = draft_completion({"claims": [{"text": "환불 가능합니다.", "citation_ids": []}]})
    client = scripted_client({DRAFT_STAGE: [rejected, citing_draft()]})
    collector = StubCollector(collection(evidence=[POLICY_EVIDENCE]))
    pipeline = pipeline_with(collector=collector, client=client)

    run(pipeline)

    assert len(collector.calls) == 1


def test_재생성_프롬프트에_기각_사유가_전부_실린다() -> None:
    """근거에 없는 대표번호 + citation_ids 빈 문장 → 사유 2종이 함께 피드백된다."""
    rejected = draft_completion(
        {"claims": [{"text": "고객센터 1588-0000 으로 연락 주세요.", "citation_ids": []}]}
    )
    client = scripted_client({DRAFT_STAGE: [rejected, citing_draft()]})
    pipeline = pipeline_with(
        collector=StubCollector(collection(evidence=[POLICY_EVIDENCE])), client=client
    )

    processed = run(pipeline)

    assert processed.attempts[0].reject_reasons == (
        RejectReason.MISSING_CITATION,
        RejectReason.PII_DETECTED,
    )
    retry_prompt = client.calls_for(DRAFT_STAGE)[1]["user"]
    assert RejectReason.MISSING_CITATION.value in retry_prompt
    assert RejectReason.PII_DETECTED.value in retry_prompt
    # 같은 근거로 재생성한다 — 근거 목록이 그대로 다시 실린다.
    assert POLICY_EVIDENCE.id in retry_prompt


def test_형식_불일치_초안은_schema_violation_으로_기각되고_재생성한다() -> None:
    """초안 생성은 형식 불일치를 재시도하지 않는다 — L1 이 잡고 기존 루프가 처리한다."""
    broken = LLMFormatError(
        stage=DRAFT_STAGE,
        detail="JSON 파싱 실패",
        raw_text="죄송합니다, 환불은 7일 이내입니다.",
        input_tokens=4,
        output_tokens=9,
    )
    client = scripted_client({DRAFT_STAGE: [broken, citing_draft()]})
    pipeline = pipeline_with(
        collector=StubCollector(collection(evidence=[POLICY_EVIDENCE])), client=client
    )

    processed = run(pipeline)

    assert processed.attempts[0].reject_reasons == (RejectReason.SCHEMA_VIOLATION,)
    assert processed.attempts[0].draft == "죄송합니다, 환불은 7일 이내입니다."
    assert processed.status is InquiryStatus.ANSWERED


def test_초안_전송_오류는_llm_call_failed_이고_실패_단계를_남긴다() -> None:
    failure = LLMCallError(stage=DRAFT_STAGE, reason="transport_error", attempts=2)
    client = scripted_client({DRAFT_STAGE: [failure]})
    pipeline = pipeline_with(
        collector=StubCollector(collection(evidence=[POLICY_EVIDENCE])), client=client
    )

    processed = run(pipeline)

    assert processed.status is InquiryStatus.ESCALATED
    assert processed.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert processed.failed_stage == DRAFT_STAGE
    assert processed.attempts == ()


def test_초안_전송_오류가_실은_과금_토큰도_생성_합산에_남는다() -> None:
    """200 으로 돌아온 거절 응답처럼 실패까지 과금된 초안 토큰도 실비용이다 — 버리지 않는다.

    "실행됐으나 실패한 호출의 토큰도 그대로 집계한다"(docs/contracts.md "토큰 집계 경계").
    판정 경로와 같은
    규칙이고, 계열도 같아야 한다 — 초안은 생성 합산(`input_tokens`/`output_tokens`)이다.
    """
    failure = LLMCallError(
        stage=DRAFT_STAGE,
        reason="refusal",
        attempts=1,
        input_tokens=12,
        output_tokens=3,
    )
    client = scripted_client({DRAFT_STAGE: [failure]})
    pipeline = pipeline_with(
        collector=StubCollector(collection(evidence=[POLICY_EVIDENCE])), client=client
    )

    processed = run(pipeline)

    assert processed.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert processed.failed_stage == DRAFT_STAGE
    # 수집 대역(11/3) + 실패한 초안 호출(12/3) — 판정 계열에는 한 톨도 새지 않는다.
    assert (processed.input_tokens, processed.output_tokens) == (23, 6)
    assert (processed.judge_input_tokens, processed.judge_output_tokens) == (0, 0)


def test_재생성_중_전송_오류가_나도_앞선_시도_기록은_남는다() -> None:
    rejected = draft_completion({"claims": [{"text": "환불 가능합니다.", "citation_ids": []}]})
    failure = LLMCallError(stage=DRAFT_STAGE, reason="transport_error", attempts=2)
    client = scripted_client({DRAFT_STAGE: [rejected, failure]})
    pipeline = pipeline_with(
        collector=StubCollector(collection(evidence=[POLICY_EVIDENCE])), client=client
    )

    processed = run(pipeline)

    assert processed.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert [a.verdict for a in processed.attempts] == [Verdict.REJECT]


def test_초안_전_인계는_초안_생성에_진입하지_않고_근거를_남긴다() -> None:
    """감사 목적 — 인계로 끝나도 그 시점까지 수집된 근거는 citations 에 남는다."""
    client = scripted_client({})
    collector = StubCollector(
        collection(evidence=[POLICY_EVIDENCE], escalation=EscalationReason.MISSING_ORDER_REF)
    )
    pipeline = pipeline_with(collector=collector, client=client)

    processed = run(pipeline)

    assert processed.status is InquiryStatus.ESCALATED
    assert processed.escalation_reason is EscalationReason.MISSING_ORDER_REF
    assert processed.attempts == ()
    assert client.calls_for(DRAFT_STAGE) == []
    assert [item.id for item in processed.evidence] == [POLICY_EVIDENCE.id]


def test_지연과_토큰이_기록되고_임베딩_토큰은_섞이지_않는다() -> None:
    rejected = draft_completion({"claims": [{"text": "환불 가능합니다.", "citation_ids": []}]})
    client = scripted_client({DRAFT_STAGE: [rejected, citing_draft()]})
    collected = collection(
        evidence=[POLICY_EVIDENCE], input_tokens=11, output_tokens=3, embedding_tokens=7
    )
    pipeline = pipeline_with(collector=StubCollector(collected), client=client)

    processed = run(pipeline)

    # 생성 계열 합산 = 수집(의도·SQL) + 초안 2회. 임베딩은 별도 필드.
    assert processed.input_tokens == 11 + 5 + 5
    assert processed.output_tokens == 3 + 2 + 2
    assert processed.embedding_tokens == 7
    assert processed.latency_ms >= 0


def test_DB_오류는_인계_사유로_바꾸지_않고_전파한다() -> None:
    """인계 사유 6종은 업무 판정이다 — 인프라 장애를 그중 하나로 위장하지 않는다."""
    pipeline = pipeline_with(collector=BrokenCollector(), client=scripted_client({}))

    with pytest.raises(psycopg.Error):
        run(pipeline)


# ── L2 배선 (DB 불필요, 판정은 대역) ────────────────────────────────────────


def l2_pipeline(client: ScriptedGenerationClient, judge: ScriptedJudge) -> InquiryPipeline:
    return pipeline_with(
        collector=StubCollector(collection(evidence=[POLICY_EVIDENCE])),
        client=client,
        judge=judge,
        l2_enabled=True,
    )


def test_L2_기각_후_재생성이_통과하면_확정된다() -> None:
    client = scripted_client(
        {DRAFT_STAGE: [citing_draft(text="첫 초안입니다."), citing_draft(text="고친 초안입니다.")]}
    )
    judge = ScriptedJudge([judge_reject(claim_text="첫 초안입니다."), judge_pass()])

    processed = run(l2_pipeline(client, judge))

    assert processed.status is InquiryStatus.ANSWERED
    assert processed.answer == "고친 초안입니다."
    assert [(a.attempt_no, a.verdict) for a in processed.attempts] == [
        (1, Verdict.REJECT),
        (2, Verdict.PASS),
    ]
    # 종합 사유는 L2 사유를 그대로 싣는다 (L1 은 통과했다).
    assert processed.attempts[0].reject_reasons == (RejectReason.UNSUPPORTED_CLAIM,)
    first_l1 = processed.attempts[0].l1_result
    first_l2 = processed.attempts[0].l2_result
    assert first_l1 is not None and first_l1.verdict is Verdict.PASS
    assert first_l2 is not None and first_l2.verdict is Verdict.REJECT
    assert first_l2.reject_reasons == (RejectReason.UNSUPPORTED_CLAIM,)
    # 재생성 초안도 판정을 받는다 — 시도당 1회 배치 판정.
    assert len(judge.calls) == 2
    # 재생성은 같은 근거로 한다.
    assert [call["evidence"] for call in judge.calls] == [(POLICY_EVIDENCE,), (POLICY_EVIDENCE,)]


def test_L2_가_두_번_기각하면_rejected_twice_로_인계한다() -> None:
    client = scripted_client(
        {DRAFT_STAGE: [citing_draft(text="초안."), citing_draft(text="초안.")]}
    )
    judge = ScriptedJudge([judge_reject(claim_text="초안."), judge_reject(claim_text="초안.")])

    processed = run(l2_pipeline(client, judge))

    assert processed.status is InquiryStatus.ESCALATED
    assert processed.escalation_reason is EscalationReason.REJECTED_TWICE
    assert processed.answer is None
    assert processed.claims == ()
    assert [a.verdict for a in processed.attempts] == [Verdict.REJECT, Verdict.REJECT]
    # 상한은 for 루프 범위다 — 초안 생성도 판정도 2회를 넘지 않는다.
    assert len(client.calls_for(DRAFT_STAGE)) == MAX_DRAFT_ATTEMPTS
    assert len(judge.calls) == MAX_DRAFT_ATTEMPTS


def test_L1_기각_뒤_L2_기각도_rejected_twice_다() -> None:
    """`rejected_twice` 는 **층 무관** 2회 연속 기각이다."""
    rejected = draft_completion({"claims": [{"text": "환불 가능합니다.", "citation_ids": []}]})
    client = scripted_client({DRAFT_STAGE: [rejected, citing_draft(text="고친 초안.")]})
    judge = ScriptedJudge([judge_reject(claim_text="고친 초안.")])

    processed = run(l2_pipeline(client, judge))

    assert processed.escalation_reason is EscalationReason.REJECTED_TWICE
    assert processed.attempts[0].reject_reasons == (RejectReason.MISSING_CITATION,)
    assert processed.attempts[1].reject_reasons == (RejectReason.UNSUPPORTED_CLAIM,)


def test_L1_이_기각한_시도에는_L2_판정이_없다() -> None:
    """L1 reject 면 L2 는 실행되지 않는다 — 판정 호출은 통과한 초안에만 붙는다."""
    rejected = draft_completion({"claims": [{"text": "환불 가능합니다.", "citation_ids": []}]})
    client = scripted_client({DRAFT_STAGE: [rejected, citing_draft()]})
    judge = ScriptedJudge([judge_pass()])

    processed = run(l2_pipeline(client, judge))

    assert processed.status is InquiryStatus.ANSWERED
    assert len(judge.calls) == 1
    assert processed.attempts[0].l2_result is None
    first_l1 = processed.attempts[0].l1_result
    assert first_l1 is not None and first_l1.verdict is Verdict.REJECT
    second_l2 = processed.attempts[1].l2_result
    assert second_l2 is not None and second_l2.verdict is Verdict.PASS


def test_L1_이_두_번_기각하면_판정자는_한_번도_불리지_않는다() -> None:
    """L1 기각만으로 끝나는 문의는 판정 비용이 0 이다 — 스위치가 켜져 있어도."""
    rejected = draft_completion({"claims": [{"text": "환불 가능합니다.", "citation_ids": []}]})
    client = scripted_client({DRAFT_STAGE: [rejected, rejected]})
    judge = ScriptedJudge([])  # 호출되면 대본 없음으로 실패한다

    processed = run(l2_pipeline(client, judge))

    assert processed.escalation_reason is EscalationReason.REJECTED_TWICE
    assert judge.calls == []
    assert [attempt.l2_result for attempt in processed.attempts] == [None, None]
    assert (processed.judge_input_tokens, processed.judge_output_tokens) == (0, 0)


def test_재생성_시도의_L2_전송_실패도_l2_judge_인계다() -> None:
    """L1 기각 뒤 재생성이 L2 에 닿아 무너진 경우 — 인프라 실패가 재기각으로 둔갑하지 않는다."""
    rejected = draft_completion({"claims": [{"text": "환불 가능합니다.", "citation_ids": []}]})
    client = scripted_client({DRAFT_STAGE: [rejected, citing_draft(text="고친 초안.")]})
    judge = ScriptedJudge([LLMCallError(stage=JUDGE_STAGE, reason="transport_error", attempts=2)])

    processed = run(l2_pipeline(client, judge))

    assert processed.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert processed.failed_stage == L2_JUDGE_STAGE
    assert processed.answer is None
    # 시도 행은 2건 — 1번은 L1 기각, 2번은 L1 통과 + L2 null.
    assert [attempt.attempt_no for attempt in processed.attempts] == [1, 2]
    assert processed.attempts[1].l2_result is None
    second_l1 = processed.attempts[1].l1_result
    assert second_l1 is not None and second_l1.verdict is Verdict.PASS


def test_스위치가_꺼지면_판정자가_있어도_사이클_1_동작이다() -> None:
    judge = ScriptedJudge([])  # 호출되면 대본 없음으로 실패한다
    pipeline = pipeline_with(
        collector=StubCollector(collection(evidence=[POLICY_EVIDENCE])),
        client=scripted_client({DRAFT_STAGE: [citing_draft()]}),
        judge=judge,
        l2_enabled=False,
    )

    processed = run(pipeline)

    assert processed.status is InquiryStatus.ANSWERED
    assert judge.calls == []
    assert processed.attempts[0].l2_result is None
    assert (processed.judge_input_tokens, processed.judge_output_tokens) == (0, 0)


def test_L2_기각_피드백에_사유_코드와_claim_상세가_실린다() -> None:
    contradiction = EvidenceContradiction(
        evidence_id_a=POLICY_EVIDENCE.id,
        evidence_id_b="policy:faq:12",
        explanation="환불 기간이 7일과 30일로 상충한다.",
    )
    client = scripted_client(
        {DRAFT_STAGE: [citing_draft(text="첫 초안입니다."), citing_draft(text="고친 초안입니다.")]}
    )
    judge = ScriptedJudge(
        [
            judge_reject(
                claim_text="첫 초안입니다.",
                explanation="국내 배송 조항은 해외 배송 주제를 다루지 않는다.",
                reasons=(RejectReason.UNSUPPORTED_CLAIM, RejectReason.CONTRADICTORY_EVIDENCE),
                contradictions=(contradiction,),
            ),
            judge_pass(),
        ]
    )

    run(l2_pipeline(client, judge))

    retry_prompt = client.calls_for(DRAFT_STAGE)[1]["user"]
    assert RejectReason.UNSUPPORTED_CLAIM.value in retry_prompt
    assert RejectReason.CONTRADICTORY_EVIDENCE.value in retry_prompt
    # claim 단위 상세 — 어느 문장이 왜.
    assert "첫 초안입니다." in retry_prompt
    assert "국내 배송 조항은 해외 배송 주제를 다루지 않는다." in retry_prompt
    # 근거쌍 모순은 쌍과 함께, 구제 방향("모순을 명시하고 두 기준을 모두 안내")이 실린다.
    assert contradiction.evidence_id_b in retry_prompt
    assert "환불 기간이 7일과 30일로 상충한다." in retry_prompt
    assert "모순을 명시" in retry_prompt
    assert "두 기준을 모두 안내" in retry_prompt
    # 같은 근거로 재생성한다.
    assert POLICY_EVIDENCE.id in retry_prompt


def test_L2_전송_오류는_llm_call_failed_이고_실패_단계는_l2_judge_다() -> None:
    client = scripted_client({DRAFT_STAGE: [citing_draft()]})
    judge = ScriptedJudge([LLMCallError(stage=JUDGE_STAGE, reason="transport_error", attempts=2)])

    processed = run(l2_pipeline(client, judge))

    assert processed.status is InquiryStatus.ESCALATED
    assert processed.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert processed.failed_stage == L2_JUDGE_STAGE == "l2_judge"
    # 검증하지 못한 답변은 내보내지 않는다.
    assert processed.answer is None
    assert processed.claims == ()
    # 그 시도 행은 L1 판정만 남고 L2 판정은 null 이다 (종합 verdict 는 pass).
    assert len(processed.attempts) == 1
    attempt = processed.attempts[0]
    assert attempt.l1_result is not None and attempt.l1_result.verdict is Verdict.PASS
    assert attempt.l2_result is None
    assert attempt.verdict is Verdict.PASS
    assert attempt.reject_reasons == ()
    # 인프라 실패는 재생성 사유가 아니다 — 초안을 다시 만들지 않는다.
    assert len(client.calls_for(DRAFT_STAGE)) == 1


def test_L2_형식_불일치_소진도_인계이고_실패_호출_토큰이_집계된다() -> None:
    """배치 판정의 형식 불일치는 반경이 문의 전체다 — 재시도 소진 = 인계."""
    exhausted = LLMFormatError(
        stage=JUDGE_STAGE,
        detail="판정 산출이 형식에 맞지 않았다",
        raw_text="이건 JSON 이 아니다",
        input_tokens=50,
        output_tokens=6,
    )
    client = scripted_client({DRAFT_STAGE: [citing_draft()]})
    judge = ScriptedJudge([exhausted])

    processed = run(l2_pipeline(client, judge))

    assert processed.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert processed.failed_stage == L2_JUDGE_STAGE
    assert processed.attempts[0].l2_result is None
    # 실패한 판정 호출이 쓴 토큰도 실비용이므로 그대로 집계한다.
    assert (processed.judge_input_tokens, processed.judge_output_tokens) == (50, 6)
    # 생성 합산은 수집(11/3) + 초안 1회(5/2) 그대로다.
    assert (processed.input_tokens, processed.output_tokens) == (16, 5)


def test_판정_토큰은_생성_합산에_섞이지_않는다() -> None:
    client = scripted_client(
        {DRAFT_STAGE: [citing_draft(text="첫 초안."), citing_draft(text="고친 초안.")]}
    )
    judge = ScriptedJudge(
        [
            judge_reject(claim_text="첫 초안.", input_tokens=100, output_tokens=20),
            judge_pass(input_tokens=30, output_tokens=4),
        ]
    )

    processed = run(l2_pipeline(client, judge))

    # 생성 계열 = 수집(11/3) + 초안 2회(5/2 씩). 판정·임베딩은 각각 별도 필드다.
    assert (processed.input_tokens, processed.output_tokens) == (21, 7)
    assert (processed.judge_input_tokens, processed.judge_output_tokens) == (130, 24)
    assert processed.embedding_tokens == 7


def test_판정_키_부재는_llm_call_failed_로_삼키지_않고_전파한다() -> None:
    """자격 증명 부재는 업무 판정이 아니다 — 인계로 위장하면 평가 지표가 오염된다."""
    client = scripted_client({DRAFT_STAGE: [citing_draft()]})
    judge = ScriptedJudge([MissingCredentialsError("ANTHROPIC_API_KEY 가 설정되지 않았다")])

    with pytest.raises(MissingCredentialsError):
        run(l2_pipeline(client, judge))


# ── 이음매: 실제 `judge.Judge` 를 배선한 루프 (외부 호출 0회, DB 불필요) ─────
#
# 위의 L2 케이스는 전부 `ScriptedJudge` 목이 받는다 — 목은 어떤 키워드 조합도 삼키므로
# 인자명(`draft=`/`evidence=`)이나 `JudgeOutcome` 필드명이 어긋나도 전부 녹색이다.
# 아래 두 건은 **실제 판정 모듈**을 대본형 생성 클라이언트 위에 얹어 그 이음매를 본다.


def judge_completion(
    *,
    claim_text: str,
    reasons: Sequence[str] = (),
    input_tokens: int = 40,
    output_tokens: int = 9,
) -> JsonCompletion:
    """실제 `Judge` 의 파서를 통과하는 판정 산출 (claim 1개짜리 초안용).

    정합성 규칙상 사유가 있으면 verdict 는 reject 이고 claim 판정도 reject 여야 한다.
    """
    verdict = "reject" if reasons else "pass"
    return JsonCompletion(
        data={
            "claim_judgments": [
                {"claim_text": claim_text, "verdict": verdict, "explanation": "판정 사유 상세."}
            ],
            "contradictions": [],
            "verdict": verdict,
            "reject_reasons": list(reasons),
        },
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def real_judge_pipeline(client: ScriptedGenerationClient) -> InquiryPipeline:
    """생성도 판정도 같은 대본형 클라이언트를 쓰는 조립 — 판정자만 **실제 `Judge`** 다."""
    return pipeline_with(
        collector=StubCollector(collection(evidence=[POLICY_EVIDENCE])),
        client=client,
        judge=Judge(client=cast(GenerationClient, client)),
        l2_enabled=True,
    )


def test_실제_판정자로도_L2_기각_후_재생성이_통과한다() -> None:
    """목 없이 파이프라인 ↔ 판정 모듈 이음매를 끝까지 돈다 (외부 호출 0회)."""
    client = scripted_client(
        {
            DRAFT_STAGE: [
                citing_draft(text="첫 초안입니다."),
                citing_draft(text="고친 초안입니다."),
            ],
            JUDGE_STAGE: [
                judge_completion(
                    claim_text="첫 초안입니다.",
                    reasons=("unsupported_claim",),
                    input_tokens=100,
                    output_tokens=20,
                ),
                judge_completion(claim_text="고친 초안입니다.", input_tokens=30, output_tokens=4),
            ],
        }
    )

    processed = run(real_judge_pipeline(client))

    assert processed.status is InquiryStatus.ANSWERED
    assert processed.answer == "고친 초안입니다."
    assert [(a.attempt_no, a.verdict) for a in processed.attempts] == [
        (1, Verdict.REJECT),
        (2, Verdict.PASS),
    ]
    assert processed.attempts[0].reject_reasons == (RejectReason.UNSUPPORTED_CLAIM,)
    # 판정 호출에 초안 claim 과 수집 근거가 실렸다 — 인자명이 어긋나면 여기서 깨진다.
    judge_calls = client.calls_for(JUDGE_STAGE)
    assert len(judge_calls) == 2
    assert "첫 초안입니다." in judge_calls[0]["user"]
    assert POLICY_EVIDENCE.evidence_text in judge_calls[0]["user"]
    # 판정 결과가 재생성 피드백으로 돌아온다 (claim 단위 상세까지).
    assert "판정 사유 상세." in client.calls_for(DRAFT_STAGE)[1]["user"]
    # 판정 토큰은 분리 집계된다 — 생성 계열은 수집(11/3) + 초안 2회(5/2 씩).
    assert (processed.judge_input_tokens, processed.judge_output_tokens) == (130, 24)
    assert (processed.input_tokens, processed.output_tokens) == (21, 7)


def test_판정_형식_실패_뒤_전송_오류라도_과금된_판정_토큰이_남는다() -> None:
    """1회차 200(형식 실패)로 과금된 뒤 2회차가 전송 오류 — 토큰이 0 으로 사라지지 않는다.

    "실행됐으나 실패한 호출의 토큰도 그대로 집계한다"(docs/contracts.md "토큰 집계 경계").
    파이프라인이
    여기서 값을 잃으면 처리 기록·API 응답이 판정 토큰을 싣는 뒤 태스크 시점에 건당 비용
    지표가 거짓말을 한다.
    """
    client = scripted_client(
        {
            DRAFT_STAGE: [citing_draft(text="첫 초안입니다.")],
            JUDGE_STAGE: [
                LLMFormatError(
                    stage=JUDGE_STAGE,
                    detail="JSON 파싱 실패",
                    raw_text="이건 JSON 이 아니다",
                    input_tokens=30,
                    output_tokens=5,
                ),
                LLMCallError(stage=JUDGE_STAGE, reason="transport_error", attempts=2),
            ],
        }
    )

    processed = run(real_judge_pipeline(client))

    assert processed.status is InquiryStatus.ESCALATED
    assert processed.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert processed.failed_stage == L2_JUDGE_STAGE
    assert processed.answer is None
    # 판정 시도는 2회(형식 실패 + 전송 오류), 과금된 것은 1회차분이다.
    assert len(client.calls_for(JUDGE_STAGE)) == 2
    assert (processed.judge_input_tokens, processed.judge_output_tokens) == (30, 5)
    # 생성 합산은 수집(11/3) + 초안 1회(5/2) 그대로 — 판정 토큰이 섞이지 않는다.
    assert (processed.input_tokens, processed.output_tokens) == (16, 5)


# ── 배선 규칙 (fail-open 차단) ──────────────────────────────────────────────


def test_판정_결과가_비면_통과로_접지_않는다() -> None:
    """L2 를 돌린 시도인데 판정이 비어 있으면 fail-closed 다 — 조용히 answered 로 가지 않는다.

    오늘의 `judge.Judge` 로는 도달할 수 없는 상태지만, fail-closed 는 **배선 실수에도**
    성립해야 한다: `l2_result is None` 을 "L2 통과"로 접으면 판정 없는 초안이 확정된다.
    """
    client = scripted_client({DRAFT_STAGE: [citing_draft()]})
    empty = JudgeOutcome(
        result=cast(JudgeResult, None), input_tokens=40, output_tokens=9, attempts=1
    )

    with pytest.raises(PipelineWiringError):
        run(l2_pipeline(client, ScriptedJudge([empty])))


def test_스위치_켜짐_판정자_미배선은_조립_시점_오류다() -> None:
    """조용히 L2 를 건너뛰는 경로를 두지 않는다 — fail-closed 는 배선 실수에도 성립한다."""
    with pytest.raises(PipelineWiringError):
        InquiryPipeline(
            collector=cast(EvidenceCollecting, StubCollector(collection())),
            drafter=cast(DraftGenerating, None),
            l2_enabled=True,
        )


def test_스위치_꺼짐이거나_판정자가_있으면_조립된다() -> None:
    """양성 대조 — 전부 거부하는 배선 규칙은 규칙이 아니다."""
    collector = cast(EvidenceCollecting, StubCollector(collection()))
    drafter = cast(DraftGenerating, None)

    assert InquiryPipeline(collector=collector, drafter=drafter, l2_enabled=False) is not None
    assert (
        InquiryPipeline(
            collector=collector,
            drafter=drafter,
            judge=cast(Judging, ScriptedJudge([])),
            l2_enabled=True,
        )
        is not None
    )


def _built_judge(settings: Settings) -> Judge:
    pipeline = build_pipeline(
        generation_client=cast(GenerationClient, scripted_client({})),
        embedding_client=cast(EmbeddingClient, LexicalEmbeddingClient(dimensions=1536)),
        settings=settings,
    )
    return cast(Judge, pipeline._judge)


def test_판정_기본_배선은_Anthropic_계열이다() -> None:
    """같은 계열로 바꾸면 self-judging bias 로 검출률이 오염된다 (결정 0004)."""
    settings = Settings(
        l2_enabled=True,
        anthropic_api_key="키가-아닌-테스트값",
        judge_effort="low",
        judge_max_output_tokens=4321,
    )
    judge = _built_judge(settings)
    client = cast(LazyJudgeClient, judge._client)
    resolved = client._resolve()

    assert isinstance(resolved, AnthropicGenerationClient)
    assert not isinstance(resolved, OpenAIGenerationClient)
    assert resolved.model == settings.judge_model
    # 설정 상한이 판정자에 실린다 (하드코딩 금지).
    assert judge._effort == "low"
    assert judge._max_output_tokens == 4321


def test_판정_클라이언트는_첫_호출_때_만들어진다() -> None:
    """조립은 키를 요구하지 않는다 — 조회 전용 경로가 키 없이도 살아 있어야 한다."""
    judge = _built_judge(Settings(l2_enabled=True, anthropic_api_key=""))
    client = cast(LazyJudgeClient, judge._client)

    # `anthropic.Anthropic(api_key="")` 는 생성자에서 예외를 던지지 않는다 — 키 검사는
    # 설정값을 명시적으로 봐야 하고, 부재는 인계가 아니라 설정 오류다.
    with pytest.raises(MissingCredentialsError):
        client._resolve()


def test_build_pipeline_은_스위치가_켜져_있어도_조립에_성공한다() -> None:
    """하위호환은 조립 함수가 진다 — 설정 기반 판정자를 항상 배선한다."""
    pipeline = build_pipeline(
        generation_client=cast(GenerationClient, scripted_client({})),
        embedding_client=cast(EmbeddingClient, LexicalEmbeddingClient(dimensions=1536)),
        settings=Settings(l2_enabled=True, anthropic_api_key=""),
    )

    assert pipeline._judge is not None


# ── 인계 사유 6종 (실제 근거 수집기 + 시딩된 DB) ────────────────────────────


@pytest.fixture
def indexed_policies(app_conn: psycopg.Connection[DictRow]) -> None:
    """저장소 정책 문서를 결정론 임베딩으로 적재한다 (픽스처 롤백으로 되돌아간다)."""
    index_policy_documents(
        conn=app_conn,
        documents=load_policy_documents(),
        embedder=LexicalEmbeddingClient(dimensions=1536),
    )


@pytest.fixture
def seeded_order_no(ro_conn: psycopg.Connection[DictRow]) -> str:
    row = ro_conn.execute("SELECT order_no FROM orders ORDER BY order_no LIMIT 1").fetchone()
    assert row is not None, "시딩된 주문이 있어야 한다"
    return str(row["order_no"])


def live_pipeline(client: ScriptedGenerationClient, *, threshold: float = 0.0) -> InquiryPipeline:
    """실제 근거 수집기 + 시딩된 DB 로 도는 조립 — **L2 는 꺼서** 사이클 1 동작으로 본다.

    인계 사유 6종·응답 골격은 판정 층과 무관하고, 판정을 켜면 확률 층 대역이 필요해진다
    (그 대역은 T9 이 `testing.py` 에 넣는다).
    """
    return build_pipeline(
        generation_client=cast(GenerationClient, client),
        embedding_client=LexicalEmbeddingClient(dimensions=1536),
        settings=Settings(
            vector_top_k=5,
            vector_similarity_threshold=threshold,
            sql_max_rows=50,
            l2_enabled=False,
        ),
    )


def run_live(
    pipeline: InquiryPipeline,
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    *,
    content: str = INQUIRY,
    order_no: str | None = None,
) -> Any:
    return pipeline.run(
        inquiry_id=new_inquiry_id(),
        content=content,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_인계_no_evidence(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """임계값을 넘는 정책 청크가 하나도 없으면 근거 0건 인계."""
    client = scripted_client({INTENT_STAGE: [intent_completion("policy")]})
    processed = run_live(live_pipeline(client, threshold=0.99), app_conn, ro_conn)

    assert processed.status is InquiryStatus.ESCALATED
    assert processed.escalation_reason is EscalationReason.NO_EVIDENCE
    assert processed.evidence == ()
    assert processed.attempts == ()


@pytest.mark.db
def test_인계_missing_order_ref(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    client = scripted_client({INTENT_STAGE: [intent_completion("order")]})
    processed = run_live(live_pipeline(client), app_conn, ro_conn, content="제 주문 어디쯤 왔나요?")

    assert processed.escalation_reason is EscalationReason.MISSING_ORDER_REF


@pytest.mark.db
def test_인계_order_not_found(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    client = scripted_client({INTENT_STAGE: [intent_completion("order")]})
    processed = run_live(
        live_pipeline(client),
        app_conn,
        ro_conn,
        content="제 주문 어디쯤 왔나요?",
        order_no=MISSING_ORDER_NO,
    )

    assert processed.escalation_reason is EscalationReason.ORDER_NOT_FOUND
    # 존재성 선검사에서 끊겼으므로 text-to-SQL 에 진입하지 않는다.
    assert client.calls_for(SQL_GENERATION_STAGE) == []


@pytest.mark.db
def test_인계_sql_failed(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    seeded_order_no: str,
) -> None:
    """안전장치가 두 번 모두 거부하면 sql_failed — 실패 내역이 처리 기록에 남는다."""
    client = scripted_client(
        {
            INTENT_STAGE: [intent_completion("order")],
            SQL_GENERATION_STAGE: [
                sql_completion("DELETE FROM orders"),
                sql_completion("SELECT * FROM pg_catalog.pg_user"),
            ],
        }
    )
    processed = run_live(
        live_pipeline(client),
        app_conn,
        ro_conn,
        content="제 주문 상태를 알려주세요.",
        order_no=seeded_order_no,
    )

    assert processed.escalation_reason is EscalationReason.SQL_FAILED
    assert len(processed.sql_failures) == 2
    assert [failure.attempt_no for failure in processed.sql_failures] == [1, 2]


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_인계_llm_call_failed(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    client = scripted_client(
        {
            INTENT_STAGE: [intent_completion("policy")],
            DRAFT_STAGE: [LLMCallError(stage=DRAFT_STAGE, reason="transport_error", attempts=2)],
        }
    )
    processed = run_live(live_pipeline(client), app_conn, ro_conn)

    assert processed.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert processed.failed_stage == DRAFT_STAGE
    # 초안 전에 모은 정책 근거는 감사 목적으로 남는다.
    assert processed.evidence != ()


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_인계_rejected_twice(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    bogus = draft_completion(
        {"claims": [{"text": "환불은 30일 이내입니다.", "citation_ids": ["policy:없는문서:1"]}]}
    )
    client = scripted_client(
        {INTENT_STAGE: [intent_completion("policy")], DRAFT_STAGE: [bogus, bogus]}
    )
    processed = run_live(live_pipeline(client), app_conn, ro_conn)

    assert processed.escalation_reason is EscalationReason.REJECTED_TWICE
    assert len(processed.attempts) == 2
    assert all(a.reject_reasons == (RejectReason.INVALID_CITATION,) for a in processed.attempts)


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_실제_정책_근거로_답변이_확정된다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    client = scripted_client(
        {INTENT_STAGE: [intent_completion("policy")], DRAFT_STAGE: [citing_draft()]}
    )
    processed = run_live(live_pipeline(client), app_conn, ro_conn)

    assert processed.status is InquiryStatus.ANSWERED
    assert processed.intent is IntentSource.POLICY
    assert processed.evidence != ()
    assert all(item.source is EvidenceSource.POLICY for item in processed.evidence)
