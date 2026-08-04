"""파이프라인 루프 테스트 — 접수 → 근거 수집 → 초안 → L1 → 종결.

생성 LLM 은 전부 목이다(실제 API 키를 쓰지 않는다). 임베딩은 `LexicalEmbeddingClient`
결정론 대역이다.

두 층으로 나눠 검증한다.

* **DB 없는 층** — 근거 수집을 대역으로 갈아 끼워 루프 계약만 본다: 재생성 1회 상한,
  근거 재수집 금지, 기각 사유 전량 피드백, 형식 불일치 → `schema_violation` 루프,
  초안 전 인계의 근거 보존.
* **DB 층(`db` 마커)** — 실제 `EvidenceCollector` + 시딩된 Postgres 로 인계 사유 6종을
  끝까지 재현한다.

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
    EscalationReason,
    Evidence,
    EvidenceSource,
    InquiryStatus,
    IntentSource,
    RejectReason,
    Verdict,
)
from reply_gate.draft import DRAFT_STAGE, DraftGenerator
from reply_gate.evidence import (
    INTENT_STAGE,
    SQL_GENERATION_STAGE,
    EvidenceCollection,
)
from reply_gate.llm import GenerationClient, JsonCompletion, LLMCallError, LLMFormatError
from reply_gate.pipeline import (
    MAX_DRAFT_ATTEMPTS,
    AcceptedInquiry,
    DraftGenerating,
    EvidenceCollecting,
    InquiryPipeline,
    ReceiptError,
    accept_inquiry,
    build_pipeline,
    new_inquiry_id,
)
from reply_gate.policy_index import index_policy_documents, load_policy_documents
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
    """

    def __init__(self, script: Mapping[str, Sequence[Any]]) -> None:
        self._script = {stage: list(outcomes) for stage, outcomes in script.items()}
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        stage = kwargs["stage"]
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
    *, collector: Any, client: ScriptedGenerationClient | None = None
) -> InquiryPipeline:
    generator = DraftGenerator(client=cast(GenerationClient, client)) if client else None
    return InquiryPipeline(
        collector=cast(EvidenceCollecting, collector),
        drafter=cast(DraftGenerating, generator),
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
    return build_pipeline(
        generation_client=cast(GenerationClient, client),
        embedding_client=LexicalEmbeddingClient(dimensions=1536),
        settings=Settings(
            vector_top_k=5,
            vector_similarity_threshold=threshold,
            sql_max_rows=50,
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
