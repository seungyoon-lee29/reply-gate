"""단계별 지연 계측 테스트 — 구간 아홉을 재고, **미측정을 0 과 가른다**.

지금까지 지연 필드는 셋뿐이었다(케이스별 전체 시간, 세트별 p50·p95). "초안 몇 초 /
판정 몇 초"를 산출물로 말할 수 없었다는 뜻이다. 이 파일이 지키는 계약은 다섯이다.

1. **구간 합이 전체 시간을 넘지 않는다** — 구간은 전부 처리 창 **안쪽**에서 잰다.
2. **미측정은 0 이 아니다** — 돌지 않은 구간은 `None` 이고, 집계 분모에서 빠진다.
3. **한 구간의 시간은 그 구간의 총 벽시계다** — 전송 재시도·형식 실패·예외로 죽은 호출의
   경과가 전부 그 구간에 든다. 이 저장소는 토큰에서 이미 같은 사고를 겪었다(전송 3회를
   `attempts=2` 로 신고). 성공한 마지막 호출만 재면 같은 사고를 지연 축에서 반복한다.
4. **재생성이 돌면 시도별로 쌓인다** — 합계와 시도별 값을 함께 알 수 있어야 한다.
5. **리포트 JSON 과 사람이 읽는 줄이 같은 값을 적는다** — 커밋된 리포트에서 두 표면이
   갈린 전례가 있다.

시간에 의존하는 단언은 **결정론**이어야 한다. 밖으로 나가는 호출 구간은 호출 래퍼가 재므로
래퍼의 시계를 목으로 고정하거나, 대역이 돌려주는 경과 값을 직접 지정해서 잰다. 실제 벽시계를
그대로 단언하는 검사는 두지 않는다 — 값이 흔들려 CI 가 무작위로 붉어진다.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import fields, replace
from typing import Any, cast

import psycopg
import pytest
from psycopg.rows import DictRow

from reply_gate import evidence, llm
from reply_gate.api import AttemptOut, InquiryResponse, MetricsOut
from reply_gate.config import Settings
from reply_gate.contracts import (
    EscalationReason,
    Evidence,
    EvidenceSource,
    GateResult,
    InquiryStatus,
    IntentSource,
    RejectReason,
    Verdict,
)
from reply_gate.draft import DRAFT_STAGE
from reply_gate.evaluation import (
    JudgeAccuracy,
    SpanAggregate,
    evaluate_judge_fixture,
    measure_judge_accuracy,
    measure_pipeline_agreement,
    render_markdown,
    report_to_json,
)
from reply_gate.evidence import (
    INQUIRY_EMBEDDING_STAGE,
    INTENT_STAGE,
    SPAN_NAMES,
    SQL_GENERATION_STAGE,
    EvidenceCollector,
    SqlFailureKind,
    StageDurations,
    classify_intent,
)
from reply_gate.judge import JUDGE_STAGE, Judge
from reply_gate.llm import EmbeddingResult, JsonCompletion, LLMCallError, LLMFormatError
from reply_gate.pipeline import (
    AttemptDurations,
    AttemptRecord,
    new_inquiry_id,
)
from reply_gate.policy_index import index_policy_documents, load_policy_documents
from reply_gate.query_rewrite import QUERY_REWRITE_STAGE
from reply_gate.records import load_inquiry, save_inquiry
from reply_gate.testing import LexicalEmbeddingClient
from tests.test_api import ATTEMPT_KEYS, METRICS_KEYS, RESPONSE_KEYS
from tests.test_evaluation import (
    GOLDEN,
    JUDGE_FIXTURES,
    OracleJudge,
    ScriptedPipeline,
    _conditions,
    _processed,
    _report,
)
from tests.test_evidence import INQUIRY as EVIDENCE_INQUIRY
from tests.test_evidence import INQUIRY_ID as EVIDENCE_INQUIRY_ID
from tests.test_evidence import MISSING_ORDER_NO as EVIDENCE_MISSING_ORDER_NO
from tests.test_evidence import _client as _evidence_client
from tests.test_evidence import _collector as _evidence_collector
from tests.test_judge import DOMESTIC, _all_pass_payload, _RecordingClient
from tests.test_judge import _draft as _judge_draft
from tests.test_llm_client import _connection_error, _generation_client, _response
from tests.test_pipeline import (
    POLICY_EVIDENCE,
    ScriptedJudge,
    StubCollector,
    citing_draft,
    collection,
    draft_completion,
    intent_completion,
    judge_pass,
    judge_reject,
    live_pipeline,
    pipeline_with,
    run,
    run_live,
    scripted_client,
)

INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"intent": {"type": "string"}},
    "required": ["intent"],
    "additionalProperties": False,
}


class _Clock:
    """읽을 때마다 대본의 다음 값을 돌려주는 결정론 시계(초 단위, `perf_counter` 대체)."""

    def __init__(self, ticks: list[float]) -> None:
        self._ticks = list(ticks)
        self.reads = 0

    def __call__(self) -> float:
        value = self._ticks[min(self.reads, len(self._ticks) - 1)]
        self.reads += 1
        return value


def _completion(data: Any, *, elapsed_ms: float) -> JsonCompletion:
    return JsonCompletion(data=data, input_tokens=1, output_tokens=1, elapsed_ms=elapsed_ms)


def _first_order_no(ro_conn: psycopg.Connection[DictRow]) -> str:
    row = ro_conn.execute("SELECT order_no FROM orders ORDER BY order_no LIMIT 1").fetchone()
    assert row is not None, "시딩된 주문이 있어야 한다"
    return str(row["order_no"])


class _QueuedClient:
    """`GenerationClient` 대역 — 단계별 대본을 순서대로 돌려주고 예외는 던진다."""

    def __init__(self, script: dict[str, list[Any]]) -> None:
        self._script = {stage: list(items) for stage, items in script.items()}

    def complete_json(self, **kwargs: Any) -> Any:
        stage = kwargs["stage"]
        queue = self._script.get(stage)
        if not queue:
            raise AssertionError(f"대본에 없는 호출이다: stage={stage!r}")
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


# ── 1. 호출 래퍼 — 재시도·형식 실패·예외의 경과가 전부 구간에 든다 ──────────


def test_전송_재시도가_난_호출의_경과는_두_시도를_합친_벽시계다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """성공한 마지막 호출만 재면 래퍼 재시도 1회가 통째로 사라진다.

    같은 사고를 이 저장소는 토큰 축에서 이미 겪었다(전송 3회를 `attempts=2` 로 신고).
    """
    clock = _Clock([0.0, 0.030])
    monkeypatch.setattr(llm, "perf_counter", clock)
    client, responses = _generation_client(
        [_connection_error(), _response(json.dumps({"intent": "policy"}))]
    )

    completion = client.complete_json(
        stage="intent", system="s", user="u", schema=INTENT_SCHEMA, schema_name="intent"
    )

    assert len(responses.calls) == 2, "재시도가 실제로 일어나야 이 검사가 의미를 갖는다"
    assert completion.elapsed_ms == pytest.approx(30.0)


def test_예외로_죽은_호출의_경과도_예외에_실려_올라온다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """토큰과 같은 자격이다 — 실패한 호출도 시간을 썼다."""
    monkeypatch.setattr(llm, "perf_counter", _Clock([0.0, 0.045]))
    client, responses = _generation_client([_connection_error(), _connection_error()])

    with pytest.raises(LLMCallError) as caught:
        client.complete_json(
            stage="intent", system="s", user="u", schema=INTENT_SCHEMA, schema_name="intent"
        )

    assert len(responses.calls) == llm.MAX_ATTEMPTS
    assert caught.value.elapsed_ms == pytest.approx(45.0)


def test_형식_불일치로_버려진_산출의_경과도_실려_올라온다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "perf_counter", _Clock([0.0, 0.012]))
    client, _responses = _generation_client([_response("JSON 이 아니다")])

    with pytest.raises(LLMFormatError) as caught:
        client.complete_json(
            stage="intent", system="s", user="u", schema=INTENT_SCHEMA, schema_name="intent"
        )

    assert caught.value.elapsed_ms == pytest.approx(12.0)


# ── 2. 의도 분류 구간 — 형식 루프가 토큰과 같은 자격으로 누적한다 ───────────


def test_의도_분류_구간은_형식_재시도의_경과를_합산한다() -> None:
    client = _QueuedClient(
        {
            INTENT_STAGE: [
                LLMFormatError(stage=INTENT_STAGE, detail="빈 응답", raw_text="", elapsed_ms=5.0),
                _completion({"source": "policy"}, elapsed_ms=7.0),
            ]
        }
    )

    result = classify_intent(client=cast(llm.GenerationClient, client), inquiry="문의")

    assert result.source is IntentSource.POLICY
    assert result.elapsed_ms == pytest.approx(12.0)


def test_의도_분류가_예외로_죽어도_앞선_시도의_경과가_따라_올라온다() -> None:
    client = _QueuedClient(
        {
            INTENT_STAGE: [
                LLMFormatError(stage=INTENT_STAGE, detail="빈 응답", raw_text="", elapsed_ms=5.0),
                LLMCallError(
                    stage=INTENT_STAGE,
                    reason="transport_error",
                    attempts=2,
                    elapsed_ms=9.0,
                ),
            ]
        }
    )

    with pytest.raises(LLMCallError) as caught:
        classify_intent(client=cast(llm.GenerationClient, client), inquiry="문의")

    assert caught.value.elapsed_ms == pytest.approx(14.0)


# ── 3. 파이프라인 — 시도별로 쌓이고, 안 돈 구간은 미측정이다 ────────────────


def _judge_client_pipeline(*, drafts: list[Any], judgments: list[Any]) -> Any:
    return pipeline_with(
        collector=StubCollector(collection(evidence=[POLICY_EVIDENCE])),
        client=scripted_client({DRAFT_STAGE: drafts}),
        judge=ScriptedJudge(judgments),
        l2_enabled=True,
    )


def test_초안_전_인계는_초안_게이트_판정_구간을_미측정으로_남긴다() -> None:
    """0 으로 찍으면 "0 초 만에 초안을 만들었다"가 되어 리포트가 거짓말을 한다."""
    pipeline = pipeline_with(
        collector=StubCollector(collection(evidence=[], escalation=EscalationReason.NO_EVIDENCE)),
        client=scripted_client({}),
    )

    processed = run(pipeline)

    assert processed.escalation_reason is EscalationReason.NO_EVIDENCE
    assert processed.stage_durations.draft_ms is None
    assert processed.stage_durations.gate_ms is None
    assert processed.stage_durations.l2_judge_ms is None


def test_판정이_돌지_않은_시도의_판정_구간은_미측정이다() -> None:
    """L2 스위치가 꺼진 실행에서 `l2_judge_ms` 를 0 으로 적으면 "공짜로 판정했다"가 된다."""
    pipeline = pipeline_with(
        collector=StubCollector(collection(evidence=[POLICY_EVIDENCE])),
        client=scripted_client({DRAFT_STAGE: [citing_draft()]}),
    )

    processed = run(pipeline)

    assert processed.attempts[0].durations.l2_judge_ms is None
    assert processed.attempts[0].durations.gate_ms is not None
    assert processed.stage_durations.l2_judge_ms is None


def test_재생성_실행은_시도별로_쌓이고_합계가_시도_합과_같다() -> None:
    rejected = draft_completion({"claims": [{"text": "환불 가능합니다.", "citation_ids": []}]})
    pipeline = _judge_client_pipeline(
        drafts=[rejected, citing_draft()],
        judgments=[judge_pass()],
    )

    processed = run(pipeline)

    assert [attempt.attempt_no for attempt in processed.attempts] == [1, 2]
    per_attempt = [attempt.durations for attempt in processed.attempts]
    # 1차는 L1 기각이라 판정이 돌지 않았다 — 미측정이고 0 이 아니다.
    assert per_attempt[0].l2_judge_ms is None
    assert per_attempt[1].l2_judge_ms is not None
    total_gate = sum(item.gate_ms or 0.0 for item in per_attempt)
    assert processed.stage_durations.gate_ms == pytest.approx(total_gate)


def test_판정_호출이_실패한_시도도_판정_구간을_남긴다() -> None:
    """실패한 호출도 시간을 썼다 — 토큰과 같은 규칙이다."""
    pipeline = _judge_client_pipeline(
        drafts=[citing_draft()],
        judgments=[
            LLMCallError(stage=JUDGE_STAGE, reason="transport_error", attempts=2, elapsed_ms=33.0)
        ],
    )

    processed = run(pipeline)

    assert processed.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert processed.stage_durations.l2_judge_ms == pytest.approx(33.0)
    assert processed.attempts[0].durations.l2_judge_ms == pytest.approx(33.0)


def test_초안_생성이_예외로_죽어도_초안_구간이_남는다() -> None:
    pipeline = pipeline_with(
        collector=StubCollector(collection(evidence=[POLICY_EVIDENCE])),
        client=scripted_client(
            {
                DRAFT_STAGE: [
                    LLMCallError(
                        stage=DRAFT_STAGE,
                        reason="transport_error",
                        attempts=2,
                        elapsed_ms=21.0,
                    )
                ]
            }
        ),
    )

    processed = run(pipeline)

    assert processed.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert processed.stage_durations.draft_ms == pytest.approx(21.0)
    # 그 시도는 게이트에 닿지도 못했다 — 0 이 아니라 미측정이다.
    assert processed.stage_durations.gate_ms is None


def test_구간_합이_전체_시간을_넘지_않는다() -> None:
    """구간은 전부 처리 창 **안쪽**에서 잰다 — 밖에서 감싸면 이 부등식이 깨진다."""
    pipeline = _judge_client_pipeline(drafts=[citing_draft()], judgments=[judge_pass()])

    processed = run(pipeline)

    measured = processed.stage_durations.measured_total_ms
    assert measured is not None
    # `latency_ms` 는 반올림된 정수라 1 ms 미만의 반올림 차를 허용한다.
    assert measured <= processed.latency_ms + 1.0


def test_게이트_구간은_호출자가_잰다() -> None:
    """`gate.py` 는 시간에 의존하지 않는다(하드 게이트 1) — 구간은 파이프라인이 잰다.

    게이트 모듈이 `time` 을 import 하지 않는다는 것은 `tests/test_gate.py` 의 AST 구조
    검사가 지킨다. 여기서 지키는 것은 그 제약 아래에서도 **구간이 실제로 채워진다**는 것이다.
    """
    pipeline = pipeline_with(
        collector=StubCollector(collection(evidence=[POLICY_EVIDENCE])),
        client=scripted_client({DRAFT_STAGE: [citing_draft()]}),
    )

    processed = run(pipeline)

    gate_ms = processed.stage_durations.gate_ms
    assert gate_ms is not None and gate_ms >= 0.0


# ── 4. 시도 기록은 DB 왕복 구조다 — 새 칸은 기본값을 갖고 실리지 않는다 ─────


def test_시도_기록은_구간_없이도_만들어진다() -> None:
    """처리 기록 복원(`records._load_attempts`)이 생성자를 직접 부른다."""
    record = AttemptRecord(
        attempt_no=1,
        verdict=Verdict.PASS,
        reject_reasons=(),
        draft={"claims": []},
    )

    assert record.durations == AttemptDurations()
    assert record.durations.draft_ms is None


@pytest.mark.db
def test_처리_기록_복원은_구간_시간을_들고_오지_않는다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """DB 스키마 무변경이 계약이다 — 계측은 결과 객체와 리포트까지만 간다."""
    base = _processed(
        attempts=(
            AttemptRecord(
                attempt_no=1,
                verdict=Verdict.PASS,
                reject_reasons=(),
                draft={"claims": [{"text": "답변", "citation_ids": ["policy:refund:2-1"]}]},
                l1_result=GateResult(verdict=Verdict.PASS),
                durations=AttemptDurations(draft_ms=12.0, gate_ms=0.4, l2_judge_ms=88.0),
            ),
        ),
    )
    processed = replace(
        base,
        inquiry_id=new_inquiry_id(),
        stage_durations=StageDurations(intent_ms=3.0, draft_ms=12.0),
    )

    save_inquiry(conn=app_conn, processed=processed)
    loaded = load_inquiry(conn=app_conn, inquiry_id=processed.inquiry_id)

    assert loaded is not None
    assert loaded.stage_durations == StageDurations()
    assert loaded.attempts[0].durations == AttemptDurations()
    # 복원 자체는 그대로 산다 — 나머지 필드는 저장한 값과 같다.
    assert loaded.attempts[0].verdict is Verdict.PASS


# ── 5. 근거 수집 모듈 안쪽의 여섯 구간 ──────────────────────────────────────


@pytest.fixture
def indexed_policies(app_conn: psycopg.Connection[DictRow]) -> None:
    """저장소 정책 문서를 결정론 임베딩으로 적재한다 (픽스처 롤백으로 되돌아간다)."""
    index_policy_documents(
        conn=app_conn,
        documents=load_policy_documents(),
        embedder=LexicalEmbeddingClient(dimensions=1536),
    )


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_정책_경로는_검색_구간을_재고_주문_구간은_미측정으로_남는다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """근거 수집 **안쪽** 여섯 구간이 바깥에서 한 칸으로 접히면 이 검사가 깨진다."""
    client = scripted_client(
        {
            INTENT_STAGE: [intent_completion("policy")],
            DRAFT_STAGE: [citing_draft()],
        }
    )
    processed = run_live(live_pipeline(client), app_conn, ro_conn)

    spans = processed.stage_durations
    assert spans.intent_ms is not None
    assert spans.inquiry_embedding_ms is not None
    assert spans.policy_search_ms is not None
    assert spans.query_rewrite_ms is not None
    # 정책 단독 문의는 조회문을 만들지도, 조회를 돌리지도 않았다 — 0 이 아니라 미측정이다.
    assert spans.sql_generation_ms is None
    assert spans.sql_execution_ms is None
    measured = spans.measured_total_ms
    assert measured is not None and measured <= processed.latency_ms + 1.0


def test_한_구간도_재지_않았으면_합계는_0_이_아니라_미측정이다() -> None:
    """**"미측정은 0 이 아니다" 의 마지막 표면.**

    구간별 칸은 각각 `None` 을 지키는데 **합계가 0 을 내면** 리포트가 "이 문의는 0 ms 를
    썼다"고 말하게 된다 — 재지 않은 것과 0 초를 다시 뭉개는 자리다. 실제로 이 계약이
    한 줄(`return 0.0 if total is None else total`)로 뒤집혀도 스위트 전체가 초록이었다.

    양성 대조를 함께 둔다 — 한 칸이라도 재면 그 값이 그대로 합계가 되어야 한다.
    """
    빈_구간 = StageDurations()

    assert all(value is None for value in 빈_구간.as_mapping().values())
    assert 빈_구간.measured_total_ms is None

    # 양성 대조 — 한 칸만 재도 합계는 그 값이다(0 으로도, None 으로도 접히지 않는다).
    한_칸 = StageDurations(gate_ms=0.25)
    assert 한_칸.measured_total_ms == pytest.approx(0.25)

    # 0 ms 로 **측정된** 구간은 미측정이 아니다 — 둘을 값으로 가른다.
    영_밀리초 = StageDurations(gate_ms=0.0)
    assert 영_밀리초.measured_total_ms == pytest.approx(0.0)
    assert 영_밀리초.measured_total_ms is not None


@pytest.mark.db
def test_주문_경로는_조회문_생성과_조회_실행_구간을_잰다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
) -> None:
    order_no = _first_order_no(ro_conn)
    client = scripted_client(
        {
            INTENT_STAGE: [intent_completion("order")],
            SQL_GENERATION_STAGE: [
                _completion(
                    {"sql": f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'"},
                    elapsed_ms=4.0,
                )
            ],
            DRAFT_STAGE: [citing_draft()],
        }
    )

    processed = run_live(live_pipeline(client), app_conn, ro_conn, order_no=order_no)

    spans = processed.stage_durations
    assert spans.sql_generation_ms == pytest.approx(4.0)
    assert spans.sql_execution_ms is not None
    # 주문 단독 의도는 정책 검색을 아예 돌리지 않는다.
    assert spans.policy_search_ms is None
    assert spans.inquiry_embedding_ms is None


def test_수집기_결과는_구간_묶음을_함께_들고_나온다() -> None:
    """근거 묶음이 구간을 들고 나오지 않으면 파이프라인이 여섯을 한 칸으로 접는다."""
    assert hasattr(EvidenceCollector, "collect")
    empty = collection()
    assert empty.stage_durations == StageDurations()


# ── 6. 리포트 — 케이스별 값과 세트별 집계, 두 표면이 같은 값 ────────────────


def _agreement_with(durations: list[StageDurations]) -> Any:
    cases = GOLDEN[: len(durations)]
    results = [replace(_processed(), stage_durations=item, latency_ms=1000) for item in durations]
    return measure_pipeline_agreement(
        cases=cases,
        pipeline=ScriptedPipeline(results),
        app_conn=cast(psycopg.Connection[DictRow], None),
        readonly_conn=cast(psycopg.Connection[DictRow], None),
    )


#: 구간 표의 제목. **가드가 보는 표면을 이 절 하나로 좁힌다.**
_STAGE_SECTION_TITLE = "### 단계별 지연 (구간 아홉)"


def _stage_duration_rows(markdown: str) -> dict[str, str]:
    """구간 표의 `구간 이름 → 나머지 셀` 을 읽는다 — **그 절 블록 안에서만** 훑는다.

    문서 전체를 훑으면 조건 지문 표의 행(구간 이름을 백틱으로 감싼 첫 칸이 같은 모양이다)에도
    매치된다. dict 컴프리헨션이라 "뒤에 오는 행이 이긴다"는 **렌더 순서 우연**에 기대게 되고, 절
    순서가 바뀌면 조용히 다른 셀을 비교하거나 실패한다. 이 저장소가 이미 배운 규칙이 그것
    이다 — 같은 사고의 다음 변종은 늘 **가드가 안 보는 표면**으로 들어오고, 반대로 가드가
    너무 넓은 표면을 보면 우연히 맞는다.

    잘라내는 기준은 제목이고, 끝은 다음 절 제목이다. 블록을 못 찾으면 빈 결과를 돌려주므로
    호출부의 `assert rows` 가 그대로 문다.
    """
    block = re.search(
        rf"^{re.escape(_STAGE_SECTION_TITLE)}$\n(?P<body>.*?)(?=^\#{{2,3}} )",
        markdown,
        flags=re.MULTILINE | re.DOTALL,
    )
    if block is None:
        return {}
    return {
        match.group("span"): match.group("cells")
        for match in re.finditer(
            r"^\| `(?P<span>[a-z_0-9]+)` \|(?P<cells>.*)\|$",
            block.group("body"),
            flags=re.MULTILINE,
        )
    }


def _aggregate(agreement: Any, span: str) -> SpanAggregate:
    found = next(item for item in agreement.stage_durations if item.span == span)
    return cast(SpanAggregate, found)


def test_세트_집계는_미측정을_분모에서_뺀다() -> None:
    """0 을 섞어 평균 내면 그 구간이 실제보다 빨라 보인다."""
    agreement = _agreement_with(
        [
            StageDurations(intent_ms=10.0),
            StageDurations(intent_ms=20.0),
            StageDurations(intent_ms=None),
        ]
    )

    intent = _aggregate(agreement, "intent")

    assert intent.measured_cases == 2
    assert intent.unmeasured_cases == 1
    assert intent.total_ms == pytest.approx(30.0)
    assert intent.mean_ms == pytest.approx(15.0)


def test_전부_미측정인_구간의_집계는_0_이_아니라_미측정이다() -> None:
    agreement = _agreement_with([StageDurations(intent_ms=10.0), StageDurations(intent_ms=20.0)])

    gate = _aggregate(agreement, "gate")

    assert gate.measured_cases == 0
    assert gate.mean_ms is None
    assert gate.total_ms is None


def test_케이스별_구간과_시도별_구간이_리포트_JSON_에_실린다() -> None:
    agreement = _agreement_with([StageDurations(intent_ms=10.0, draft_ms=4.0)])
    payload = report_to_json(_report(pipeline=agreement))
    outcome = payload["measurement_2_pipeline_agreement"]["outcomes"][0]

    assert outcome["stage_durations"]["intent"] == pytest.approx(10.0)
    assert outcome["stage_durations"]["gate"] is None
    assert outcome["attempt_durations"] == []


def test_리포트_JSON_과_사람이_읽는_줄이_같은_값을_적는다() -> None:
    """정본이 갈린 전례가 있다 — 두 표면을 같은 값으로 못박는다."""
    agreement = _agreement_with(
        [StageDurations(intent_ms=10.0, draft_ms=4.0), StageDurations(intent_ms=20.0)]
    )
    report = _report(pipeline=agreement)
    markdown = render_markdown(report)
    payload = report_to_json(report)["measurement_2_pipeline_agreement"]["stage_durations"]

    rows = _stage_duration_rows(markdown)
    assert rows, "구간 표가 비면 이 검사는 아무것도 지키지 않는다"
    for item in payload:
        assert item["span"] in rows, f"{item['span']} 구간이 사람이 읽는 줄에 없다"
        cells = [cell.strip() for cell in rows[item["span"]].split("|")]
        assert cells[0] == str(item["measured_cases"])
        assert cells[1] == str(item["unmeasured_cases"])
        assert cells[3] == ("미측정" if item["mean_ms"] is None else f"{item['mean_ms']:.1f}")


def test_구간_표_대조는_그_절_블록만_본다() -> None:
    """가드가 보는 표면을 좁힌 것이 **실제로 일을 한다**는 증명.

    조건 지문 표도 구간 이름을 백틱으로 감싼 첫 칸을 갖는다(`query_rewrite` · `top_k` 등).
    문서 전체를 훑으면 그 행들이 함께 걸려, dict 컴프리헨션이 "뒤에 오는 행이 이긴다"는
    **렌더 순서 우연**으로 통과한다 — 절 순서가 바뀌면 조용히 다른 셀을 비교한다.
    """
    report = _report(pipeline=_agreement_with([StageDurations(intent_ms=10.0)]))
    markdown = render_markdown(report)

    narrowed = _stage_duration_rows(markdown)
    whole_document = {
        match.group("span")
        for match in re.finditer(
            r"^\| `(?P<span>[a-z_0-9]+)` \|.*\|$", markdown, flags=re.MULTILINE
        )
    }

    assert set(narrowed) == set(SPAN_NAMES)
    # 좁히기가 없으면 조건 지문 표의 행이 함께 걸린다 — 이 차집합이 비면 좁히기가
    # 아무것도 하지 않는다는 뜻이므로, 그때는 이 검사가 스스로 실패해야 한다.
    assert whole_document - set(SPAN_NAMES)


def test_구간_표_블록을_못_찾으면_대조가_비어_실패한다() -> None:
    """음성 대조 — 잘라내기가 빗나가면 `assert rows` 가 물도록 빈 결과를 돌려준다."""
    assert _stage_duration_rows("## 다른 리포트\n\n| `intent` | 1 | 0 |\n") == {}


def test_구간_이름_목록은_자료형에서_유도되고_비면_실패한다() -> None:
    """검사 대상 목록을 손으로 관리하지 않는다 — 자료형이 정본이다."""
    derived = tuple(field.name.removesuffix("_ms") for field in fields(StageDurations))

    assert derived, "유도한 목록이 비면 이 가드는 아무것도 지키지 않는다"
    assert derived == SPAN_NAMES
    assert len(SPAN_NAMES) == 9


def test_구간_이름_유도가_빠진_칸을_실제로_잡는다() -> None:
    """음성 대조 — 자료형에서 칸이 하나 빠지면 유도 결과가 정본과 갈린다."""
    derived = tuple(
        field.name.removesuffix("_ms")
        for field in fields(StageDurations)
        if field.name != "gate_ms"
    )

    assert derived != SPAN_NAMES
    assert "gate" not in derived


# ── 7. 판정 픽스처 측정도 판정 호출 지연을 기록한다 ─────────────────────────


def test_판정_픽스처_측정은_판정_호출_지연을_기록한다() -> None:
    accuracy = measure_judge_accuracy(fixtures=JUDGE_FIXTURES, judge=OracleJudge(JUDGE_FIXTURES))

    assert isinstance(accuracy, JudgeAccuracy)
    latency = accuracy.judge_latency
    assert latency is not None
    assert latency.measured_cases == len(JUDGE_FIXTURES)
    assert latency.mean_ms is not None


def test_판정에_실패한_픽스처도_경과를_남긴다() -> None:
    class _FailingJudge:
        def judge(self, **kwargs: Any) -> Any:
            del kwargs
            raise LLMFormatError(
                stage=JUDGE_STAGE, detail="형식 불일치", raw_text="", elapsed_ms=17.0
            )

    outcome = evaluate_judge_fixture(fixture=JUDGE_FIXTURES[0], judge=_FailingJudge())

    assert outcome.error is not None
    assert outcome.elapsed_ms == pytest.approx(17.0)


def test_판정_지연이_리포트_두_표면에_같은_값으로_실린다() -> None:
    accuracy = measure_judge_accuracy(fixtures=JUDGE_FIXTURES, judge=OracleJudge(JUDGE_FIXTURES))
    report = _report(
        pipeline=_agreement_with([StageDurations(intent_ms=1.0)]),
        judge=accuracy,
        conditions=_conditions(judge_is_real=True),
    )
    markdown = render_markdown(report)
    latency = report_to_json(report)["measurement_3_l2_judge_accuracy"]["judge_latency"]

    assert latency["measured_cases"] == len(JUDGE_FIXTURES)
    assert f"{latency['mean_ms']:.1f}" in markdown


# ── 8. 대역은 밖으로 나가지 않는다 — 그 구간은 0.0 이고 미측정이 아니다 ────


def test_대역의_밖으로_나가는_구간은_0_이고_결정론이다() -> None:
    """대역이 실제 시계를 재면 산출이 비결정론이 되어 `--stub-llm` 실행이 재현되지 않는다.

    **0.0 은 "재지 않았다"가 아니라 "밖으로 나간 시간이 0"이라는 측정값이다** — 그래서
    집계 분모에 들어가고, 미측정(`None`)과 구분된다. 대역 결정론은 이 저장소의 계약이고
    `tests/test_testing_doubles.py` 가 그것을 지킨다.
    """
    from reply_gate.testing import StubJudge

    fixture = JUDGE_FIXTURES[0]
    first = StubJudge().judge(draft=fixture.draft, evidence=fixture.evidences)
    second = StubJudge().judge(draft=fixture.draft, evidence=fixture.evidences)

    assert first.elapsed_ms == 0.0
    assert first == second


def test_임베딩_대역도_같은_규칙을_따른다() -> None:
    embedder = LexicalEmbeddingClient(dimensions=64)

    assert embedder.embed(stage="x", texts=["문의"]) == embedder.embed(stage="x", texts=["문의"])
    assert embedder.embed(stage="x", texts=["문의"]).elapsed_ms == 0.0


def test_대역_실행에서도_코드만_도는_구간은_실제로_잰다() -> None:
    """대역이 0 을 신고하는 것은 **밖으로 나가는 구간**뿐이다 — 게이트는 실제로 돌았다."""
    pipeline = pipeline_with(
        collector=StubCollector(collection(evidence=[POLICY_EVIDENCE])),
        client=scripted_client({DRAFT_STAGE: [citing_draft()]}),
    )

    processed = run(pipeline)

    assert processed.stage_durations.gate_ms is not None
    assert processed.stage_durations.draft_ms == 0.0


# ── 9. 다른 축이 흔들리지 않는다 ───────────────────────────────────────────


def test_계측은_판정을_바꾸지_않는다() -> None:
    """계측이 판정 경로를 건드리면 헤드라인 지표가 오염된다."""
    rejected = draft_completion({"claims": [{"text": "환불 가능합니다.", "citation_ids": []}]})
    pipeline = _judge_client_pipeline(
        drafts=[rejected, citing_draft()],
        judgments=[judge_reject(claim_text="안내드립니다.")],
    )

    processed = run(pipeline)

    assert processed.status is InquiryStatus.ESCALATED
    assert processed.escalation_reason is EscalationReason.REJECTED_TWICE
    assert [attempt.verdict for attempt in processed.attempts] == [
        Verdict.REJECT,
        Verdict.REJECT,
    ]
    assert RejectReason.MISSING_CITATION in processed.attempts[0].reject_reasons


def test_근거_묶음의_다른_칸은_그대로다() -> None:
    """계측 배선이 같은 자리를 지나는 다른 필드를 밀어내지 않는다."""
    item = collection(evidence=[POLICY_EVIDENCE])

    assert item.retrieval_fallback_reason is None
    assert item.abstention_undefined_reason is None
    assert isinstance(item.evidence[0], Evidence)
    assert item.evidence[0].source is EvidenceSource.POLICY


def test_응답_계약에_구간_칸이_생기지_않았다() -> None:
    """HTTP 응답 계약 확장 0 — 결과 객체가 넓어져도 표면은 그대로다.

    기대 키 목록은 응답 계약 검사(`tests/test_api.py`)가 소유한 것을 그대로 빌린다 —
    여기서 두 벌로 적으면 한쪽만 늘어난다.
    """
    assert set(MetricsOut.model_fields) == METRICS_KEYS
    assert set(InquiryResponse.model_fields) == RESPONSE_KEYS
    assert set(AttemptOut.model_fields) == ATTEMPT_KEYS
    assert not any(name.endswith("_ms") for name in AttemptOut.model_fields)


@pytest.mark.db
def test_처리_기록_스키마에_구간_칸이_생기지_않았다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """DB 스키마 무변경 — 칸을 더하면 볼륨 재생성과 유료 정책 재색인이 따라온다."""
    rows = app_conn.execute(
        "SELECT table_name, column_name FROM information_schema.columns"
        " WHERE table_schema = 'public' AND table_name IN ('inquiries', 'inquiry_attempts')"
    ).fetchall()
    columns = {(row["table_name"], row["column_name"]) for row in rows}

    assert columns, "검사 대상이 비면 이 가드는 아무것도 지키지 않는다"
    timing_columns = {item for item in columns if item[1].endswith("_ms")}
    # 전체 처리 시간 한 칸만 남는다 — 구간 아홉은 어느 테이블에도 없다.
    assert timing_columns == {("inquiries", "latency_ms")}


def test_구간_시간_누적은_미측정을_0_으로_접지_않는다() -> None:
    assert llm.accumulate_optional_ms(None, None) is None
    assert llm.accumulate_optional_ms(None, 3.0) == pytest.approx(3.0)
    assert llm.accumulate_optional_ms(2.0, None) == pytest.approx(2.0)
    assert llm.accumulate_optional_ms(2.0, 3.0) == pytest.approx(5.0)


def test_질의_재작성_단계_이름은_구간_이름과_같은_자리를_가리킨다() -> None:
    assert QUERY_REWRITE_STAGE in SPAN_NAMES
    assert INTENT_STAGE in SPAN_NAMES
    assert SQL_GENERATION_STAGE in SPAN_NAMES


# ── 10. 예외·`finally` 경로의 구간 적재 — 지우면 실패해야 한다 ──────────────
#
# 뮤테이션 검사가 이 절을 만들게 했다. 아래 일곱 자리는 **코드는 옳은데 지워도 스위트가
# 초록**이었다 — 전부 예외 경로이거나 `finally` 라 정상 흐름 검사가 지나가지 않는 자리다.
# 명세의 행동 계약이 "예외로 죽은 호출의 경과 시간도 토큰과 같은 자격으로 올려보낸다"를
# 명시하므로, 한 줄만 지워도 아무도 모르는 상태로 두지 않는다. 이 계측 위에서 유료 실측을
# 산다는 것이 그 규율의 이유다.


def _judge_client(outcomes: list[Any]) -> Judge:
    return Judge(client=cast(llm.GenerationClient, _RecordingClient(outcomes)))


def test_판정_구간은_형식_재시도로_버려진_시도의_경과를_합산한다() -> None:
    """`judge.py` 의 형식 루프 누적 — 마지막 시도만 재면 버려진 호출의 시간이 사라진다."""
    draft = _judge_draft(("국내 배송은 3일 이내입니다.", (DOMESTIC.id,)))
    judge = _judge_client(
        [
            LLMFormatError(stage=JUDGE_STAGE, detail="빈 응답", raw_text="", elapsed_ms=5.0),
            JsonCompletion(
                data=_all_pass_payload(draft),
                input_tokens=1,
                output_tokens=1,
                elapsed_ms=7.0,
            ),
        ]
    )

    outcome = judge.judge(draft=draft, evidence=[DOMESTIC])

    assert outcome.result.verdict is Verdict.PASS
    assert outcome.elapsed_ms == pytest.approx(12.0)


def test_판정이_전송_오류로_죽어도_앞선_형식_시도의_경과가_따라_올라온다() -> None:
    """토큰 축의 "전송 3회를 2회로 신고" 와 정확히 같은 실패 모양을 지연 축에서 막는다.

    다시 던지는 예외에 **누적분**을 싣지 않고 마지막 예외의 값만 실으면, 형식 실패로
    버려진 1차의 시간이 통째로 사라지는데 종결 기록은 멀쩡해 보인다.
    """
    draft = _judge_draft(("국내 배송은 3일 이내입니다.", (DOMESTIC.id,)))
    judge = _judge_client(
        [
            LLMFormatError(stage=JUDGE_STAGE, detail="빈 응답", raw_text="", elapsed_ms=5.0),
            LLMCallError(stage=JUDGE_STAGE, reason="transport_error", attempts=2, elapsed_ms=9.0),
        ]
    )

    with pytest.raises(LLMCallError) as caught:
        judge.judge(draft=draft, evidence=[DOMESTIC])

    assert caught.value.elapsed_ms == pytest.approx(14.0)


def test_의도_분류가_예외로_죽은_문의도_그_구간을_남긴다() -> None:
    """근거 수집이 인계로 끝나도 **어느 구간이 죽었는지**가 산출물에 남아야 한다."""
    client = _evidence_client(
        {
            INTENT_STAGE: [
                LLMCallError(
                    stage=INTENT_STAGE, reason="transport_error", attempts=2, elapsed_ms=31.0
                )
            ]
        }
    )

    result = _evidence_collector(client).collect(
        inquiry_id=EVIDENCE_INQUIRY_ID,
        content=EVIDENCE_INQUIRY,
        order_no=None,
        app_conn=cast(psycopg.Connection[DictRow], None),
        readonly_conn=cast(psycopg.Connection[DictRow], None),
    )

    assert result.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert result.failed_stage == INTENT_STAGE
    assert result.stage_durations.intent_ms == pytest.approx(31.0)


class _FailingEmbeddingClient:
    """질의 임베딩이 전송 오류로 죽는 대역 — 경과를 예외에 실어 올린다."""

    def __init__(self, error: LLMCallError) -> None:
        self._error = error

    @property
    def model(self) -> str:
        return "stub:failing"

    @property
    def dimensions(self) -> int:
        return 1536

    def embed(self, *, stage: str, texts: Sequence[str]) -> EmbeddingResult:
        del stage, texts
        raise self._error


def test_질의_임베딩이_예외로_죽은_문의도_그_구간을_남긴다() -> None:
    client = _evidence_client({INTENT_STAGE: [intent_completion("policy")]})
    collector = EvidenceCollector(
        generation_client=cast(llm.GenerationClient, client),
        embedding_client=_FailingEmbeddingClient(
            LLMCallError(
                stage=INQUIRY_EMBEDDING_STAGE,
                reason="transport_error",
                attempts=2,
                elapsed_ms=23.0,
            )
        ),
        settings=Settings(vector_top_k=5, vector_similarity_threshold=0.0, sql_max_rows=50),
        rewrite_client=cast(llm.GenerationClient, client),
    )

    result = collector.collect(
        inquiry_id=EVIDENCE_INQUIRY_ID,
        content=EVIDENCE_INQUIRY,
        order_no=None,
        app_conn=cast(psycopg.Connection[DictRow], None),
        readonly_conn=cast(psycopg.Connection[DictRow], None),
    )

    assert result.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert result.failed_stage == INQUIRY_EMBEDDING_STAGE
    assert result.stage_durations.inquiry_embedding_ms == pytest.approx(23.0)


@pytest.mark.db
def test_조회문_생성이_예외로_죽은_문의도_그_구간을_남긴다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    order_no = _first_order_no(ro_conn)
    client = _evidence_client(
        {
            INTENT_STAGE: [intent_completion("order")],
            SQL_GENERATION_STAGE: [
                LLMCallError(
                    stage=SQL_GENERATION_STAGE,
                    reason="transport_error",
                    attempts=2,
                    elapsed_ms=44.0,
                )
            ],
        }
    )

    result = _evidence_collector(client).collect(
        inquiry_id=EVIDENCE_INQUIRY_ID,
        content=EVIDENCE_INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert result.failed_stage == SQL_GENERATION_STAGE
    assert result.stage_durations.sql_generation_ms == pytest.approx(44.0)


@pytest.mark.db
def test_주문이_없는_문의도_존재성_선검사의_조회_실행_구간을_남긴다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """선검사는 근거가 되지 않지만 **코드가 부르는 DB 왕복**이고 시간을 쓴다.

    빼면 `order_not_found` 로 끝난 문의의 DB 시간이 아홉 구간 어디에도 잡히지 않는다.
    시계를 목으로 고정해 값까지 못박는다 — 실제 벽시계로 재면 CI 가 흔들린다.
    """
    monkeypatch.setattr(evidence, "perf_counter", _Clock([0.0, 0.002]))
    client = _evidence_client({INTENT_STAGE: [intent_completion("order")]})

    result = _evidence_collector(client).collect(
        inquiry_id=EVIDENCE_INQUIRY_ID,
        content=EVIDENCE_INQUIRY,
        order_no=EVIDENCE_MISSING_ORDER_NO,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is EscalationReason.ORDER_NOT_FOUND
    assert result.stage_durations.sql_execution_ms == pytest.approx(2.0)


@pytest.mark.db
def test_실행에_실패한_조회의_경과도_조회_실행_구간에_든다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`finally` 로 재는 이유 — 성공 경로에서만 재면 실패한 조회의 시간이 사라진다.

    시계 대본은 (선검사 1.0 · 실패한 1차 4.0 · 성공한 2차 8.0) 이고 합은 13.0 이다.
    `finally` 를 성공 경로로 옮기면 1차의 창이 통째로 빠져 다른 값이 나온다.
    """
    monkeypatch.setattr(evidence, "perf_counter", _Clock([0.0, 0.001, 0.100, 0.104, 0.200, 0.208]))
    order_no = _first_order_no(ro_conn)
    broken = f"SELECT order_no FROM orders WHERE order_no = '{order_no}' AND quantity = 'abc'"
    good = f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'"
    client = _evidence_client(
        {
            INTENT_STAGE: [intent_completion("order")],
            SQL_GENERATION_STAGE: [
                _completion({"sql": broken}, elapsed_ms=0.0),
                _completion({"sql": good}, elapsed_ms=0.0),
            ],
        }
    )

    result = _evidence_collector(client).collect(
        inquiry_id=EVIDENCE_INQUIRY_ID,
        content=EVIDENCE_INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    # 양성 대조 — 1차가 실제로 실행 단계에서 죽어야 이 검사가 무언가를 지킨다.
    assert result.sql_failures[0].kind is SqlFailureKind.EXECUTION_ERROR
    assert result.escalation_reason is None
    assert result.stage_durations.sql_execution_ms == pytest.approx(13.0)
