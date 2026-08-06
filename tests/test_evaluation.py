"""평가 하네스 테스트 — 두 측정이 분리되어 있고, 미실행이 미실행으로 남는지 못박는다.

이 파일이 지키는 계약은 넷이다.

* **측정 1 은 LLM 을 호출하지 않는다** — 소켓을 막아 두고도 끝까지 돈다(결정론 층).
* **저장소 골든셋·픽스처의 라벨이 데이터·코드와 어긋나지 않는다** — 라벨이 ground truth
  이므로 어긋나면 지표가 거짓말을 한다.
* **하네스 배관이 30건을 끝까지 흘린다** — 대역 LLM 으로 리포트 산출까지 검증한다.
  API 키가 생겼을 때 바로 돌아야 하기 때문이다.
* **미실행은 0 으로 채우지 않는다** — 리포트에 사유가 남는다.

DB 를 쓰는 테스트는 정책 인덱싱 픽스처(트랜잭션 롤백) 말고는 행을 남기지 않는다.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, cast

import psycopg
import pytest
from psycopg.rows import DictRow
from scripts import evaluate

from reply_gate.config import Settings
from reply_gate.contracts import (
    Claim,
    EscalationReason,
    Evidence,
    EvidenceSource,
    InquiryStatus,
    IntentSource,
    RejectReason,
    Verdict,
)
from reply_gate.evaluation import (
    DEFAULT_GOLDEN_SET_PATH,
    DEFAULT_L1_FIXTURES_PATH,
    REASON_ORDER,
    TARGETS,
    EvaluationReport,
    ExpectedOutcomeSet,
    GoldenCase,
    L1Fixture,
    MetricTarget,
    ReportStemError,
    RunConditions,
    SkippedMeasurement,
    StubGenerationClient,
    build_report,
    load_golden_set,
    load_l1_fixtures,
    measure_gate_accuracy,
    measure_pipeline_agreement,
    render_markdown,
    report_to_json,
    resolve_report_stem,
    utc_now_iso,
    write_report,
)
from reply_gate.llm import GenerationClient
from reply_gate.order_ref import is_valid_order_no
from reply_gate.pipeline import AttemptRecord, ProcessedInquiry, build_pipeline
from reply_gate.policy_index import index_policy_documents, load_policy_documents
from reply_gate.testing import LexicalEmbeddingClient

_NO_CONN = cast(psycopg.Connection[DictRow], None)

FIXTURES = load_l1_fixtures(DEFAULT_L1_FIXTURES_PATH)
GOLDEN = load_golden_set(DEFAULT_GOLDEN_SET_PATH)


# ── 대역 ────────────────────────────────────────────────────────────────────


def _processed(
    *,
    status: InquiryStatus = InquiryStatus.ANSWERED,
    escalation: EscalationReason | None = None,
    attempts: tuple[AttemptRecord, ...] = (),
    latency_ms: int = 10,
    input_tokens: int = 0,
    output_tokens: int = 0,
    embedding_tokens: int = 0,
) -> ProcessedInquiry:
    return ProcessedInquiry(
        inquiry_id="00000000-0000-4000-8000-000000000000",
        order_no=None,
        content="문의",
        intent=IntentSource.POLICY,
        status=status,
        answer="답변" if status is InquiryStatus.ANSWERED else None,
        claims=(Claim(text="답변", citation_ids=("policy:refund:2-1",)),),
        escalation_reason=escalation,
        failed_stage=None,
        evidence=(),
        sql_snapshots=(),
        sql_failures=(),
        attempts=attempts,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        embedding_tokens=embedding_tokens,
    )


class ScriptedPipeline:
    """`PipelineRunning` 대역 — 미리 정한 결과를 순서대로 돌려준다(커넥션은 쓰지 않는다)."""

    def __init__(self, results: list[ProcessedInquiry]) -> None:
        self._results = list(results)

    def run(self, **kwargs: Any) -> ProcessedInquiry:
        del kwargs
        return self._results.pop(0)


def _case(
    case_id: str,
    *,
    statuses: tuple[InquiryStatus, ...] = (InquiryStatus.ANSWERED,),
    reasons: tuple[EscalationReason, ...] = (),
    expect_reject: bool = False,
    forbidden: tuple[RejectReason, ...] = (),
) -> GoldenCase:
    return GoldenCase(
        id=case_id,
        category="normal",
        order_no=None,
        content="문의",
        expected=ExpectedOutcomeSet(
            statuses=frozenset(statuses),
            escalation_reasons=frozenset(reasons),
            expect_reject=expect_reject,
            forbidden_reject_reasons=frozenset(forbidden),
        ),
        note="",
    )


def _conditions(*, is_real: bool = False) -> RunConditions:
    return RunConditions(
        started_at=utc_now_iso(),
        generation="대역",
        embedding="대역",
        similarity_threshold=0.05,
        top_k=5,
        l1_fixture_count=len(FIXTURES),
        golden_case_count=len(GOLDEN),
        l1_fixtures_path=str(DEFAULT_L1_FIXTURES_PATH),
        golden_set_path=str(DEFAULT_GOLDEN_SET_PATH),
        api_key_present=False,
        measurement2_is_real=is_real,
    )


# ── 측정 1 — 결정론 층 ──────────────────────────────────────────────────────


def test_저장소_픽스처_셋은_사유_4종과_정상_통과를_모두_덮는다() -> None:
    """사유별로 여러 건이 있어야 검출률이 의미 있는 수치가 된다."""
    expected_counts = {
        reason: sum(1 for fixture in FIXTURES if reason in fixture.expected_reasons)
        for reason in REASON_ORDER
    }
    assert all(count >= 2 for count in expected_counts.values()), expected_counts
    assert sum(1 for fixture in FIXTURES if not fixture.is_violation) >= 2
    # 오탐률의 절반은 "근거 유래 PII 정상 에코가 통과하는가" 다.
    echoes = [fixture for fixture in FIXTURES if fixture.category == "pii_echo"]
    assert echoes and all(fixture.expected_verdict is Verdict.PASS for fixture in echoes)


def test_측정1_은_저장소_픽스처_전체를_기대대로_판정한다() -> None:
    """헤드라인 수치 — 구조적 오류 검출률과 정상 초안 오탐률."""
    accuracy = measure_gate_accuracy(FIXTURES)

    assert accuracy.total == len(FIXTURES)
    assert accuracy.detection_rate == 1.0
    assert accuracy.false_positive_rate == 0.0
    mismatched = [
        outcome.fixture_id for outcome in accuracy.outcomes if not outcome.reasons_matched
    ]
    assert mismatched == []
    assert all(item.spurious_count == 0 for item in accuracy.breakdown)


def test_측정1_은_소켓을_막아도_끝까지_돈다(monkeypatch: pytest.MonkeyPatch) -> None:
    """L1 은 LLM 호출 0회다 — 네트워크를 막아 두고도 측정 1 은 완주한다."""

    def _blocked(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("측정 1 은 네트워크를 타면 안 된다")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)

    accuracy = measure_gate_accuracy(FIXTURES)

    assert accuracy.total == len(FIXTURES)
    assert accuracy.detection_rate == 1.0


def test_측정1_은_재실행해도_같은_수치를_낸다() -> None:
    first = measure_gate_accuracy(FIXTURES)
    second = measure_gate_accuracy(FIXTURES)
    assert first == second


def test_검출_실패와_오탐이_수치에_그대로_드러난다() -> None:
    """놓친 위반과 잘못 기각한 정상 초안이 각각 분자에 반영되는지."""
    evidence = Evidence(
        id="policy:refund:2-1",
        source=EvidenceSource.POLICY,
        content="환불은 7일 이내입니다.",
        evidence_text="환불은 7일 이내입니다.",
    )
    missed = L1Fixture(
        id="X1",
        category="missing_citation",
        note="실제로는 통과하는데 기각을 기대한 라벨",
        raw_draft={"claims": [{"text": "환불은 7일 이내입니다.", "citation_ids": [evidence.id]}]},
        evidences=(evidence,),
        expected_verdict=Verdict.REJECT,
        expected_reasons=(RejectReason.MISSING_CITATION,),
    )
    false_positive = L1Fixture(
        id="X2",
        category="clean",
        note="실제로는 기각되는데 통과를 기대한 라벨",
        raw_draft={"claims": [{"text": "환불은 7일 이내입니다.", "citation_ids": []}]},
        evidences=(evidence,),
        expected_verdict=Verdict.PASS,
        expected_reasons=(),
    )
    accuracy = measure_gate_accuracy([missed, false_positive])

    assert accuracy.detection_rate == 0.0
    assert accuracy.false_positive_rate == 1.0
    breakdown = {item.reason: item for item in accuracy.breakdown}
    assert breakdown[RejectReason.MISSING_CITATION].detected_count == 0
    assert breakdown[RejectReason.MISSING_CITATION].spurious_count == 1


def test_픽스처_라벨이_판정과_모순이면_로드에서_막는다(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "F00",
                "category": "clean",
                "evidences": [],
                "raw_draft": {"claims": []},
                "expected": {"verdict": "pass", "reject_reasons": ["schema_violation"]},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="사유가"):
        load_l1_fixtures(path)


# ── 골든셋 라벨이 데이터·코드와 어긋나지 않는가 ─────────────────────────────


def test_골든셋은_30건이고_네_범주를_섞어_담는다() -> None:
    assert len(GOLDEN) == 30
    categories = {case.category for case in GOLDEN}
    assert categories == {"normal", "reject_bait", "no_evidence", "escalation"}


def test_기각_유발_문의만_기각을_기대한다() -> None:
    """`expect_reject` 는 기각 재현율의 분모다 — 범주와 어긋나면 지표가 흐려진다."""
    baits = [case for case in GOLDEN if case.expected.expect_reject]
    assert {case.id for case in baits} == {
        case.id for case in GOLDEN if case.category == "reject_bait"
    }
    assert len(baits) >= 5


def test_골든셋_주문번호는_전부_접수_형식을_만족한다() -> None:
    for case in GOLDEN:
        if case.order_no is not None:
            assert is_valid_order_no(case.order_no), case.id


def test_허용_인계_사유는_상태_집합과_일관된다() -> None:
    for case in GOLDEN:
        escalated = InquiryStatus.ESCALATED in case.expected.statuses
        assert escalated == bool(case.expected.escalation_reasons), case.id


def test_정상_PII_에코_케이스가_오탐_감시로_등록되어_있다() -> None:
    """근거 유래 PII 를 되돌려주는 문의에서 pii_detected 가 나면 오탐이다."""
    watched = [case for case in GOLDEN if case.expected.forbidden_reject_reasons]
    assert watched, "금지 기각 사유가 걸린 케이스가 최소 1건은 있어야 한다"
    assert any(
        RejectReason.PII_DETECTED in case.expected.forbidden_reject_reasons for case in watched
    )


@pytest.mark.db
def test_골든셋_주문번호_라벨이_시딩_데이터와_맞는다(ro_conn: psycopg.Connection[DictRow]) -> None:
    """`order_not_found` 를 기대한 건만 실제로 없는 주문이어야 한다."""
    for case in GOLDEN:
        if case.order_no is None:
            continue
        row = ro_conn.execute(
            "SELECT 1 FROM orders WHERE order_no = %s", (case.order_no,)
        ).fetchone()
        expects_not_found = EscalationReason.ORDER_NOT_FOUND in case.expected.escalation_reasons
        assert (row is None) == expects_not_found, f"{case.id}: {case.order_no}"


def test_무근거_문의는_정책_문서에_실제로_근거가_없다() -> None:
    """무근거 라벨이 데이터와 어긋나면 지표가 거짓말을 한다 — 조항 본문 전체로 확인한다."""
    corpus = "\n".join(
        chunk.embedding_text for document in load_policy_documents() for chunk in document.chunks
    )
    forbidden_terms = {
        "G21": ("해외", "국제"),
        "G22": ("대량", "기업"),
        "G23": ("매장", "방문 수령"),
        "G24": ("선물 포장", "포장 서비스"),
    }
    for case_id, terms in forbidden_terms.items():
        assert any(case.id == case_id for case in GOLDEN)
        for term in terms:
            assert term not in corpus, (
                f"{case_id}: 정책 문서에 '{term}' 가 있어 무근거 라벨이 틀렸다"
            )


# ── 측정 2 — 허용 결과 집합 대조 ────────────────────────────────────────────


def test_허용_상태_집합_밖이면_불일치다() -> None:
    agreement = measure_pipeline_agreement(
        cases=[_case("A", statuses=(InquiryStatus.ANSWERED,))],
        pipeline=ScriptedPipeline(
            [_processed(status=InquiryStatus.ESCALATED, escalation=EscalationReason.NO_EVIDENCE)]
        ),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )
    assert agreement.match_rate == 0.0
    assert "허용 집합" in agreement.outcomes[0].mismatches[0]


def test_상태는_맞고_인계_사유가_집합_밖이면_불일치다() -> None:
    case = _case(
        "A",
        statuses=(InquiryStatus.ESCALATED,),
        reasons=(EscalationReason.NO_EVIDENCE,),
    )
    agreement = measure_pipeline_agreement(
        cases=[case],
        pipeline=ScriptedPipeline(
            [_processed(status=InquiryStatus.ESCALATED, escalation=EscalationReason.SQL_FAILED)]
        ),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )
    assert agreement.matched == 0
    assert "인계 사유" in agreement.outcomes[0].mismatches[0]


def test_허용_결과_집합_안이면_상태가_달라도_일치다() -> None:
    """초안이 확률적이므로 라벨은 단일 정답이 아니다."""
    case = _case(
        "A",
        statuses=(InquiryStatus.ANSWERED, InquiryStatus.ESCALATED),
        reasons=(EscalationReason.REJECTED_TWICE,),
    )
    agreement = measure_pipeline_agreement(
        cases=[case],
        pipeline=ScriptedPipeline(
            [_processed(status=InquiryStatus.ESCALATED, escalation=EscalationReason.REJECTED_TWICE)]
        ),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )
    assert agreement.match_rate == 1.0


def test_기각_재현율은_시도_판정에서_나온다() -> None:
    """미끼 문의는 최종 상태가 answered 여도 기각이 한 번은 나와야 한다."""
    bait = _case(
        "B",
        statuses=(InquiryStatus.ANSWERED, InquiryStatus.ESCALATED),
        reasons=(EscalationReason.REJECTED_TWICE,),
        expect_reject=True,
    )
    rejected_then_passed = _processed(
        attempts=(
            AttemptRecord(
                attempt_no=1,
                verdict=Verdict.REJECT,
                reject_reasons=(RejectReason.PII_DETECTED,),
                draft="원문",
            ),
            AttemptRecord(attempt_no=2, verdict=Verdict.PASS, reject_reasons=(), draft={}),
        )
    )
    never_rejected = _processed(
        attempts=(AttemptRecord(attempt_no=1, verdict=Verdict.PASS, reject_reasons=(), draft={}),)
    )

    agreement = measure_pipeline_agreement(
        cases=[bait, bait],
        pipeline=ScriptedPipeline([rejected_then_passed, never_rejected]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert agreement.bait_total == 2
    assert agreement.bait_reject_reproduced == 1
    assert agreement.bait_reject_recall == 0.5
    assert agreement.outcomes[0].matched is True
    assert agreement.outcomes[1].matched is False


def test_금지_기각_사유가_발화하면_오탐으로_집계된다() -> None:
    case = _case("P", forbidden=(RejectReason.PII_DETECTED,))
    processed = _processed(
        attempts=(
            AttemptRecord(
                attempt_no=1,
                verdict=Verdict.REJECT,
                reject_reasons=(RejectReason.PII_DETECTED,),
                draft="원문",
            ),
            AttemptRecord(attempt_no=2, verdict=Verdict.PASS, reject_reasons=(), draft={}),
        )
    )
    agreement = measure_pipeline_agreement(
        cases=[case],
        pipeline=ScriptedPipeline([processed]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert agreement.forbidden_watch_total == 1
    assert agreement.forbidden_violations == 1
    assert agreement.matched == 0
    assert "오탐" in agreement.outcomes[0].mismatches[0]


def test_지연_백분위와_건당_토큰은_생성과_임베딩을_구분해_집계한다() -> None:
    latencies = [10, 20, 30, 40, 100]
    results = [
        _processed(latency_ms=value, input_tokens=100, output_tokens=50, embedding_tokens=10)
        for value in latencies
    ]
    agreement = measure_pipeline_agreement(
        cases=[_case(f"C{index}") for index in range(len(results))],
        pipeline=ScriptedPipeline(results),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert agreement.latency_p50_ms == 30
    assert agreement.latency_p95_ms == 100
    assert agreement.generation_tokens_per_inquiry == 150.0
    assert agreement.embedding_tokens_per_inquiry == 10.0
    assert agreement.total_tokens_per_inquiry == 160.0


# ── 리포트 ──────────────────────────────────────────────────────────────────


def test_미실행_측정2_는_0_이_아니라_사유로_남는다(tmp_path: Path) -> None:
    report = build_report(
        conditions=_conditions(),
        gate_accuracy=measure_gate_accuracy(FIXTURES),
        pipeline=SkippedMeasurement(reason="OPENAI_API_KEY 가 없다"),
    )
    markdown = render_markdown(report)
    payload = report_to_json(report)

    assert "**미실행 (사유: OPENAI_API_KEY 가 없다)**" in markdown
    assert payload["measurement_2_pipeline_agreement"] == {
        "executed": False,
        "skip_reason": "OPENAI_API_KEY 가 없다",
    }
    # 측정 1 은 그대로 수치가 나와야 한다.
    assert payload["measurement_1_l1_gate_accuracy"]["detection_rate"] == 1.0

    markdown_path, json_path = write_report(report, out_dir=tmp_path)
    assert markdown_path.exists() and json_path.exists()


def test_확정된_목표치가_리포트에_실린다() -> None:
    report = build_report(
        conditions=_conditions(),
        gate_accuracy=measure_gate_accuracy(FIXTURES),
        pipeline=SkippedMeasurement(reason="미요청"),
    )
    markdown = render_markdown(report)
    assert "## 목표치 대비" in markdown
    assert "| 측정 1 구조적 오류 검출률 | ≥ 100% |" in markdown
    assert "| 측정 1 정상 초안 오탐률 | ≤ 0% |" in markdown
    assert "| 측정 2 허용 결과 집합 대비 일치율 | ≥ 75% |" in markdown

    metrics = report_to_json(report)["targets"]["metrics"]
    by_key = {metric["key"]: metric for metric in metrics}
    assert by_key["detection_rate"]["bound"] == 1.0
    assert by_key["false_positive_rate"]["direction"] == "at_most"
    assert by_key["match_rate"]["bound"] == 0.75


def test_측정하지_않은_지표는_미달로_적지_않는다() -> None:
    """돌지 않은 측정을 "미달"로 찍으면 리포트가 거짓말을 한다 — `None` 이어야 한다."""
    report = build_report(
        conditions=_conditions(),
        gate_accuracy=measure_gate_accuracy(FIXTURES),
        pipeline=SkippedMeasurement(reason="미요청"),
    )
    metrics = report_to_json(report)["targets"]["metrics"]
    match_rate = next(metric for metric in metrics if metric["key"] == "match_rate")
    assert match_rate["measured"] is None
    assert match_rate["met"] is None
    assert "미측정" in render_markdown(report)


def test_목표치는_경계값을_달성으로_본다() -> None:
    at_least = MetricTarget(key="k", label="l", bound=0.75)
    at_most = MetricTarget(key="k", label="l", bound=0.0, at_most=True)
    assert at_least.met(0.75) is True
    assert at_least.met(0.7499) is False
    assert at_most.met(0.0) is True
    assert at_most.met(0.0001) is False
    assert at_least.met(None) is None


def test_목표치_키와_실측값_키는_항상_정렬돼_있다() -> None:
    """TARGETS 에 지표를 추가하고 measured() 갱신을 빠뜨리면 여기서 먼저 깨져야 한다."""
    report = build_report(
        conditions=_conditions(),
        gate_accuracy=measure_gate_accuracy(FIXTURES),
        pipeline=SkippedMeasurement(reason="미요청"),
    )
    assert {target.key for target in TARGETS} == set(report.measured().keys())


def _real_agreement_report(*, matched: bool) -> EvaluationReport:
    """실측(measurement2_is_real=True) 1건짜리 일치율 리포트 — 달성/미달 판정 검증용."""
    expected = (InquiryStatus.ANSWERED,) if matched else (InquiryStatus.ESCALATED,)
    agreement = measure_pipeline_agreement(
        cases=[_case("A", statuses=expected)],
        pipeline=ScriptedPipeline([_processed()]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )
    return build_report(
        conditions=_conditions(is_real=True),
        gate_accuracy=measure_gate_accuracy(FIXTURES),
        pipeline=agreement,
    )


def test_실측된_일치율은_달성과_미달을_실제로_판정한다() -> None:
    """판정 배선을 뒤집는 회귀(달성↔미달)를 렌더링과 JSON 양쪽에서 잡는다."""
    met_report = _real_agreement_report(matched=True)  # 일치율 1/1 = 100% ≥ 75%
    missed_report = _real_agreement_report(matched=False)  # 일치율 0/1 = 0% < 75%

    met_json = next(
        metric
        for metric in report_to_json(met_report)["targets"]["metrics"]
        if metric["key"] == "match_rate"
    )
    missed_json = next(
        metric
        for metric in report_to_json(missed_report)["targets"]["metrics"]
        if metric["key"] == "match_rate"
    )
    assert met_json["met"] is True and met_json["verdict"] == "달성"
    assert missed_json["met"] is False and missed_json["verdict"] == "미달"

    assert "| 100.0% | 달성 |" in render_markdown(met_report)
    assert "| 0.0% | **미달** |" in render_markdown(missed_report)


def test_대역_일치율은_달성으로_판정하지_않는다() -> None:
    """대역 수치로 확률 층 목표를 "달성" 이라 찍으면 리포트가 거짓말을 한다."""
    agreement = measure_pipeline_agreement(
        cases=[_case("A")],
        pipeline=ScriptedPipeline([_processed()]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )
    report = build_report(
        conditions=_conditions(is_real=False),
        gate_accuracy=measure_gate_accuracy(FIXTURES),
        pipeline=agreement,
    )
    match_rate = next(
        metric
        for metric in report_to_json(report)["targets"]["metrics"]
        if metric["key"] == "match_rate"
    )
    assert match_rate["measured"] is not None  # 값은 남긴다 — 배관 검증용
    assert match_rate["met"] is None  # 판정은 하지 않는다
    assert match_rate["verdict"] == "대역 — 판정 없음"
    assert "대역 — 판정 없음" in render_markdown(report)


# ── 리포트 이름 가드 — 라이브 실측 보존 계약 ─────────────────────────────────


def test_기본_이름은_실측일_때만_라이브다(tmp_path: Path) -> None:
    resolve = lambda real: resolve_report_stem(  # noqa: E731
        requested=None, live_requested=True, measurement2_is_real=real, out_dir=tmp_path
    )
    assert resolve(True) == "evaluation-live"
    # --live 를 줬어도 키·DB 문제로 측정 2 가 미실행이면 비실측 — 라이브 이름을 쓰면 안 된다.
    assert resolve(False) == "evaluation"


def test_비실측은_라이브_이름을_쓸_수_없다(tmp_path: Path) -> None:
    for stem in ("evaluation-live", "evaluation-live-9", "Evaluation-live", "EVALUATION-LIVE-2"):
        with pytest.raises(ReportStemError):
            resolve_report_stem(
                requested=stem, live_requested=False, measurement2_is_real=False, out_dir=tmp_path
            )


def test_이름_조각이_아닌_stem_은_거부된다(tmp_path: Path) -> None:
    """`./`·`../`·절대경로·빈 문자열로 가드를 우회해 같은 파일명에 쓰는 것을 막는다."""
    for stem in (
        "./evaluation-live-x",
        "../reports/evaluation-live",
        "/tmp/evaluation-live",
        "sub/evaluation-live",
        "..",
        "",
        " evaluation-live",
        ".hidden",
    ):
        with pytest.raises(ReportStemError):
            resolve_report_stem(
                requested=stem, live_requested=False, measurement2_is_real=False, out_dir=tmp_path
            )


def test_실측은_라이브_이름만_쓸_수_있다(tmp_path: Path) -> None:
    """실측 결과가 gitignore 되는 이름으로 새면 다음 기본 실행이 그대로 덮는다."""
    with pytest.raises(ReportStemError):
        resolve_report_stem(
            requested="evaluation", live_requested=True, measurement2_is_real=True, out_dir=tmp_path
        )
    assert (
        resolve_report_stem(
            requested="evaluation-live-9",
            live_requested=True,
            measurement2_is_real=True,
            out_dir=tmp_path,
        )
        == "evaluation-live-9"
    )


def test_실측은_기존_라이브_리포트를_덮어쓸_수_없다(tmp_path: Path) -> None:
    """실측은 비결정론이라 덮이면 그 수치는 재생성되지 않는다 — 빈 이름을 제안해야 한다."""
    (tmp_path / "evaluation-live.md").write_text("기존 실측", encoding="utf-8")
    (tmp_path / "evaluation-live-1.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReportStemError, match="evaluation-live-2"):
        resolve_report_stem(
            requested=None, live_requested=True, measurement2_is_real=True, out_dir=tmp_path
        )
    with pytest.raises(ReportStemError):
        resolve_report_stem(
            requested="evaluation-live-1",
            live_requested=True,
            measurement2_is_real=True,
            out_dir=tmp_path,
        )


def test_리포트는_실행_조건과_한계를_함께_적는다() -> None:
    markdown = render_markdown(
        build_report(
            conditions=_conditions(),
            gate_accuracy=measure_gate_accuracy(FIXTURES),
            pipeline=SkippedMeasurement(reason="미요청"),
        )
    )
    assert "유사도 임계값: 0.05" in markdown
    assert "top k: 5" in markdown
    assert "패턴형 PII 만 본다" in markdown
    assert "L2 사이클로 이월" in markdown
    assert "재실행하면 값이 달라진다" in markdown


def test_대역으로_돈_측정2_는_실제_수치가_아니라고_경고한다() -> None:
    agreement = measure_pipeline_agreement(
        cases=[_case("A")],
        pipeline=ScriptedPipeline([_processed()]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )
    stub_markdown = render_markdown(
        build_report(
            conditions=_conditions(is_real=False),
            gate_accuracy=measure_gate_accuracy(FIXTURES),
            pipeline=agreement,
        )
    )
    real_markdown = render_markdown(
        build_report(
            conditions=_conditions(is_real=True),
            gate_accuracy=measure_gate_accuracy(FIXTURES),
            pipeline=agreement,
        )
    )
    assert "실제 모델 수치가 아니다" in stub_markdown
    assert "실제 모델 수치가 아니다" not in real_markdown
    # 일치율은 초안 전 인계 포함임을 리포트가 명시해야 한다.
    assert "초안 전 인계 경로 포함" in stub_markdown


def test_리포트는_사람용과_기계용_두_형식을_낸다(tmp_path: Path) -> None:
    report = build_report(
        conditions=_conditions(),
        gate_accuracy=measure_gate_accuracy(FIXTURES),
        pipeline=SkippedMeasurement(reason="미요청"),
    )
    markdown_path, json_path = write_report(report, out_dir=tmp_path, stem="run")

    assert markdown_path.name == "run.md"
    assert json_path.name == "run.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["measurement_1_l1_gate_accuracy"]["llm_calls"] == 0
    assert len(payload["measurement_1_l1_gate_accuracy"]["outcomes"]) == len(FIXTURES)


# ── 실행 진입점의 가드 (scripts/evaluate.py — DB·LLM 없이 돈다) ──────────────
#
# 가드는 얇지만 **돌릴 수 없는 실행을 산출물 이전에 죽이는** 자리다. 여기가 새면
# 무엇을 잰 것인지 알 수 없는 리포트가 남고, 라이브 리포트는 덮어쓸 수 없다.


def _l2_settings(*, live_keys: bool = True, judge_key: bool = True) -> Settings:
    key = "키가-아닌-테스트값"
    return Settings(
        l2_enabled=True,
        openai_api_key=key if live_keys else "",
        anthropic_api_key=key if judge_key else "",
    )


def _block_outbound_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    """가드가 걷혀도 이 테스트가 실제 API 를 부르는 일이 없게 막는다 (측정 1 패턴과 같다)."""

    def _blocked(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("가드가 막아야 할 실행이 외부 호출까지 갔다")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


def test_stub_llm_은_L2_켜짐에서_산출물을_남기지_않고_멈춘다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """결정론 판정 대역이 없으므로 대역 실행이 실제 판정 모델을 부르게 된다 — 명시적으로 죽는다.

    조용히 `l2_enabled=False` 로 낮춰 돌리면 대역 수치가 사이클 1 을 잰 것인지 L2 를
    포함해 잰 것인지 알 수 없게 된다.
    """
    monkeypatch.setattr(evaluate, "get_settings", lambda: _l2_settings())
    _block_outbound_sockets(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        evaluate.main(["--stub-llm", "--out-dir", str(tmp_path)])

    assert "결정론 판정" in str(excinfo.value)
    # 측정 1(무료)조차 시작하기 전에 죽는다 — 리포트 파일이 남으면 안 된다.
    assert list(tmp_path.iterdir()) == []


def test_live_는_실측이_돌_조건에서만_L2_켜짐으로_멈춘다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """라이브 리포트는 덮어쓸 수 없다 — "L2 포함 여부 불명"인 실측이 영구히 남으면 안 된다.

    가드가 발화하는 조건은 **실측 이름을 받을 실행**, 즉 키도 DB 도 갖춰져 측정 2 가
    실제로 돌 때다. 그래서 여기서는 미실행 사유를 없애(`database_unavailable_reason`
    → None) 실측이 돌 상태를 만든다.
    """
    monkeypatch.setattr(evaluate, "get_settings", lambda: _l2_settings())
    monkeypatch.setattr(evaluate, "database_unavailable_reason", lambda *, settings: None)
    _block_outbound_sockets(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        evaluate.main(["--live", "--out-dir", str(tmp_path)])

    message = str(excinfo.value)
    assert "L2 판정이 켜져 있는데" in message
    # 이 가드가 임시라는 사실이 오류 메시지에 남아야 뒤 태스크가 걷어낼 자리를 안다.
    assert "하네스 확장 태스크" in message
    assert list(tmp_path.iterdir()) == []


def test_키_없는_live_는_죽지_않고_측정1_과_미실행_사유를_남긴다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """키가 없으면 측정 2 는 미실행이라 실측 이름을 받지 못한다 — 막을 피해가 없다.

    이때 가드가 죽이면 무료인 측정 1 산출물까지 잃는다. 리포트는 덮어쓸 수 있는
    `evaluation` 계열로 남고, 측정 2 는 **미실행 + 사유**로 기록된다
    (`scripts/AGENTS.md` 불변식 5).
    """
    monkeypatch.setattr(evaluate, "get_settings", lambda: _l2_settings(live_keys=False))
    _block_outbound_sockets(monkeypatch)

    assert evaluate.main(["--live", "--out-dir", str(tmp_path)]) == 0

    payload = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["measurement_2_pipeline_agreement"]["executed"] is False
    assert "OPENAI_API_KEY" in payload["measurement_2_pipeline_agreement"]["skip_reason"]
    # 측정 1 은 그대로 산출된다 — 무료 측정을 라이브 가드가 인질로 잡지 않는다.
    assert payload["measurement_1_l1_gate_accuracy"]["llm_calls"] == 0
    # 비실측이므로 라이브 이름 계열은 만들어지지 않는다.
    assert not list(tmp_path.glob("evaluation-live*"))


def test_판정_키가_없으면_측정2_를_시작하지_않는다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """시작해 놓고 첫 판정 호출에서 죽으면 30건 중 일부만 과금하고 산출물은 없다.

    이 사유는 **CLI 경로에서 실제로 관측된다** — 라이브 가드가 먼저 죽이면 도달할 수 없는
    죽은 분기가 되므로, 판정을 private 함수 직접 호출이 아니라 `main` 산출물로 확인한다.
    """
    monkeypatch.setattr(evaluate, "get_settings", lambda: _l2_settings(judge_key=False))
    monkeypatch.setattr(evaluate, "database_unavailable_reason", lambda *, settings: "DB 사유")
    _block_outbound_sockets(monkeypatch)

    assert evaluate.main(["--live", "--out-dir", str(tmp_path)]) == 0

    payload = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert "ANTHROPIC_API_KEY" in payload["measurement_2_pipeline_agreement"]["skip_reason"]

    # 양성 대조 — L2 가 꺼져 있으면 판정 키 부재는 사유가 아니고 뒤 검사(DB)로 넘어간다.
    args = evaluate.build_parser().parse_args(["--live"])
    settings = _l2_settings(judge_key=False).model_copy(update={"l2_enabled": False})
    assert evaluate._skip_reason(args=args, settings=settings) == "DB 사유"


# ── 배관 전체 (대역 LLM + 시딩된 DB) ────────────────────────────────────────


@pytest.fixture
def indexed_policies(app_conn: psycopg.Connection[DictRow]) -> None:
    """정책 문서를 결정론 임베딩으로 적재한다 (픽스처 롤백으로 되돌아간다)."""
    index_policy_documents(
        conn=app_conn,
        documents=load_policy_documents(),
        embedder=LexicalEmbeddingClient(dimensions=1536),
    )


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_하네스가_골든셋_30건을_대역으로_끝까지_흘려_리포트를_낸다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    tmp_path: Path,
) -> None:
    """API 키가 없어도 배관은 끝까지 검증된다 — 키가 생기면 바로 돌아야 하기 때문이다.

    **L2 는 꺼서** 돌린다: 결정론 판정 대역이 아직 없어(하네스 확장 태스크 몫) 켜면 실제
    판정 모델을 부르게 된다. L2 켜짐 대역 커버리지는 그 태스크가 복원한다.
    """
    stub = StubGenerationClient()
    pipeline = build_pipeline(
        generation_client=cast(GenerationClient, stub),
        embedding_client=LexicalEmbeddingClient(dimensions=1536),
        settings=Settings(
            vector_top_k=5,
            vector_similarity_threshold=0.05,
            sql_max_rows=50,
            l2_enabled=False,
        ),
    )

    agreement = measure_pipeline_agreement(
        cases=GOLDEN, pipeline=pipeline, app_conn=app_conn, readonly_conn=ro_conn
    )

    assert agreement.total == 30
    assert [outcome.case_id for outcome in agreement.outcomes] == [case.id for case in GOLDEN]
    assert all(outcome.error is None for outcome in agreement.outcomes)
    assert agreement.latency_p50_ms is not None and agreement.latency_p95_ms is not None
    # 기각 유발 문의 5건에서 L1 기각이 실제로 재현되어야 재현율이 의미를 가진다.
    assert agreement.bait_total == 5
    assert agreement.bait_reject_reproduced == 5
    assert agreement.bait_reject_recall == 1.0
    # 근거 유래 PII 에코는 통과해야 한다 (오탐 0).
    assert agreement.forbidden_violations == 0
    # 초안 전 인계 경로도 분모에 들어간다.
    assert agreement.escalation_counts.get("missing_order_ref") == 2
    assert agreement.escalation_counts.get("order_not_found") == 2

    report = build_report(
        conditions=_conditions(),
        gate_accuracy=measure_gate_accuracy(FIXTURES),
        pipeline=agreement,
    )
    markdown_path, json_path = write_report(report, out_dir=tmp_path, stem="e2e")
    markdown = markdown_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "## 측정 1 — L1 게이트 단위 정확도 (결정론)" in markdown
    assert "## 측정 2 — 파이프라인 판정 일치율 (end-to-end)" in markdown
    assert "기각 유발 문의의 기각 재현율" in markdown
    assert payload["measurement_2_pipeline_agreement"]["executed"] is True
    assert len(payload["measurement_2_pipeline_agreement"]["outcomes"]) == 30


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_대역_LLM_은_미끼_조항에서_기각을_재현한다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """미끼 조항 문의 → 근거에 없는 번호 생성 → `pii_detected` 기각 → 재생성 통과.

    L1 층의 기각 재현이므로 **L2 는 꺼서** 사이클 1 동작으로 확인한다.
    """
    stub = StubGenerationClient()
    pipeline = build_pipeline(
        generation_client=cast(GenerationClient, stub),
        embedding_client=LexicalEmbeddingClient(dimensions=1536),
        settings=Settings(
            vector_top_k=5,
            vector_similarity_threshold=0.05,
            sql_max_rows=50,
            l2_enabled=False,
        ),
    )
    bait = next(case for case in GOLDEN if case.id == "G16")

    agreement = measure_pipeline_agreement(
        cases=[bait], pipeline=pipeline, app_conn=app_conn, readonly_conn=ro_conn
    )
    outcome = agreement.outcomes[0]

    assert outcome.attempt_verdicts[0] is Verdict.REJECT
    assert RejectReason.PII_DETECTED in outcome.reject_reasons
    assert outcome.matched is True
