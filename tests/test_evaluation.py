"""평가 하네스 테스트 — 세 측정이 분리되어 있고, 미실행이 미실행으로 남는지 못박는다.

이 파일이 지키는 계약은 다섯이다.

* **측정 1 은 LLM 을 호출하지 않는다** — 소켓을 막아 두고도 끝까지 돈다(결정론 층).
* **저장소 골든셋·픽스처의 라벨이 데이터·코드와 어긋나지 않는다** — 라벨이 ground truth
  이므로 어긋나면 지표가 거짓말을 한다.
* **하네스 배관이 30건을 끝까지 흘린다** — 대역 LLM·대역 판정으로 리포트 산출까지
  검증한다. API 키가 생겼을 때 바로 돌아야 하기 때문이다.
* **측정 3(L2 판정 단위 정확도)은 확률 층이고 과금된다** — 대역으로 배관만 검증하고,
  목표치는 미확정으로 남는다.
* **미실행은 0 으로 채우지 않는다** — 리포트에 사유가 남는다.

DB 를 쓰는 테스트는 정책 인덱싱 픽스처(트랜잭션 롤백) 말고는 행을 남기지 않는다.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import psycopg
import pytest
from psycopg.rows import DictRow
from scripts import evaluate

from reply_gate.config import Settings, get_settings
from reply_gate.contracts import (
    Claim,
    ClaimJudgment,
    Draft,
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
from reply_gate.evaluation import (
    DEFAULT_GOLDEN_SET_PATH,
    DEFAULT_JUDGE_FIXTURES_PATH,
    DEFAULT_L1_FIXTURES_PATH,
    REASON_ORDER,
    TARGETS,
    EvaluationReport,
    ExpectedOutcomeSet,
    GoldenCase,
    JudgeAccuracy,
    JudgeFixture,
    L1Fixture,
    MetricTarget,
    PipelineAgreement,
    ReportStemError,
    RunConditions,
    SkippedMeasurement,
    StubGenerationClient,
    assess_targets,
    attach_regression_guard,
    build_report,
    load_golden_set,
    load_judge_fixtures,
    load_l1_fixtures,
    measure_gate_accuracy,
    measure_judge_accuracy,
    measure_pipeline_agreement,
    render_markdown,
    report_to_json,
    resolve_report_stem,
    utc_now_iso,
    write_report,
)
from reply_gate.gate import evaluate_draft
from reply_gate.judge import JUDGE_STAGE, JudgeOutcome
from reply_gate.llm import GenerationClient, LLMFormatError
from reply_gate.order_ref import is_valid_order_no
from reply_gate.pipeline import L2_JUDGE_STAGE, AttemptRecord, ProcessedInquiry
from reply_gate.policy_index import index_policy_documents, load_policy_documents
from reply_gate.testing import LexicalEmbeddingClient, StubJudge, build_stub_pipeline

_NO_CONN = cast(psycopg.Connection[DictRow], None)

FIXTURES = load_l1_fixtures(DEFAULT_L1_FIXTURES_PATH)
GOLDEN = load_golden_set(DEFAULT_GOLDEN_SET_PATH)
JUDGE_FIXTURES = load_judge_fixtures(DEFAULT_JUDGE_FIXTURES_PATH)


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
    judge_input_tokens: int = 0,
    judge_output_tokens: int = 0,
    evidence: tuple[Evidence, ...] = (),
    failed_stage: str | None = None,
    intent: IntentSource | None = IntentSource.POLICY,
) -> ProcessedInquiry:
    return ProcessedInquiry(
        inquiry_id="00000000-0000-4000-8000-000000000000",
        order_no=None,
        content="문의",
        intent=intent,
        status=status,
        answer="답변" if status is InquiryStatus.ANSWERED else None,
        claims=(Claim(text="답변", citation_ids=("policy:refund:2-1",)),),
        escalation_reason=escalation,
        failed_stage=failed_stage,
        evidence=evidence,
        sql_snapshots=(),
        sql_failures=(),
        attempts=attempts,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        embedding_tokens=embedding_tokens,
        judge_input_tokens=judge_input_tokens,
        judge_output_tokens=judge_output_tokens,
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
    category: str = "normal",
) -> GoldenCase:
    return GoldenCase(
        id=case_id,
        category=category,
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


def _conditions(
    *,
    is_real: bool = False,
    judge_is_real: bool = False,
    l2_enabled: bool = True,
    judge: str = "결정론 대역",
    retrieval_strategy: str = "vector+rewrite",
    fingerprint: dict[str, str] | None = None,
    declared: tuple[str, ...] = (),
) -> RunConditions:
    return RunConditions(
        started_at=utc_now_iso(),
        generation="대역",
        embedding="대역",
        embedding_dimensions=1536,
        judge=judge,
        retrieval_strategy=retrieval_strategy,
        similarity_threshold=0.05,
        top_k=5,
        l1_fixture_count=len(FIXTURES),
        golden_case_count=len(GOLDEN),
        judge_fixture_count=len(JUDGE_FIXTURES),
        l1_fixtures_path=str(DEFAULT_L1_FIXTURES_PATH),
        golden_set_path=str(DEFAULT_GOLDEN_SET_PATH),
        judge_fixtures_path=str(DEFAULT_JUDGE_FIXTURES_PATH),
        api_key_present=False,
        judge_api_key_present=False,
        l2_enabled=l2_enabled,
        measurement2_is_real=is_real,
        measurement3_is_real=judge_is_real,
        billed=is_real,
        condition_fingerprint=fingerprint or {},
        declared_experiment_fields=declared,
    )


def _report(
    *,
    pipeline: PipelineAgreement | SkippedMeasurement,
    conditions: RunConditions | None = None,
    judge: JudgeAccuracy | SkippedMeasurement | None = None,
) -> EvaluationReport:
    """리포트 조립 — 측정 3 은 명시하지 않으면 "미실행 + 사유" 다."""
    return build_report(
        conditions=conditions if conditions is not None else _conditions(),
        gate_accuracy=measure_gate_accuracy(FIXTURES),
        pipeline=pipeline,
        judge_accuracy=judge if judge is not None else SkippedMeasurement(reason="미요청"),
    )


# ── 판정 대역 (측정 3 용) ───────────────────────────────────────────────────


class OracleJudge:
    """픽스처의 **기대 판정을 그대로** 돌려주는 `Judging` 대역 — 검출률 100%·오탐률 0% 의 상한."""

    def __init__(self, fixtures: Sequence[JudgeFixture]) -> None:
        self._by_text = {fixture.draft.answer_text: fixture for fixture in fixtures}
        self.calls: list[Draft] = []

    def judge(self, *, draft: Draft, evidence: Sequence[Evidence]) -> JudgeOutcome:
        del evidence
        self.calls.append(draft)
        fixture = self._by_text[draft.answer_text]
        return JudgeOutcome(
            result=JudgeResult(
                verdict=fixture.expected_verdict,
                reject_reasons=fixture.expected_reasons,
                claim_judgments=tuple(
                    ClaimJudgment(claim_text=claim.text, verdict=verdict, explanation="기대 판정")
                    for claim, verdict in zip(
                        fixture.claims, fixture.expected_claim_verdicts, strict=True
                    )
                ),
                contradictions=tuple(
                    EvidenceContradiction(evidence_id_a=a, evidence_id_b=b, explanation="기대 모순")
                    for a, b in fixture.expected_contradiction_pairs
                ),
            ),
            input_tokens=100,
            output_tokens=20,
            attempts=1,
        )


class AlwaysPassJudge:
    """무엇이든 통과시키는 `Judging` 대역 — 검출률 0%·오탐률 0% 의 하한(양성 대조)."""

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
            input_tokens=7,
            output_tokens=3,
            attempts=1,
        )


class BrokenJudge:
    """형식 불일치 재시도까지 소진한 판정 — 측정 3 이 이것을 어떻게 세는지 못박는다."""

    def judge(self, *, draft: Draft, evidence: Sequence[Evidence]) -> JudgeOutcome:
        del draft, evidence
        raise LLMFormatError(
            stage=JUDGE_STAGE,
            detail="판정 산출이 형식에 맞지 않았다",
            raw_text="{}",
            input_tokens=11,
            output_tokens=2,
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


def test_미끼_라벨은_기각을_요구하지_않는다() -> None:
    """결정 0008 회귀 방지 — 재현율 분모는 범주(`reject_bait`)이고 라벨이 아니다.

    정면 조항을 딛은 "없다" 안내는 결정 0007 이 통과시키기로 정한 쪽이므로, 미끼 라벨이
    기각을 요구하면 채점표가 설계와 어긋난다. 기각은 허용하되 요구하지 않는다.
    """
    baits = [case for case in GOLDEN if case.category == "reject_bait"]
    assert {case.id for case in baits} == {"G16", "G17", "G18", "G19", "G20"}
    assert not any(case.expected.expect_reject for case in baits)


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
    """재현율 분모는 범주(`reject_bait`)이고, `expect_reject` 는 케이스 단위 요구다.

    기각이 한 번도 없으면 재현율에서 빠지고, `expect_reject` 라벨이 켜져 있으면
    그 케이스는 불일치로도 찍힌다 — 두 축이 분리되어 있음을 함께 고정한다.
    """
    bait = _case(
        "B",
        statuses=(InquiryStatus.ANSWERED, InquiryStatus.ESCALATED),
        reasons=(EscalationReason.REJECTED_TWICE,),
        expect_reject=True,
        category="reject_bait",
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
    # 반대 방향 고정: 미끼 범주가 아니면 expect_reject 가 켜져 있어도 분모에 들어가지 않는다.
    nonbait_expecting = _case(
        "N",
        statuses=(InquiryStatus.ANSWERED, InquiryStatus.ESCALATED),
        reasons=(EscalationReason.REJECTED_TWICE,),
        expect_reject=True,
    )

    agreement = measure_pipeline_agreement(
        cases=[bait, bait, nonbait_expecting],
        pipeline=ScriptedPipeline([rejected_then_passed, never_rejected, never_rejected]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert agreement.bait_total == 2
    assert agreement.bait_reject_reproduced == 1
    assert agreement.bait_reject_recall == 0.5
    assert agreement.bait_unmeasured == 0
    assert agreement.outcomes[0].matched is True
    assert agreement.outcomes[1].matched is False
    # 비미끼 케이스도 expect_reject 요구는 그대로 받는다 — 불일치로는 찍히되 분모 밖이다.
    assert agreement.outcomes[2].matched is False


def _bait_case(*, expect_reject: bool = False) -> GoldenCase:
    return _case(
        "B",
        statuses=(InquiryStatus.ANSWERED, InquiryStatus.ESCALATED),
        reasons=(EscalationReason.REJECTED_TWICE,),
        expect_reject=expect_reject,
        category="reject_bait",
    )


def _judge_call_died() -> ProcessedInquiry:
    """L2 호출이 무너진 시도 — L1 통과, L2 판정 없음, 종합 verdict 는 `pass` 다."""
    return _processed(
        status=InquiryStatus.ESCALATED,
        escalation=EscalationReason.LLM_CALL_FAILED,
        failed_stage=L2_JUDGE_STAGE,
        attempts=(AttemptRecord(attempt_no=1, verdict=Verdict.PASS, reject_reasons=(), draft={}),),
    )


def _gate_ran_and_missed() -> ProcessedInquiry:
    return _processed(
        attempts=(AttemptRecord(attempt_no=1, verdict=Verdict.PASS, reject_reasons=(), draft={}),)
    )


def test_판정이_돌지_못한_미끼는_게이트가_놓친_것과_구분된다() -> None:
    """두 상태의 `attempt_verdicts` 는 똑같이 `['pass']` 다.

    구분하지 않으면 인프라 실패가 "게이트가 미끼를 놓쳤다"로 접혀 분모에 0 으로 채워진다
    (`scripts/AGENTS.md` 불변식 5 — 미실행을 0 으로 채우지 않는다).
    """
    died, missed = _judge_call_died(), _gate_ran_and_missed()
    assert [a.verdict for a in died.attempts] == [a.verdict for a in missed.attempts]

    unmeasured = measure_pipeline_agreement(
        cases=[_bait_case()],
        pipeline=ScriptedPipeline([died]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )
    measured = measure_pipeline_agreement(
        cases=[_bait_case()],
        pipeline=ScriptedPipeline([missed]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    # 판정이 없었던 건은 분모 밖 — 재현율은 0.0 이 아니라 "미실행"이다.
    assert unmeasured.bait_total == 0
    assert unmeasured.bait_unmeasured == 1
    assert unmeasured.bait_reject_recall is None
    assert unmeasured.outcomes[0].gate_never_ran is True

    # 게이트가 실제로 돌았고 놓친 건은 그대로 0.0 이다.
    assert measured.bait_total == 1
    assert measured.bait_unmeasured == 0
    assert measured.bait_reject_recall == 0.0
    assert measured.outcomes[0].gate_never_ran is False


def test_판정이_없었던_케이스의_불일치_사유는_기각_실패가_아니다() -> None:
    """`어떤 시도도 기각되지 않았다` 는 거짓이다 — 게이트에 닿지 못했다."""
    agreement = measure_pipeline_agreement(
        cases=[_bait_case(expect_reject=True)],
        pipeline=ScriptedPipeline([_judge_call_died()]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    mismatches = agreement.outcomes[0].mismatches
    assert any("판정이 없었다(미측정)" in item for item in mismatches)
    assert not any("어떤 시도도 기각되지 않았다" in item for item in mismatches)


def test_분모에서_뺀_미측정은_리포트와_JSON_에_남는다(tmp_path: Path) -> None:
    """분모에서 빼기만 하고 말하지 않으면 그냥 사라진 것이다."""
    agreement = measure_pipeline_agreement(
        cases=[_bait_case()],
        pipeline=ScriptedPipeline([_judge_call_died()]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    markdown_path, json_path = write_report(
        _report(pipeline=agreement), out_dir=tmp_path, stem="unmeasured"
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))["measurement_2_pipeline_agreement"]

    assert "미측정 1건은 분모 밖이다" in markdown
    assert payload["bait_unmeasured"] == 1
    assert payload["outcomes"][0]["gate_never_ran"] is True


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


def test_지연_백분위와_건당_토큰은_생성_임베딩_판정을_구분해_집계한다() -> None:
    """판정 토큰이 생성 합산에 섞이면 건당 비용 지표가 무너진다 — 계열은 셋이다."""
    latencies = [10, 20, 30, 40, 100]
    results = [
        _processed(
            latency_ms=value,
            input_tokens=100,
            output_tokens=50,
            embedding_tokens=10,
            judge_input_tokens=200,
            judge_output_tokens=40,
        )
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
    assert agreement.judge_tokens_per_inquiry == 240.0
    assert agreement.total_tokens_per_inquiry == 400.0
    assert agreement.judge_input_tokens_total == 1000
    assert agreement.judge_output_tokens_total == 200


def test_L2_기각도_미끼_기각_재현율에_들어간다() -> None:
    """재현율 정의는 **시도 중 최소 1건 기각, 층 무관**이다 — L2 기각도 자동 포함된다."""
    bait = _case(
        "B",
        statuses=(InquiryStatus.ANSWERED, InquiryStatus.ESCALATED),
        reasons=(EscalationReason.REJECTED_TWICE,),
        category="reject_bait",
    )
    l2_rejected = _processed(
        attempts=(
            AttemptRecord(
                attempt_no=1,
                verdict=Verdict.REJECT,
                reject_reasons=(RejectReason.UNSUPPORTED_CLAIM,),
                draft={},
            ),
            AttemptRecord(attempt_no=2, verdict=Verdict.PASS, reject_reasons=(), draft={}),
        )
    )
    agreement = measure_pipeline_agreement(
        cases=[bait],
        pipeline=ScriptedPipeline([l2_rejected]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert agreement.bait_reject_reproduced == 1
    assert agreement.bait_reject_recall == 1.0
    assert agreement.outcomes[0].matched is True


def test_골든_산출물은_파이프라인이_채택한_근거_ID를_보존한다() -> None:
    evidence = Evidence(
        id="policy:support:4-1",
        source=EvidenceSource.POLICY,
        content="고객센터 운영 안내",
        evidence_text="고객센터 운영 안내",
    )
    agreement = measure_pipeline_agreement(
        cases=[_case("G17")],
        pipeline=ScriptedPipeline([_processed(evidence=(evidence,))]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert agreement.outcomes[0].adopted_evidence_ids == ("policy:support:4-1",)
    payload = report_to_json(_report(pipeline=agreement))
    outcome = payload["measurement_2_pipeline_agreement"]["outcomes"][0]
    assert outcome["adopted_evidence_ids"] == ["policy:support:4-1"]
    markdown = render_markdown(_report(pipeline=agreement))
    assert "### 케이스별 채택 근거" in markdown
    assert "`G17`: `policy:support:4-1`" in markdown


def test_평가_보고서는_검색_실패_생성_문제_빈_정답_정상_경로를_분해한다(
    tmp_path: Path,
) -> None:
    support = Evidence(
        id="policy:support:4-1",
        source=EvidenceSource.POLICY,
        content="고객센터 운영 안내",
        evidence_text="고객센터 운영 안내",
    )
    rejected = AttemptRecord(
        attempt_no=1,
        verdict=Verdict.REJECT,
        reject_reasons=(RejectReason.PII_DETECTED,),
        draft={},
    )
    l2_rejected = AttemptRecord(
        attempt_no=1,
        verdict=Verdict.REJECT,
        reject_reasons=(RejectReason.UNSUPPORTED_CLAIM,),
        draft={},
    )
    channel = Evidence(
        id="policy:support:4-2",
        source=EvidenceSource.POLICY,
        content="문의 접수 채널",
        evidence_text="문의 접수 채널",
    )
    # G01 은 정답이 2조항(2-1·2-2)인 케이스다 — 일부만 찾은 경로를 만들려면 분모가 2 여야 한다.
    refund_period = Evidence(
        id="policy:refund:2-1",
        source=EvidenceSource.POLICY,
        content="단순 변심 환불 기간",
        evidence_text="단순 변심 환불 기간",
    )
    escalated_statuses = (InquiryStatus.ESCALATED,)
    escalated_reasons = (EscalationReason.NO_EVIDENCE, EscalationReason.REJECTED_TWICE)
    agreement = measure_pipeline_agreement(
        cases=[
            _case("G01", statuses=escalated_statuses, reasons=escalated_reasons),
            _case("G17", statuses=escalated_statuses, reasons=escalated_reasons),
            _case("G20", statuses=escalated_statuses, reasons=escalated_reasons),
            _case("G19"),
            _case(
                "G21",
                statuses=escalated_statuses,
                reasons=escalated_reasons,
                category="no_evidence",
            ),
            _case(
                "G22",
                statuses=escalated_statuses,
                reasons=escalated_reasons,
                category="no_evidence",
            ),
        ],
        pipeline=ScriptedPipeline(
            [
                # G01: 정답 2조항 중 1조항만 채택 — 필요한 조항이 빠진 채 생성했다.
                _processed(
                    status=InquiryStatus.ESCALATED,
                    escalation=EscalationReason.REJECTED_TWICE,
                    attempts=(rejected,),
                    evidence=(refund_period,),
                ),
                # G17: 정답 조항을 하나도 못 찾았다.
                _processed(
                    status=InquiryStatus.ESCALATED,
                    escalation=EscalationReason.NO_EVIDENCE,
                ),
                # G20: 정답 2조항을 전부 채택했는데도 기각·인계 — 생성 문제.
                _processed(
                    status=InquiryStatus.ESCALATED,
                    escalation=EscalationReason.REJECTED_TWICE,
                    attempts=(rejected,),
                    evidence=(support, channel),
                ),
                # G19: 정답 조항 없이 답변이 확정됐다 — 게이트를 통과한 근거 부족.
                _processed(status=InquiryStatus.ANSWERED, evidence=(support,)),
                _processed(
                    status=InquiryStatus.ESCALATED,
                    escalation=EscalationReason.NO_EVIDENCE,
                ),
                _processed(
                    status=InquiryStatus.ESCALATED,
                    escalation=EscalationReason.REJECTED_TWICE,
                    attempts=(l2_rejected,),
                    evidence=(support,),
                ),
            ]
        ),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    markdown_path, json_path = write_report(_report(pipeline=agreement), out_dir=tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    breakdown = payload["failure_attribution"]
    assert breakdown["computed"] is True
    assert breakdown["generation_issue_count"] == 1
    assert breakdown["retrieval_failure_count"] == 1
    assert breakdown["partial_retrieval_failure_count"] == 1
    assert breakdown["retrieval_failure_total"] == 2
    assert breakdown["answered_without_relevant_evidence_count"] == 1
    assert breakdown["expected_no_answer_count"] == 2
    by_id = {item["case_id"]: item for item in breakdown["cases"]}
    # 정답 조항 일부만 찾은 것은 생성 문제가 아니다 — 필요한 근거가 빠져 있었다.
    assert by_id["G01"]["classification"] == "partial_retrieval_failure"
    assert by_id["G01"]["missing_relevant_evidence_ids"] == ["policy:refund:2-2"]
    assert by_id["G17"]["classification"] == "retrieval_failure"
    assert by_id["G20"]["classification"] == "generation_issue"
    assert by_id["G19"]["classification"] == "answered_without_relevant_evidence"
    assert by_id["G21"]["ended_with_zero_evidence"] is True
    assert by_id["G21"]["l2_caught_with_evidence"] is False
    assert by_id["G21"]["normal_behavior"] is True
    assert by_id["G21"]["normal_behavior_path"] == "retrieval_zero_evidence"
    assert by_id["G22"]["ended_with_zero_evidence"] is False
    assert by_id["G22"]["l2_caught_with_evidence"] is True
    assert by_id["G22"]["normal_behavior"] is True
    assert by_id["G22"]["normal_behavior_path"] == "l2_rejected_with_evidence"
    assert by_id["G21"]["classification"] == "expected_no_answer"
    assert by_id["G22"]["classification"] == "expected_no_answer"

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "### 검색 실패 / 생성 문제 분해" in markdown
    assert "생성 문제: **1건**" in markdown
    assert "검색 실패 합계: **2건**" in markdown
    assert "일부 누락 1건" in markdown
    assert "근거 없이 답변 확정: **1건**" in markdown
    assert "빈 정답 정상 인계: **2건**" in markdown
    assert "`G21`" in markdown and "검색 0건 종료" in markdown
    assert "`G22`" in markdown and "근거 채택 후 L2 검출" in markdown


def test_빈_정답의_LLM_실패와_L1만_기각은_정상_인계로_위장하지_않는다(
    tmp_path: Path,
) -> None:
    unrelated = Evidence(
        id="policy:support:4-1",
        source=EvidenceSource.POLICY,
        content="고객센터 운영 안내",
        evidence_text="고객센터 운영 안내",
    )
    l1_rejected = AttemptRecord(
        attempt_no=1,
        verdict=Verdict.REJECT,
        reject_reasons=(RejectReason.PII_DETECTED,),
        draft={},
    )
    agreement = measure_pipeline_agreement(
        cases=[
            _case(
                "G23",
                statuses=(InquiryStatus.ESCALATED,),
                reasons=(EscalationReason.LLM_CALL_FAILED,),
                category="no_evidence",
            ),
            _case(
                "G24",
                statuses=(InquiryStatus.ESCALATED,),
                reasons=(EscalationReason.REJECTED_TWICE,),
                category="no_evidence",
            ),
        ],
        pipeline=ScriptedPipeline(
            [
                _processed(
                    status=InquiryStatus.ESCALATED,
                    escalation=EscalationReason.LLM_CALL_FAILED,
                    failed_stage="intent",
                ),
                _processed(
                    status=InquiryStatus.ESCALATED,
                    escalation=EscalationReason.REJECTED_TWICE,
                    attempts=(l1_rejected,),
                    evidence=(unrelated,),
                ),
            ]
        ),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    markdown_path, json_path = write_report(_report(pipeline=agreement), out_dir=tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    breakdown = payload["failure_attribution"]
    assert breakdown["expected_no_answer_count"] == 0
    assert breakdown["expected_no_answer_anomaly_count"] == 2
    by_id = {item["case_id"]: item for item in breakdown["cases"]}
    assert by_id["G23"]["classification"] == "expected_no_answer"
    assert by_id["G23"]["normal_behavior"] is False
    assert by_id["G23"]["anomaly_reason"] == ("검색 0건이지만 no_evidence·시도 0건 종료가 아님")
    assert by_id["G24"]["classification"] == "expected_no_answer"
    assert by_id["G24"]["normal_behavior"] is False
    assert by_id["G24"]["anomaly_reason"] == "근거를 채택했지만 L2 기각 사유 없음"

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "빈 정답 정상 인계: **0건**" in markdown
    assert "빈 정답 비정상 종결: **2건**" in markdown
    assert "`G23`: **빈 정답 비정상 종결**" in markdown
    assert "`G24`: **빈 정답 비정상 종결**" in markdown


def test_검색_라벨이_없으면_분해만_미산출하고_측정_1_2_3은_유지한다(
    tmp_path: Path,
) -> None:
    agreement = measure_pipeline_agreement(
        cases=[_case("G17")],
        pipeline=ScriptedPipeline([_processed()]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )
    judge_accuracy = measure_judge_accuracy(
        fixtures=JUDGE_FIXTURES,
        judge=OracleJudge(JUDGE_FIXTURES),
    )
    report = build_report(
        conditions=_conditions(judge_is_real=True),
        gate_accuracy=measure_gate_accuracy(FIXTURES),
        pipeline=agreement,
        judge_accuracy=judge_accuracy,
        retrieval_labels_path=tmp_path / "missing-retrieval-labels.jsonl",
    )

    markdown_path, json_path = write_report(report, out_dir=tmp_path, stem="missing-labels")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    breakdown = payload["failure_attribution"]
    assert breakdown["computed"] is False
    assert "FileNotFoundError" in breakdown["reason"]
    assert "generation_issue_count" not in breakdown
    assert payload["measurement_1_l1_gate_accuracy"]["total"] == len(FIXTURES)
    assert payload["measurement_2_pipeline_agreement"]["executed"] is True
    assert payload["measurement_2_pipeline_agreement"]["total"] == 1
    assert payload["measurement_3_l2_judge_accuracy"]["executed"] is True
    assert payload["measurement_3_l2_judge_accuracy"]["total"] == len(JUDGE_FIXTURES)

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "**미산출 (사유: 검색 정답 라벨 로드 실패(FileNotFoundError):" in markdown
    assert "## 측정 1 —" in markdown
    assert "## 측정 2 —" in markdown
    assert "## 측정 3 —" in markdown


# ── 측정 3 — L2 판정 단위 정확도 (확률 층·과금) ─────────────────────────────


def test_판정_픽스처는_의미_정책_4분면을_모두_덮는다() -> None:
    """로더가 저장소 픽스처를 그대로 소비하는지 + 4분면이 라벨로 구분되는지."""
    assert len(JUDGE_FIXTURES) == 11
    by_category = {fixture.category for fixture in JUDGE_FIXTURES}
    quadrants = {
        "adjacent_citation": Verdict.REJECT,
        "direct_no_value": Verdict.PASS,
        "contradiction_unstated": Verdict.REJECT,
        "contradiction_stated": Verdict.PASS,
    }
    assert quadrants.keys() <= by_category
    for category, verdict in quadrants.items():
        members = [fixture for fixture in JUDGE_FIXTURES if fixture.category == category]
        assert members and all(fixture.expected_verdict is verdict for fixture in members), category

    # 사유 2종 각각 양성이 있고, 정상 통과 음성도 있어야 검출률·오탐률이 의미를 가진다.
    for reason in (RejectReason.UNSUPPORTED_CLAIM, RejectReason.CONTRADICTORY_EVIDENCE):
        assert any(reason in fixture.expected_reasons for fixture in JUDGE_FIXTURES), reason
    assert sum(1 for fixture in JUDGE_FIXTURES if not fixture.is_violation) >= 2
    # 모순쌍 라벨이 실제로 실린다(근거쌍 단위 기록).
    assert any(fixture.expected_contradictions for fixture in JUDGE_FIXTURES)


def test_판정_픽스처는_L1_을_통과한_초안이다() -> None:
    """L2 는 L1 통과분에만 도는 층이다 — 픽스처가 L1 에 걸리면 측정 3 이 무엇을 재는지 흐려진다."""
    for fixture in JUDGE_FIXTURES:
        result = evaluate_draft(
            raw_draft={
                "claims": [
                    {"text": claim.text, "citation_ids": list(claim.citation_ids)}
                    for claim in fixture.claims
                ]
            },
            evidences=fixture.evidences,
        )
        assert result.verdict is Verdict.PASS, (fixture.id, result.reject_reasons)


def test_판정_픽스처_라벨이_판정과_모순이면_로드에서_막는다(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "J00",
                "category": "clean",
                "evidences": [
                    {"id": "policy:refund:2-1", "source": "policy", "content": "환불은 7일."}
                ],
                "claims": [{"text": "환불은 7일.", "citation_ids": ["policy:refund:2-1"]}],
                "expected": {
                    "verdict": "pass",
                    "reject_reasons": ["unsupported_claim"],
                    "claim_judgments": [{"claim_text": "환불은 7일.", "verdict": "pass"}],
                    "contradictions": [],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="사유가"):
        load_judge_fixtures(path)


def test_판정_픽스처의_claim_판정_라벨은_claim_전부와_대응해야_한다(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "J00",
                "category": "clean",
                "evidences": [
                    {"id": "policy:refund:2-1", "source": "policy", "content": "환불은 7일."}
                ],
                "claims": [{"text": "환불은 7일.", "citation_ids": ["policy:refund:2-1"]}],
                "expected": {
                    "verdict": "pass",
                    "reject_reasons": [],
                    "claim_judgments": [],
                    "contradictions": [],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="claim"):
        load_judge_fixtures(path)


def test_한_픽스처_안의_claim_텍스트가_중복이면_로드에서_막는다(tmp_path: Path) -> None:
    """판정 계약이 claim 을 **텍스트로 식별**하므로 중복 텍스트는 라벨 쪽에서 막아야 한다.

    막지 않으면 대조 딕셔너리가 두 claim 을 한 판정으로 접는데 `claim_total` 은 2 로 세어,
    두 건이 같은 verdict 로 채점된다 — claim 단위 일치율이 조용히 거짓이 된다.
    """
    claim = {"text": "환불은 7일.", "citation_ids": ["policy:refund:2-1"]}
    path = tmp_path / "dup.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "J00",
                "category": "clean",
                "evidences": [
                    {"id": "policy:refund:2-1", "source": "policy", "content": "환불은 7일."}
                ],
                "claims": [claim, dict(claim)],
                "expected": {
                    "verdict": "pass",
                    "reject_reasons": [],
                    "claim_judgments": [
                        {"claim_text": "환불은 7일.", "verdict": "pass"},
                        {"claim_text": "환불은 7일.", "verdict": "pass"},
                    ],
                    "contradictions": [],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="유일"):
        load_judge_fixtures(path)


def test_측정3_은_기대대로_판정하면_검출률_100_오탐률_0_이다() -> None:
    accuracy = measure_judge_accuracy(fixtures=JUDGE_FIXTURES, judge=OracleJudge(JUDGE_FIXTURES))

    assert accuracy.total == len(JUDGE_FIXTURES)
    assert accuracy.error_total == 0
    assert accuracy.detection_rate == 1.0
    assert accuracy.false_positive_rate == 0.0
    assert accuracy.reason_set_exact_rate == 1.0
    assert accuracy.claim_verdict_match_rate == 1.0
    assert accuracy.contradiction_recall == 1.0
    assert accuracy.contradiction_extra_total == 0
    assert accuracy.input_tokens_total == 100 * len(JUDGE_FIXTURES)


def test_측정3_은_전부_통과시키는_판정에서_검출률_0_이다() -> None:
    """양성 대조 — 전부 통과시키는 판정기는 오탐 0 이지만 검출도 0 이다."""
    accuracy = measure_judge_accuracy(fixtures=JUDGE_FIXTURES, judge=AlwaysPassJudge())

    assert accuracy.detection_rate == 0.0
    assert accuracy.false_positive_rate == 0.0
    assert accuracy.violation_total > 0
    missed = {outcome.fixture_id for outcome in accuracy.outcomes if not outcome.verdict_matched}
    assert missed == {
        fixture.id for fixture in JUDGE_FIXTURES if fixture.expected_verdict is Verdict.REJECT
    }


def test_측정3_은_판정_실패를_분모에서_빼고_따로_센다() -> None:
    """형식 불일치 소진은 "판정하지 못했다"이지 "통과시켰다"가 아니다 — 0 으로 채우지 않는다."""
    accuracy = measure_judge_accuracy(fixtures=JUDGE_FIXTURES, judge=BrokenJudge())

    assert accuracy.error_total == len(JUDGE_FIXTURES)
    assert accuracy.judged_total == 0
    assert accuracy.detection_rate is None
    assert accuracy.false_positive_rate is None
    # 실패한 호출이 쓴 토큰도 실비용이므로 그대로 집계한다.
    assert accuracy.input_tokens_total == 11 * len(JUDGE_FIXTURES)
    assert all(outcome.error is not None for outcome in accuracy.outcomes)


def test_측정3_은_결정론_판정_대역으로도_끝까지_돈다() -> None:
    """대역은 실제 판정 모델이 아니다 — 여기서 보는 것은 배관이 끝까지 도는지다."""
    seen: list[str] = []
    accuracy = measure_judge_accuracy(
        fixtures=JUDGE_FIXTURES,
        judge=StubJudge(),
        on_outcome=lambda outcome: seen.append(outcome.fixture_id),
    )

    assert seen == [fixture.id for fixture in JUDGE_FIXTURES]
    assert accuracy.error_total == 0
    assert accuracy.detection_rate is not None
    assert accuracy.input_tokens_total > 0


# ── 리포트 ──────────────────────────────────────────────────────────────────


def test_미실행_측정2_는_0_이_아니라_사유로_남는다(tmp_path: Path) -> None:
    report = _report(pipeline=SkippedMeasurement(reason="OPENAI_API_KEY 가 없다"))
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


def test_미실행_측정3_은_0_이_아니라_사유로_남는다() -> None:
    report = _report(
        pipeline=SkippedMeasurement(reason="미요청"),
        judge=SkippedMeasurement(reason="--live 가 아니다"),
    )
    markdown = render_markdown(report)
    payload = report_to_json(report)

    assert "## 측정 3 — L2 판정 단위 정확도 (확률 층)" in markdown
    assert "**미실행 (사유: --live 가 아니다)**" in markdown
    assert payload["measurement_3_l2_judge_accuracy"] == {
        "executed": False,
        "skip_reason": "--live 가 아니다",
    }


def test_실측된_측정3_리포트는_확률층과_과금과_무목표를_적는다() -> None:
    """측정 1 과 같은 형태의 수치지만 **재현되지 않고 과금된다** — 리포트가 그 사실을 들고 있다."""
    accuracy = measure_judge_accuracy(fixtures=JUDGE_FIXTURES, judge=OracleJudge(JUDGE_FIXTURES))
    report = _report(
        pipeline=SkippedMeasurement(reason="미요청"),
        judge=accuracy,
        conditions=_conditions(judge_is_real=True),
    )
    markdown = render_markdown(report)
    payload = report_to_json(report)["measurement_3_l2_judge_accuracy"]

    assert "확률 층이고 과금된다" in markdown
    assert "목표치: **없음**" in markdown
    assert "L2 검출률: 100.0%" in markdown
    assert "L2 오탐률: 0.0%" in markdown
    # 목표치 대비 표는 3종 그대로다 — 측정 3 지표를 목표로 승격하지 않는다.
    assert {target.key for target in TARGETS} == {
        "detection_rate",
        "false_positive_rate",
        "match_rate",
    }
    assert payload["executed"] is True
    assert payload["is_real"] is True
    assert payload["deterministic"] is False
    assert payload["billed"] is True
    assert payload["target"] is None
    assert payload["detection_rate"] == 1.0
    assert payload["false_positive_rate"] == 0.0
    assert len(payload["outcomes"]) == len(JUDGE_FIXTURES)
    assert payload["tokens"]["input_total"] > 0


def test_대역으로_만든_측정3_은_과금된_실측으로_적히지_않는다() -> None:
    """리포트가 스스로 "과금된 실측"이라고 거짓 신고할 수 있으면 이 제품의 유일한 주장이 무너진다.

    같은 대역 산출물이라도 실측 플래그가 갈리면 JSON 의 `billed`·`deterministic` 과
    마크다운의 확률층·과금 문구가 **통째로 뒤집혀야** 한다 — 구분이 실행 조건의 판정 모델
    문자열 한 줄에만 있으면 형식만으로는 실측과 구분되지 않는다.
    """
    accuracy = measure_judge_accuracy(fixtures=JUDGE_FIXTURES, judge=StubJudge())
    report = _report(
        pipeline=SkippedMeasurement(reason="미요청"),
        judge=accuracy,
        conditions=_conditions(judge_is_real=False),
    )
    markdown = render_markdown(report)
    payload = report_to_json(report)

    assert "실제 판정 모델 수치가 아니다" in markdown
    assert "과금되지 않았고 재실행해도 같은 값" in markdown
    assert "확률 층이고 과금된다" not in markdown
    judged = payload["measurement_3_l2_judge_accuracy"]
    assert judged["executed"] is True
    assert judged["is_real"] is False
    assert judged["billed"] is False
    assert judged["deterministic"] is True
    assert payload["conditions"]["measurement3_is_real"] is False


def test_리포트_토큰_표는_생성_임베딩_판정_세_계열을_적는다() -> None:
    """L2 켜짐 실측에서 판정 비용을 빠뜨리면 건당 비용 지표가 거짓이 된다."""
    agreement = measure_pipeline_agreement(
        cases=[_case("A")],
        pipeline=ScriptedPipeline(
            [
                _processed(
                    input_tokens=100,
                    output_tokens=50,
                    embedding_tokens=10,
                    judge_input_tokens=200,
                    judge_output_tokens=40,
                )
            ]
        ),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )
    report = _report(pipeline=agreement, conditions=_conditions(is_real=True))
    markdown = render_markdown(report)
    tokens = report_to_json(report)["measurement_2_pipeline_agreement"]["tokens"]

    assert "| 생성 소계 | 150 |" in markdown
    assert "| 임베딩 | 10 |" in markdown
    assert "| 판정 소계 | 240 |" in markdown
    assert "| **합산** | 400 |" in markdown
    assert tokens["judge_input_total"] == 200
    assert tokens["judge_output_total"] == 40
    assert tokens["judge_per_inquiry"] == 240.0
    assert tokens["total_per_inquiry"] == 400.0

    # 대역 실행이면 같은 표의 판정 행이 대역임을 스스로 말한다 — `--stub-llm` 은 판정자까지
    # 대역으로 갈아 끼우고 그 휴리스틱 값이 합산에까지 들어간다.
    stub_markdown = render_markdown(_report(pipeline=agreement, conditions=_conditions()))
    assert "| 판정 입력 (대역) | 200 |" in stub_markdown
    assert "| 판정 소계 (대역) | 240 |" in stub_markdown
    assert "| 판정 소계 | 240 |" not in stub_markdown


def test_확정된_목표치가_리포트에_실린다() -> None:
    report = _report(pipeline=SkippedMeasurement(reason="미요청"))
    markdown = render_markdown(report)
    assert "## 목표치 대비" in markdown
    assert "| 측정 1 구조적 오류 검출률 | 달성 목표 | ≥ 100% |" in markdown
    assert "| 측정 1 정상 초안 오탐률 | 달성 목표 | ≤ 0% |" in markdown
    assert "| 측정 2 허용 결과 집합 대비 일치율 | 하한 경보선 | ≥ 75% |" in markdown

    metrics = report_to_json(report)["targets"]["metrics"]
    by_key = {metric["key"]: metric for metric in metrics}
    assert by_key["detection_rate"]["bound"] == 1.0
    assert by_key["false_positive_rate"]["direction"] == "at_most"
    assert by_key["match_rate"]["bound"] == 0.75


def test_측정하지_않은_지표는_미달로_적지_않는다() -> None:
    """돌지 않은 측정을 "미달"로 찍으면 리포트가 거짓말을 한다 — `None` 이어야 한다."""
    report = _report(pipeline=SkippedMeasurement(reason="미요청"))
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
    report = _report(pipeline=SkippedMeasurement(reason="미요청"))
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
    return _report(pipeline=agreement, conditions=_conditions(is_real=True))


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
    # 합산 일치율은 **하한 경보선**이라 미달 문면이 "경보"다(결정 0006 재확정).
    # `met` 은 여전히 False 다 — 경계 판정 자체가 사라진 것이 아니라 성격이 바뀐 것이다.
    assert missed_json["met"] is False and missed_json["verdict"] == "경보"

    assert "| 100.0% | 달성 |" in render_markdown(met_report)
    assert "| 0.0% | **경보** |" in render_markdown(missed_report)


def test_대역_일치율은_달성으로_판정하지_않는다() -> None:
    """대역 수치로 확률 층 목표를 "달성" 이라 찍으면 리포트가 거짓말을 한다."""
    agreement = measure_pipeline_agreement(
        cases=[_case("A")],
        pipeline=ScriptedPipeline([_processed()]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )
    report = _report(pipeline=agreement, conditions=_conditions(is_real=False))
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


def _stem(
    tmp_path: Path, *, requested: str | None = None, real: bool = False, l2: bool = False
) -> str:
    return resolve_report_stem(
        requested=requested,
        live_requested=True,
        billed=real,
        l2_enabled=l2,
        out_dir=tmp_path,
    )


def test_기본_이름은_실측일_때만_라이브다(tmp_path: Path) -> None:
    assert _stem(tmp_path, real=True) == "evaluation-live"
    # --live 를 줬어도 키·DB 문제로 측정 2 가 미실행이면 비실측 — 라이브 이름을 쓰면 안 된다.
    assert _stem(tmp_path, real=False) == "evaluation"


def test_L2_켜짐_실측의_기본_이름은_l2_계열로_자동_넘버링된다(tmp_path: Path) -> None:
    """3회 반복 실측이 기본 이름 충돌로 죽으면 안 된다 — 코드가 빈 번호를 찾는다."""
    assert _stem(tmp_path, real=True, l2=True) == "evaluation-live-l2-1"
    (tmp_path / "evaluation-live-l2-1.md").write_text("1회차", encoding="utf-8")
    assert _stem(tmp_path, real=True, l2=True) == "evaluation-live-l2-2"
    (tmp_path / "evaluation-live-l2-2.json").write_text("{}", encoding="utf-8")
    assert _stem(tmp_path, real=True, l2=True) == "evaluation-live-l2-3"
    # 꺼짐 기준선 계열은 그대로다 — 두 계열이 섞이지 않는다.
    assert _stem(tmp_path, real=True, l2=False) == "evaluation-live"


def test_비실측은_라이브_이름을_쓸_수_없다(tmp_path: Path) -> None:
    for stem in (
        "evaluation-live",
        "evaluation-live-9",
        "Evaluation-live",
        "EVALUATION-LIVE-2",
        "evaluation-live-l2-1",
        "Evaluation-Live-L2-4",
    ):
        for l2 in (False, True):
            with pytest.raises(ReportStemError):
                _stem(tmp_path, requested=stem, real=False, l2=l2)


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
            _stem(tmp_path, requested=stem, real=False)


def test_실측은_라이브_이름만_쓸_수_있다(tmp_path: Path) -> None:
    """실측 결과가 gitignore 되는 이름으로 새면 다음 기본 실행이 그대로 덮는다."""
    with pytest.raises(ReportStemError):
        _stem(tmp_path, requested="evaluation", real=True)
    assert _stem(tmp_path, requested="evaluation-live-9", real=True) == "evaluation-live-9"


def test_L2_켜짐_실측은_명시_스템도_l2_계열이어야_한다(tmp_path: Path) -> None:
    """기본 이름만 가르면 `--report-stem evaluation-live-4` 가 꺼짐 기준선을 오염시킨다."""
    for stem in ("evaluation-live", "evaluation-live-4", "evaluation"):
        with pytest.raises(ReportStemError):
            _stem(tmp_path, requested=stem, real=True, l2=True)
    assert (
        _stem(tmp_path, requested="evaluation-live-l2-9", real=True, l2=True)
        == "evaluation-live-l2-9"
    )


def test_L2_꺼짐_실측은_l2_이름을_쓸_수_없다(tmp_path: Path) -> None:
    """반대 방향도 막아야 양방향이다 — 꺼짐 기준선이 L2 계열로 새면 계열이 뒤섞인다."""
    for stem in ("evaluation-live-l2-1", "evaluation-live-l2", "Evaluation-Live-L2-1"):
        with pytest.raises(ReportStemError):
            _stem(tmp_path, requested=stem, real=True, l2=False)


def test_실측은_기존_라이브_리포트를_덮어쓸_수_없다(tmp_path: Path) -> None:
    """실측은 비결정론이라 덮이면 그 수치는 재생성되지 않는다 — 빈 이름을 제안해야 한다."""
    (tmp_path / "evaluation-live.md").write_text("기존 실측", encoding="utf-8")
    (tmp_path / "evaluation-live-1.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReportStemError, match="evaluation-live-2"):
        _stem(tmp_path, real=True)
    with pytest.raises(ReportStemError):
        _stem(tmp_path, requested="evaluation-live-1", real=True)


def test_L2_켜짐_실측의_빈_이름_제안도_l2_계열을_따른다(tmp_path: Path) -> None:
    (tmp_path / "evaluation-live-l2-1.md").write_text("기존 실측", encoding="utf-8")
    with pytest.raises(ReportStemError, match="evaluation-live-l2-2"):
        _stem(tmp_path, requested="evaluation-live-l2-1", real=True, l2=True)


def test_리포트는_실행_조건과_한계를_함께_적는다() -> None:
    markdown = render_markdown(_report(pipeline=SkippedMeasurement(reason="미요청")))
    assert "유사도 임계값: 0.05" in markdown
    assert "top k: 5" in markdown
    assert "패턴형 PII 만 본다" in markdown
    assert "재실행하면 값이 달라지고" in markdown
    # L2 를 끈 실행에는 의미 층이 통째로 없다는 사실이 한계 절에 남아야 한다.
    assert "L2 를 끈 실행에는 그 층이 통째로 없다" in markdown


def test_실행_조건에_L2_켜짐_여부와_판정_모델이_남는다() -> None:
    """L2 켜짐 실측과 꺼짐 기준선은 같은 형태의 리포트를 낸다 — 구분은 실행 조건이 들고 있다."""
    on = render_markdown(
        _report(
            pipeline=SkippedMeasurement(reason="미요청"),
            conditions=_conditions(l2_enabled=True, judge="Anthropic `claude-sonnet-5`"),
        )
    )
    off = render_markdown(
        _report(
            pipeline=SkippedMeasurement(reason="미요청"),
            conditions=_conditions(l2_enabled=False, judge="L2 꺼짐 (판정 미실행)"),
        )
    )
    assert "- L2 판정: 켜짐" in on
    assert "판정 모델: Anthropic `claude-sonnet-5`" in on
    assert "- L2 판정: 꺼짐" in off

    payload = report_to_json(
        _report(
            pipeline=SkippedMeasurement(reason="미요청"),
            conditions=_conditions(l2_enabled=True, judge="Anthropic `claude-sonnet-5`"),
        )
    )
    assert payload["conditions"]["l2_enabled"] is True
    assert payload["conditions"]["judge"] == "Anthropic `claude-sonnet-5`"
    assert payload["conditions"]["judge_api_key_present"] is False


def test_대역으로_돈_측정2_는_실제_수치가_아니라고_경고한다() -> None:
    agreement = measure_pipeline_agreement(
        cases=[_case("A")],
        pipeline=ScriptedPipeline([_processed()]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )
    stub_markdown = render_markdown(_report(pipeline=agreement, conditions=_conditions()))
    real_markdown = render_markdown(
        _report(pipeline=agreement, conditions=_conditions(is_real=True))
    )
    assert "실제 모델 수치가 아니다" in stub_markdown
    assert "실제 모델 수치가 아니다" not in real_markdown
    # 일치율은 초안 전 인계 포함임을 리포트가 명시해야 한다.
    assert "초안 전 인계 경로 포함" in stub_markdown
    # `--stub-llm` 은 판정자까지 대역으로 갈아 끼운다 — 경고가 판정 대역을 열거하고,
    # 안내 항목에 판정 모델이 들어가야 실행 조건 어디를 읽어야 하는지가 맞다.
    assert "판정 대역" in stub_markdown
    assert "임베딩·판정 모델 항목" in stub_markdown


def test_리포트는_사람용과_기계용_두_형식을_낸다(tmp_path: Path) -> None:
    report = _report(pipeline=SkippedMeasurement(reason="미요청"))
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


def _l2_settings(
    *, live_keys: bool = True, judge_key: bool = True, l2_enabled: bool = True
) -> Settings:
    key = "키가-아닌-테스트값"
    return Settings(
        l2_enabled=l2_enabled,
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


def test_stub_llm_은_L2_켜짐에서도_산출물을_낸다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """결정론 판정 대역이 생겼으므로 대역 실행은 L2 켜짐에서도 죽지 않는다.

    (여기서는 DB 를 없애 측정 2·3 을 미실행으로 만든다 — 확인 대상은 "일시 가드가
    걷혔는가"이고, L2 켜짐 대역 배관 자체는 `db` 마커 e2e 가 검증한다.)
    """
    monkeypatch.setattr(evaluate, "get_settings", lambda: _l2_settings())
    monkeypatch.setattr(evaluate, "database_unavailable_reason", lambda *, settings: "DB 사유")
    _block_outbound_sockets(monkeypatch)

    assert evaluate.main(["--stub-llm", "--out-dir", str(tmp_path)]) == 0

    payload = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["measurement_1_l1_gate_accuracy"]["llm_calls"] == 0
    assert payload["conditions"]["l2_enabled"] is True
    # 대역 실행은 실측이 아니므로 라이브 이름 계열은 만들어지지 않는다.
    assert not list(tmp_path.glob("evaluation-live*"))


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


def test_판정_키가_없으면_측정2_와_측정3_을_모두_건너뛴다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """시작해 놓고 첫 판정 호출에서 죽으면 30건 중 일부만 과금하고 산출물은 없다.

    그리고 **강등 실행은 금지**다 — 판정 키가 없다고 L2 를 꺼서 측정 2 만 돌리면
    기준선과 실행 조건이 오염된다. 둘 다 같은 사유로 미실행이어야 한다.
    """
    monkeypatch.setattr(evaluate, "get_settings", lambda: _l2_settings(judge_key=False))
    monkeypatch.setattr(evaluate, "database_unavailable_reason", lambda *, settings: "DB 사유")
    _block_outbound_sockets(monkeypatch)

    assert evaluate.main(["--live", "--out-dir", str(tmp_path)]) == 0

    payload = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert "ANTHROPIC_API_KEY" in payload["measurement_2_pipeline_agreement"]["skip_reason"]
    judged = payload["measurement_3_l2_judge_accuracy"]
    assert judged["executed"] is False
    assert "ANTHROPIC_API_KEY" in judged["skip_reason"]
    # 강등되지 않았다 — 실행 조건에 L2 는 켜짐 그대로 남는다.
    assert payload["conditions"]["l2_enabled"] is True

    # 양성 대조 — L2 가 꺼져 있으면 판정 키 부재는 사유가 아니고 뒤 검사(DB)로 넘어간다.
    args = evaluate.build_parser().parse_args(["--live"])
    settings = _l2_settings(judge_key=False).model_copy(update={"l2_enabled": False})
    assert evaluate._skip_reason(args=args, settings=settings) == "DB 사유"


def test_stub_llm_실행에서_측정3_은_미실행_사유로_남는다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """측정 3 은 실제 판정 모델을 부르는 확률 층이라 `--live` 로만 돈다."""
    monkeypatch.setattr(evaluate, "get_settings", lambda: _l2_settings())
    monkeypatch.setattr(evaluate, "database_unavailable_reason", lambda *, settings: None)
    _block_outbound_sockets(monkeypatch)

    args = evaluate.build_parser().parse_args(["--stub-llm"])
    reason = evaluate._judge_skip_reason(args=args, settings=_l2_settings(), skip=None)

    assert reason is not None
    assert "--live" in reason


def _canned_measurement_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """측정 2 를 DB·실제 LLM 없이 "돌았다"로 만든다 — 여기서 보는 것은 측정 3 배선이다."""

    def _fake(*, cases: Any, args: Any, settings: Any) -> Any:
        del args, settings
        agreement = measure_pipeline_agreement(
            cases=[_case(case.id) for case in cases[:1]],
            pipeline=ScriptedPipeline([_processed()]),
            app_conn=_NO_CONN,
            readonly_conn=_NO_CONN,
        )
        return agreement, "실제 생성(대체됨)", "실제 임베딩(대체됨)"

    monkeypatch.setattr(evaluate, "_run_measurement_two", _fake)


def test_live_실행은_측정3_을_실제로_돌려_l2_계열_산출물에_싣는다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**유일하게 과금되는 경로**의 회귀 방어 — 배선이 끊겨도 전체가 녹색이면 안 된다.

    실 API 호출은 아웃바운드 소켓 차단으로 막고 판정자만 대역으로 갈아 끼운다. 측정 2 는
    과금·DB 없이 "돌았다"로 대체해 L2 켜짐 실측 조건(= l2 계열 이름)을 만든다.
    """
    monkeypatch.setattr(evaluate, "get_settings", lambda: _l2_settings())
    monkeypatch.setattr(evaluate, "database_unavailable_reason", lambda *, settings: None)
    monkeypatch.setattr(evaluate, "build_judge", lambda settings: OracleJudge(JUDGE_FIXTURES))
    _canned_measurement_two(monkeypatch)
    _block_outbound_sockets(monkeypatch)

    assert evaluate.main(["--live", "--out-dir", str(tmp_path)]) == 0

    # L2 켜짐 실측이므로 l2 계열 이름으로 떨어진다.
    json_path = tmp_path / "evaluation-live-l2-1.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    judged = payload["measurement_3_l2_judge_accuracy"]
    assert judged["executed"] is True
    assert judged["total"] == len(JUDGE_FIXTURES)
    assert judged["error_total"] == 0
    assert judged["detection_rate"] == 1.0
    assert payload["conditions"]["judge_fixture_count"] == len(JUDGE_FIXTURES)


def test_측정3_의_예상_밖_예외는_이미_과금된_측정2_산출물을_날리지_않는다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """측정 2 가 끝난 뒤의 예외로 트레이스백을 내면 골든셋 30건을 다시 사야 한다.

    `evaluate_judge_fixture` 가 잡는 것은 판정 형식·전송 오류뿐이라, SDK 스큐(TypeError)나
    Ctrl-C 는 그대로 올라온다 — 미실행으로 강등하고 사유에 예외 종류와 메시지를 남긴 뒤
    리포트는 **반드시** 쓴다.

    `SystemExit` 도 같은 경로다: `BaseException` 직계라 `except (Exception, KeyboardInterrupt)`
    로는 잡히지 않아, 판정 SDK 나 그 의존이 `sys.exit()` 를 부르면 리포트 없이 프로세스가
    끝난다 — 과금된 산출물을 남기는 것이 종료 신호를 전달하는 것보다 우선이다.
    """
    for index, error in enumerate(
        (
            TypeError("SDK 버전 스큐"),
            KeyboardInterrupt(),
            SystemExit("판정 SDK 의존이 sys.exit() 를 불렀다"),
        )
    ):
        out_dir = tmp_path / f"run{index}"

        def _boom(settings: Any, error: BaseException = error) -> Any:
            del settings
            raise error

        monkeypatch.setattr(evaluate, "get_settings", lambda: _l2_settings())
        monkeypatch.setattr(evaluate, "database_unavailable_reason", lambda *, settings: None)
        monkeypatch.setattr(evaluate, "build_judge", _boom)
        _canned_measurement_two(monkeypatch)
        _block_outbound_sockets(monkeypatch)

        # 산출물을 남기고 실패를 종료 코드로 알린다 — 조용한 성공이 아니다.
        assert evaluate.main(["--live", "--out-dir", str(out_dir)]) == 1

        payload = json.loads((out_dir / "evaluation-live-l2-1.json").read_text(encoding="utf-8"))
        # 과금이 끝난 측정 2 는 그대로 남는다.
        assert payload["measurement_2_pipeline_agreement"]["executed"] is True
        judged = payload["measurement_3_l2_judge_accuracy"]
        assert judged["executed"] is False
        assert type(error).__name__ in judged["skip_reason"]


@pytest.mark.parametrize(
    "error",
    [
        psycopg.OperationalError("서버가 연결을 끊었다"),
        KeyboardInterrupt(),
        SystemExit("의존이 sys.exit() 를 불렀다"),
    ],
    ids=["인프라 예외", "Ctrl-C", "SystemExit"],
)
def test_측정2_의_예상_밖_예외도_리포트를_남기고_측정3_을_잇지_않는다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    """**가장 비싼 측정**이 죽어도 산출물은 남는다 — 그리고 판정을 더 사지 않는다.

    `measure_pipeline_agreement` 는 인프라 예외를 그대로 터뜨리는 것이 설계다(지표가 아니라
    환경 고장이다). 30건 중 25건째에 DB 컨테이너가 재기동하면 그 예외가 `main` 밖으로 나가
    **이미 과금된 완주분과 무료인 측정 1 산출물까지** 트레이스백과 함께 사라진다 — 측정 3 에서
    고친 것과 정확히 같은 실패 모양이라 같은 패턴으로 막는다.

    이어서 측정 3 을 돌리지도 않는다: 중단 사유가 Ctrl-C 면 이어 도는 것이 곧 추가 과금이다.
    """
    judge_builds: list[object] = []

    def _boom(*, cases: Any, args: Any, settings: Any) -> Any:
        del cases, args, settings
        raise error

    monkeypatch.setattr(evaluate, "get_settings", lambda: _l2_settings())
    monkeypatch.setattr(evaluate, "database_unavailable_reason", lambda *, settings: None)
    monkeypatch.setattr(evaluate, "_run_measurement_two", _boom)
    monkeypatch.setattr(evaluate, "build_judge", lambda settings: judge_builds.append(settings))
    _block_outbound_sockets(monkeypatch)

    # 산출물을 남기고 실패를 종료 코드로 알린다 — 조용한 성공이 아니다.
    assert evaluate.main(["--live", "--out-dir", str(tmp_path)]) == 1

    payload = json.loads((tmp_path / "evaluation-live-l2-1.json").read_text(encoding="utf-8"))
    agreement = payload["measurement_2_pipeline_agreement"]
    assert agreement["executed"] is False
    # 사유에 예외 종류와 메시지가 그대로 남는다 — 실패가 조용한 성공으로 보이면 안 된다.
    assert type(error).__name__ in agreement["skip_reason"]
    assert str(error) in agreement["skip_reason"]
    # 판정 모델을 조립조차 하지 않는다 — 중단된 실행이 판정 비용을 더 쓰면 안 된다.
    assert not judge_builds
    judged = payload["measurement_3_l2_judge_accuracy"]
    assert judged["executed"] is False
    assert "측정 2" in judged["skip_reason"]
    # 무료인 측정 1 은 비싼 측정의 사고에 인질로 잡히지 않는다.
    assert payload["measurement_1_l1_gate_accuracy"]["detection_rate"] == 1.0
    # 실행 조건에 **무엇으로 돌리던 중이었는지**가 남는다 — 어느 모델에 돈을 썼는지가
    # 산출물에서 사라지면 중단 리포트를 읽을 수 없다.
    assert "OpenAI" in payload["conditions"]["generation"]
    assert "측정 2 중단" in payload["conditions"]["generation"]
    assert "측정 2 중단" in payload["conditions"]["judge"]


def test_비과금_대역_실행의_Ctrl_C_도_종료_코드가_0_이_아니다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """중단은 사용자가 알아야 하는 사실이다 — 과금 실행이 아니어도 성공으로 읽히면 안 된다.

    중단 강등(`except BaseException`)이 산출물을 남기려고 예외를 삼키므로, 종료 코드까지
    0 이면 `--stub-llm` 실행의 Ctrl-C 가 아무 흔적 없는 성공이 된다. 산출물은 그대로
    남기되(과금분 보존) 중단 사실은 종료 코드가 들고 간다.
    """
    label = "--stub-llm"
    argv = ["--stub-llm"]

    def _interrupt(*, cases: Any, args: Any, settings: Any) -> Any:
        del cases, args, settings
        raise KeyboardInterrupt

    monkeypatch.setattr(evaluate, "get_settings", lambda: _l2_settings())
    monkeypatch.setattr(evaluate, "database_unavailable_reason", lambda *, settings: None)
    monkeypatch.setattr(evaluate, "_run_measurement_two", _interrupt)
    _block_outbound_sockets(monkeypatch)

    assert evaluate.main([*argv, "--out-dir", str(tmp_path)]) != 0, label

    # 산출물은 남는다 — 실패를 트레이스백이 아니라 리포트와 종료 코드로 알린다.
    payload = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["measurement_1_l1_gate_accuracy"]["detection_rate"] == 1.0
    assert "KeyboardInterrupt" in payload["measurement_2_pipeline_agreement"]["skip_reason"]
    # 비과금이므로 라이브 계열 이름은 받지 못한다.
    assert not list(tmp_path.glob("evaluation-live*")), label


def test_과금_실행에서_판정_픽스처_로드_실패는_종료_코드_1_이다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """골든셋 30건을 사고 측정 3 은 못 낸 실행이 `exit 0` 으로 성공처럼 읽히면 안 된다.

    구조가 같은 중단 강등은 이미 1 을 세우는데 로드 실패만 경고로 끝나면 종료 코드 규칙이
    갈린다 — 산출물은 저장소가 추적하는 `evaluation-live-l2-1` 이고 종료 코드는 0 이라,
    래퍼·CI 는 "L2 실측이 성공했다"로 읽는다.
    """
    monkeypatch.setattr(evaluate, "get_settings", lambda: _l2_settings())
    monkeypatch.setattr(evaluate, "database_unavailable_reason", lambda *, settings: None)
    _canned_measurement_two(monkeypatch)
    _block_outbound_sockets(monkeypatch)
    missing = tmp_path / "오타.jsonl"

    assert (
        evaluate.main(["--live", "--judge-fixtures", str(missing), "--out-dir", str(tmp_path)]) == 1
    )

    payload = json.loads((tmp_path / "evaluation-live-l2-1.json").read_text(encoding="utf-8"))
    # 과금이 끝난 측정 2 는 그대로 남는다 — 실패는 종료 코드와 미실행 사유가 든다.
    assert payload["measurement_2_pipeline_agreement"]["executed"] is True
    judged = payload["measurement_3_l2_judge_accuracy"]
    assert judged["executed"] is False
    assert "FileNotFoundError" in judged["skip_reason"]
    # 세지 못한 픽스처 수를 0 으로 채우지 않는다.
    assert payload["conditions"]["judge_fixture_count"] is None


#: 매트릭스 안에서 "tmp_path 아래의 없는 파일"을 가리키는 자리표시자.
_MISSING_FIXTURES = "<없는-판정-픽스처>"

#: `--live` 조합의 종료 코드 — **측정 3 이 돌 조건(과금 실행 + L2 켜짐)이었는데 미실행이면 1**.
#: L2 꺼짐 기준선은 측정 3 이 설계상 안 도는 것이므로 과금 실행이어도 0 이다.
#: (설명, argv, 키, 측정 2, 측정 3, 기대 종료 코드, 리포트 stem)
_EXIT_CODE_MATRIX: tuple[tuple[str, list[str], str, str, str, int, str], ...] = (
    ("과금·측정 2·3 완주", ["--live"], "both", "canned", "oracle", 0, "evaluation-live-l2-1"),
    ("과금·측정 3 중단", ["--live"], "both", "canned", "boom", 1, "evaluation-live-l2-1"),
    (
        "과금·판정 픽스처 로드 실패",
        ["--live", "--judge-fixtures", _MISSING_FIXTURES],
        "both",
        "canned",
        "oracle",
        1,
        "evaluation-live-l2-1",
    ),
    ("과금·측정 2 중단", ["--live"], "both", "boom", "oracle", 1, "evaluation-live-l2-1"),
    # L2 꺼짐 기준선: 측정 2 는 실측(과금)이지만 측정 3 은 설계상 미실행이다 — 정상이므로 0.
    ("과금·L2 꺼짐 기준선", ["--live"], "l2-off", "canned", "oracle", 0, "evaluation-live"),
    ("비과금·생성 키 없음", ["--live"], "none", "unused", "oracle", 0, "evaluation"),
    ("비과금·판정 키 없음", ["--live"], "no-judge", "unused", "oracle", 0, "evaluation"),
    ("비과금·대역 실행", ["--stub-llm"], "both", "canned", "oracle", 0, "evaluation"),
    ("비과금·기본 실행", [], "both", "unused", "oracle", 0, "evaluation"),
)


def _apply_exit_code_scenario(
    monkeypatch: pytest.MonkeyPatch, *, keys: str, m2: str, m3: str
) -> None:
    """실 API 호출 없이 매트릭스 한 줄을 세운다 — 소켓 차단 + 대역 주입만 쓴다."""
    settings = {
        "both": _l2_settings(),
        "none": _l2_settings(live_keys=False),
        "no-judge": _l2_settings(judge_key=False),
        "l2-off": _l2_settings(l2_enabled=False),
    }[keys]
    monkeypatch.setattr(evaluate, "get_settings", lambda: settings)
    monkeypatch.setattr(evaluate, "database_unavailable_reason", lambda *, settings: None)

    if m2 == "canned":
        _canned_measurement_two(monkeypatch)
    elif m2 == "boom":

        def _m2_boom(*, cases: Any, args: Any, settings: Any) -> Any:
            del cases, args, settings
            raise psycopg.OperationalError("DB 컨테이너가 재기동했다")

        monkeypatch.setattr(evaluate, "_run_measurement_two", _m2_boom)

    if m3 == "oracle":
        monkeypatch.setattr(evaluate, "build_judge", lambda settings: OracleJudge(JUDGE_FIXTURES))
    elif m3 == "boom":

        def _m3_boom(settings: Any) -> Any:
            del settings
            raise SystemExit("판정 SDK 가 죽었다")

        monkeypatch.setattr(evaluate, "build_judge", _m3_boom)

    _block_outbound_sockets(monkeypatch)


@pytest.mark.parametrize(
    ("label", "argv", "keys", "m2", "m3", "expected_code", "stem"),
    _EXIT_CODE_MATRIX,
    ids=[row[0] for row in _EXIT_CODE_MATRIX],
)
def test_종료_코드는_측정3_이_돌_조건이었는데_미실행일_때만_1_이다(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    argv: list[str],
    keys: str,
    m2: str,
    m3: str,
    expected_code: int,
    stem: str,
) -> None:
    """종료 코드 규칙이 갈리지 않는지 `--live` 조합 전체로 못박는다 (실 API 호출 0회).

    규칙: **측정 3 이 돌 조건(= `--live` 로 선검사를 통과해 측정 2 가 실측으로 돌 조건이었고
    L2 도 켜져 있던 실행)이었는데 미실행이면 1, 그 밖에는 0.** `--live` 없이 도는 평범한
    실행에서 측정 3 이 "`--live` 아님" 사유로 미실행인 것도, **L2 꺼짐 기준선에서 측정 3 이
    설계상 안 도는 것**도 정상이므로 0 이다.

    어느 조합이든 **리포트는 반드시 쓰인다** — 실패를 트레이스백이 아니라 산출물과 종료
    코드로 알리는 것이 이 하네스의 계약이다.
    """
    _apply_exit_code_scenario(monkeypatch, keys=keys, m2=m2, m3=m3)
    resolved = [
        str(tmp_path / "없는-판정-픽스처.jsonl") if item == _MISSING_FIXTURES else item
        for item in argv
    ]

    assert evaluate.main([*resolved, "--out-dir", str(tmp_path)]) == expected_code, label

    # 산출물은 어느 조합에서도 남는다.
    assert (tmp_path / f"{stem}.md").exists(), label
    payload = json.loads((tmp_path / f"{stem}.json").read_text(encoding="utf-8"))
    assert payload["measurement_1_l1_gate_accuracy"]["llm_calls"] == 0, label
    # 규칙 자체를 산출물에 대고 다시 확인한다 — 종료 코드와 리포트가 갈리면 안 된다.
    billed = payload["conditions"]["measurement2_is_real"]
    l2_on = payload["conditions"]["l2_enabled"]
    judged = payload["measurement_3_l2_judge_accuracy"]["executed"]
    assert (expected_code == 1) == (billed and l2_on and not judged), label
    # 비과금 실행은 라이브 계열 이름을 받지 못한다(라이브 이름 ⇔ 실측, 양방향).
    assert bool(list(tmp_path.glob("evaluation-live*"))) is billed, label


def test_판정_픽스처를_읽지_못해도_측정1_산출물은_남는다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """측정 1 은 키도 DB 도 픽스처 파일도 없이 도는 무료 측정이다 — 인질로 잡히면 안 된다.

    로드 실패(파일 부재·라벨 불일치)는 **측정 3 의 미실행 사유**로 강등되고, 픽스처 수는
    0 이 아니라 미실행으로 적힌다.
    """
    monkeypatch.setattr(evaluate, "get_settings", lambda: _l2_settings())
    _block_outbound_sockets(monkeypatch)
    missing = tmp_path / "없는-판정-픽스처.jsonl"

    assert (
        evaluate.main(
            ["--judge-fixtures", str(missing), "--out-dir", str(tmp_path)],
        )
        == 0
    )

    payload = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["measurement_1_l1_gate_accuracy"]["detection_rate"] == 1.0
    judged = payload["measurement_3_l2_judge_accuracy"]
    assert judged["executed"] is False
    assert "FileNotFoundError" in judged["skip_reason"]
    # 세지 못한 픽스처 수를 0 으로 채우지 않는다.
    assert payload["conditions"]["judge_fixture_count"] is None
    assert "판정 픽스처: 미실행" in (tmp_path / "evaluation.md").read_text(encoding="utf-8")


def test_라벨이_깨진_판정_픽스처도_측정1_산출물을_막지_않는다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ValueError`(라벨 불일치)도 파일 부재와 같은 강등 경로를 탄다."""
    monkeypatch.setattr(evaluate, "get_settings", lambda: _l2_settings())
    _block_outbound_sockets(monkeypatch)
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{ 이건 JSON 이 아니다\n", encoding="utf-8")

    assert evaluate.main(["--judge-fixtures", str(broken), "--out-dir", str(tmp_path)]) == 0

    payload = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["measurement_1_l1_gate_accuracy"]["detection_rate"] == 1.0
    assert "ValueError" in payload["measurement_3_l2_judge_accuracy"]["skip_reason"]


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

    **L2 를 켜서** 돌린다: 결정론 판정 대역(`testing.StubJudge`)이 있으므로 외부 호출
    0회로 L2 를 포함한 배관을 잰다.
    """
    stub = StubGenerationClient()
    stub_judge = StubJudge()
    pipeline = build_stub_pipeline(
        generation_client=cast(GenerationClient, stub),
        embedding_client=LexicalEmbeddingClient(dimensions=1536),
        judge=stub_judge,
        settings=Settings(
            vector_top_k=5,
            vector_similarity_threshold=0.05,
            sql_max_rows=50,
            l2_enabled=True,
        ),
    )

    agreement = measure_pipeline_agreement(
        cases=GOLDEN, pipeline=pipeline, app_conn=app_conn, readonly_conn=ro_conn
    )

    assert agreement.total == 30
    assert [outcome.case_id for outcome in agreement.outcomes] == [case.id for case in GOLDEN]
    assert all(outcome.error is None for outcome in agreement.outcomes)
    assert agreement.latency_p50_ms is not None and agreement.latency_p95_ms is not None
    # 미끼 범주 5건에서 L1 기각이 실제로 재현되어야 재현율이 의미를 가진다(분모는 범주다).
    assert agreement.bait_total == 5
    assert agreement.bait_reject_reproduced == 5
    assert agreement.bait_reject_recall == 1.0
    # 근거 유래 PII 에코는 통과해야 한다 (오탐 0).
    assert agreement.forbidden_violations == 0
    # 초안 전 인계 경로도 분모에 들어간다.
    assert agreement.escalation_counts.get("missing_order_ref") == 2
    assert agreement.escalation_counts.get("order_not_found") == 2
    # L2 가 실제로 돌았다 — 판정 토큰이 생성 토큰과 **분리되어** 집계된다.
    assert stub_judge.calls
    assert agreement.judge_input_tokens_total > 0
    assert agreement.judge_output_tokens_total > 0
    assert agreement.judge_tokens_per_inquiry is not None

    report = _report(pipeline=agreement)
    markdown_path, json_path = write_report(report, out_dir=tmp_path, stem="e2e")
    markdown = markdown_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "## 측정 1 — L1 게이트 단위 정확도 (결정론)" in markdown
    assert "## 측정 2 — 파이프라인 판정 일치율 (end-to-end)" in markdown
    assert "미끼 문의(reject_bait)의 기각 재현율" in markdown
    assert payload["measurement_2_pipeline_agreement"]["executed"] is True
    assert len(payload["measurement_2_pipeline_agreement"]["outcomes"]) == 30
    tokens = payload["measurement_2_pipeline_agreement"]["tokens"]
    assert tokens["judge_input_total"] > 0


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_대역_LLM_은_미끼_조항에서_기각을_재현한다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """미끼 조항 문의 → 근거에 없는 번호 생성 → `pii_detected` 기각 → 재생성 통과.

    **L2 를 켠 채** 확인한다: L1 기각 시도에서는 L2 가 호출되지 않고(층 순서), 재생성
    초안에서만 판정이 돈다 — 그 두 사실이 한 번에 고정된다.
    """
    stub = StubGenerationClient()
    stub_judge = StubJudge()
    pipeline = build_stub_pipeline(
        generation_client=cast(GenerationClient, stub),
        embedding_client=LexicalEmbeddingClient(dimensions=1536),
        judge=stub_judge,
        settings=Settings(
            vector_top_k=5,
            vector_similarity_threshold=0.05,
            sql_max_rows=50,
            l2_enabled=True,
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
    # L1 기각 시도는 L2 를 부르지 않는다 — 판정은 재생성 초안 1회뿐이다.
    assert len(stub_judge.calls) == 1
    assert outcome.judge_input_tokens > 0


@pytest.mark.db
@pytest.mark.usefixtures("seeded_order_count")
def test_stub_llm_진입점이_실제_DB_에서_판정_대역까지_배선한다(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--stub-llm` 의 **판정 대역 배선을 진입점 경로로** 밟는 유일한 테스트다.

    이게 없으면 `evaluate.py` 의 `build_stub_pipeline` 을 `build_pipeline` 으로 바꿔도 전체가
    녹색이고, 실제 `--stub-llm` 실행만 진짜 판정 모델을 부른다(= 과금). 여기서 못박는 것은
    셋이다: 아웃바운드 소켓을 막은 채 완주하는가, 측정 2 가 실행되고 **판정 토큰이 실제로
    쌓이는가**(판정 층이 배선됐다는 증거), 측정 3 은 `--live` 사유로 미실행인가.
    """
    settings = get_settings().model_copy(update={"l2_enabled": True})
    monkeypatch.setattr(evaluate, "get_settings", lambda: settings)
    _block_outbound_sockets(monkeypatch)

    assert evaluate.main(["--stub-llm", "--out-dir", str(tmp_path)]) == 0

    payload = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    agreement = payload["measurement_2_pipeline_agreement"]
    assert agreement["executed"] is True
    assert agreement["total"] == len(GOLDEN)
    tokens = agreement["tokens"]
    assert tokens["judge_input_total"] + tokens["judge_output_total"] > 0
    judged = payload["measurement_3_l2_judge_accuracy"]
    assert judged["executed"] is False
    assert "--live" in judged["skip_reason"]
    # 대역 실행은 실측이 아니다 — L2 를 켰어도 라이브 이름 계열은 만들어지지 않는다.
    assert payload["conditions"]["l2_enabled"] is True
    assert payload["conditions"]["measurement2_is_real"] is False
    assert not list(tmp_path.glob("evaluation-live*"))


# ── 케이스 귀인 개선 3건 (docs/tracking/findings.md 20번) ─────────────────────


def test_귀인이_정책_검색_실행_여부를_명시_필드로_적는다(tmp_path: Path) -> None:
    """의도가 `order` 로 라우팅해 **검색이 아예 안 돈** 케이스와 검색이 걸러진 케이스를
    가른다. 지금까지는 검색 토큰 0 + `sql:` 단독 채택으로 역추론해야 했다."""
    sql_row = Evidence(
        id="sql:11111111-1111-4111-8111-111111111111:1",
        source=EvidenceSource.SQL,
        content="SELECT ...",
        evidence_text="배송중",
    )
    support = Evidence(
        id="policy:support:4-1",
        source=EvidenceSource.POLICY,
        content="고객센터 운영 안내",
        evidence_text="고객센터 운영 안내",
    )
    agreement = measure_pipeline_agreement(
        cases=[_case("G09"), _case("G19"), _case("G17")],
        pipeline=ScriptedPipeline(
            [
                # G09: 의도 분류가 order 로 보내 정책 검색이 실행되지 않았다.
                _processed(evidence=(sql_row,), intent=IntentSource.ORDER),
                # G19: 검색은 돌았는데 정답 조항이 걸러졌다.
                _processed(evidence=(support,), intent=IntentSource.BOTH),
                # G17: 의도 해석 자체가 무너졌다 — 실행 여부를 알 수 없다.
                _processed(
                    status=InquiryStatus.ESCALATED,
                    escalation=EscalationReason.LLM_CALL_FAILED,
                    intent=None,
                ),
            ]
        ),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert agreement.outcomes[0].intent == "order"
    payload = report_to_json(_report(pipeline=agreement))
    by_id = {item["case_id"]: item for item in payload["failure_attribution"]["cases"]}
    assert by_id["G09"]["policy_retrieval_ran"] is False
    assert "order" in (by_id["G09"]["policy_retrieval_note"] or "")
    assert by_id["G19"]["policy_retrieval_ran"] is True
    # 미상은 False 가 아니다 — 0 으로 채우지 않는다는 규칙이 케이스 단위에도 적용된다.
    assert by_id["G17"]["policy_retrieval_ran"] is None

    markdown = render_markdown(_report(pipeline=agreement))
    assert "정책 검색 미실행" in markdown


def test_주문_단계_사전_인계는_빈_정답의_기대_경로다(tmp_path: Path) -> None:
    """G28 의 `order_not_found` 종결은 "구조적 사유가 이긴다" 계약상 정상이다
    (docs/business-rules.md). 기대 경로 목록에 없으면 매번 비정상으로 찍힌다."""
    agreement = measure_pipeline_agreement(
        cases=[
            _case(
                "G28",
                statuses=(InquiryStatus.ESCALATED,),
                reasons=(EscalationReason.ORDER_NOT_FOUND,),
                category="no_evidence",
            )
        ],
        pipeline=ScriptedPipeline(
            [
                _processed(
                    status=InquiryStatus.ESCALATED,
                    escalation=EscalationReason.ORDER_NOT_FOUND,
                    intent=IntentSource.BOTH,
                )
            ]
        ),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    payload = report_to_json(_report(pipeline=agreement))
    breakdown = payload["failure_attribution"]
    case = breakdown["cases"][0]
    assert case["classification"] == "expected_no_answer"
    assert case["normal_behavior"] is True
    assert case["normal_behavior_path"] == "order_stage_pre_handoff"
    assert case["anomaly_reason"] is None
    assert breakdown["expected_no_answer_anomaly_count"] == 0
    assert "주문 단계 사전 인계" in render_markdown(_report(pipeline=agreement))


def test_generation_issue_는_최종_상태와_matched_를_병기한다(tmp_path: Path) -> None:
    """1차 기각 후 재생성으로 회복한 케이스가 종결 실패처럼 읽히면 안 된다."""
    support = Evidence(
        id="policy:support:4-1",
        source=EvidenceSource.POLICY,
        content="고객센터 운영 안내",
        evidence_text="고객센터 운영 안내",
    )
    channel = Evidence(
        id="policy:support:4-2",
        source=EvidenceSource.POLICY,
        content="문의 접수 채널",
        evidence_text="문의 접수 채널",
    )
    recovered = _processed(
        status=InquiryStatus.ANSWERED,
        attempts=(
            AttemptRecord(
                attempt_no=1,
                verdict=Verdict.REJECT,
                reject_reasons=(RejectReason.UNSUPPORTED_CLAIM,),
                draft={},
            ),
            AttemptRecord(attempt_no=2, verdict=Verdict.PASS, reject_reasons=(), draft={}),
        ),
        evidence=(support, channel),
    )
    agreement = measure_pipeline_agreement(
        cases=[_case("G20")],
        pipeline=ScriptedPipeline([recovered]),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    payload = report_to_json(_report(pipeline=agreement))
    case = payload["failure_attribution"]["cases"][0]
    assert case["classification"] == "generation_issue"
    assert case["final_status"] == "answered"
    assert case["matched"] is True
    markdown = render_markdown(_report(pipeline=agreement))
    assert "최종 answered" in markdown and "라벨 일치" in markdown


# ── 리포트 이름 규칙 — 자격 판정 근거는 "과금 실행 여부" ──────────────────────


def test_측정_2_를_건너뛴_과금_실행도_라이브_이름을_받는다(tmp_path: Path) -> None:
    """측정 3 단독 실측이 `evaluation` 스템으로 떨어지면 gitignore 에 걸려 다음 실행에
    덮인다 — 과금된 산출물이 그렇게 사라진다. 이름 계열은 늘리지 않는다."""
    assert _stem(tmp_path, real=True, l2=True) == "evaluation-live-l2-1"
    assert (
        _stem(tmp_path, requested="evaluation-live-l2-7", real=True, l2=True)
        == "evaluation-live-l2-7"
    )


def test_과금이_아닌_실행은_여전히_라이브_이름을_못_쓴다(tmp_path: Path) -> None:
    """양방향 불변식은 그대로다 — 근거만 `measurement2_is_real` 에서 과금 여부로 바뀐다."""
    with pytest.raises(ReportStemError):
        _stem(tmp_path, requested="evaluation-live-l2-1", real=False, l2=True)


# ── 합산 일치율은 하한 경보선이다 (결정 0006 재확정) ─────────────────────────


def test_합산_일치율_목표는_하한_경보선이고_판정은_케이스가_한다() -> None:
    by_key = {target.key: target for target in TARGETS}
    assert by_key["match_rate"].alert_only is True
    assert by_key["detection_rate"].alert_only is False
    assert by_key["false_positive_rate"].alert_only is False


def test_하한_경보선_미달은_경보로_적힌다() -> None:
    report = _real_agreement_report(matched=False)
    verdicts = {item.target.key: item.verdict for item in assess_targets(report)}
    assert verdicts["match_rate"] == "경보"
    markdown = render_markdown(report)
    assert "하한 경보선" in markdown
    assert "케이스 단위" in markdown


# ── 회귀 가드가 리포트에 실린다 ──────────────────────────────────────────────


def test_리포트가_회귀_가드_두_줄을_싣는다(tmp_path: Path) -> None:
    report = _real_agreement_report(matched=True)
    markdown = render_markdown(report)
    assert "## 회귀 가드 — 이중 기준선 두 줄 보고" in markdown
    payload = report_to_json(report)
    # 넘기지 않은 가드는 "미산출 + 사유"다 — 통과로 대체하지 않는다.
    assert payload["regression_guard"]["computed"] is False
    assert payload["regression_guard"]["reason"]


def test_실행_조건에_조건_지문과_선언된_실험_변인이_실린다() -> None:
    conditions = _conditions(
        is_real=True,
        fingerprint={"abstention_tau": "0.42", "abstention_gate_statistic": "mean_top_k"},
        declared=("abstention_tau", "abstention_gate_statistic"),
    )
    report = _report(pipeline=SkippedMeasurement(reason="미요청"), conditions=conditions)
    payload = report_to_json(report)
    assert payload["conditions"]["condition_fingerprint"]["abstention_tau"] == "0.42"
    assert payload["conditions"]["declared_experiment_fields"] == [
        "abstention_tau",
        "abstention_gate_statistic",
    ]
    assert payload["conditions"]["billed"] is True
    assert payload["conditions"]["measurement_scope"]
    markdown = render_markdown(report)
    assert "조건 지문" in markdown
    assert "abstention_tau" in markdown
    assert "선언된 실험 변인" in markdown


def test_attach_regression_guard_가_리포트에_두_줄을_붙인다(tmp_path: Path) -> None:
    """실행 진입점이 쓰는 이음매 — 승격 참조가 없으면 "기준선 미등재"가 산출물에 남는다."""
    report = _real_agreement_report(matched=True)
    attached = attach_regression_guard(
        report,
        stem="evaluation-live-l2-9",
        reports_dir=tmp_path,
        promoted_reference_path=tmp_path / "없는-참조.json",
    )
    payload = report_to_json(attached)["regression_guard"]
    assert payload["computed"] is True
    assert payload["verdict"] == "기준선 미등재"
    assert payload["promotion"]["registered"] is False
    assert payload["binding"]["role"] == "구속"
    assert payload["alert"]["role"] == "경보"
    assert "기준선 미등재" in render_markdown(attached)
