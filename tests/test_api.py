"""API 표면 테스트 — 엔드포인트 4개와 응답 스키마.

응답 스키마의 계약은 "**모든 키가 항상 존재한다**" 이다: `answer`·`escalation_reason` 은
해당 없을 때 null, `claims`·`citations`·`attempts` 는 해당 없을 때 빈 배열. 초안 전
인계라도 그때까지 모은 근거는 `citations` 에 남는다.

`attempts[]` 는 **종합 판정(기존 키)과 층별 판정(`l1`/`l2`)을 함께** 싣는다. `l2` 는
미실행이면 `null` 이고 키는 사라지지 않는다 — 미실행 3종(L1 reject · 스위치 꺼짐 ·
L2 호출 실패)을 각각 확인한다. 특히 **L2 호출 실패 시도**는 층 결합 정의상 종합 verdict 가
`pass` 인데 문의는 인계되므로(docs/contracts.md "층별 판정 키"), 응답이 "판정이 없었다"를
`l2: null` 로 드러내지
못하면 화면이 "통과했는데 왜 인계?"로 읽힌다.

`metrics.tokens` 의 `input`/`output` 은 **생성 LLM 합산**으로 의미가 불변이고, 판정 토큰은
`judge_input`/`judge_output` 으로 분리된 키다(L2 미실행이면 0).

접수 검증(빈 내용·주문번호 형식)은 **파이프라인을 돌리지 않고** 422 로 끝난다 — 인계가
아니다. 그 증거로 대역 서비스의 호출 기록이 비어 있는지 확인한다.

자격 증명 부재는 **503 설정 오류**다(인계가 아니다). 판정 키(`ANTHROPIC_API_KEY`)는
L2 가 켜져 있을 때 **POST 진입 시 선검사**한다 — 이 선검사가 조회·422 경로까지 번지면
키 없는 환경에서 조회가 죽으므로, 그 경계를 여기서 못박는다.

DB 가 필요한 테스트는 `db` 마커가 붙고, 쓰기는 전부 `app_conn` 트랜잭션 안에서만 일어나
픽스처 롤백으로 되돌아간다.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any, cast

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import DictRow

from reply_gate.api import (
    InquiryResponse,
    InquiryService,
    ServiceOpener,
    app,
    build_embedding_client,
    build_generation_client,
    get_service,
)
from reply_gate.config import Settings, get_settings
from reply_gate.contracts import (
    ClaimJudgment,
    Draft,
    EscalationReason,
    Evidence,
    EvidenceContradiction,
    InquiryStatus,
    JudgeResult,
    RejectReason,
    Verdict,
)
from reply_gate.draft import DRAFT_STAGE, DraftGenerator
from reply_gate.evidence import INTENT_STAGE, EvidenceCollector
from reply_gate.judge import JUDGE_STAGE, JudgeOutcome
from reply_gate.llm import GenerationClient, LLMCallError
from reply_gate.pipeline import (
    InquiryPipeline,
    Judging,
    ProcessedInquiry,
    build_pipeline,
    new_inquiry_id,
)
from reply_gate.records import save_inquiry
from reply_gate.testing import LexicalEmbeddingClient
from tests.test_pipeline import (
    ScriptedGenerationClient,
    ScriptedJudge,
    citing_draft,
    draft_completion,
    intent_completion,
    live_pipeline,
    scripted_client,
)
from tests.test_records import _processed

INQUIRY = "환불은 언제까지 신청할 수 있나요?"

RESPONSE_KEYS = {
    "inquiry_id",
    "status",
    "answer",
    "claims",
    "citations",
    "attempts",
    "escalation_reason",
    "metrics",
}

#: 시도 1건의 키 — 종합(기존 키) + 층별. 미실행 층은 `null` 이고 키는 사라지지 않는다.
ATTEMPT_KEYS = {"verdict", "reject_reasons", "l1", "l2"}
L1_KEYS = {"verdict", "reject_reasons"}
L2_KEYS = {"verdict", "reject_reasons", "claim_judgments", "contradictions"}
#: 생성 합산(기존 키·기존 의미) + 판정·검색 계열 분리 키.
TOKEN_KEYS = {
    "input",
    "output",
    "judge_input",
    "judge_output",
    "retrieval_input",
    "retrieval_output",
}
#: `metrics` 의 키 — 폴백 사유는 인계 사유가 아니라 검색 층의 관측이므로 여기 산다.
METRICS_KEYS = {"latency_ms", "tokens", "retrieval_fallback_reason"}


def layer(verdict: Verdict, reasons: Sequence[RejectReason] = ()) -> dict[str, Any]:
    """층별 판정 1건의 기대 형태 (L1, 그리고 L2 의 상세 없는 부분)."""
    return {"verdict": verdict.value, "reject_reasons": [reason.value for reason in reasons]}


def attempt(
    verdict: Verdict,
    reasons: Sequence[RejectReason] = (),
    *,
    l1: dict[str, Any] | None,
    l2: dict[str, Any] | None,
) -> dict[str, Any]:
    """시도 1건의 기대 형태. `l1`/`l2` 는 **명시 인자**다 — 기본값을 두면 미실행 null 을
    빠뜨린 기대값이 조용히 통과한다."""
    return {
        "verdict": verdict.value,
        "reject_reasons": [reason.value for reason in reasons],
        "l1": l1,
        "l2": l2,
    }


# ── 대역 서비스 (DB 불필요) ─────────────────────────────────────────────────


class RecordingService:
    """`InquiryService` 대역 — 호출만 기록한다. 접수 거부가 파이프라인을 막는지 본다."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def process(self, *, content: str, order_no: str | None) -> ProcessedInquiry:
        self.calls.append({"content": content, "order_no": order_no})
        raise AssertionError("접수 거부 케이스에서는 파이프라인이 돌면 안 된다")

    def fetch(self, inquiry_id: str) -> ProcessedInquiry | None:
        self.calls.append({"fetch": inquiry_id})
        return None


@pytest.fixture
def recorder() -> Iterator[RecordingService]:
    service = RecordingService()

    def _override() -> ServiceOpener:
        return lambda: nullcontext(cast(InquiryService, service))

    app.dependency_overrides[get_service] = _override
    try:
        yield service
    finally:
        app.dependency_overrides.pop(get_service, None)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def api_settings(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> Settings:
    """`api.get_settings` 를 고정한다 — POST 선검사가 보는 설정이다.

    선검사는 프로세스 설정(`.env`)을 읽으므로, 고정하지 않으면 로컬에 판정 키가 있느냐로
    테스트 결과가 갈린다. 다른 값(DB 접속 등)은 평소대로 `.env` 에서 온다.
    """
    settings = Settings(**overrides)
    monkeypatch.setattr("reply_gate.api.get_settings", lambda: settings)
    return settings


@pytest.fixture(autouse=True)
def default_api_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """기본값은 **L2 꺼짐** — 판정 키 경로를 보는 테스트가 각자 다시 주입한다."""
    return api_settings(monkeypatch, l2_enabled=False)


@contextmanager
def service_client(
    *,
    pipeline: InquiryPipeline,
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
) -> Iterator[TestClient]:
    """주어진 파이프라인 + 테스트 커넥션으로 앱을 돌린다 (의존성 주입 지점).

    주입한 커넥션은 **이미 열려 있으므로** 여는 계층은 `nullcontext` 로 비운다 — 테스트
    커넥션의 수명은 픽스처가 소유하고, 요청이 끝날 때 닫히면 안 된다.
    """

    def _override() -> ServiceOpener:
        return lambda: nullcontext(
            InquiryService(pipeline=pipeline, app_conn=app_conn, readonly_conn=ro_conn)
        )

    app.dependency_overrides[get_service] = _override
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_service, None)


@contextmanager
def live_client(
    *,
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    generation: ScriptedGenerationClient,
    threshold: float = 0.0,
) -> Iterator[TestClient]:
    """실제 파이프라인 + 테스트 커넥션으로 앱을 돌린다 (LLM 만 목). **L2 는 꺼짐.**"""
    with service_client(
        pipeline=live_pipeline(generation, threshold=threshold),
        app_conn=app_conn,
        ro_conn=ro_conn,
    ) as test_client:
        yield test_client


def l2_live_pipeline(
    generation: ScriptedGenerationClient, judge: Any, *, threshold: float = 0.0
) -> InquiryPipeline:
    """실제 근거 수집기 + 판정 대역으로 **L2 를 켠** 조립.

    `tests.test_pipeline.live_pipeline` 은 L2 를 꺼 둔다(인계 사유 6종·응답 골격은 판정 층과
    무관하므로). 응답의 층 구분과 L2 상세는 판정이 실제로 돌아야 나오므로, 여기서는
    `build_pipeline` 대신 협력자를 직접 조립해 판정 대역을 끼운다 — `build_pipeline` 은
    실제 Anthropic 판정자를 배선하기 때문이다.
    """
    settings = Settings(
        vector_top_k=5,
        vector_similarity_threshold=threshold,
        sql_max_rows=50,
        l2_enabled=True,
    )
    client = cast(GenerationClient, generation)
    return InquiryPipeline(
        collector=EvidenceCollector(
            generation_client=client,
            embedding_client=LexicalEmbeddingClient(dimensions=1536),
            settings=settings,
            rewrite_client=client,
        ),
        drafter=DraftGenerator(client=client, effort=settings.generation_effort),
        judge=cast(Judging, judge),
        l2_enabled=True,
    )


@contextmanager
def l2_client(
    *,
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    generation: ScriptedGenerationClient,
    judge: Any,
) -> Iterator[TestClient]:
    """L2 켜짐 + 판정 대역으로 앱을 돌린다.

    `api.get_settings`(= POST 판정 키 선검사)는 autouse 픽스처가 L2 꺼짐으로 고정해 둔
    그대로다. 선검사와 파이프라인 스위치는 서로 다른 설정 객체를 보므로 독립이고, 여기서
    보려는 것은 선검사가 아니라 **응답에 실리는 층 구분**이다.
    """
    with service_client(
        pipeline=l2_live_pipeline(generation, judge), app_conn=app_conn, ro_conn=ro_conn
    ) as test_client:
        yield test_client


class ContradictionJudge:
    """`Judging` 대역 — 수집 근거의 **실제 ID** 로 모순쌍을 만들어 기각한다.

    `tests.test_pipeline.ScriptedJudge` 는 미리 만든 결과를 돌려주므로 근거 ID 를 알 수
    없다. 응답·화면에 실리는 모순쌍이 실제 근거 ID 인지 보려면 판정 시점에 만들어야 한다.
    """

    def __init__(self, *, input_tokens: int = 140, output_tokens: int = 37) -> None:
        self.calls: list[dict[str, Any]] = []
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    def judge(self, *, draft: Draft, evidence: Sequence[Evidence]) -> JudgeOutcome:
        self.calls.append({"draft": draft, "evidence": tuple(evidence)})
        assert len(evidence) >= 2, "모순쌍을 만들려면 근거가 2건 이상이어야 한다"
        return JudgeOutcome(
            result=JudgeResult(
                verdict=Verdict.REJECT,
                reject_reasons=(
                    RejectReason.UNSUPPORTED_CLAIM,
                    RejectReason.CONTRADICTORY_EVIDENCE,
                ),
                claim_judgments=tuple(
                    ClaimJudgment(
                        claim_text=claim.text,
                        verdict=Verdict.REJECT,
                        explanation="인용한 조항이 이 문장의 주제를 다루지 않는다.",
                    )
                    for claim in draft.claims
                ),
                contradictions=(
                    EvidenceContradiction(
                        evidence_id_a=evidence[0].id,
                        evidence_id_b=evidence[1].id,
                        explanation="두 조항이 같은 사안에 다른 기간을 말한다.",
                    ),
                ),
            ),
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            attempts=1,
        )


# ── 헬스 체크 / 웹 폼 ───────────────────────────────────────────────────────


def test_health_는_200_이다(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_웹_폼은_한_장짜리_HTML_이다(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_웹_폼에_기각_사유가_보이는_자리가_있다(client: TestClient) -> None:
    """데모 요구사항 — 시도별 판정과 기각 사유가 화면에서 바로 보여야 한다."""
    html = client.get("/").text

    assert 'id="attempts"' in html  # 시도별 pass/reject 가 실리는 자리
    assert 'id="escalation"' in html  # 인계 사유가 실리는 자리
    assert 'id="citations"' in html  # 근거 목록이 실리는 자리
    assert "게이트 판정" in html
    assert "기각" in html
    assert "인계 사유" in html


def test_웹_폼의_판정_제목은_층_중립이다(client: TestClient) -> None:
    """목록이 L1 사유와 L2 사유를 함께 찍으므로 "L1 게이트 판정"은 틀린 제목이다."""
    html = client.get("/").text

    assert "L1 게이트 판정 — 시도 이력" not in html
    assert "게이트 판정 — 시도별 층별 이력" in html


def test_웹_폼은_층_구분과_L2_상세를_표시한다(client: TestClient) -> None:
    """층 배지·claim 단위 판정·근거쌍 모순이 화면에 있어야 한다.

    docs/contracts.md "층별 판정 키"·"토큰 집계 경계".
    """
    html = client.get("/").text

    assert "L2 미실행" in html  # 층 배지 — 미실행은 통과가 아니다
    assert "claim 단위 판정" in html  # 데모의 "왜 기각됐는지" 장면
    assert "근거쌍 모순" in html
    assert "판정 토큰" in html  # 생성 합산과 분리된 계열
    assert "tokens.judge_input" in html
    assert "tokens.judge_output" in html


def test_웹_폼은_L2_호출_실패_시도를_통과로_읽히게_두지_않는다(client: TestClient) -> None:
    """종합 verdict 는 pass 인데 문의는 인계되는 경우(docs/contracts.md
    "층별 판정 키") — 값은 그대로 두고
    "판정이 없었다"를 화면이 따로 말해야 "통과했는데 왜 인계?"로 읽히지 않는다."""
    html = client.get("/").text

    assert "판정 호출이 재시도 후에도 실패했다" in html
    assert "unverified" in html  # 종합 pass 지만 미검증인 시도를 따로 칠하는 자리


def test_웹_폼은_외부_리소스를_불러오지_않는다(client: TestClient) -> None:
    """오프라인에서 열려야 한다 — CDN·폰트·외부 스크립트 금지."""
    html = client.get("/").text

    assert "http://" not in html
    assert "https://" not in html


# ── 접수 검증 (파이프라인 미실행) ───────────────────────────────────────────


def test_content_가_없으면_422(client: TestClient, recorder: RecordingService) -> None:
    response = client.post("/inquiries", json={"order_no": "ORD-20260315-0001"})

    assert response.status_code == 422
    assert recorder.calls == []


def test_content_가_빈_문자열이면_422(client: TestClient, recorder: RecordingService) -> None:
    response = client.post("/inquiries", json={"content": "   "})

    assert response.status_code == 422
    assert recorder.calls == []


def test_주문번호_형식이_틀리면_422_이고_파이프라인이_돌지_않는다(
    client: TestClient, recorder: RecordingService
) -> None:
    response = client.post("/inquiries", json={"content": INQUIRY, "order_no": "12345"})

    assert response.status_code == 422
    assert recorder.calls == []


def test_없는_문의는_404(client: TestClient, recorder: RecordingService) -> None:
    response = client.get("/inquiries/3b0a5a1e-0000-4000-8000-000000000000")

    assert response.status_code == 404


# ── 응답 스키마 (실제 파이프라인 + DB) ──────────────────────────────────────


@pytest.fixture
def indexed_policies(app_conn: psycopg.Connection[DictRow]) -> None:
    from reply_gate.policy_index import index_policy_documents, load_policy_documents
    from reply_gate.testing import LexicalEmbeddingClient

    index_policy_documents(
        conn=app_conn,
        documents=load_policy_documents(),
        embedder=LexicalEmbeddingClient(dimensions=1536),
    )


def _assert_skeleton(payload: dict[str, Any]) -> None:
    """모든 키가 항상 존재하고 타입이 계약대로인지 — 층별 키와 판정 토큰 키 포함."""
    assert set(payload) == RESPONSE_KEYS
    assert isinstance(payload["inquiry_id"], str)
    assert payload["status"] in {InquiryStatus.ANSWERED.value, InquiryStatus.ESCALATED.value}
    assert isinstance(payload["claims"], list)
    assert isinstance(payload["citations"], list)
    assert isinstance(payload["attempts"], list)
    for item in payload["attempts"]:
        # 층별 키는 **미실행이어도 사라지지 않는다** — 소비자는 키 존재가 아니라 값으로 분기한다.
        assert set(item) == ATTEMPT_KEYS
        assert item["l1"] is None or set(item["l1"]) == L1_KEYS
        assert item["l2"] is None or set(item["l2"]) == L2_KEYS
    assert set(payload["metrics"]) == METRICS_KEYS
    assert set(payload["metrics"]["tokens"]) == TOKEN_KEYS
    assert isinstance(payload["metrics"]["latency_ms"], int)


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_answered_응답_스키마(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    generation = scripted_client(
        {INTENT_STAGE: [intent_completion("policy")], DRAFT_STAGE: [citing_draft()]}
    )
    with live_client(app_conn=app_conn, ro_conn=ro_conn, generation=generation) as test_client:
        payload = test_client.post("/inquiries", json={"content": INQUIRY}).json()

    _assert_skeleton(payload)
    assert payload["status"] == InquiryStatus.ANSWERED.value
    assert isinstance(payload["answer"], str) and payload["answer"]
    assert payload["escalation_reason"] is None
    assert payload["claims"] and payload["claims"][0]["citation_ids"]
    assert payload["citations"] and payload["citations"][0]["source"] == "policy"
    # 스위치 꺼짐 = L2 미실행 → `l2` 는 null 이고 판정 토큰은 0 이다.
    assert payload["attempts"] == [
        attempt(Verdict.PASS, l1=layer(Verdict.PASS), l2=None),
    ]
    assert payload["metrics"]["tokens"]["judge_input"] == 0
    assert payload["metrics"]["tokens"]["judge_output"] == 0


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_escalated_응답_스키마는_answer_가_null_이고_claims_가_빈_배열이다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    bogus = draft_completion(
        {"claims": [{"text": "환불은 30일 이내입니다.", "citation_ids": ["policy:없는문서:1"]}]}
    )
    generation = scripted_client(
        {INTENT_STAGE: [intent_completion("policy")], DRAFT_STAGE: [bogus, bogus]}
    )
    with live_client(app_conn=app_conn, ro_conn=ro_conn, generation=generation) as test_client:
        response = test_client.post("/inquiries", json={"content": INQUIRY})
        payload = response.json()

    assert response.status_code == 200
    _assert_skeleton(payload)
    assert payload["status"] == InquiryStatus.ESCALATED.value
    assert payload["answer"] is None
    assert payload["claims"] == []
    assert payload["escalation_reason"] == EscalationReason.REJECTED_TWICE.value
    # 데모의 주인공 — 기각 사유가 응답에 그대로 실린다. 종합 사유와 L1 사유가 같다.
    rejected = attempt(
        Verdict.REJECT,
        (RejectReason.INVALID_CITATION,),
        l1=layer(Verdict.REJECT, (RejectReason.INVALID_CITATION,)),
        l2=None,
    )
    assert payload["attempts"] == [rejected, rejected]


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_초안_전_인계도_수집된_근거를_citations_에_남긴다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """의도가 both 인데 주문번호가 없다 → 정책 근거를 모은 뒤 missing_order_ref 인계."""
    generation = scripted_client({INTENT_STAGE: [intent_completion("both")]})
    with live_client(app_conn=app_conn, ro_conn=ro_conn, generation=generation) as test_client:
        payload = test_client.post("/inquiries", json={"content": INQUIRY}).json()

    _assert_skeleton(payload)
    assert payload["escalation_reason"] == EscalationReason.MISSING_ORDER_REF.value
    assert payload["attempts"] == []
    assert payload["claims"] == []
    assert payload["citations"] != []


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_POST_결과는_GET_으로_같은_골격이_다시_나온다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    generation = scripted_client(
        {INTENT_STAGE: [intent_completion("policy")], DRAFT_STAGE: [citing_draft()]}
    )
    with live_client(app_conn=app_conn, ro_conn=ro_conn, generation=generation) as test_client:
        created = test_client.post("/inquiries", json={"content": INQUIRY}).json()
        fetched = test_client.get(f"/inquiries/{created['inquiry_id']}").json()

    assert fetched == created


@pytest.mark.db
def test_GET_은_메모리가_아니라_저장된_기록에서_재구성한다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """이 앱 인스턴스가 처리한 적 없는 문의를 DB 에 직접 넣고 조회한다."""
    stored = _processed(new_inquiry_id())
    save_inquiry(conn=app_conn, processed=stored)

    generation = scripted_client({})
    with live_client(app_conn=app_conn, ro_conn=ro_conn, generation=generation) as test_client:
        response = test_client.get(f"/inquiries/{stored.inquiry_id}")
        payload = response.json()

    assert response.status_code == 200
    _assert_skeleton(payload)
    assert payload["inquiry_id"] == stored.inquiry_id
    assert payload["answer"] == stored.answer
    # 판정·검색 토큰은 생성 합산(910/210)에 섞이지 않고 **분리된 키**로 복원된다.
    assert payload["metrics"] == {
        "latency_ms": 1234,
        "tokens": {
            "input": 910,
            "output": 210,
            "judge_input": 433,
            "judge_output": 91,
            "retrieval_input": 17,
            "retrieval_output": 5,
        },
        "retrieval_fallback_reason": None,
    }
    l1_reasons = (RejectReason.MISSING_CITATION, RejectReason.PII_DETECTED)
    stored_l2 = stored.attempts[1].l2_result
    assert stored_l2 is not None
    assert payload["attempts"] == [
        # 시도 1 — L1 reject 라 L2 는 돌지 않았다(null).
        attempt(Verdict.REJECT, l1_reasons, l1=layer(Verdict.REJECT, l1_reasons), l2=None),
        # 시도 2 — L1 pass 뒤 L2 실행. claim 판정·모순쌍이 저장→복원을 거쳐 그대로 실린다.
        attempt(
            Verdict.PASS,
            l1=layer(Verdict.PASS),
            l2={
                "verdict": Verdict.PASS.value,
                "reject_reasons": [],
                "claim_judgments": [
                    {
                        "claim_text": judgment.claim_text,
                        "verdict": judgment.verdict.value,
                        "explanation": judgment.explanation,
                    }
                    for judgment in stored_l2.claim_judgments
                ],
                "contradictions": [
                    {
                        "evidence_id_a": contradiction.evidence_id_a,
                        "evidence_id_b": contradiction.evidence_id_b,
                        "explanation": contradiction.explanation,
                    }
                    for contradiction in stored_l2.contradictions
                ],
            },
        ),
    ]
    assert [citation["id"] for citation in payload["citations"]] == [
        item.id for item in stored.evidence
    ]


@pytest.mark.db
def test_처리_기록이_DB_에_남는다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    generation = scripted_client({INTENT_STAGE: [intent_completion("order")]})
    with live_client(app_conn=app_conn, ro_conn=ro_conn, generation=generation) as test_client:
        payload = test_client.post("/inquiries", json={"content": "제 주문 어디쯤 왔나요?"}).json()

    row = app_conn.execute(
        "SELECT status, escalation_reason, latency_ms FROM inquiries WHERE id = %s",
        (payload["inquiry_id"],),
    ).fetchone()

    assert row is not None
    assert row["status"] == InquiryStatus.ESCALATED.value
    assert row["escalation_reason"] == EscalationReason.MISSING_ORDER_REF.value
    assert row["latency_ms"] >= 0


# ── 층 구분과 L2 상세 (docs/contracts.md "층별 판정 키") ─────────────────────


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_L2_기각_시도는_claim_단위_판정과_모순쌍을_응답에_싣는다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """데모의 "왜 기각됐는지" 장면 — 사유 코드만이 아니라 어느 문장이 왜인지가 실린다."""
    generation = scripted_client(
        {INTENT_STAGE: [intent_completion("policy")], DRAFT_STAGE: [citing_draft(), citing_draft()]}
    )
    judge = ContradictionJudge()

    with l2_client(
        app_conn=app_conn, ro_conn=ro_conn, generation=generation, judge=judge
    ) as test_client:
        payload = test_client.post("/inquiries", json={"content": INQUIRY}).json()

    _assert_skeleton(payload)
    assert payload["status"] == InquiryStatus.ESCALATED.value
    assert payload["escalation_reason"] == EscalationReason.REJECTED_TWICE.value
    assert len(payload["attempts"]) == 2

    for item in payload["attempts"]:
        # 종합 사유 = 두 층 사유의 합집합. L1 은 통과했으므로 종합 사유는 L2 사유 그대로다.
        assert item["verdict"] == Verdict.REJECT.value
        assert item["reject_reasons"] == [
            RejectReason.UNSUPPORTED_CLAIM.value,
            RejectReason.CONTRADICTORY_EVIDENCE.value,
        ]
        assert item["l1"] == layer(Verdict.PASS)
        assert item["l2"]["verdict"] == Verdict.REJECT.value
        assert item["l2"]["reject_reasons"] == item["reject_reasons"]

    judgments = payload["attempts"][0]["l2"]["claim_judgments"]
    assert judgments and judgments[0]["verdict"] == Verdict.REJECT.value
    assert judgments[0]["explanation"]
    # claim_text 는 초안 문장 참조다 — 판정이 어느 문장을 가리키는지 화면이 알아야 한다.
    assert judgments[0]["claim_text"] == "안내드립니다."

    contradictions = payload["attempts"][0]["l2"]["contradictions"]
    citation_ids = {citation["id"] for citation in payload["citations"]}
    assert len(contradictions) == 1
    assert {contradictions[0]["evidence_id_a"], contradictions[0]["evidence_id_b"]} <= citation_ids
    assert contradictions[0]["explanation"]

    # 판정 토큰은 별도 키다 — 생성 합산에 섞이지 않는다.
    assert payload["metrics"]["tokens"]["judge_input"] == 140 * 2
    assert payload["metrics"]["tokens"]["judge_output"] == 37 * 2


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_L2_기각_응답도_GET_으로_같은_골격이_다시_나온다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """재구성 동등성 — claim 단위 판정·모순쌍·판정 토큰까지 저장→복원을 왕복한다."""
    generation = scripted_client(
        {INTENT_STAGE: [intent_completion("policy")], DRAFT_STAGE: [citing_draft(), citing_draft()]}
    )

    with l2_client(
        app_conn=app_conn, ro_conn=ro_conn, generation=generation, judge=ContradictionJudge()
    ) as test_client:
        created = test_client.post("/inquiries", json={"content": INQUIRY}).json()
        fetched = test_client.get(f"/inquiries/{created['inquiry_id']}").json()

    assert fetched == created


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_L2_호출이_실패한_시도는_종합_pass_인데_L2_가_null_이다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """docs/contracts.md "층별 판정 키" 가 정한 동작 — 값은 바꾸지 않는다.
    대신 `l2: null` 이 "판정이 없었다"를
    드러내야 화면이 "통과했는데 왜 인계?"로 읽히지 않는다."""
    generation = scripted_client(
        {INTENT_STAGE: [intent_completion("policy")], DRAFT_STAGE: [citing_draft()]}
    )
    judge = ScriptedJudge([LLMCallError(stage=JUDGE_STAGE, reason="transport_error", attempts=2)])

    with l2_client(
        app_conn=app_conn, ro_conn=ro_conn, generation=generation, judge=judge
    ) as test_client:
        created = test_client.post("/inquiries", json={"content": INQUIRY}).json()
        fetched = test_client.get(f"/inquiries/{created['inquiry_id']}").json()

    _assert_skeleton(created)
    assert created["status"] == InquiryStatus.ESCALATED.value
    assert created["escalation_reason"] == EscalationReason.LLM_CALL_FAILED.value
    assert created["answer"] is None
    assert created["attempts"] == [attempt(Verdict.PASS, l1=layer(Verdict.PASS), l2=None)]
    assert fetched == created


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_L1_이_기각한_시도는_L2_가_null_이고_판정자가_불리지_않는다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """미실행 3종 중 두 번째 — L2 는 L1 통과분에만 돈다."""
    bogus = draft_completion(
        {"claims": [{"text": "환불은 30일 이내입니다.", "citation_ids": ["policy:없는문서:1"]}]}
    )
    generation = scripted_client(
        {INTENT_STAGE: [intent_completion("policy")], DRAFT_STAGE: [bogus, bogus]}
    )
    # 대본이 빈 판정 대역 — 한 번이라도 불리면 AssertionError 로 죽는다.
    judge = ScriptedJudge([])

    with l2_client(
        app_conn=app_conn, ro_conn=ro_conn, generation=generation, judge=judge
    ) as test_client:
        payload = test_client.post("/inquiries", json={"content": INQUIRY}).json()

    _assert_skeleton(payload)
    assert judge.calls == []
    rejected = attempt(
        Verdict.REJECT,
        (RejectReason.INVALID_CITATION,),
        l1=layer(Verdict.REJECT, (RejectReason.INVALID_CITATION,)),
        l2=None,
    )
    assert payload["attempts"] == [rejected, rejected]
    assert payload["metrics"]["tokens"]["judge_input"] == 0
    assert payload["metrics"]["tokens"]["judge_output"] == 0


def test_판정_토큰은_생성_합산에_섞이지_않는다() -> None:
    """`metrics.tokens` 의 기존 키 의미는 불변이다 — 생성 합산이 판정 토큰을 흡수하면
    건당 비용 지표가 무너진다(DB 없이 조립기만 본다)."""
    processed = _processed(new_inquiry_id())

    tokens = InquiryResponse.of(processed).model_dump()["metrics"]["tokens"]

    assert processed.judge_input_tokens and processed.judge_output_tokens  # 양성 대조
    assert processed.retrieval_input_tokens and processed.retrieval_output_tokens  # 양성 대조
    assert tokens == {
        "input": processed.input_tokens,
        "output": processed.output_tokens,
        "judge_input": processed.judge_input_tokens,
        "judge_output": processed.judge_output_tokens,
        "retrieval_input": processed.retrieval_input_tokens,
        "retrieval_output": processed.retrieval_output_tokens,
    }


# ── API 키가 없는 환경 (런타임 스모크가 잡아낸 회귀) ────────────────────────
#
# 생성 클라이언트를 의존성 해석 시점에 만들면, 자격 증명이 없을 때 **LLM 을 전혀 쓰지 않는
# 경로**(조회 전용 GET, 접수 거부 422)까지 500 으로 무너진다. 클라이언트는 실제 호출 직전에
# 만들고, 자격 증명 부재는 인계(`llm_call_failed`)가 아니라 **설정 오류(503)** 로 끝나야 한다.


def keyless_pipeline() -> InquiryPipeline:
    """생성 키도 판정 키도 없는 조립 — **스위치는 켜 둔다**(조립이 키를 요구하지 않아야 한다)."""
    settings = Settings(
        openai_api_key="",
        anthropic_api_key="",
        l2_enabled=True,
        vector_top_k=5,
        vector_similarity_threshold=0.0,
        sql_max_rows=50,
    )
    return build_pipeline(
        generation_client=build_generation_client(settings),
        embedding_client=build_embedding_client(settings),
        settings=settings,
    )


@contextmanager
def keyless_client(
    *, app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> Iterator[TestClient]:
    with service_client(
        pipeline=keyless_pipeline(), app_conn=app_conn, ro_conn=ro_conn
    ) as test_client:
        yield test_client


def test_API_키가_없어도_앱_조립은_성공한다() -> None:
    """클라이언트는 첫 호출 때 만들어진다 — 조립 시점에 자격 증명을 요구하지 않는다."""
    assert keyless_pipeline() is not None


@pytest.mark.db
def test_API_키가_없어도_조회는_동작한다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """`GET /inquiries/{id}` 는 DB 만 읽는다 — 생성 LLM 자격 증명과 무관해야 한다."""
    stored = _processed(new_inquiry_id())
    save_inquiry(conn=app_conn, processed=stored)

    with keyless_client(app_conn=app_conn, ro_conn=ro_conn) as test_client:
        response = test_client.get(f"/inquiries/{stored.inquiry_id}")

    assert response.status_code == 200
    _assert_skeleton(response.json())


@pytest.mark.db
def test_API_키가_없으면_접수_거부는_그대로_422다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    with keyless_client(app_conn=app_conn, ro_conn=ro_conn) as test_client:
        response = test_client.post("/inquiries", json={"content": INQUIRY, "order_no": "12345"})

    assert response.status_code == 422


@pytest.mark.db
def test_API_키가_없으면_POST_는_503_이고_인계로_기록하지_않는다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """설정 오류는 업무 판정이 아니다 — `llm_call_failed` 로 위장하면 평가 지표가 오염된다."""
    before = app_conn.execute("SELECT count(*) AS n FROM inquiries").fetchone()
    assert before is not None

    with keyless_client(app_conn=app_conn, ro_conn=ro_conn) as test_client:
        response = test_client.post("/inquiries", json={"content": INQUIRY})

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.text
    after = app_conn.execute("SELECT count(*) AS n FROM inquiries").fetchone()
    assert after is not None
    assert after["n"] == before["n"]


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_인덱스와_다른_모델로_질의하면_503_이고_인계로_기록하지_않는다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """낡은 정책 인덱스는 설정 오류다 — `no_evidence` 로 집계되면 재색인 누락이 검색 품질로 위장한다.

    `indexed_policies` 는 기본 모델 이름으로 적재하고, 여기서는 다른 이름의 임베더로 질의한다.
    차원은 같으므로 pgvector 는 아무 불평 없이 코사인을 계산해 줄 것이다 — 그래서 코드가 막는다.
    """
    before = app_conn.execute("SELECT count(*) AS n FROM inquiries").fetchone()
    assert before is not None
    pipeline = build_pipeline(
        generation_client=cast(
            GenerationClient, scripted_client({INTENT_STAGE: [intent_completion("policy")]})
        ),
        embedding_client=LexicalEmbeddingClient(dimensions=1536, model="다른-모델"),
        settings=Settings(vector_top_k=5, vector_similarity_threshold=0.0, l2_enabled=False),
    )

    with service_client(pipeline=pipeline, app_conn=app_conn, ro_conn=ro_conn) as test_client:
        response = test_client.post("/inquiries", json={"content": INQUIRY})

    assert response.status_code == 503
    assert "index_policies" in response.text
    after = app_conn.execute("SELECT count(*) AS n FROM inquiries").fetchone()
    assert after is not None
    assert after["n"] == before["n"]


# ── 판정 키가 없는 환경 (L2 켜짐 = POST 진입 선검사) ─────────────────────────
#
# lazy(L2 에 도달해서야 실패)로 하면 생성 토큰을 태운 뒤 503 이 나고, L1 이 2회 기각하면
# 키 없이도 정상 종결되어 이 규칙이 조건부가 된다. 그래서 검사는 eager 다 — 단,
# **POST 경로에만**: 조회·422 는 판정 키와 무관하다.


def test_판정_키가_없으면_POST_는_503_이고_파이프라인이_돌지_않는다(
    client: TestClient, recorder: RecordingService, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_settings(monkeypatch, l2_enabled=True, anthropic_api_key="")

    response = client.post("/inquiries", json={"content": INQUIRY})

    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.text
    assert recorder.calls == []


def test_판정_키가_없어도_접수_거부는_그대로_422다(
    client: TestClient, recorder: RecordingService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """선검사가 422 경로까지 번지면 형식 오류가 설정 오류로 둔갑한다."""
    api_settings(monkeypatch, l2_enabled=True, anthropic_api_key="")

    response = client.post("/inquiries", json={"content": INQUIRY, "order_no": "12345"})

    assert response.status_code == 422
    assert recorder.calls == []


def test_판정_키가_없어도_조회는_동작한다(
    client: TestClient, recorder: RecordingService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`GET /inquiries/{id}` 는 DB 만 읽는다 — 판정 자격 증명과 무관해야 한다."""
    api_settings(monkeypatch, l2_enabled=True, anthropic_api_key="")

    response = client.get("/inquiries/3b0a5a1e-0000-4000-8000-000000000000")

    assert response.status_code == 404
    assert recorder.calls == [{"fetch": "3b0a5a1e-0000-4000-8000-000000000000"}]


@pytest.mark.db
def test_판정_키_부재_503_은_처리_기록을_남기지_않는다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """설정 오류는 `llm_call_failed` 인계가 아니다 — 기록으로 남으면 지표가 오염된다."""
    api_settings(monkeypatch, l2_enabled=True, anthropic_api_key="")
    before = app_conn.execute("SELECT count(*) AS n FROM inquiries").fetchone()
    assert before is not None

    # 대본이 빈 생성 대역 — 선검사가 뚫리면 파이프라인이 돌다가 실패해 이 테스트가 깨진다.
    generation = scripted_client({})
    with live_client(app_conn=app_conn, ro_conn=ro_conn, generation=generation) as test_client:
        response = test_client.post("/inquiries", json={"content": INQUIRY})

    assert response.status_code == 503
    after = app_conn.execute("SELECT count(*) AS n FROM inquiries").fetchone()
    assert after is not None
    assert after["n"] == before["n"]


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_스위치가_꺼져_있으면_판정_키_없이도_POST_가_처리된다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """양성 대조 — 선검사는 스위치가 켜져 있을 때만 건다."""
    api_settings(monkeypatch, l2_enabled=False, anthropic_api_key="")
    generation = scripted_client(
        {INTENT_STAGE: [intent_completion("policy")], DRAFT_STAGE: [citing_draft()]}
    )

    with live_client(app_conn=app_conn, ro_conn=ro_conn, generation=generation) as test_client:
        response = test_client.post("/inquiries", json={"content": INQUIRY})

    assert response.status_code == 200
    assert response.json()["status"] == InquiryStatus.ANSWERED.value


# ── DB 가 없을 때의 순서 (요청 오류·설정 오류가 500 에 가리지 않는다) ───────
#
# 이 절의 테스트는 **DB 가 없어야 의미가 있다** — `db` 마커를 붙이지 않고, 살아 있는
# 컨테이너를 건드리는 대신 설정을 죽은 포트로 돌려 "DB 없음"을 만든다. 의존성 override 도
# 하지 않는다: 실제 `get_service` 가 언제 연결하는지가 검사 대상이기 때문이다.


def _dbless(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> Settings:
    """DB 가 없는 환경. 포트 1 은 어떤 Postgres 도 듣지 않는다."""
    return api_settings(monkeypatch, postgres_port=1, **overrides)


def test_의존성_해석만으로는_커넥션을_열지_않는다(monkeypatch: pytest.MonkeyPatch) -> None:
    """`get_service()` 자체가 연결하면 라우트의 선검사가 무엇을 하든 늦는다.

    프레임워크의 의존성 해석 순서에 기대지 않는 지점이 여기다 — 해석이 공짜면 순서가
    무엇이든 선검사가 먼저 끝난다.
    """
    _dbless(monkeypatch)

    open_service = get_service()  # DB 가 없는데도 여기서 터지면 안 된다

    with pytest.raises(psycopg.OperationalError), open_service():  # 열 때에야 터진다
        pass


def test_DB_가_없어도_접수_거부는_422다(monkeypatch: pytest.MonkeyPatch) -> None:
    """형식이 틀린 주문번호는 요청 오류다 — DB 가 떠 있느냐와 무관하다."""
    _dbless(monkeypatch, l2_enabled=False)

    with TestClient(app) as test_client:
        response = test_client.post("/inquiries", json={"content": INQUIRY, "order_no": "12345"})

    assert response.status_code == 422


def test_DB_가_없어도_빈_내용은_422다(monkeypatch: pytest.MonkeyPatch) -> None:
    _dbless(monkeypatch, l2_enabled=False)

    with TestClient(app) as test_client:
        response = test_client.post("/inquiries", json={"content": "   "})

    assert response.status_code == 422


def test_DB_가_없어도_판정키_부재는_503이다(monkeypatch: pytest.MonkeyPatch) -> None:
    """설정 오류는 DB 상태와 무관하게 설정 오류다."""
    _dbless(monkeypatch, l2_enabled=True, anthropic_api_key="")

    with TestClient(app) as test_client:
        response = test_client.post("/inquiries", json={"content": INQUIRY})

    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.text


def test_DB_가_진짜_필요한_경로는_그대로_500이다(monkeypatch: pytest.MonkeyPatch) -> None:
    """DB 장애를 업무 판정으로 바꾸지 않는다(docs/standards.md).

    위 세 건을 "DB 오류를 삼켜서" 통과시키면 이 테스트가 깨진다.
    """
    _dbless(monkeypatch, l2_enabled=False)

    with TestClient(app, raise_server_exceptions=False) as test_client:
        lookup = test_client.get("/inquiries/3b0a5a1e-0000-4000-8000-000000000000")
        process = test_client.post("/inquiries", json={"content": INQUIRY})

    assert lookup.status_code == 500
    assert process.status_code == 500


# ── 커넥션 배선 자체 (목으로 덮이는 구간) ────────────────────────────────────


@pytest.mark.db
def test_get_service_가_계정_두_개로_커넥션을_열고_닫는다() -> None:
    """다른 테스트는 이 의존성을 통째로 override 하므로 배선이 목 뒤에 숨는다.

    text-to-SQL 이 앱 계정 커넥션을 집으면 안전장치 1층이 무의미해지므로,
    실제로 열리는 커넥션이 각각 어느 계정인지 한 번은 DB 에 물어 확인한다.
    """
    settings = get_settings()
    open_service = get_service()
    with open_service() as service:
        with service._app_conn.cursor() as cur:
            cur.execute("SELECT current_user AS who")
            row = cur.fetchone()
            assert row is not None
            assert row["who"] == settings.postgres_app_user

        with service._readonly_conn.cursor() as cur:
            cur.execute("SELECT current_user AS who")
            row = cur.fetchone()
            assert row is not None
            assert row["who"] == settings.postgres_ro_user

    # 블록이 끝나면 두 커넥션 모두 닫혀 있어야 한다 — 요청마다 새는 것을 막는다.
    assert service._app_conn.closed
    assert service._readonly_conn.closed
