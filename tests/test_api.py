"""API 표면 테스트 — 엔드포인트 4개와 응답 스키마.

응답 스키마의 계약은 "**모든 키가 항상 존재한다**" 이다: `answer`·`escalation_reason` 은
해당 없을 때 null, `claims`·`citations`·`attempts` 는 해당 없을 때 빈 배열. 초안 전
인계라도 그때까지 모은 근거는 `citations` 에 남는다.

접수 검증(빈 내용·주문번호 형식)은 **파이프라인을 돌리지 않고** 422 로 끝난다 — 인계가
아니다. 그 증거로 대역 서비스의 호출 기록이 비어 있는지 확인한다.

DB 가 필요한 테스트는 `db` 마커가 붙고, 쓰기는 전부 `app_conn` 트랜잭션 안에서만 일어나
픽스처 롤백으로 되돌아간다.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import DictRow

from reply_gate.api import (
    InquiryService,
    app,
    build_embedding_client,
    build_generation_client,
    get_service,
)
from reply_gate.config import Settings
from reply_gate.contracts import EscalationReason, InquiryStatus, RejectReason, Verdict
from reply_gate.draft import DRAFT_STAGE
from reply_gate.evidence import INTENT_STAGE
from reply_gate.pipeline import (
    InquiryPipeline,
    ProcessedInquiry,
    build_pipeline,
    new_inquiry_id,
)
from reply_gate.records import save_inquiry
from tests.test_pipeline import (
    ScriptedGenerationClient,
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

    def _override() -> RecordingService:
        return service

    app.dependency_overrides[get_service] = _override
    try:
        yield service
    finally:
        app.dependency_overrides.pop(get_service, None)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@contextmanager
def live_client(
    *,
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    generation: ScriptedGenerationClient,
    threshold: float = 0.0,
) -> Iterator[TestClient]:
    """실제 파이프라인 + 테스트 커넥션으로 앱을 돌린다 (LLM 만 목)."""

    def _override() -> InquiryService:
        return InquiryService(
            pipeline=live_pipeline(generation, threshold=threshold),
            app_conn=app_conn,
            readonly_conn=ro_conn,
        )

    app.dependency_overrides[get_service] = _override
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_service, None)


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
    assert "L1 게이트 판정" in html
    assert "기각" in html
    assert "인계 사유" in html


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
    """모든 키가 항상 존재하고 타입이 계약대로인지."""
    assert set(payload) == RESPONSE_KEYS
    assert isinstance(payload["inquiry_id"], str)
    assert payload["status"] in {InquiryStatus.ANSWERED.value, InquiryStatus.ESCALATED.value}
    assert isinstance(payload["claims"], list)
    assert isinstance(payload["citations"], list)
    assert isinstance(payload["attempts"], list)
    assert set(payload["metrics"]) == {"latency_ms", "tokens"}
    assert set(payload["metrics"]["tokens"]) == {"input", "output"}
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
    assert payload["attempts"] == [{"verdict": Verdict.PASS.value, "reject_reasons": []}]


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
    # 데모의 주인공 — 기각 사유가 응답에 그대로 실린다.
    assert payload["attempts"] == [
        {"verdict": Verdict.REJECT.value, "reject_reasons": [RejectReason.INVALID_CITATION.value]},
        {"verdict": Verdict.REJECT.value, "reject_reasons": [RejectReason.INVALID_CITATION.value]},
    ]


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
    assert payload["metrics"] == {"latency_ms": 1234, "tokens": {"input": 910, "output": 210}}
    assert payload["attempts"] == [
        {
            "verdict": Verdict.REJECT.value,
            "reject_reasons": [
                RejectReason.MISSING_CITATION.value,
                RejectReason.PII_DETECTED.value,
            ],
        },
        {"verdict": Verdict.PASS.value, "reject_reasons": []},
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


# ── API 키가 없는 환경 (런타임 스모크가 잡아낸 회귀) ────────────────────────
#
# 생성 클라이언트를 의존성 해석 시점에 만들면, 자격 증명이 없을 때 **LLM 을 전혀 쓰지 않는
# 경로**(조회 전용 GET, 접수 거부 422)까지 500 으로 무너진다. 클라이언트는 실제 호출 직전에
# 만들고, 자격 증명 부재는 인계(`llm_call_failed`)가 아니라 **설정 오류(503)** 로 끝나야 한다.


def keyless_pipeline() -> InquiryPipeline:
    settings = Settings(
        openai_api_key="",
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
    def _override() -> InquiryService:
        return InquiryService(pipeline=keyless_pipeline(), app_conn=app_conn, readonly_conn=ro_conn)

    app.dependency_overrides[get_service] = _override
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_service, None)


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
