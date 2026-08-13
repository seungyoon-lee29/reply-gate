"""근거 수집 테스트 — 의도 분류·벡터 검색·존재성 선검사·text-to-SQL.

생성 LLM 은 전부 목이다. 임베딩은 `LexicalEmbeddingClient` 결정론 대역을 써서 API 키 없이
벡터 검색 배관까지 돌린다. DB 가 필요한 테스트는 `db` 마커가 붙고, 쓰기는 하지 않는다
(정책 인덱싱은 `app_conn` 트랜잭션 안에서만 일어나고 픽스처가 롤백한다).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import psycopg
import pytest
from psycopg.rows import DictRow

from reply_gate.config import Settings
from reply_gate.contracts import (
    EscalationReason,
    EvidenceSource,
    IntentSource,
    RejectReason,
    Verdict,
)
from reply_gate.db import readonly_connect
from reply_gate.evidence import (
    INQUIRY_EMBEDDING_STAGE,
    INTENT_STAGE,
    SQL_GENERATION_STAGE,
    EvidenceCollector,
    SqlFailureKind,
    _sql_evidence_texts,
    classify_intent,
    generate_sql,
    order_exists,
)
from reply_gate.gate import evaluate_draft
from reply_gate.llm import GenerationClient, JsonCompletion, LLMCallError, LLMFormatError
from reply_gate.policy_index import index_policy_documents, load_policy_documents
from reply_gate.query_rewrite import QUERY_REWRITE_STAGE
from reply_gate.testing import LexicalEmbeddingClient

INQUIRY_ID = "11111111-2222-3333-4444-555555555555"
INQUIRY = "환불 신청은 언제까지 가능한가요? 제 주문도 확인해 주세요."
MISSING_ORDER_NO = "ORD-20991231-9999"


# ── 목 / 픽스처 ─────────────────────────────────────────────────────────────


#: 대본에 재작성이 없을 때 기본으로 돌려주는 토큰 — 0 이 아니어야 "검색 계열이 생성
#: 합산에 섞이지 않는다"를 대조할 수 있다.
_DEFAULT_REWRITE_TOKENS = (7, 3)


class _ScriptedClient:
    """`GenerationClient` 대역 — 단계별로 미리 정한 결과를 순서대로 돌려준다.

    **재작성 단계만 예외**다: 대본에 아예 없으면 "원문 그대로"를 돌려준다. 재작성은 정책
    경로의 기본 호출이라 대본에 넣기를 강제하면 관계없는 테스트 수십 개가 재작성 대본을
    이고 다니게 되고, 원문 그대로는 픽스처 계약이 명시적으로 허용하는 산출이라
    (`rewritten` == `original`) 수집기가 검색을 한 번만 돌아 기존 기대가 그대로 선다.
    **대본이 재작성을 지정했으면 그것이 이기고, 큐가 마르면 그때는 오류다.**
    """

    def __init__(self, script: Mapping[str, Sequence[Any]]) -> None:
        self._script = {stage: list(outcomes) for stage, outcomes in script.items()}
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        stage = kwargs["stage"]
        if stage == QUERY_REWRITE_STAGE and stage not in self._script:
            return _echo_rewrite(kwargs["user"])
        queue = self._script.get(stage)
        if not queue:
            raise AssertionError(f"대본에 없는 호출이다: stage={stage!r}")
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def calls_for(self, stage: str) -> list[dict[str, Any]]:
        return [call for call in self.calls if call["stage"] == stage]


def _echo_rewrite(user_prompt: str) -> JsonCompletion:
    """재작성 대역의 기본 산출 — 프롬프트에 실린 문의 원문을 그대로 되돌려준다."""
    inquiry = user_prompt.removeprefix("[문의]\n")
    return JsonCompletion(
        data={"rewritten": inquiry},
        input_tokens=_DEFAULT_REWRITE_TOKENS[0],
        output_tokens=_DEFAULT_REWRITE_TOKENS[1],
    )


def _rewrite(text: str, *, input_tokens: int = 7, output_tokens: int = 3) -> JsonCompletion:
    return JsonCompletion(
        data={"rewritten": text}, input_tokens=input_tokens, output_tokens=output_tokens
    )


def _client(script: Mapping[str, Sequence[Any]]) -> _ScriptedClient:
    return _ScriptedClient(script)


def _intent(value: str, *, input_tokens: int = 10, output_tokens: int = 2) -> JsonCompletion:
    return JsonCompletion(
        data={"source": value}, input_tokens=input_tokens, output_tokens=output_tokens
    )


def _sql(text: str, *, input_tokens: int = 30, output_tokens: int = 12) -> JsonCompletion:
    return JsonCompletion(
        data={"sql": text}, input_tokens=input_tokens, output_tokens=output_tokens
    )


def _collector(
    client: _ScriptedClient,
    *,
    threshold: float = 0.0,
    top_k: int = 5,
    max_rows: int = 50,
    query_rewrite_enabled: bool = True,
) -> EvidenceCollector:
    """기본은 **제품 기본값(재작성 켜짐)** 이다 — 테스트가 제품이 쓰지 않는 구성을 재면
    배관 검증이 실제 실행과 다른 것을 검증한다."""
    return EvidenceCollector(
        generation_client=cast(GenerationClient, client),
        embedding_client=LexicalEmbeddingClient(dimensions=1536),
        settings=Settings(
            vector_top_k=top_k,
            vector_similarity_threshold=threshold,
            sql_max_rows=max_rows,
            query_rewrite_enabled=query_rewrite_enabled,
        ),
        rewrite_client=cast(GenerationClient, client) if query_rewrite_enabled else None,
    )


_NO_CONN = cast(psycopg.Connection[DictRow], None)


@pytest.fixture
def indexed_policies(app_conn: psycopg.Connection[DictRow]) -> None:
    """저장소의 정책 문서를 결정론 임베딩으로 적재한다 (픽스처 롤백으로 되돌아간다)."""
    index_policy_documents(
        conn=app_conn,
        documents=load_policy_documents(),
        embedder=LexicalEmbeddingClient(dimensions=1536),
    )


@pytest.fixture
def sample_order(ro_conn: psycopg.Connection[DictRow]) -> dict[str, Any]:
    row = ro_conn.execute(
        "SELECT order_no, customer_name, customer_phone, customer_email, status"
        " FROM orders ORDER BY order_no LIMIT 1"
    ).fetchone()
    assert row is not None, "시딩된 주문이 있어야 한다"
    return dict(row)


# ── 의도 분류 (DB 불필요) ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("policy", IntentSource.POLICY),
        ("order", IntentSource.ORDER),
        ("both", IntentSource.BOTH),
    ],
)
def test_의도_분류_구조화_출력을_파싱한다(value: str, expected: IntentSource) -> None:
    client = _client({INTENT_STAGE: [_intent(value)]})

    result = classify_intent(client=cast(GenerationClient, client), inquiry=INQUIRY)

    assert result.source is expected
    assert result.error is None
    assert (result.input_tokens, result.output_tokens) == (10, 2)


def test_형식_불일치는_1회_재시도한_뒤_성공한다() -> None:
    broken = LLMFormatError(
        stage=INTENT_STAGE,
        detail="JSON 파싱 실패",
        raw_text="정책이랑 주문 둘 다요",
        input_tokens=7,
        output_tokens=1,
    )
    client = _client({INTENT_STAGE: [broken, _intent("both")]})

    result = classify_intent(client=cast(GenerationClient, client), inquiry=INQUIRY)

    assert result.source is IntentSource.BOTH
    assert len(client.calls_for(INTENT_STAGE)) == 2
    # 재시도 프롬프트는 직전 산출과 실패 사유를 피드백으로 싣는다.
    retry_prompt = client.calls_for(INTENT_STAGE)[1]["user"]
    assert "정책이랑 주문 둘 다요" in retry_prompt
    assert "JSON 파싱 실패" in retry_prompt
    # 실패한 호출의 토큰도 합산에서 빠지지 않는다.
    assert (result.input_tokens, result.output_tokens) == (17, 3)


def test_재시도해도_형식이_안_맞으면_실패로_돌려준다() -> None:
    broken = LLMFormatError(
        stage=INTENT_STAGE, detail="형식 불일치", input_tokens=1, output_tokens=1
    )
    client = _client({INTENT_STAGE: [broken, broken]})

    result = classify_intent(client=cast(GenerationClient, client), inquiry=INQUIRY)

    assert result.source is None
    assert result.error is not None
    # 재시도는 코드가 1회로 상한을 강제한다.
    assert len(client.calls_for(INTENT_STAGE)) == 2


def test_열거값이_아닌_source_도_형식_불일치로_본다() -> None:
    client = _client({INTENT_STAGE: [_intent("shipping"), _intent("policy")]})

    result = classify_intent(client=cast(GenerationClient, client), inquiry=INQUIRY)

    assert result.source is IntentSource.POLICY
    assert len(client.calls_for(INTENT_STAGE)) == 2


def test_전송_오류는_재시도하지_않고_그대로_올린다() -> None:
    """래퍼가 이미 1회 재시도했다 — 여기서 또 재시도하면 상한이 무너진다."""
    client = _client(
        {INTENT_STAGE: [LLMCallError(stage=INTENT_STAGE, reason="transport_error", attempts=2)]}
    )

    with pytest.raises(LLMCallError):
        classify_intent(client=cast(GenerationClient, client), inquiry=INQUIRY)

    assert len(client.calls_for(INTENT_STAGE)) == 1


def test_형식_실패_뒤_전송_오류면_앞선_전송까지_세어_올린다() -> None:
    """토큰은 누적하면서 횟수만 버리면 '1회 재시도'라고 적힌 기록이 실제와 갈린다.

    1차 시도가 형식 오류(전송 1회) + 2차 시도가 전송 오류(래퍼가 2회 전송) = 실제 3회.
    """
    broken = LLMFormatError(
        stage=INTENT_STAGE,
        detail="형식 불일치",
        input_tokens=7,
        output_tokens=1,
        transport_attempts=1,
    )
    died = LLMCallError(
        stage=INTENT_STAGE, reason="transport_error", attempts=2, input_tokens=3, output_tokens=0
    )
    client = _client({INTENT_STAGE: [broken, died]})

    with pytest.raises(LLMCallError) as excinfo:
        classify_intent(client=cast(GenerationClient, client), inquiry=INQUIRY)

    assert excinfo.value.attempts == 3
    # 토큰 누적은 원래도 맞았다 — 같은 자리에서 횟수만 빠져 있었다.
    assert (excinfo.value.input_tokens, excinfo.value.output_tokens) == (10, 1)


# ── SQL 생성 (DB 불필요) ────────────────────────────────────────────────────


def test_SQL_생성_프롬프트에_화이트리스트와_주문번호가_실린다() -> None:
    client = _client({SQL_GENERATION_STAGE: [_sql("SELECT order_no FROM orders")]})

    generate_sql(
        client=cast(GenerationClient, client),
        inquiry=INQUIRY,
        order_no="ORD-20260201-0001",
        max_rows=50,
    )

    prompt = client.calls_for(SQL_GENERATION_STAGE)[0]["user"]
    assert "orders" in prompt
    assert "customer_phone" in prompt
    assert "ORD-20260201-0001" in prompt
    assert "50" in prompt


def test_빈_SQL_은_생성_실패로_본다() -> None:
    client = _client({SQL_GENERATION_STAGE: [_sql("   ")]})

    result = generate_sql(
        client=cast(GenerationClient, client),
        inquiry=INQUIRY,
        order_no="ORD-20260201-0001",
        max_rows=50,
    )

    assert result.sql is None
    assert result.error is not None
    # SQL 생성은 여기서 재시도하지 않는다 — 재시도 상한은 수집기가 통제한다.
    assert len(client.calls_for(SQL_GENERATION_STAGE)) == 1


def test_SQL_생성_형식_불일치도_생성_실패로_본다() -> None:
    broken = LLMFormatError(
        stage=SQL_GENERATION_STAGE,
        detail="JSON 아님",
        raw_text="SELECT",
        input_tokens=4,
        output_tokens=2,
    )
    client = _client({SQL_GENERATION_STAGE: [broken]})

    result = generate_sql(
        client=cast(GenerationClient, client),
        inquiry=INQUIRY,
        order_no="ORD-20260201-0001",
        max_rows=50,
    )

    assert result.sql is None
    assert (result.input_tokens, result.output_tokens) == (4, 2)


# ── 인계 경로 중 DB 를 건드리지 않는 것들 ───────────────────────────────────


def test_order_의도인데_주문번호가_없으면_missing_order_ref() -> None:
    """DB 도 SQL 생성도 건드리지 않는다 — 커넥션 자리에 None 을 넣어 그것을 증명한다."""
    client = _client({INTENT_STAGE: [_intent("order")]})

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=None,
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert result.escalation_reason is EscalationReason.MISSING_ORDER_REF
    assert result.evidence == ()
    assert client.calls_for(SQL_GENERATION_STAGE) == []


def test_형식이_깨진_주문번호도_missing_order_ref() -> None:
    client = _client({INTENT_STAGE: [_intent("order")]})

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no="ORD-20260230-0001",  # 달력에 없는 날짜
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert result.escalation_reason is EscalationReason.MISSING_ORDER_REF


def test_의도_분류_실패는_llm_call_failed_와_실패_단계를_남긴다() -> None:
    broken = LLMFormatError(stage=INTENT_STAGE, detail="형식 불일치")
    client = _client({INTENT_STAGE: [broken, broken]})

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=None,
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert result.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert result.failed_stage == INTENT_STAGE
    assert result.intent is None


def test_의도_분류_전송_오류도_llm_call_failed() -> None:
    client = _client(
        {INTENT_STAGE: [LLMCallError(stage=INTENT_STAGE, reason="transport_error", attempts=2)]}
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=None,
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert result.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert result.failed_stage == INTENT_STAGE


def test_수집_전송_오류가_실은_과금_토큰도_수집_합산에_남는다() -> None:
    """거절처럼 200 으로 돌아온 실패도 과금된다 — 수집 경로만 그 토큰을 버리면 안 된다.

    "실행됐으나 실패한 호출의 토큰도 그대로 집계한다"(docs/contracts.md "토큰 집계 경계").
    초안·판정 경로는
    이미 집계하므로, 같은 거절이 어느 단계에서 났느냐로 실비용 기록이 갈리면 안 된다.
    """
    client = _client(
        {
            INTENT_STAGE: [
                LLMCallError(
                    stage=INTENT_STAGE,
                    reason="refusal",
                    attempts=1,
                    input_tokens=9,
                    output_tokens=4,
                )
            ]
        }
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=None,
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert result.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert (result.input_tokens, result.output_tokens) == (9, 4)


def test_의도_형식_실패_뒤_전송_오류라도_과금된_토큰이_수집_합산에_남는다() -> None:
    """의도 해석의 재시도 루프도 앞선 시도의 과금분을 실어 보낸다.

    1회차가 200 으로 돌아왔지만 형식이 깨졌고(과금됨) 2회차가 거절로 죽으면, 1회차
    토큰은 예외에 실리지 않는 한 사라진다 — `judge.Judge.judge` 와 같은 형태로 막는다.
    """
    client = _client(
        {
            INTENT_STAGE: [
                LLMFormatError(
                    stage=INTENT_STAGE,
                    detail="형식 불일치",
                    input_tokens=10,
                    output_tokens=5,
                ),
                LLMCallError(
                    stage=INTENT_STAGE,
                    reason="refusal",
                    attempts=1,
                    input_tokens=9,
                    output_tokens=4,
                ),
            ]
        }
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=None,
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert result.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert (result.input_tokens, result.output_tokens) == (19, 9)


# ── 존재성 선검사 (DB 필요) ─────────────────────────────────────────────────


@pytest.mark.db
def test_존재성_선검사는_존재_여부만_판정한다(
    ro_conn: psycopg.Connection[DictRow], sample_order: dict[str, Any]
) -> None:
    assert order_exists(conn=ro_conn, order_no=sample_order["order_no"]) is True
    assert order_exists(conn=ro_conn, order_no=MISSING_ORDER_NO) is False


@pytest.mark.db
def test_없는_주문은_order_not_found_이고_text_to_SQL_에_진입하지_않는다(
    ro_conn: psycopg.Connection[DictRow], app_conn: psycopg.Connection[DictRow]
) -> None:
    client = _client({INTENT_STAGE: [_intent("order")]})

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=MISSING_ORDER_NO,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is EscalationReason.ORDER_NOT_FOUND
    assert client.calls_for(SQL_GENERATION_STAGE) == []
    assert result.sql_snapshots == ()
    assert result.evidence == ()


# ── 정책 벡터 검색 (DB 필요) ────────────────────────────────────────────────


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_정책_의도는_벡터_검색_결과를_근거로_삼는다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    client = _client({INTENT_STAGE: [_intent("policy")]})

    result = _collector(client, threshold=0.0).collect(
        inquiry_id=INQUIRY_ID,
        content="환불 신청 기간이 어떻게 되나요",
        order_no=None,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is None
    assert result.intent is IntentSource.POLICY
    assert result.evidence
    assert all(item.source is EvidenceSource.POLICY for item in result.evidence)
    assert all(item.id.startswith("policy:") for item in result.evidence)
    assert result.embedding_tokens > 0


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_임계값_미달_결과는_버려지고_근거가_없으면_no_evidence(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    client = _client({INTENT_STAGE: [_intent("policy")]})

    result = _collector(client, threshold=0.99).collect(
        inquiry_id=INQUIRY_ID,
        content="환불 신청 기간이 어떻게 되나요",
        order_no=None,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.evidence == ()
    assert result.escalation_reason is EscalationReason.NO_EVIDENCE


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_상위_k_개까지만_근거로_삼는다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    client = _client({INTENT_STAGE: [_intent("policy")]})

    result = _collector(client, threshold=0.0, top_k=2).collect(
        inquiry_id=INQUIRY_ID,
        content="환불 신청 기간이 어떻게 되나요",
        order_no=None,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert len(result.evidence) == 2


def test_SQL_원문은_표시용_content에만_남고_allowlist_근거는_결과_행만_쓴다() -> None:
    sql = "SELECT status FROM orders WHERE order_no = '<선검사 주문>' AND '010-9999-9999' <> ''"

    content, evidence_text = _sql_evidence_texts(
        sql=sql,
        rows=({"status": "배송중"},),
        pii_safe_output_columns=("status",),
    )

    assert sql in content
    assert "010-9999-9999" in content
    assert evidence_text == "1) status=배송중"


def test_SQL_계산_결과의_PII는_evidence_text에서_출처로_승인하지_않는다() -> None:
    sql = "SELECT concat('010', '-9999-', '9999') AS customer_phone FROM orders"

    content, evidence_text = _sql_evidence_texts(
        sql=sql,
        rows=({"customer_phone": "010-9999-9999"},),
        pii_safe_output_columns=(),
    )

    assert "customer_phone=010-9999-9999" in content
    assert "010-9999-9999" not in evidence_text


def test_출력_별칭_이름에_심은_PII도_allowlist_근거가_되지_않는다() -> None:
    """출력 이름은 LLM 이 정한다 — 값만 거르면 같은 공격이 이름 자리로 내려온다.

    `SELECT status AS "010-9999-9999"` 는 가드를 통과하고 별칭이 결과 dict 의 **키**가 되며,
    직접 컬럼으로 증명된 이름 집합에도 그 별칭이 들어온다. `key=value` 로 렌더되는 순간
    지어낸 번호가 근거 유래로 승인됐다.
    """
    sql = "SELECT status AS \"010-9999-9999\" FROM orders WHERE order_no = '<선검사 주문>'"

    content, evidence_text = _sql_evidence_texts(
        sql=sql,
        rows=({"010-9999-9999": "배송중"},),
        pii_safe_output_columns=("010-9999-9999",),
    )

    assert "010-9999-9999" in content
    assert "010-9999-9999" not in evidence_text


def test_SQL_계산_결과가_비PII면_L2용_evidence_text에_계속_남긴다() -> None:
    content, evidence_text = _sql_evidence_texts(
        sql="SELECT upper(status) AS status_label FROM orders",
        rows=({"status_label": "배송중"},),
        pii_safe_output_columns=(),
    )

    assert "status_label=배송중" in content
    assert evidence_text == "1) status_label=배송중"


@pytest.mark.parametrize(
    ("safe_name", "row_name"),
    [
        pytest.param("phone", "phone", id="따옴표_없는_별칭은_소문자"),
        pytest.param("Phone", "Phone", id="따옴표_별칭은_대소문자_보존"),
    ],
)
def test_DB가_노출하는_별칭과_직접_컬럼_provenance가_일치하면_PII를_보존한다(
    safe_name: str, row_name: str
) -> None:
    _, evidence_text = _sql_evidence_texts(
        sql="SELECT customer_phone FROM orders",
        rows=({row_name: "010-1234-5678"},),
        pii_safe_output_columns=(safe_name,),
    )

    assert evidence_text == f"1) {row_name}=010-1234-5678"


# ── text-to-SQL (DB 필요) ───────────────────────────────────────────────────


@pytest.mark.db
def test_채택된_SQL_근거는_순번_ID_와_스냅샷을_받는다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    order_no = sample_order["order_no"]
    client = _client(
        {
            INTENT_STAGE: [_intent("order")],
            SQL_GENERATION_STAGE: [
                _sql(f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'")
            ],
        }
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is None
    assert [item.id for item in result.evidence] == [f"sql:{INQUIRY_ID}:1"]
    assert result.evidence[0].source is EvidenceSource.SQL
    # 선검사 쿼리는 근거가 아니므로 스냅샷도 ID 도 받지 않는다.
    assert len(result.sql_snapshots) == 1
    snapshot = result.sql_snapshots[0]
    assert snapshot.sequence == 1
    assert snapshot.evidence_id == f"sql:{INQUIRY_ID}:1"
    assert order_no in snapshot.query_sql
    assert snapshot.result_rows and snapshot.result_rows[0]["order_no"] == order_no
    assert result.sql_failures == ()


@pytest.mark.db
def test_SQL_근거의_evidence_text_에_연락처_원문이_그대로_들어간다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    """마스킹하면 L1 PII allowlist 가 정상 에코를 오기각한다 (docs/business-rules.md "PII 규칙")."""
    order_no = sample_order["order_no"]
    client = _client(
        {
            INTENT_STAGE: [_intent("order")],
            SQL_GENERATION_STAGE: [
                _sql(
                    "SELECT order_no, customer_phone, customer_email FROM orders"
                    f" WHERE order_no = '{order_no}'"
                )
            ],
        }
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    evidence = result.evidence[0]
    assert sample_order["customer_phone"] in evidence.evidence_text
    assert sample_order["customer_email"] in evidence.evidence_text

    # 실제로 L1 이 정상 에코를 통과시키는지까지 확인한다 — 이게 이 요구사항의 존재 이유다.
    draft = {
        "claims": [
            {
                "text": f"등록된 연락처는 {sample_order['customer_phone']} 입니다.",
                "citation_ids": [evidence.id],
            }
        ]
    }
    assert evaluate_draft(raw_draft=draft, evidences=result.evidence).verdict is Verdict.PASS


@pytest.mark.db
def test_SQL_WHERE_리터럴은_표시용_content에만_남고_PII_allowlist에는_들어가지_않는다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    order_no = sample_order["order_no"]
    sql = f"SELECT status FROM orders WHERE order_no = '{order_no}' AND '010-9999-9999' <> ''"
    client = _client(
        {
            INTENT_STAGE: [_intent("order")],
            SQL_GENERATION_STAGE: [_sql(sql)],
        }
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    evidence = result.evidence[0]
    assert sql in evidence.content
    assert sql not in evidence.evidence_text
    assert "010-9999-9999" not in evidence.evidence_text

    fabricated = {
        "claims": [
            {
                "text": "연락처는 010-9999-9999입니다.",
                "citation_ids": [evidence.id],
            }
        ]
    }
    status_echo = {
        "claims": [
            {
                "text": f"주문 상태는 {sample_order['status']}입니다.",
                "citation_ids": [evidence.id],
            }
        ]
    }

    assert evaluate_draft(raw_draft=fabricated, evidences=result.evidence).reject_reasons == (
        RejectReason.PII_DETECTED,
    )
    assert evaluate_draft(raw_draft=status_echo, evidences=result.evidence).verdict is Verdict.PASS


@pytest.mark.db
def test_안전장치가_거부하면_피드백으로_1회_재시도해_성공한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    order_no = sample_order["order_no"]
    bad = "SELECT id, content FROM inquiries"
    good = f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'"
    client = _client(
        {INTENT_STAGE: [_intent("order")], SQL_GENERATION_STAGE: [_sql(bad), _sql(good)]}
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is None
    assert [item.id for item in result.evidence] == [f"sql:{INQUIRY_ID}:1"]
    assert len(result.sql_failures) == 1
    failure = result.sql_failures[0]
    assert failure.attempt_no == 1
    assert failure.kind is SqlFailureKind.GUARD_REJECTED
    assert failure.query_sql == bad

    retry_prompt = client.calls_for(SQL_GENERATION_STAGE)[1]["user"]
    assert bad in retry_prompt
    assert "unknown_table" in retry_prompt


@pytest.mark.db
def test_재시도까지_실패하면_sql_failed_이고_근거_ID_는_부여되지_않는다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    client = _client(
        {
            INTENT_STAGE: [_intent("order")],
            SQL_GENERATION_STAGE: [
                _sql("DELETE FROM orders"),
                _sql("SELECT content FROM policy_chunks"),
            ],
        }
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=sample_order["order_no"],
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is EscalationReason.SQL_FAILED
    assert result.evidence == ()
    assert result.sql_snapshots == ()
    assert [failure.attempt_no for failure in result.sql_failures] == [1, 2]
    assert [failure.query_sql for failure in result.sql_failures] == [
        "DELETE FROM orders",
        "SELECT content FROM policy_chunks",
    ]
    # 재시도 상한은 코드가 강제한다 — 3번째 생성 호출은 없다.
    assert len(client.calls_for(SQL_GENERATION_STAGE)) == 2


@pytest.mark.db
def test_주문_범위를_벗어난_SQL_은_거부되고_피드백으로_1회_재시도해_성공한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    """WHERE 없는 쿼리는 무관한 고객 50명의 연락처를 근거로 만든다 — 실행 전에 막아야 한다."""
    order_no = sample_order["order_no"]
    unscoped = "SELECT order_no, customer_name, customer_phone, customer_email FROM orders LIMIT 3"
    good = f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'"
    client = _client(
        {INTENT_STAGE: [_intent("order")], SQL_GENERATION_STAGE: [_sql(unscoped), _sql(good)]}
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is None
    # 거부는 기존 경로를 탄다 — guard_rejected → 재시도 1회.
    assert len(result.sql_failures) == 1
    failure = result.sql_failures[0]
    assert failure.kind is SqlFailureKind.GUARD_REJECTED
    assert failure.query_sql == unscoped
    assert "order_scope" in failure.error

    retry_prompt = client.calls_for(SQL_GENERATION_STAGE)[1]["user"]
    assert "order_scope" in retry_prompt
    assert order_no in retry_prompt

    # 채택된 근거는 그 주문 1건뿐이다.
    assert len(result.sql_snapshots) == 1
    assert {row["order_no"] for row in result.sql_snapshots[0].result_rows} == {order_no}


@pytest.mark.db
def test_외부_조인_우회는_거부되고_피드백으로_1회_재시도해_성공한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    """외부 조인의 ON 절은 보존측을 거르지 않아 무관한 주문 50건이 근거가 된다.

    거부 사유 코드는 **가드 내부 코드**로 남고, 새 인계 사유를 만들지 않는다
    (재시도 상한 1회 그대로).
    """
    order_no = sample_order["order_no"]
    bypass = (
        "SELECT o.order_no, o.customer_name, o.customer_phone FROM orders o"
        f" LEFT JOIN (SELECT 1 AS x) d ON o.order_no = '{order_no}'"
    )
    good = f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'"
    client = _client(
        {INTENT_STAGE: [_intent("order")], SQL_GENERATION_STAGE: [_sql(bypass), _sql(good)]}
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is None
    assert len(result.sql_failures) == 1
    failure = result.sql_failures[0]
    assert failure.kind is SqlFailureKind.GUARD_REJECTED
    assert failure.query_sql == bypass
    assert "unsupported_join" in failure.error

    retry_prompt = client.calls_for(SQL_GENERATION_STAGE)[1]["user"]
    assert "unsupported_join" in retry_prompt
    assert "INNER JOIN" in retry_prompt

    # 채택된 근거는 선검사를 통과한 주문 1건뿐이다.
    assert len(result.sql_snapshots) == 1
    assert {row["order_no"] for row in result.sql_snapshots[0].result_rows} == {order_no}


@pytest.mark.db
def test_다른_주문번호를_계속_가리키면_sql_failed_로_인계한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    """재시도 상한은 1회 그대로다 — 새 인계 사유를 만들지 않는다."""
    order_no = sample_order["order_no"]
    other = "SELECT order_no, customer_phone FROM orders WHERE order_no = 'ORD-20991231-9999'"
    both = (
        f"SELECT order_no, customer_phone FROM orders WHERE order_no = '{order_no}'"
        " OR order_no = 'ORD-20991231-9999'"
    )
    client = _client(
        {INTENT_STAGE: [_intent("order")], SQL_GENERATION_STAGE: [_sql(other), _sql(both)]}
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is EscalationReason.SQL_FAILED
    assert result.evidence == ()
    assert result.sql_snapshots == ()
    assert [failure.attempt_no for failure in result.sql_failures] == [1, 2]
    assert all(failure.kind is SqlFailureKind.GUARD_REJECTED for failure in result.sql_failures)
    assert all("order_scope" in failure.error for failure in result.sql_failures)
    # 세 번째 생성 호출은 없다.
    assert len(client.calls_for(SQL_GENERATION_STAGE)) == 2


@pytest.mark.db
def test_pg_sleep_은_실행되지_않고_안전장치가_먼저_거부한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    """DB 를 건드리기 전에 거부되므로 이 테스트는 2초를 자지 않는다."""
    order_no = sample_order["order_no"]
    sleeping = f"SELECT order_no FROM orders WHERE order_no = '{order_no}' AND pg_sleep(2) IS NULL"
    good = f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'"
    client = _client(
        {INTENT_STAGE: [_intent("order")], SQL_GENERATION_STAGE: [_sql(sleeping), _sql(good)]}
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is None
    assert result.sql_failures[0].kind is SqlFailureKind.GUARD_REJECTED
    assert "forbidden_function" in result.sql_failures[0].error


@pytest.mark.db
def test_실행_오류도_SQL_실패_경로를_탄다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    """안전장치는 통과하지만 DB 가 거부하는 쿼리 — 검증만으로는 못 잡는 층이다."""
    order_no = sample_order["order_no"]
    broken = f"SELECT order_no FROM orders WHERE order_no = '{order_no}' AND quantity = 'abc'"
    good = f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'"
    client = _client(
        {INTENT_STAGE: [_intent("order")], SQL_GENERATION_STAGE: [_sql(broken), _sql(good)]}
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is None
    assert result.sql_failures[0].kind is SqlFailureKind.EXECUTION_ERROR
    assert result.sql_failures[0].error
    assert [item.id for item in result.evidence] == [f"sql:{INQUIRY_ID}:1"]


@pytest.mark.db
def test_유효_SQL_생성_실패도_SQL_실패_경로다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    broken = LLMFormatError(stage=SQL_GENERATION_STAGE, detail="JSON 아님", raw_text="SELECT")
    client = _client({INTENT_STAGE: [_intent("order")], SQL_GENERATION_STAGE: [broken, broken]})

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=sample_order["order_no"],
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is EscalationReason.SQL_FAILED
    assert all(failure.kind is SqlFailureKind.GENERATION_FAILED for failure in result.sql_failures)
    assert result.sql_failures[0].query_sql is None


@pytest.mark.db
def test_SQL_생성의_전송_오류는_llm_call_failed_다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    """전송 오류는 SQL 실패 경로가 아니라 공통 실패 정책 소관이다
    (docs/standards.md "재시도 상한")."""
    client = _client(
        {
            INTENT_STAGE: [_intent("order")],
            SQL_GENERATION_STAGE: [
                LLMCallError(stage=SQL_GENERATION_STAGE, reason="transport_error", attempts=2)
            ],
        }
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=sample_order["order_no"],
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert result.failed_stage == SQL_GENERATION_STAGE


# ── 정상 실행 0건 (실패가 아니다) ───────────────────────────────────────────


@pytest.mark.db
def test_SQL_정상_실행_0건은_order_단독이면_no_evidence(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    order_no = sample_order["order_no"]
    client = _client(
        {
            INTENT_STAGE: [_intent("order")],
            SQL_GENERATION_STAGE: [
                _sql(
                    f"SELECT order_no FROM orders WHERE order_no = '{order_no}'"
                    " AND status = '그런상태없음'"
                )
            ],
        }
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    # 실패가 아니라 "주문 소스 근거 0건" 이다 — order_not_found 도 sql_failed 도 아니다.
    assert result.escalation_reason is EscalationReason.NO_EVIDENCE
    assert result.sql_failures == ()
    assert result.sql_snapshots == ()
    assert len(client.calls_for(SQL_GENERATION_STAGE)) == 1


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_both_에서_SQL_0건이면_정책_근거만으로_진행한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    order_no = sample_order["order_no"]
    client = _client(
        {
            INTENT_STAGE: [_intent("both")],
            SQL_GENERATION_STAGE: [
                _sql(
                    f"SELECT order_no FROM orders WHERE order_no = '{order_no}'"
                    " AND status = '그런상태없음'"
                )
            ],
        }
    )

    result = _collector(client, threshold=0.0).collect(
        inquiry_id=INQUIRY_ID,
        content="환불 신청 기간이 어떻게 되나요",
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is None
    assert result.evidence
    assert all(item.source is EvidenceSource.POLICY for item in result.evidence)


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_both_에서_주문_측_구조적_실패는_정책_근거가_있어도_인계가_우선한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    client = _client(
        {
            INTENT_STAGE: [_intent("both")],
            SQL_GENERATION_STAGE: [
                _sql("DROP TABLE orders"),
                _sql("UPDATE orders SET status = 'x'"),
            ],
        }
    )

    result = _collector(client, threshold=0.0).collect(
        inquiry_id=INQUIRY_ID,
        content="환불 신청 기간이 어떻게 되나요",
        order_no=sample_order["order_no"],
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is EscalationReason.SQL_FAILED
    # 그 시점까지 모은 근거는 그대로 실려 나간다 (처리 기록·감사용).
    assert result.evidence
    assert all(item.source is EvidenceSource.POLICY for item in result.evidence)


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_both_에서_주문번호가_없으면_정책_근거가_있어도_missing_order_ref(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    client = _client({INTENT_STAGE: [_intent("both")]})

    result = _collector(client, threshold=0.0).collect(
        inquiry_id=INQUIRY_ID,
        content="환불 신청 기간이 어떻게 되나요",
        order_no=None,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.escalation_reason is EscalationReason.MISSING_ORDER_REF
    assert result.evidence


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_모든_소스_근거_0건이면_no_evidence(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    order_no = sample_order["order_no"]
    client = _client(
        {
            INTENT_STAGE: [_intent("both")],
            SQL_GENERATION_STAGE: [
                _sql(
                    f"SELECT order_no FROM orders WHERE order_no = '{order_no}'"
                    " AND status = '그런상태없음'"
                )
            ],
        }
    )

    result = _collector(client, threshold=0.99).collect(
        inquiry_id=INQUIRY_ID,
        content="환불 신청 기간이 어떻게 되나요",
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.evidence == ()
    assert result.escalation_reason is EscalationReason.NO_EVIDENCE


# ── 실행 시간 상한 (read-only 커넥션의 statement_timeout) ───────────────────


@pytest.mark.db
def test_read_only_커넥션에_statement_timeout_이_걸려_있다(
    ro_conn: psycopg.Connection[DictRow], db_settings: Settings
) -> None:
    """가드가 함수를 막아도 실행 시간까지 코드가 예측할 수는 없다 — DB 에 직접 물어본다.

    `0` 은 무제한이다. 무제한이면 쿼리 하나가 워커를 몇 분씩 묶는다.
    """
    row = ro_conn.execute("SHOW statement_timeout").fetchone()

    assert row is not None
    assert row["statement_timeout"] not in ("0", "")
    assert db_settings.sql_statement_timeout_ms > 0


@pytest.mark.db
def test_상한을_넘긴_쿼리는_실제로_끊긴다(db_settings: Settings) -> None:
    """상한이 **동작하는지**까지 확인한다. 250ms 상한 + 2초 sleep 이라 매달리지 않는다."""
    settings = db_settings.model_copy(update={"sql_statement_timeout_ms": 250})

    with readonly_connect(settings=settings) as conn:
        assert conn.execute("SHOW statement_timeout").fetchone() == {"statement_timeout": "250ms"}
        with pytest.raises(psycopg.errors.QueryCanceled):
            conn.execute("SELECT pg_sleep(2)")


# ── 토큰 집계 ───────────────────────────────────────────────────────────────


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_토큰은_생성과_임베딩을_분리해_집계한다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    order_no = sample_order["order_no"]
    client = _client(
        {
            INTENT_STAGE: [_intent("both", input_tokens=100, output_tokens=5)],
            SQL_GENERATION_STAGE: [
                _sql(
                    f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'",
                    input_tokens=200,
                    output_tokens=20,
                )
            ],
        }
    )

    result = _collector(client, threshold=0.0).collect(
        inquiry_id=INQUIRY_ID,
        content="환불 신청 기간이 어떻게 되나요",
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    # 생성 토큰은 의도 분류 + SQL 생성의 합이다.
    assert (result.input_tokens, result.output_tokens) == (300, 25)
    # 임베딩 토큰은 생성 토큰과 섞이지 않는다 (건당 비용을 따로 산출한다).
    assert result.embedding_tokens == len("환불 신청 기간이 어떻게 되나요")


@pytest.mark.db
def test_임베딩은_정책_검색이_필요할_때만_호출된다(
    app_conn: psycopg.Connection[DictRow],
    ro_conn: psycopg.Connection[DictRow],
    sample_order: dict[str, Any],
) -> None:
    order_no = sample_order["order_no"]
    client = _client(
        {
            INTENT_STAGE: [_intent("order")],
            SQL_GENERATION_STAGE: [
                _sql(f"SELECT order_no, status FROM orders WHERE order_no = '{order_no}'")
            ],
        }
    )

    result = _collector(client).collect(
        inquiry_id=INQUIRY_ID,
        content=INQUIRY,
        order_no=order_no,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    # order 단독 의도는 정책 검색을 하지 않으므로 임베딩 호출 자체가 없다.
    assert result.embedding_tokens == 0
    assert result.escalation_reason is None
    assert INQUIRY_EMBEDDING_STAGE == "inquiry_embedding"
    # 재작성도 정책 검색의 일부다 — 검색하지 않는 의도에서는 부르지 않는다(과금 0).
    assert client.calls_for(QUERY_REWRITE_STAGE) == []
    assert (result.retrieval_input_tokens, result.retrieval_output_tokens) == (0, 0)


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_검색_토큰은_생성_임베딩_어디에도_합산되지_않는다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """계열이 섞이면 초안을 만들지도 않은 문의가 초안 생성 토큰을 쓴 것으로 찍힌다."""
    content = "환불 언제까지 되나요"
    client = _client(
        {
            INTENT_STAGE: [_intent("policy", input_tokens=100, output_tokens=5)],
            QUERY_REWRITE_STAGE: [_rewrite("환불 신청 기간", input_tokens=40, output_tokens=8)],
        }
    )

    result = _collector(client, threshold=0.0).collect(
        inquiry_id=INQUIRY_ID,
        content=content,
        order_no=None,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    # 생성 합산은 의도 해석 몫 그대로다 — 재작성 40/8 이 여기 들어오지 않는다.
    assert (result.input_tokens, result.output_tokens) == (100, 5)
    assert (result.retrieval_input_tokens, result.retrieval_output_tokens) == (40, 8)
    # 임베딩 계열도 따로다 — 두 질의를 실었으므로 두 문자열의 길이 합이다.
    assert result.embedding_tokens == len(content) + len("환불 신청 기간")
    assert result.retrieval_fallback_reason is None


# ── 질의 재작성 (DB 필요) ───────────────────────────────────────────────────


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_원문과_재작성문을_둘_다_검색해_합집합으로_모은다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """재작성이 주제를 옮겨도 원문이 안전망이다 (spec §5-2).

    두 질의를 따로 돌린 결과의 합집합이 실제 채택 근거와 같은지 본다 — 어느 한쪽만 쓰면
    이 등식이 깨진다.
    """
    original = "환불 신청 기간이 어떻게 되나요"
    rewritten = "교환 신청 조건"

    def adopted(content: str) -> set[str]:
        """재작성을 끄고 그 질의 하나만으로 검색한 결과."""
        result = _collector(
            _client({INTENT_STAGE: [_intent("policy")]}),
            threshold=0.0,
            top_k=3,
            query_rewrite_enabled=False,
        ).collect(
            inquiry_id=INQUIRY_ID,
            content=content,
            order_no=None,
            app_conn=app_conn,
            readonly_conn=ro_conn,
        )
        return {item.id for item in result.evidence}

    only_original = adopted(original)
    only_rewritten = adopted(rewritten)

    union = _collector(
        _client({INTENT_STAGE: [_intent("policy")], QUERY_REWRITE_STAGE: [_rewrite(rewritten)]}),
        threshold=0.0,
        top_k=3,
    ).collect(
        inquiry_id=INQUIRY_ID,
        content=original,
        order_no=None,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )
    merged = {item.id for item in union.evidence}

    # 두 질의가 실제로 다른 조항을 물어야 합집합이 관측된다(양성 대조).
    assert only_rewritten - only_original
    assert merged <= only_original | only_rewritten
    assert merged & only_rewritten and merged & only_original
    # 합집합이어도 채택 상한은 그대로다 — 재작성이 후보를 2배로 늘리지 않는다.
    assert len(union.evidence) == 3


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_재작성이_원문과_같으면_임베딩을_한_번만_태운다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """픽스처 계약이 허용하는 산출이다(`rewritten` == `original`) — 같은 벡터를 두 번 사지 않는다."""
    content = "환불 신청 기간이 어떻게 되나요"
    client = _client({INTENT_STAGE: [_intent("policy")], QUERY_REWRITE_STAGE: [_rewrite(content)]})

    result = _collector(client, threshold=0.0).collect(
        inquiry_id=INQUIRY_ID,
        content=content,
        order_no=None,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    # `LexicalEmbeddingClient` 는 문자 수를 토큰으로 센다 — 두 번 실렸으면 2배가 된다.
    assert result.embedding_tokens == len(content)


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_재작성_실패는_인계가_아니라_폴백이고_기록에_남는다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """검색은 검증이 아니라 재료 수집이다 — 재료가 덜 좋아진 것은 검증 없이 내보내는 것과 다르다.

    같은 문의를 재작성 성공 조건에서도 돌려 **폴백이 결과를 망치지 않았다**를 함께 본다.
    """
    content = "환불 신청 기간이 어떻게 되나요"
    client = _client(
        {
            INTENT_STAGE: [_intent("policy")],
            QUERY_REWRITE_STAGE: [
                LLMCallError(stage=QUERY_REWRITE_STAGE, reason="연결 실패", attempts=2)
            ],
        }
    )

    result = _collector(client, threshold=0.0).collect(
        inquiry_id=INQUIRY_ID,
        content=content,
        order_no=None,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    # 인계가 아니다 — 원문 질의로 그대로 진행해 근거를 모았다.
    assert result.escalation_reason is None
    assert result.failed_stage is None
    assert result.evidence
    # 조용하지도 않다 — 사유가 수집 결과에 실려 처리 기록·리포트로 흘러간다.
    assert result.retrieval_fallback_reason is not None
    assert "전송 오류" in result.retrieval_fallback_reason
    # 폴백은 사이클 2 동작과 같다 — 원문만 검색한 것과 임베딩 비용이 같다.
    assert result.embedding_tokens == len(content)
