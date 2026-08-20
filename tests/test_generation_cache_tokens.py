"""생성·검색 계열의 캐시 적중 토큰 — **계열별 별도 칸**으로 리포트까지 간다 (외부 호출 없음).

**이 파일이 지키는 것은 달러가 실측이라는 주장의 전제다.** 생성 쪽 클라이언트가 입력·출력
토큰만 읽고 캐시 적중분을 읽지 않으면, 환산이 전부 정가를 곱해 **실제 청구액이 아니라 그
이상일 수 없는 값**(상한)이 된다. 판정 계열에는 같은 배선이 이미 있고, 이 파일은 그것을 본으로
삼아 생성·검색 두 계열에 같은 규칙을 세운다.

고정하는 계약 다섯:

1. **응답의 캐시 계열 토큰을 읽어 별도 칸으로 싣는다.** 입력 칸에 접지 않는다 — 접으면 같은
   은폐가 반대 방향으로 생긴다(적중이 "입력 토큰 감소"로 위장하거나, 이미 포함된 값이 두 번
   세어진다).
2. **칸은 계열별이다 — 생성과 검색 두 쌍.** 두 계열이 **같은 클라이언트**를 지나므로 한 쌍으로
   묶으면 어느 계열의 캐시였는지 되짚을 수 없다. 계열 자체는 늘어나지 않는다(생성·임베딩·
   판정·검색 넷 그대로).
3. **없는 값은 0 이 아니라 미측정이다.** 캐시 계열을 싣지 않는 응답에서 0 을 적으면 "캐시가
   0 토큰 적중했다"와 "캐시를 잰 적이 없다"가 같은 값이 된다.
4. **응답 계약은 변하지 않는다.** 캐시 칸은 평가 리포트에만 있다 — 판정 계열의 캐시 칸이
   리포트에만 있는 것과 같은 경계다.
5. **단가 문서가 상한과 실측을 가른다.** 캐시를 재지 않고 낸 커밋된 산출물은 영구히 상한이고,
   그 구분이 문서에 서 있지 않으면 다음 사람이 두 값을 같은 자로 읽는다.

OpenAI 응답의 캐시 계열은 `usage.input_tokens_details` 안에 있고, 그 묶음은 **입력 토큰의
내역**이다 — 즉 캐시 칸의 값은 `input_tokens` 에 이미 **포함**돼 있다. Anthropic 은 반대로
`input_tokens` 가 캐시 적중분을 **제외한** 값이다. 두 계열의 포함 관계가 반대라는 사실이
달러 환산의 전제이므로, 여기서 "입력 칸이 그대로 남는다"를 못박는다.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import psycopg
import pytest
from psycopg.rows import DictRow

from reply_gate.contracts import EscalationReason, IntentSource
from reply_gate.draft import DRAFT_STAGE, DraftGenerator
from reply_gate.evaluation import (
    measure_pipeline_agreement,
    render_markdown,
    report_to_json,
)
from reply_gate.evidence import INTENT_STAGE, classify_intent, generate_sql
from reply_gate.llm import (
    JsonCompletion,
    LLMCallError,
    LLMFormatError,
    OpenAIGenerationClient,
)
from reply_gate.policy_index import index_policy_documents, load_policy_documents
from reply_gate.query_rewrite import QUERY_REWRITE_STAGE, rewrite_query
from reply_gate.testing import LexicalEmbeddingClient
from tests.test_api import TOKEN_KEYS
from tests.test_evaluation import GOLDEN, ScriptedPipeline, _processed, _report
from tests.test_pipeline import (
    POLICY_EVIDENCE,
    StubCollector,
    citing_draft,
    collection,
    draft_completion,
    intent_completion,
    live_pipeline,
    pipeline_with,
    run,
    run_live,
    scripted_client,
)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PRICING_DOC = _ROOT / "docs" / "tracking" / "pricing.md"

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}},
    "required": ["verdict"],
    "additionalProperties": False,
}

INQUIRY = "환불 규정이 어떻게 되나요?"


# ── 대역 ────────────────────────────────────────────────────────────────────


class _RecordingCalls:
    """SDK 엔드포인트 대역 — 정해진 결과를 순서대로 돌려준다."""

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
    input_tokens: int = 1200,
    output_tokens: int = 7,
    cached: int | None = None,
    cache_write: int | None = None,
) -> SimpleNamespace:
    """OpenAI usage 대역. **캐시 칸에 `None` 을 주면 그 필드가 아예 없는 응답**이 된다.

    캐싱이 걸리지 않은 호출·옛 SDK 응답에는 내역 묶음이 없거나 칸이 비어 있다. 그때 0 으로
    접으면 "적중 0"과 "잰 적이 없다"가 구분되지 않는다.
    """
    fields: dict[str, Any] = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    details: dict[str, Any] = {}
    if cached is not None:
        details["cached_tokens"] = cached
    if cache_write is not None:
        details["cache_write_tokens"] = cache_write
    if details:
        fields["input_tokens_details"] = SimpleNamespace(**details)
    return SimpleNamespace(**fields)


def _openai_response(
    text: str, *, refusal: str | None = None, usage: SimpleNamespace | None = None
) -> SimpleNamespace:
    if refusal is not None:
        output = [SimpleNamespace(content=[SimpleNamespace(type="refusal", refusal=refusal)])]
    else:
        output = [SimpleNamespace(content=[SimpleNamespace(type="output_text", text=text)])]
    return SimpleNamespace(
        output=output, output_text=text, usage=_usage() if usage is None else usage
    )


def _openai_client(outcomes: list[Any]) -> OpenAIGenerationClient:
    from pydantic import SecretStr

    calls = _RecordingCalls(outcomes)
    fake_sdk = cast(Any, SimpleNamespace(responses=calls))
    return OpenAIGenerationClient(api_key=SecretStr("test"), model="gpt-5.6-terra", client=fake_sdk)


class _Client:
    """`GenerationClient` 대역 — 정해진 산출 하나를 돌려주거나 예외를 던진다."""

    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome

    def complete_json(self, **kwargs: Any) -> Any:
        del kwargs
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


def _completion(
    data: Any,
    *,
    input_tokens: int = 100,
    output_tokens: int = 10,
    cache_creation: int | None = None,
    cache_read: int | None = None,
) -> JsonCompletion:
    return JsonCompletion(
        data=data,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )


# ── 계약 1 — 호출 래퍼가 캐시 계열을 읽고, 입력 칸에 접지 않는다 ─────────────


def test_생성_래퍼가_캐시_계열_토큰을_별도_칸으로_싣는다() -> None:
    """읽지 않으면 환산이 전부 정가를 곱해 달러가 상한이 된다."""
    client = _openai_client(
        [_openai_response('{"verdict": "pass"}', usage=_usage(cached=800, cache_write=90))]
    )

    result = client.complete_json(stage=DRAFT_STAGE, system="s", user="u", schema=SCHEMA)

    assert result.cache_read_input_tokens == 800
    assert result.cache_creation_input_tokens == 90


def test_캐시_적중은_생성_입력_토큰_칸에_접히지_않는다() -> None:
    """입력 칸에 접으면 같은 은폐가 반대 방향으로 생긴다 — 옛 산출물과 대조도 끊긴다."""
    client = _openai_client(
        [
            _openai_response(
                '{"verdict": "pass"}', usage=_usage(input_tokens=1200, cached=800, cache_write=90)
            )
        ]
    )

    result = client.complete_json(stage=DRAFT_STAGE, system="s", user="u", schema=SCHEMA)

    # 벤더가 보고한 값 그대로다. 빼지도(1200-800) 더하지도(1200+800) 않는다.
    assert result.input_tokens == 1200


def test_내역_묶음이_없는_응답은_0_이_아니라_미측정이다() -> None:
    client = _openai_client([_openai_response('{"verdict": "pass"}', usage=_usage())])

    result = client.complete_json(stage=DRAFT_STAGE, system="s", user="u", schema=SCHEMA)

    assert result.cache_read_input_tokens is None
    assert result.cache_creation_input_tokens is None


def test_칸_하나만_실린_응답은_나머지_칸을_0_으로_채우지_않는다() -> None:
    client = _openai_client([_openai_response('{"verdict": "pass"}', usage=_usage(cached=0))])

    result = client.complete_json(stage=DRAFT_STAGE, system="s", user="u", schema=SCHEMA)

    # 0 은 **측정값**이고, 보고되지 않은 칸은 미측정이다 — 둘을 같은 값으로 적지 않는다.
    assert result.cache_read_input_tokens == 0
    assert result.cache_creation_input_tokens is None


def test_거절_실패에도_생성_캐시_계열이_실린다() -> None:
    """200 으로 온 응답이라 이미 과금됐다 — 실패에 실어 올린다(토큰과 같은 규칙)."""
    client = _openai_client(
        [
            _openai_response(
                "", refusal="안전 정책", usage=_usage(input_tokens=40, cached=30, cache_write=0)
            )
        ]
    )

    with pytest.raises(LLMCallError) as excinfo:
        client.complete_json(stage=DRAFT_STAGE, system="s", user="u", schema=SCHEMA)

    assert excinfo.value.cache_read_input_tokens == 30
    assert excinfo.value.cache_creation_input_tokens == 0


def test_형식_오류에도_생성_캐시_계열이_실린다() -> None:
    client = _openai_client(
        [_openai_response("JSON 이 아니다", usage=_usage(input_tokens=40, cached=30))]
    )

    with pytest.raises(LLMFormatError) as excinfo:
        client.complete_json(stage=DRAFT_STAGE, system="s", user="u", schema=SCHEMA)

    assert excinfo.value.cache_read_input_tokens == 30


# ── 계약 2 — 단계별 산출이 캐시 칸을 그대로 들고 나온다 ──────────────────────


def test_의도_해석_산출이_캐시_칸을_들고_나온다() -> None:
    result = classify_intent(
        client=cast(
            Any, _Client(_completion({"source": "policy"}, cache_read=11, cache_creation=2))
        ),
        inquiry=INQUIRY,
    )

    assert result.cache_read_input_tokens == 11
    assert result.cache_creation_input_tokens == 2


def test_의도_해석은_형식_재시도의_캐시_칸을_합산한다() -> None:
    """앞선 시도의 비용을 버리지 않는 규칙 — 토큰·전송 수·경과와 같은 자리다."""

    class _Twice:
        def __init__(self) -> None:
            self._queue = [
                _completion({"source": "없는값"}, cache_read=5),
                _completion({"source": "policy"}, cache_read=7),
            ]

        def complete_json(self, **kwargs: Any) -> Any:
            del kwargs
            return self._queue.pop(0)

    result = classify_intent(client=cast(Any, _Twice()), inquiry=INQUIRY)

    assert result.cache_read_input_tokens == 12


def test_의도_해석이_예외로_죽어도_앞선_시도의_캐시_칸이_따라_올라온다() -> None:
    class _ThenBoom:
        def __init__(self) -> None:
            self._queue: list[Any] = [
                _completion({"source": "없는값"}, cache_read=5),
                LLMCallError(
                    stage=INTENT_STAGE,
                    reason="transport_error",
                    attempts=2,
                    input_tokens=1,
                    output_tokens=1,
                    cache_read_input_tokens=3,
                ),
            ]

        def complete_json(self, **kwargs: Any) -> Any:
            del kwargs
            outcome = self._queue.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    with pytest.raises(LLMCallError) as excinfo:
        classify_intent(client=cast(Any, _ThenBoom()), inquiry=INQUIRY)

    assert excinfo.value.cache_read_input_tokens == 8


def test_조회문_생성_산출이_캐시_칸을_들고_나온다() -> None:
    result = generate_sql(
        client=cast(Any, _Client(_completion({"sql": "SELECT 1"}, cache_read=13))),
        inquiry=INQUIRY,
        order_no="ORD-20260315-0001",
        max_rows=50,
    )

    assert result.cache_read_input_tokens == 13


def test_초안_생성_산출이_캐시_칸을_들고_나온다() -> None:
    generator = DraftGenerator(
        client=cast(
            Any,
            _Client(_completion({"claims": []}, cache_read=17, cache_creation=4)),
        )
    )

    generation = generator.generate(inquiry=INQUIRY, evidence=(POLICY_EVIDENCE,))

    assert generation.cache_read_input_tokens == 17
    assert generation.cache_creation_input_tokens == 4


def test_재작성_산출이_캐시_칸을_들고_나온다() -> None:
    outcome = rewrite_query(
        client=cast(Any, _Client(_completion({"rewritten": "환불 규정"}, cache_read=19))),
        inquiry=INQUIRY,
    )

    assert outcome.cache_read_input_tokens == 19


def test_재작성_폴백에도_캐시_칸이_실린다() -> None:
    """실패한 호출의 토큰도 실비용이다 — 캐시 칸도 같은 자격이다."""
    outcome = rewrite_query(
        client=cast(Any, _Client(_completion({"rewritten": ""}, cache_read=21))),
        inquiry=INQUIRY,
    )

    assert outcome.fell_back
    assert outcome.cache_read_input_tokens == 21


# ── 계약 3 — 계열 분리: 생성과 검색은 같은 클라이언트를 지나도 갈린다 ────────


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
def test_같은_클라이언트를_지나도_생성과_검색_캐시가_갈린다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """**한 쌍으로 묶으면 이 검사가 깨진다** — 어느 계열의 캐시였는지 되짚을 수 없어진다.

    재작성 클라이언트는 생성 클라이언트 그대로다(`rewrite_client=generation_client`).
    그래서 계열을 가르는 것은 provider 가 아니라 **집계하는 자리**다.
    """
    client = scripted_client(
        {
            INTENT_STAGE: [intent_completion("policy")],
            QUERY_REWRITE_STAGE: [_completion({"rewritten": "환불 규정"}, cache_read=500)],
            DRAFT_STAGE: [citing_draft()],
        }
    )
    processed = run_live(live_pipeline(client), app_conn, ro_conn, content=INQUIRY)

    # 초안 대역(`citing_draft`)은 캐시 칸을 싣지 않으므로 생성 쪽 합계는 의도 해석분뿐이다.
    assert processed.retrieval_cache_read_tokens == 500
    assert processed.generation_cache_read_tokens is None


def test_생성_계열_캐시는_수집과_초안을_합산한다() -> None:
    pipeline = pipeline_with(
        collector=StubCollector(
            replace(
                collection(evidence=(POLICY_EVIDENCE,)),
                generation_cache_read_tokens=40,
                generation_cache_creation_tokens=2,
            )
        ),
        client=scripted_client(
            {
                DRAFT_STAGE: [
                    draft_completion(
                        {
                            "claims": [
                                {"text": "안내드립니다.", "citation_ids": [POLICY_EVIDENCE.id]}
                            ]
                        },
                    )
                ]
            }
        ),
    )
    # 초안 대역이 캐시 칸을 싣지 않아도 수집분은 그대로 살아 있어야 한다.
    processed = run(pipeline)

    assert processed.generation_cache_read_tokens == 40
    assert processed.generation_cache_creation_tokens == 2


def test_초안_생성이_예외로_죽어도_그_호출의_캐시_칸이_남는다() -> None:
    pipeline = pipeline_with(
        collector=StubCollector(collection(evidence=(POLICY_EVIDENCE,))),
        client=scripted_client(
            {
                DRAFT_STAGE: [
                    LLMCallError(
                        stage=DRAFT_STAGE,
                        reason="refusal",
                        attempts=1,
                        input_tokens=5,
                        output_tokens=0,
                        cache_read_input_tokens=33,
                    )
                ]
            }
        ),
    )
    processed = run(pipeline)

    assert processed.escalation_reason is EscalationReason.LLM_CALL_FAILED
    assert processed.generation_cache_read_tokens == 33


def test_캐시를_보고하지_않은_처리는_0_이_아니라_미측정이다() -> None:
    pipeline = pipeline_with(
        collector=StubCollector(collection(evidence=(POLICY_EVIDENCE,))),
        client=scripted_client({DRAFT_STAGE: [citing_draft()]}),
    )
    processed = run(pipeline)

    assert processed.generation_cache_read_tokens is None
    assert processed.generation_cache_creation_tokens is None
    assert processed.retrieval_cache_read_tokens is None
    assert processed.retrieval_cache_creation_tokens is None


# ── 계약 4 — 평가 리포트: 계열별 네 칸이 두 표면에 같은 값으로 실린다 ────────


def _agreement(results: list[Any]) -> Any:
    return measure_pipeline_agreement(
        cases=GOLDEN[: len(results)],
        pipeline=ScriptedPipeline(results),
        app_conn=cast(psycopg.Connection[DictRow], None),
        readonly_conn=cast(psycopg.Connection[DictRow], None),
    )


def _cached_processed(**kwargs: Any) -> Any:
    return replace(_processed(input_tokens=100, output_tokens=10), **kwargs)


def test_리포트_JSON_이_계열별_캐시_칸_넷을_싣는다() -> None:
    agreement = _agreement(
        [
            _cached_processed(generation_cache_read_tokens=100, retrieval_cache_read_tokens=7),
            _cached_processed(generation_cache_read_tokens=50, generation_cache_creation_tokens=9),
        ]
    )
    tokens = report_to_json(_report(pipeline=agreement))["measurement_2_pipeline_agreement"][
        "tokens"
    ]

    assert tokens["generation_cache_read_total"] == 150
    assert tokens["generation_cache_creation_total"] == 9
    assert tokens["retrieval_cache_read_total"] == 7
    # 검색 쪽 write 를 보고한 케이스가 하나도 없다 — 0 이 아니라 미측정이다.
    assert tokens["retrieval_cache_creation_total"] is None


def test_한_건도_보고하지_않은_세트의_캐시_칸은_0_이_아니라_미측정이다() -> None:
    agreement = _agreement([_cached_processed()])
    tokens = report_to_json(_report(pipeline=agreement))["measurement_2_pipeline_agreement"][
        "tokens"
    ]

    assert tokens["generation_cache_read_total"] is None
    assert tokens["generation_cache_creation_total"] is None
    assert tokens["retrieval_cache_read_total"] is None
    assert tokens["retrieval_cache_creation_total"] is None


def test_케이스별_캐시_칸도_리포트_JSON_에_남는다() -> None:
    agreement = _agreement([_cached_processed(generation_cache_read_tokens=100)])
    outcome = report_to_json(_report(pipeline=agreement))["measurement_2_pipeline_agreement"][
        "outcomes"
    ][0]

    assert outcome["generation_cache_read_tokens"] == 100
    assert outcome["retrieval_cache_read_tokens"] is None


_CACHE_LINE = re.compile(
    r"^- (?P<family>생성|검색) 계열 프롬프트 캐시[^:]*: (?P<body>.+)$", re.MULTILINE
)


def _cache_lines(markdown: str) -> dict[str, str]:
    return {match.group("family"): match.group("body") for match in _CACHE_LINE.finditer(markdown)}


def test_사람이_읽는_줄과_리포트_JSON_이_같은_값을_적는다() -> None:
    """두 표면이 갈린 전례가 있다 — 같은 원본에서 같은 값을 적는지 못박는다."""
    agreement = _agreement(
        [_cached_processed(generation_cache_read_tokens=123, retrieval_cache_read_tokens=45)]
    )
    report = _report(pipeline=agreement)
    lines = _cache_lines(render_markdown(report))
    tokens = report_to_json(report)["measurement_2_pipeline_agreement"]["tokens"]

    assert set(lines) == {"생성", "검색"}, "계열 두 줄이 모두 있어야 한다"
    assert str(tokens["generation_cache_read_total"]) in lines["생성"]
    assert str(tokens["retrieval_cache_read_total"]) in lines["검색"]
    # 보고되지 않은 칸은 사람이 읽는 줄에서도 0 이 아니라 "미측정" 이다.
    assert "미측정" in lines["생성"]
    assert "미측정" in lines["검색"]


def test_한_건도_재지_않은_계열은_사람이_읽는_줄이_미측정이다() -> None:
    lines = _cache_lines(render_markdown(_report(pipeline=_agreement([_cached_processed()]))))

    assert set(lines) == {"생성", "검색"}
    assert all("미측정" in body for body in lines.values())


def test_캐시_칸은_계열_합산을_두_번_세지_않는다() -> None:
    """OpenAI 의 캐시 칸은 `input_tokens` **안에** 든 값이다 — 합산에 다시 더하면 거짓이 된다."""
    plain = _agreement([_cached_processed()])
    cached = _agreement(
        [_cached_processed(generation_cache_read_tokens=100, retrieval_cache_read_tokens=7)]
    )

    assert cached.total_tokens_per_inquiry == plain.total_tokens_per_inquiry


def test_토큰_계열은_넷_그대로다() -> None:
    """캐시 칸이 계열을 늘리지 않는다 — 생성·임베딩·판정·검색 넷이다."""
    markdown = render_markdown(_report(pipeline=_agreement([_cached_processed()])))

    assert "### 문의 1건당 토큰 (생성·임베딩·판정·검색 구분)" in markdown


# ── 계약 5 — 응답 계약은 변하지 않는다 ──────────────────────────────────────


def test_응답_계약에_캐시_칸이_생기지_않았다() -> None:
    """캐시 칸은 리포트에만 있다 — 판정 계열의 캐시 칸과 같은 경계다."""
    from reply_gate.api import InquiryResponse

    processed = _cached_processed(
        generation_cache_read_tokens=100, retrieval_cache_read_tokens=7, intent=IntentSource.POLICY
    )
    tokens = InquiryResponse.of(processed).model_dump()["metrics"]["tokens"]

    assert set(tokens) == TOKEN_KEYS
    assert not any("cache" in key for key in tokens)


# ── 계약 6 — 단가 문서가 상한과 실측을 가른다 ───────────────────────────────


def _pricing_text() -> str:
    return _PRICING_DOC.read_text(encoding="utf-8")


def _section(markdown: str, title: str) -> str:
    """제목 줄부터 **같거나 더 높은 수준의 다음 제목** 직전까지를 돌려준다 (없으면 빈 문자열)."""
    level = len(title) - len(title.lstrip("#"))
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != title:
            continue
        body: list[str] = []
        for following in lines[index + 1 :]:
            stripped = following.lstrip("#")
            depth = len(following) - len(stripped)
            if following.startswith("#") and 0 < depth <= level:
                break
            body.append(following)
        return "\n".join(body)
    return ""


_CANON_TITLE = "### 상한과 실측을 가른다"


def _canon_split_is_stated(markdown: str) -> bool:
    """단가 문서에 상한/실측 정본 구분이 서 있는가 — **기준일과 함께**."""
    section = _section(markdown, _CANON_TITLE)
    return bool(section) and all(token in section for token in ("상한", "실측", "2026-08-19"))


def test_단가_문서에_상한과_실측의_정본_구분이_서_있다() -> None:
    """구분이 없으면 다음 사람이 캐시 미배선 산출물과 실측을 같은 자로 읽는다."""
    assert _canon_split_is_stated(_pricing_text())


def test_정본_구분_검사는_구분이_없는_문서를_실제로_잡는다() -> None:
    """음성 대조 — 이 가드가 무엇도 지키지 않는 상태로 초록이 되지 않는다."""
    assert not _canon_split_is_stated("# 단가\n\n## 1. 단가 표\n\n아무 말도 없다.\n")
    assert not _canon_split_is_stated(f"{_CANON_TITLE}\n\n상한과 실측을 가른다.\n")


def test_단가_매핑_표가_계열별_캐시_칸을_싣는다() -> None:
    """칸이 생겼는데 매핑이 없으면 달러 환산이 그 칸을 그냥 버린다."""
    section = _section(_pricing_text(), "### 계열 ↔ 단가 매핑")

    assert section, "매핑 절을 찾지 못했다"
    for key in (
        "generation_cache_read",
        "generation_cache_creation",
        "retrieval_cache_read",
        "retrieval_cache_creation",
    ):
        assert key in section, f"{key} 가 단가 매핑에 없다"


def test_단가_문서가_캐시를_읽지_않는다는_옛_서술을_들고_있지_않다() -> None:
    """배선이 생겼는데 문서가 "읽지 않는다"고 적혀 있으면 그 자체가 거짓 신고다."""
    text = _pricing_text()

    assert "읽지 않는다" not in text
