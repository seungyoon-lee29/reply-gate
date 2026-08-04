"""LLM·임베딩 래퍼의 실패 정책 단위 테스트 (외부 호출 없음, 목 사용)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx
import openai
import pytest

from reply_gate.llm import (
    MAX_ATTEMPTS,
    AnthropicClient,
    LLMCallError,
    LLMFormatError,
    OpenAIEmbeddingClient,
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"intent": {"type": "string"}},
    "required": ["intent"],
    "additionalProperties": False,
}


def _message(text: str, *, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )


class _RecordingMessages:
    """`client.messages.create` 대역 — 호출 횟수를 세고 정해진 결과를 순서대로 돌려준다."""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self._outcomes[min(len(self.calls) - 1, len(self._outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _anthropic_client(outcomes: list[Any]) -> tuple[AnthropicClient, _RecordingMessages]:
    messages = _RecordingMessages(outcomes)
    fake_sdk = cast(anthropic.Anthropic, SimpleNamespace(messages=messages))
    client = AnthropicClient(api_key="test", model="claude-opus-5", client=fake_sdk)
    return client, messages


def _connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))


def _status_error(status_code: int) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://example.invalid")
    response = httpx.Response(status_code, request=request, json={"error": {"message": "x"}})
    return anthropic.APIStatusError("boom", response=response, body=None)


def test_sdk_자동재시도를_끈다() -> None:
    """래퍼 재시도와 SDK 재시도가 중첩되면 스펙의 '1회 재시도'가 깨진다."""
    client = AnthropicClient(api_key="test", model="claude-opus-5")
    assert client._client.max_retries == 0

    embedder = OpenAIEmbeddingClient(
        api_key="test", model="text-embedding-3-small", dimensions=1536
    )
    assert embedder._client.max_retries == 0


def test_정상_응답은_데이터와_토큰을_돌려준다() -> None:
    client, messages = _anthropic_client([_message(json.dumps({"intent": "policy"}))])

    result = client.complete_json(stage="intent", system="s", user="u", schema=SCHEMA)

    assert result.data == {"intent": "policy"}
    assert (result.input_tokens, result.output_tokens) == (11, 7)
    assert len(messages.calls) == 1
    # 샘플링 파라미터는 claude-opus-5 에서 400 이므로 절대 보내지 않는다.
    assert not {"temperature", "top_p", "top_k"} & messages.calls[0].keys()


def test_전송오류는_1회만_재시도한다() -> None:
    client, messages = _anthropic_client(
        [_connection_error(), _message(json.dumps({"intent": "order"}))]
    )

    result = client.complete_json(stage="intent", system="s", user="u", schema=SCHEMA)

    assert result.data == {"intent": "order"}
    assert len(messages.calls) == MAX_ATTEMPTS


def test_전송오류가_지속되면_인계용_예외를_던진다() -> None:
    client, messages = _anthropic_client([_connection_error()])

    with pytest.raises(LLMCallError) as excinfo:
        client.complete_json(stage="draft", system="s", user="u", schema=SCHEMA)

    assert excinfo.value.stage == "draft"
    assert excinfo.value.reason == "transport_error"
    assert excinfo.value.attempts == MAX_ATTEMPTS
    assert len(messages.calls) == MAX_ATTEMPTS


def test_5xx_는_전송오류로_재시도한다() -> None:
    client, messages = _anthropic_client(
        [_status_error(503), _message(json.dumps({"intent": "both"}))]
    )

    assert client.complete_json(stage="sql", system="s", user="u", schema=SCHEMA).data == {
        "intent": "both"
    }
    assert len(messages.calls) == MAX_ATTEMPTS


def test_4xx_는_재시도하지_않고_즉시_실패한다() -> None:
    client, messages = _anthropic_client([_status_error(400)])

    with pytest.raises(LLMCallError) as excinfo:
        client.complete_json(stage="intent", system="s", user="u", schema=SCHEMA)

    assert excinfo.value.reason == "api_error"
    assert excinfo.value.attempts == 1
    assert len(messages.calls) == 1


def test_거절_응답은_사용가능한_산출이_없으므로_실패다() -> None:
    client, _ = _anthropic_client([_message("", stop_reason="refusal")])

    with pytest.raises(LLMCallError) as excinfo:
        client.complete_json(stage="draft", system="s", user="u", schema=SCHEMA)

    assert excinfo.value.reason == "refusal"


def test_형식오류는_재시도하지_않고_호출자에게_위임한다() -> None:
    client, messages = _anthropic_client([_message("이건 JSON 이 아니다")])

    with pytest.raises(LLMFormatError) as excinfo:
        client.complete_json(stage="intent", system="s", user="u", schema=SCHEMA)

    assert excinfo.value.stage == "intent"
    assert len(messages.calls) == 1
    # 초안 생성은 이 원문을 그대로 L1 에 넘겨 schema_violation 으로 판정시킨다.
    assert excinfo.value.raw_text == "이건 JSON 이 아니다"
    assert (excinfo.value.input_tokens, excinfo.value.output_tokens) == (11, 7)


# ── 임베딩 ──────────────────────────────────────────────────────────────────


class _RecordingEmbeddings:
    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self._outcomes[min(len(self.calls) - 1, len(self._outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _embedding_response(vectors: list[tuple[int, list[float]]]) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(index=index, embedding=vector) for index, vector in vectors],
        usage=SimpleNamespace(total_tokens=42),
    )


def _openai_client(outcomes: list[Any]) -> tuple[OpenAIEmbeddingClient, _RecordingEmbeddings]:
    embeddings = _RecordingEmbeddings(outcomes)
    fake_sdk = cast(openai.OpenAI, SimpleNamespace(embeddings=embeddings))
    client = OpenAIEmbeddingClient(
        api_key="test", model="text-embedding-3-small", dimensions=3, client=fake_sdk
    )
    return client, embeddings


def test_임베딩은_입력_순서대로_정렬해_돌려준다() -> None:
    client, _ = _openai_client([_embedding_response([(1, [0.0, 1.0, 0.0]), (0, [1.0, 0.0, 0.0])])])

    result = client.embed(stage="policy_index", texts=["가", "나"])

    assert result.vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert result.total_tokens == 42


def test_임베딩_전송오류도_1회만_재시도한다() -> None:
    error = openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
    client, embeddings = _openai_client([error, _embedding_response([(0, [1.0, 0.0, 0.0])])])

    result = client.embed(stage="inquiry", texts=["문의"])

    assert result.vectors == [[1.0, 0.0, 0.0]]
    assert len(embeddings.calls) == MAX_ATTEMPTS


def test_빈_입력은_호출하지_않는다() -> None:
    client, embeddings = _openai_client([])

    result = client.embed(stage="inquiry", texts=[])

    assert result.vectors == []
    assert embeddings.calls == []
