"""생성 LLM·임베딩 클라이언트 공통 래퍼.

docs/standards.md "재시도 상한" 중 **전송 오류 1회 재시도**만 담당한다.
프롬프트는 담지 않는다 — 프롬프트는 근거 수집·초안 생성 모듈이 소유한다.

구조화 출력 형식 불일치는 여기서 재시도하지 않는다: 단계별 정책이 다르기 때문이다
(의도 해석은 1회 재시도, 초안 생성은 재시도 없이 L1 으로 넘김, SQL 생성은 SQL 실패 경로).
따라서 형식 오류는 `LLMFormatError` 로 올려보내고 호출자가 정책을 적용한다.

주의: OpenAI·Anthropic SDK 모두 전송 오류·429·5xx 를 기본 2회 자동 재시도한다. 중첩되면
docs/standards.md "재시도 상한"의
"1회 재시도"가 실제로는 최대 6회 전송 시도가 되어 지연·처리 기록이 어긋나므로,
`max_retries=0` 을 명시해 재시도를 래퍼가 단독 통제한다.

샘플링 파라미터(temperature 등)는 보내지 않는다 — 결정론을 샘플링 파라미터로 보장하지
않으며, 모델 계열에 따라 아예 받지 않는 경우도 있다
(docs/standards.md "샘플링 파라미터를 보내지 않는다").
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import anthropic
import openai
from anthropic.types import Message
from openai.types.responses import Response

__all__ = [
    "AnthropicGenerationClient",
    "EmbeddingClient",
    "EmbeddingResult",
    "GenerationClient",
    "JsonCompletion",
    "LLMCallError",
    "LLMFormatError",
    "OpenAIEmbeddingClient",
    "OpenAIGenerationClient",
]

#: 최초 호출 + 재시도 1회 = 최대 2회 전송 시도 (docs/standards.md "재시도 상한").
MAX_ATTEMPTS = 2


class LLMCallError(RuntimeError):
    """전송 오류가 재시도 후에도 지속됨 → 인계 사유 `llm_call_failed`.

    `stage` 는 실패한 단계 이름이며 처리 기록에 그대로 남는다.

    토큰(`input_tokens`/`output_tokens`)은 **이 실패까지 실제로 과금된 분**이고 기본값은
    0 이다. 전송이 아예 성립하지 않은 실패(연결 오류·4xx)는 과금이 없어 0 이지만, 응답이
    200 으로 돌아온 뒤 쓸 수 있는 산출이 없는 실패(거절)와 여러 번 호출한 뒤의 실패는
    과금분이 있다 — 그것을 여기 싣지 않으면 호출자가 실비용을 0 으로 기록한다
    (docs/contracts.md "토큰 집계 경계": 실행됐으나 실패한 호출의 토큰도 그대로 집계한다).
    """

    def __init__(
        self,
        *,
        stage: str,
        reason: str,
        attempts: int,
        cause: BaseException | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        super().__init__(f"LLM 호출 실패 (stage={stage}, reason={reason}, attempts={attempts})")
        self.stage = stage
        self.reason = reason
        self.attempts = attempts
        self.cause = cause
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class LLMFormatError(ValueError):
    """구조화 출력이 기대 형식과 다름. 재시도 정책은 호출자가 정한다.

    원문(`raw_text`)과 토큰 사용량을 함께 실어 보낸다 — 초안 생성은 재시도하지 않고
    이 산출을 그대로 L1 에 넘겨 `schema_violation` 으로 판정시키기 때문이다
    (docs/standards.md "재시도 상한").
    """

    def __init__(
        self,
        *,
        stage: str,
        detail: str,
        raw_text: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        super().__init__(f"구조화 출력 형식 불일치 (stage={stage}): {detail}")
        self.stage = stage
        self.detail = detail
        self.raw_text = raw_text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


@dataclass(frozen=True)
class JsonCompletion:
    """구조화 출력 1건 + 토큰 사용량."""

    data: Any
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class EmbeddingResult:
    """임베딩 벡터 목록 + 사용 토큰 수."""

    vectors: list[list[float]]
    total_tokens: int


class GenerationClient(Protocol):
    """생성 LLM 클라이언트 계약 — 실제 구현과 테스트 대역이 공유한다."""

    def complete_json(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str = ...,
        effort: str | None = ...,
        max_output_tokens: int = ...,
    ) -> JsonCompletion: ...


class EmbeddingClient(Protocol):
    """임베딩 클라이언트 계약 — 실제 구현과 테스트 대역이 공유한다."""

    @property
    def dimensions(self) -> int: ...

    def embed(self, *, stage: str, texts: Sequence[str]) -> EmbeddingResult: ...


def _call_with_one_retry[T](
    *,
    stage: str,
    call: Callable[[], T],
    is_transport_error: Callable[[Exception], bool],
    is_api_error: Callable[[Exception], bool],
) -> T:
    """전송 오류면 1회 재시도, 재실패하면 `LLMCallError`.

    전송 오류가 아닌 API 오류(4xx 등)는 재시도해도 결과가 같으므로 즉시 실패시킨다.
    두 경우 모두 호출자에게는 `LLMCallError` 로 보여 인계 사유가 `llm_call_failed` 로 통일된다.
    """
    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return call()
        except Exception as exc:
            if is_transport_error(exc):
                last = exc
                continue
            if is_api_error(exc):
                raise LLMCallError(
                    stage=stage, reason="api_error", attempts=attempt, cause=exc
                ) from exc
            raise
    raise LLMCallError(
        stage=stage, reason="transport_error", attempts=MAX_ATTEMPTS, cause=last
    ) from last


def _parse_json_completion(
    *, stage: str, text: str, input_tokens: int, output_tokens: int
) -> JsonCompletion:
    """구조화 출력 원문을 파싱한다 — 빈 응답·비 JSON 은 `LLMFormatError` (양 래퍼 공통)."""
    if not text.strip():
        raise LLMFormatError(
            stage=stage,
            detail="빈 응답",
            raw_text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMFormatError(
            stage=stage,
            detail=f"JSON 파싱 실패: {exc}",
            raw_text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ) from exc
    return JsonCompletion(data=data, input_tokens=input_tokens, output_tokens=output_tokens)


def _is_openai_transport_error(exc: Exception) -> bool:
    # APITimeoutError 는 APIConnectionError 의 하위 타입이다.
    if isinstance(exc, openai.APIConnectionError | openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    return False


def _is_openai_api_error(exc: Exception) -> bool:
    return isinstance(exc, openai.APIError)


def _refusal_text(response: Response) -> str | None:
    """안전 분류기 거절이면 그 사유를, 아니면 None."""
    for item in getattr(response, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            if getattr(part, "type", None) == "refusal":
                refusal = getattr(part, "refusal", "")
                return str(refusal) if refusal else "refusal"
    return None


class OpenAIGenerationClient:
    """생성 계열(의도 해석·초안 생성·SQL 생성) 공통 호출 래퍼."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        client: openai.OpenAI | None = None,
    ) -> None:
        # max_retries=0: 재시도는 이 래퍼가 단독 통제한다 (모듈 docstring 참조).
        self._client = client or openai.OpenAI(api_key=api_key, max_retries=0, timeout=timeout)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def complete_json(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str = "response",
        effort: str | None = None,
        max_output_tokens: int = 8000,
    ) -> JsonCompletion:
        """JSON 스키마로 제약된 구조화 출력을 1건 받는다.

        `effort` 는 reasoning 계열 모델에서만 의미가 있으므로 **지정했을 때만** 보낸다
        — 모델 등급은 조정 가능 기본값이라 계열이 바뀔 수 있고, 지원하지 않는 모델에
        보내면 요청 자체가 거부된다.
        """
        request: dict[str, Any] = {
            "model": self._model,
            "instructions": system,
            "input": user,
            "max_output_tokens": max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        if effort is not None:
            request["reasoning"] = {"effort": effort}

        def _call() -> Response:
            # `**request` 로 넘기면 오버로드 추론이 풀려 Any 가 되므로 반환 타입을 고정한다.
            return cast(Response, self._client.responses.create(**request))

        response = _call_with_one_retry(
            stage=stage,
            call=_call,
            is_transport_error=_is_openai_transport_error,
            is_api_error=_is_openai_api_error,
        )

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

        # 안전 분류기가 거절하면 사용 가능한 산출이 없으므로 실패다.
        # 응답은 200 으로 왔으므로 **토큰은 이미 과금됐다** — 실패에 실어 보낸다.
        refusal = _refusal_text(response)
        if refusal is not None:
            raise LLMCallError(
                stage=stage,
                reason="refusal",
                attempts=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        text = response.output_text or ""
        return _parse_json_completion(
            stage=stage, text=text, input_tokens=input_tokens, output_tokens=output_tokens
        )


def _is_anthropic_transport_error(exc: Exception) -> bool:
    # APITimeoutError 는 APIConnectionError 의 하위 타입이다 (OpenAI SDK 와 같은 구조).
    if isinstance(exc, anthropic.APIConnectionError | anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500
    return False


def _is_anthropic_api_error(exc: Exception) -> bool:
    return isinstance(exc, anthropic.APIError)


class AnthropicGenerationClient:
    """L2 판정용 Anthropic 호출 래퍼 — OpenAI 래퍼와 같은 실패 정책을 따른다."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        # max_retries=0: 재시도는 이 래퍼가 단독 통제한다 (모듈 docstring 참조).
        self._client = client or anthropic.Anthropic(
            api_key=api_key, max_retries=0, timeout=timeout
        )
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def complete_json(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str = "response",
        effort: str | None = None,
        max_output_tokens: int = 8000,
    ) -> JsonCompletion:
        """JSON 스키마로 제약된 구조화 출력을 1건 받는다.

        - `schema_name` 은 Anthropic 구조화 출력에 대응 필드가 없어 전송하지 않는다
          (`GenerationClient` 프로토콜 서명 유지용).
        - `thinking` 설정은 보내지 않는다 — 미전송은 '끔'이 아니라 **adaptive thinking
          켜짐**이 모델 기본이다. 판정 토큰에 thinking 이 포함되고 `max_output_tokens`
          (와이어의 `max_tokens`)는 thinking+응답 합산 상한이므로 여유 있게 받는다.
        - `effort` 는 **지정했을 때만** `output_config` 에 실어 보낸다
          (docs/standards.md "샘플링 파라미터를 보내지 않는다").
        """
        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": schema},
        }
        if effort is not None:
            output_config["effort"] = effort
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_output_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }

        def _call() -> Message:
            # `**request` 로 넘기면 오버로드 추론이 풀려 Any 가 되므로 반환 타입을 고정한다.
            return cast(Message, self._client.messages.create(**request))

        response = _call_with_one_retry(
            stage=stage,
            call=_call,
            is_transport_error=_is_anthropic_transport_error,
            is_api_error=_is_anthropic_api_error,
        )

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

        # 안전 분류기 거절은 HTTP 200 + stop_reason="refusal" 로 온다.
        # 사용 가능한 산출이 없으므로 실패다 (OpenAI 래퍼의 거절 처리와 동수준).
        # 200 으로 온 응답이라 **입력 토큰은 이미 과금됐다** — 실패에 실어 보낸다.
        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMCallError(
                stage=stage,
                reason="refusal",
                attempts=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        # adaptive thinking 이 켜져 있으면 text 블록 앞에 thinking 블록이 올 수 있다 —
        # 위치가 아니라 블록 타입으로 골라낸다.
        text = next(
            (
                str(getattr(block, "text", "") or "")
                for block in getattr(response, "content", []) or []
                if getattr(block, "type", None) == "text"
            ),
            "",
        )
        return _parse_json_completion(
            stage=stage, text=text, input_tokens=input_tokens, output_tokens=output_tokens
        )


class OpenAIEmbeddingClient:
    """임베딩 호출 래퍼 (문의 임베딩·정책 청크 인덱싱 공통)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        timeout: float = 60.0,
        client: openai.OpenAI | None = None,
    ) -> None:
        self._client = client or openai.OpenAI(api_key=api_key, max_retries=0, timeout=timeout)
        self._model = model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, *, stage: str, texts: Sequence[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=[], total_tokens=0)

        inputs = list(texts)

        def _call() -> openai.types.CreateEmbeddingResponse:
            return self._client.embeddings.create(
                model=self._model,
                input=inputs,
                dimensions=self._dimensions,
            )

        response = _call_with_one_retry(
            stage=stage,
            call=_call,
            is_transport_error=_is_openai_transport_error,
            is_api_error=_is_openai_api_error,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return EmbeddingResult(
            vectors=[list(item.embedding) for item in ordered],
            total_tokens=response.usage.total_tokens,
        )
