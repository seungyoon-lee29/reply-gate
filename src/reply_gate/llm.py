"""LLM·임베딩 클라이언트 공통 래퍼.

spec "LLM 호출 공통 실패 정책" 중 **전송 오류 1회 재시도**만 담당한다.
프롬프트는 담지 않는다 — 프롬프트는 근거 수집·초안 생성 모듈이 소유한다.

구조화 출력 형식 불일치는 여기서 재시도하지 않는다: 단계별 정책이 다르기 때문이다
(의도 해석은 1회 재시도, 초안 생성은 재시도 없이 L1 으로 넘김, SQL 생성은 SQL 실패 경로).
따라서 형식 오류는 `LLMFormatError` 로 올려보내고 호출자가 정책을 적용한다.

주의: Anthropic·OpenAI SDK 는 전송 오류·429·5xx 를 기본 2회 자동 재시도한다.
중첩되면 스펙의 "1회 재시도"가 실제로는 최대 6회 전송 시도가 되어 지연·처리 기록이
어긋나므로, 두 클라이언트 모두 `max_retries=0` 을 명시해 재시도를 래퍼가 단독 통제한다.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import anthropic
import openai
from anthropic.types import Message, OutputConfigParam

__all__ = [
    "AnthropicClient",
    "Effort",
    "EmbeddingClient",
    "EmbeddingResult",
    "JsonCompletion",
    "LLMCallError",
    "LLMFormatError",
    "OpenAIEmbeddingClient",
]

#: `output_config.effort` 가 받는 값. 사고 깊이와 토큰 지출을 함께 조절한다.
type Effort = Literal["low", "medium", "high", "xhigh", "max"]

#: 최초 호출 + 재시도 1회 = 최대 2회 전송 시도 (spec "LLM 호출 공통 실패 정책").
MAX_ATTEMPTS = 2


class LLMCallError(RuntimeError):
    """전송 오류가 재시도 후에도 지속됨 → 인계 사유 `llm_call_failed`.

    `stage` 는 실패한 단계 이름이며 처리 기록에 그대로 남는다.
    """

    def __init__(
        self,
        *,
        stage: str,
        reason: str,
        attempts: int,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(f"LLM 호출 실패 (stage={stage}, reason={reason}, attempts={attempts})")
        self.stage = stage
        self.reason = reason
        self.attempts = attempts
        self.cause = cause


class LLMFormatError(ValueError):
    """구조화 출력이 기대 형식과 다름. 재시도 정책은 호출자가 정한다.

    원문(`raw_text`)과 토큰 사용량을 함께 실어 보낸다 — 초안 생성은 재시도하지 않고
    이 산출을 그대로 L1 에 넘겨 `schema_violation` 으로 판정시키기 때문이다(spec).
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


def _is_anthropic_transport_error(exc: Exception) -> bool:
    # APITimeoutError 는 APIConnectionError 의 하위 타입이다.
    if isinstance(exc, anthropic.APIConnectionError | anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500
    return False


def _is_openai_transport_error(exc: Exception) -> bool:
    if isinstance(exc, openai.APIConnectionError | openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    return False


class AnthropicClient:
    """생성 계열(의도 해석·초안 생성·SQL 생성) 공통 호출 래퍼."""

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

    def complete_json(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        effort: Effort = "medium",
        max_tokens: int = 8000,
    ) -> JsonCompletion:
        """JSON 스키마로 제약된 구조화 출력을 1건 받는다.

        `claude-opus-5` 는 temperature 등 샘플링 파라미터를 받지 않으므로 보내지 않는다
        (보내면 400). 사고 깊이·토큰 지출은 `effort` 로 조절한다.
        """
        output_config: OutputConfigParam = {
            "effort": effort,
            "format": {"type": "json_schema", "schema": schema},
        }

        def _call() -> Message:
            return self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config=output_config,
            )

        message = _call_with_one_retry(
            stage=stage,
            call=_call,
            is_transport_error=_is_anthropic_transport_error,
            is_api_error=lambda exc: isinstance(exc, anthropic.APIError),
        )

        # 안전 분류기가 거절하면 본문이 비거나 잘린다 — 사용 가능한 산출이 없으므로 실패다.
        if message.stop_reason == "refusal":
            raise LLMCallError(stage=stage, reason="refusal", attempts=1)

        text = "".join(block.text for block in message.content if block.type == "text")
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
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

        return JsonCompletion(
            data=data,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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
            is_api_error=lambda exc: isinstance(exc, openai.APIError),
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return EmbeddingResult(
            vectors=[list(item.embedding) for item in ordered],
            total_tokens=response.usage.total_tokens,
        )
