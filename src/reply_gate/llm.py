"""생성 LLM·임베딩 클라이언트 공통 래퍼.

docs/standards.md "재시도 상한" 중 **전송 오류 1회 재시도**만 담당한다.
프롬프트는 담지 않는다 — 프롬프트는 근거 수집·초안 생성 모듈이 소유한다.

구조화 출력 형식 불일치는 여기서 재시도하지 않는다: 단계별 정책이 다르기 때문이다
(의도 해석은 1회 재시도, 초안 생성은 재시도 없이 L1 으로 넘김, SQL 생성은 SQL 실패 경로).
따라서 형식 오류는 `LLMFormatError` 로 올려보내고 호출자가 정책을 적용한다.

주의: OpenAI SDK 는 전송 오류·429·5xx 를 기본 2회 자동 재시도한다. 중첩되면 스펙의
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

import openai
from openai.types.responses import Response

__all__ = [
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
        refusal = _refusal_text(response)
        if refusal is not None:
            raise LLMCallError(stage=stage, reason="refusal", attempts=1)

        text = response.output_text or ""
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
