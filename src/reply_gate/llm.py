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
from importlib import import_module
from typing import Any, Protocol, cast

import anthropic
import openai
from anthropic.types import Message
from openai.types.responses import Response

__all__ = [
    "AnthropicGenerationClient",
    "BgeM3EmbeddingClient",
    "EmbeddingClient",
    "EmbeddingResult",
    "GenerationClient",
    "JsonCompletion",
    "LLMCallError",
    "LLMFormatError",
    "OpenAIEmbeddingClient",
    "OpenAIGenerationClient",
    "OptionalEmbeddingDependencyError",
]

#: 최초 호출 + 재시도 1회 = 최대 2회 전송 시도 (docs/standards.md "재시도 상한").
MAX_ATTEMPTS = 2


def _pin_transport_policy[C](client: C, *, timeout: float) -> C:
    """SDK 자동 재시도를 끄고 타임아웃을 래퍼 값으로 고정한다 — **주입 여부와 무관하게.**

    구 코드는 `client or <SDK>(..., max_retries=0, timeout=timeout)` 이었다. 주입하면
    `or` **우변 전체가 평가되지 않아** 두 값이 적용되지 않는다. 두 SDK 기본값이 모두
    `max_retries=2` 라, 래퍼의 "1회 재시도"가 SDK 재시도와 **곱해져 최대 6회 전송**이
    되면서 `LLMCallError.attempts` 는 래퍼 루프만 세어 2 로 신고했다(실측: 전송 6 / 기록 2).
    타임아웃도 같은 표현식의 같은 구멍이라 120초 의도가 SDK 기본 600초가 됐다.

    그래서 값을 `or` 우변이 아니라 **관문**에 둔다. 기본 생성 경로도 이 관문을 지나므로,
    생성자에서 `max_retries=0` 리터럴이 사라져도 정책은 코드가 되돌린다.

    `with_options`·`max_retries` 가 없는 테스트 대역은 그대로 통과한다 — 대역 주입은 정상
    용법이다. 끌 수단이 없는데 재시도가 켜진 객체만 거부한다(fail-closed).
    """
    candidate: Any = client
    with_options = getattr(candidate, "with_options", None)
    if callable(with_options):  # 진짜 SDK — 공식 API 로 사본에 정책을 박는다
        candidate = with_options(max_retries=0, timeout=timeout)
    if getattr(candidate, "max_retries", 0) != 0:
        raise ValueError(
            "주입된 클라이언트의 SDK 자동 재시도를 끌 수 없다 — 재시도는 래퍼가 단독 통제한다"
        )
    return cast(C, candidate)


class LLMCallError(RuntimeError):
    """전송 오류가 재시도 후에도 지속됨 → 인계 사유 `llm_call_failed`.

    `stage` 는 실패한 단계 이름이며 처리 기록에 그대로 남는다.

    토큰(`input_tokens`/`output_tokens`)은 **이 실패까지 실제로 과금된 분**이고 기본값은
    0 이다. 전송이 아예 성립하지 않은 실패(연결 오류·4xx)는 과금이 없어 0 이지만, 응답이
    200 으로 돌아온 뒤 쓸 수 있는 산출이 없는 실패(거절)와 여러 번 호출한 뒤의 실패는
    과금분이 있다 — 그것을 여기 싣지 않으면 호출자가 실비용을 0 으로 기록한다.
    규칙은 **실행됐으나 실패한 호출의 토큰도 그대로 집계한다**이며, 출처는
    `docs/contracts.md` "토큰 집계 경계" 다.
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

    `transport_attempts` 는 이 형식 오류가 나오기까지 **실제로 나간 전송 수**다. 형식 루프를
    도는 호출자(판정·의도 해석)가 토큰과 **같은 이유로** 누적해야 하는 값이다 — 앞선 시도의
    비용을 세면서 그 시도가 있었다는 사실을 세지 않으면 기록과 실제가 갈린다.
    """

    def __init__(
        self,
        *,
        stage: str,
        detail: str,
        raw_text: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        transport_attempts: int = 1,
    ) -> None:
        super().__init__(f"구조화 출력 형식 불일치 (stage={stage}): {detail}")
        self.stage = stage
        self.detail = detail
        self.raw_text = raw_text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.transport_attempts = transport_attempts


@dataclass(frozen=True)
class JsonCompletion:
    """구조화 출력 1건 + 토큰 사용량.

    `transport_attempts` 는 이 산출을 얻기까지 실제로 나간 전송 수다(전송 오류 재시도 포함).
    성공한 호출에도 싣는 이유는 형식 루프가 **성공한 시도의 전송까지** 세어야 하기 때문이다 —
    1차가 200/비 JSON 이고 2차가 전송 실패면 실제 전송은 3회다.
    """

    data: Any
    input_tokens: int
    output_tokens: int
    transport_attempts: int = 1


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
    """임베딩 클라이언트 계약 — 실제 구현과 테스트 대역이 공유한다.

    `model` 은 편의 정보가 아니라 **벡터의 출처**다. 저장된 벡터와 질의 벡터가 같은 공간에서
    나왔는지는 차원만으로 알 수 없고(같은 차원의 다른 모델이 흔하다), 그 판정을 코드가
    하려면 클라이언트가 자기 출처를 말할 수 있어야 한다
    (`policy_index.search_policy_chunks`).
    """

    @property
    def model(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, *, stage: str, texts: Sequence[str]) -> EmbeddingResult: ...


class OptionalEmbeddingDependencyError(RuntimeError):
    """선택 설치인 로컬 임베딩 의존성이 없어 해당 비교 행만 실행할 수 없음."""


class _SentenceTransformerModel(Protocol):
    def encode(self, sentences: Sequence[str], *, normalize_embeddings: bool) -> object: ...


def _call_with_one_retry[T](
    *,
    stage: str,
    call: Callable[[], T],
    is_transport_error: Callable[[Exception], bool],
    is_api_error: Callable[[Exception], bool],
) -> tuple[T, int]:
    """전송 오류면 1회 재시도, 재실패하면 `LLMCallError`. 결과와 **실제 전송 수**를 돌려준다.

    전송 오류가 아닌 API 오류(4xx 등)는 재시도해도 결과가 같으므로 즉시 실패시킨다.
    두 경우 모두 호출자에게는 `LLMCallError` 로 보여 인계 사유가 `llm_call_failed` 로 통일된다.

    전송 수를 **세어서** 돌려주는 이유: 이 값을 상수(`MAX_ATTEMPTS`)로 되읽으면 "몇 번 돌았나"가
    아니라 "몇 번까지 돌 수 있나"를 신고하게 된다. 호출자가 형식 루프를 돌면 그 차이가 누적된다.
    """
    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return call(), attempt
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
    *, stage: str, text: str, input_tokens: int, output_tokens: int, transport_attempts: int = 1
) -> JsonCompletion:
    """구조화 출력 원문을 파싱한다 — 빈 응답·비 JSON 은 `LLMFormatError` (양 래퍼 공통).

    성공하든 형식 오류든 **실제 전송 수를 그대로 실어 보낸다** — 형식 루프를 도는 호출자가
    토큰과 같은 자격으로 누적한다.
    """
    if not text.strip():
        raise LLMFormatError(
            stage=stage,
            detail="빈 응답",
            raw_text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            transport_attempts=transport_attempts,
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
            transport_attempts=transport_attempts,
        ) from exc
    return JsonCompletion(
        data=data,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        transport_attempts=transport_attempts,
    )


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
        # 재시도·타임아웃은 이 래퍼가 단독 통제한다 (모듈 docstring 참조). 값을 `or`
        # 우변에만 두면 주입 시 우회되므로 **관문을 지나게** 한다.
        self._client = _pin_transport_policy(
            client or openai.OpenAI(api_key=api_key, max_retries=0, timeout=timeout),
            timeout=timeout,
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

        response, sent = _call_with_one_retry(
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
                # 실측이다. 1차가 전송 오류로 실패하고 2차가 200+거절이면 전송은 2회다.
                attempts=sent,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        text = response.output_text or ""
        return _parse_json_completion(
            stage=stage,
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            transport_attempts=sent,
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
        # 재시도·타임아웃은 이 래퍼가 단독 통제한다 (모듈 docstring 참조). 값을 `or`
        # 우변에만 두면 주입 시 우회되므로 **관문을 지나게** 한다.
        self._client = _pin_transport_policy(
            client or anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=timeout),
            timeout=timeout,
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

        response, sent = _call_with_one_retry(
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
                # 실측이다. 1차가 전송 오류로 실패하고 2차가 200+거절이면 전송은 2회다.
                attempts=sent,
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
            stage=stage,
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            transport_attempts=sent,
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
        self._client = _pin_transport_policy(
            client or openai.OpenAI(api_key=api_key, max_retries=0, timeout=timeout),
            timeout=timeout,
        )
        self._model = model
        self._dimensions = dimensions

    @property
    def model(self) -> str:
        return self._model

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

        response, _ = _call_with_one_retry(
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


class BgeM3EmbeddingClient:
    """선택 설치인 Sentence Transformers 경계로 BGE-M3 dense 임베딩을 만든다.

    모듈 import와 모델 로드는 생성 시점까지 미룬다. 기본 설치가 torch 계열을 끌어들이거나,
    선택 의존성이 없는 환경에서 다른 비교 행까지 함께 죽는 것을 막기 위해서다.
    """

    MODEL = "BAAI/bge-m3"
    DIMENSIONS = 1024
    #: 허브 스냅샷을 고정한다. 리비전을 열어두면 같은 커밋이 다른 벡터를 만들고, 캐시 키가
    #: 모델 이름만 담으므로 그 변경을 무효화하지도 못한다 — 재현성이 조용히 깨지는 경로다.
    REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

    def __init__(self, *, model: _SentenceTransformerModel | None = None) -> None:
        if model is None:
            try:
                module = import_module("sentence_transformers")
            except ModuleNotFoundError as exc:
                # 하위 의존성(torch 등) 누락도 "선택 의존성 미설치"다. 그대로 전파하면
                # 부분 설치 환경에서 그 행만 미측정이 아니라 실행 전체가 트레이스백으로 죽는다.
                raise OptionalEmbeddingDependencyError(
                    f"BGE-M3 미측정 — 로컬 의존성 미설치({exc.name}) "
                    "(`uv sync --extra rag-local`로 설치)"
                ) from exc
            model_type = getattr(module, "SentenceTransformer", None)
            if model_type is None:
                raise OptionalEmbeddingDependencyError(
                    "BGE-M3 미측정 — sentence-transformers에 SentenceTransformer가 없다"
                )
            model = cast(_SentenceTransformerModel, model_type(self.MODEL, revision=self.REVISION))
        self._model = model

    @property
    def model(self) -> str:
        return self.MODEL

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS

    def embed(self, *, stage: str, texts: Sequence[str]) -> EmbeddingResult:
        del stage
        if not texts:
            return EmbeddingResult(vectors=[], total_tokens=0)
        encoded = self._model.encode(list(texts), normalize_embeddings=True)
        tolist = getattr(encoded, "tolist", None)
        raw = tolist() if callable(tolist) else encoded
        if not isinstance(raw, list) or len(raw) != len(texts):
            raise ValueError("BGE-M3 임베딩 개수가 입력 텍스트 수와 다르다")
        vectors: list[list[float]] = []
        for vector in raw:
            if not isinstance(vector, list) or len(vector) != self.DIMENSIONS:
                raise ValueError(f"BGE-M3 임베딩은 {self.DIMENSIONS}차원이어야 한다")
            if any(not isinstance(value, int | float) for value in vector):
                raise ValueError("BGE-M3 임베딩 값은 숫자여야 한다")
            vectors.append([float(value) for value in vector])
        # 로컬 추론은 provider 토큰 과금이 없으므로 기존 EmbeddingResult 계약에서 0이다.
        return EmbeddingResult(vectors=vectors, total_tokens=0)
