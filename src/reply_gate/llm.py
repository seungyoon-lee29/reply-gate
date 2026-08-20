"""생성 LLM·임베딩 클라이언트 공통 래퍼.

docs/standards.md "재시도 상한" 중 **전송 오류 1회 재시도**만 담당한다.
프롬프트는 담지 않는다 — 프롬프트는 근거 수집·초안 생성 모듈이 소유한다.

구조화 출력 형식 불일치는 여기서 재시도하지 않는다: 단계별 정책이 다르기 때문이다
(의도 해석은 1회 재시도, 초안 생성은 재시도 없이 L1 으로 넘김, SQL 생성은 SQL 실패 경로).
따라서 형식 오류는 `LLMFormatError` 로 올려보내고 호출자가 정책을 적용한다.

**밖으로 나가는 호출의 경과 시간(`elapsed_ms`)을 재서 산출·예외에 함께 싣는다.** 재는 자리는
전송 재시도 루프의 **바깥**이라 재시도로 날아간 시간이 그 구간에 그대로 든다. 규칙은 토큰과
같다 — 실행됐으나 실패한 호출의 시간도 그 구간이 쓴 시간이므로 0 으로 접지 않는다.

주의: OpenAI·Anthropic SDK 모두 전송 오류·429·5xx 를 기본 2회 자동 재시도한다. 중첩되면
docs/standards.md "재시도 상한"의
"1회 재시도"가 실제로는 최대 6회 전송 시도가 되어 지연·처리 기록이 어긋나므로,
`max_retries=0` 을 명시해 재시도를 래퍼가 단독 통제한다.

샘플링 파라미터(temperature 등)는 보내지 않는다 — 결정론을 샘플링 파라미터로 보장하지
않으며, 모델 계열에 따라 아예 받지 않는 경우도 있다
(docs/standards.md "샘플링 파라미터를 보내지 않는다").

**API 키는 비밀 전용 타입(`SecretStr`)으로 받는다.** 평문이 되는 자리는 각 래퍼 생성자의
SDK 호출 인자 **한 줄뿐**이고(`api_key.get_secret_value()`), 그 세 줄이 이 패키지에서
API 키가 평문이 되는 자리의 전부다. 꺼내는 자리를 눈으로 셀 수 있게 두는 것이 요점이라,
편의로 `str` 도 받게 넓히지 않는다 — 넓히는 순간 어디서 이미 꺼내졌는지 알 수 없어진다
(docs/security.md "비밀 관리").
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from time import perf_counter
from typing import Any, Protocol, cast

import anthropic
import openai
from anthropic.types import Message
from openai.types.responses import Response
from pydantic import SecretStr

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
    "accumulate_optional_ms",
    "accumulate_optional_tokens",
]

#: 최초 호출 + 재시도 1회 = 최대 2회 전송 시도 (docs/standards.md "재시도 상한").
MAX_ATTEMPTS = 2


def _optional_token_count(usage: object, field: str) -> int | None:
    """캐시 계열 토큰 1개 — **없으면 0 이 아니라 `None`(해당 없음/미측정)이다.**

    프롬프트 캐싱을 쓰지 않는 provider(OpenAI 생성 계열)와 캐시 계열 필드를 싣지 않는
    응답에는 이 값이 아예 없다. 그때 0 으로 접으면 "캐시가 0 토큰 적중했다"(측정했고 0)와
    "캐시를 잰 적이 없다"(미측정)가 같은 값이 되어, 리포트가 재지도 않은 축을 잰 것처럼
    적는다 — 미실행·미측정을 0 으로 채우지 않는 규칙과 같은 자리다.
    """
    value = getattr(usage, field, None)
    if value is None:
        return None
    return int(value)


def accumulate_optional_tokens(total: int | None, value: int | None) -> int | None:
    """캐시 계열 토큰 누적 — **미측정(`None`)을 0 으로 접지 않는다.**

    한 번이라도 측정값이 있으면 합계는 측정값이고, 끝까지 없으면 합계도 미측정이다.
    `None + 0` 을 0 으로 만들면 재지 않은 실행이 "0 토큰"으로 신고된다.
    """
    if value is None:
        return total
    return value if total is None else total + value


def accumulate_optional_ms(total: float | None, value: float | None) -> float | None:
    """구간 시간 누적 — 토큰과 **같은 규칙**이다: 미측정(`None`)을 0 으로 접지 않는다.

    돌지 않은 구간(재작성을 쓰지 않은 문의의 재작성 구간, L2 가 돌지 않은 시도의 판정
    구간)은 "0 밀리초"가 아니라 **미측정**이다. 0 으로 채우면 집계 평균이 그 구간을 실제
    보다 빠르게 적고, 리포트가 재지도 않은 축을 잰 것처럼 보인다.
    """
    if value is None:
        return total
    return value if total is None else total + value


def _elapsed_ms(started: float) -> float:
    """시작 시각(`perf_counter`)부터 지금까지의 벽시계 밀리초."""
    return max((perf_counter() - started) * 1000.0, 0.0)


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

    캐시 계열(`cache_creation_input_tokens`/`cache_read_input_tokens`)은 같은 규칙을 따르되
    **기본값이 0 이 아니라 `None`(해당 없음/미측정)** 이다 — 캐싱을 쓰지 않는 경로에서 0 을
    싣으면 재지 않은 축이 측정값으로 신고된다.

    `elapsed_ms` 는 **이 실패까지 실제로 흐른 벽시계**(밀리초)이고 토큰과 같은 자격이다:
    예외로 죽은 호출도 시간을 썼고, 재시도한 시도의 시간도 그 구간이 쓴 시간이다. 성공한
    마지막 호출만 재면 래퍼 재시도 1회가 통째로 사라져, 이 저장소가 토큰 축에서 이미 겪은
    사고(전송 3회를 `attempts=2` 로 신고)를 지연 축에서 반복한다. 형식 루프를 도는 호출자는
    토큰·전송 수와 **같은 줄에서** 이 값을 누적한다.
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
        cache_creation_input_tokens: int | None = None,
        cache_read_input_tokens: int | None = None,
        elapsed_ms: float = 0.0,
    ) -> None:
        super().__init__(f"LLM 호출 실패 (stage={stage}, reason={reason}, attempts={attempts})")
        self.stage = stage
        self.reason = reason
        self.attempts = attempts
        self.cause = cause
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.elapsed_ms = elapsed_ms


class LLMFormatError(ValueError):
    """구조화 출력이 기대 형식과 다름. 재시도 정책은 호출자가 정한다.

    원문(`raw_text`)과 토큰 사용량을 함께 실어 보낸다 — 초안 생성은 재시도하지 않고
    이 산출을 그대로 L1 에 넘겨 `schema_violation` 으로 판정시키기 때문이다
    (docs/standards.md "재시도 상한").

    `transport_attempts` 는 이 형식 오류가 나오기까지 **실제로 나간 전송 수**다. 형식 루프를
    도는 호출자(판정·의도 해석)가 토큰과 **같은 이유로** 누적해야 하는 값이다 — 앞선 시도의
    비용을 세면서 그 시도가 있었다는 사실을 세지 않으면 기록과 실제가 갈린다.

    `elapsed_ms` 도 같은 자격이다 — 형식이 어긋나 버려진 산출도 그만큼 시간을 썼다.
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
        cache_creation_input_tokens: int | None = None,
        cache_read_input_tokens: int | None = None,
        elapsed_ms: float = 0.0,
    ) -> None:
        super().__init__(f"구조화 출력 형식 불일치 (stage={stage}): {detail}")
        self.stage = stage
        self.detail = detail
        self.raw_text = raw_text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.transport_attempts = transport_attempts
        #: 캐시 계열은 **0 이 아니라 `None`** 이 미측정이다 (`LLMCallError` 와 같은 규칙).
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.elapsed_ms = elapsed_ms


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
    #: 이 산출을 얻기까지 흐른 **벽시계 밀리초 — 전송 재시도를 포함한다.** 성공한 마지막
    #: 호출만 재면 래퍼 재시도 1회가 통째로 사라진다. 대역이 직접 만드는 산출에서는 0.0
    #: 이고, 그것은 "재지 않았다"가 아니라 **밖으로 나간 시간이 없다**는 뜻이다.
    elapsed_ms: float = 0.0
    #: 캐시에 **쓴** 토큰(약 1.25배 단가). 캐싱을 쓰지 않는 provider·응답에서는 `None`
    #: (해당 없음)이고 **0 이 아니다** — 재지 않은 축을 0 으로 신고하지 않기 위해서다.
    cache_creation_input_tokens: int | None = None
    #: 캐시에서 **읽은** 토큰(약 0.1배 단가). 켜짐 조건에서 `input_tokens` 는 이 값을
    #: **제외한** 비캐시 입력이므로, 둘을 뭉뚱그리면 적중이 "입력 토큰 감소"로 위장한다.
    cache_read_input_tokens: int | None = None


@dataclass(frozen=True)
class EmbeddingResult:
    """임베딩 벡터 목록 + 사용 토큰 수 + 호출에 흐른 벽시계(전송 재시도 포함)."""

    vectors: list[list[float]]
    total_tokens: int
    #: 질의 임베딩 구간의 시간. 부를 것이 없어 즉시 돌아온 호출은 0.0 이다(측정값이다).
    elapsed_ms: float = 0.0


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
) -> tuple[T, int, float]:
    """전송 오류면 1회 재시도, 재실패하면 `LLMCallError`.

    결과와 함께 **실제 전송 수**와 **재시도를 포함한 경과 벽시계(ms)** 를 돌려준다.

    전송 오류가 아닌 API 오류(4xx 등)는 재시도해도 결과가 같으므로 즉시 실패시킨다.
    두 경우 모두 호출자에게는 `LLMCallError` 로 보여 인계 사유가 `llm_call_failed` 로 통일된다.

    전송 수를 **세어서** 돌려주는 이유: 이 값을 상수(`MAX_ATTEMPTS`)로 되읽으면 "몇 번 돌았나"가
    아니라 "몇 번까지 돌 수 있나"를 신고하게 된다. 호출자가 형식 루프를 돌면 그 차이가 누적된다.

    **경과도 같은 이유로 루프 **바깥**에서 잰다.** 성공한 마지막 `call()` 만 재면 재시도로
    날아간 시간이 통째로 사라져, 토큰 축에서 이미 겪은 사고(전송 3회를 2회로 신고)를 지연
    축에서 반복한다. 실패로 끝나는 두 경로도 그때까지 흐른 시간을 예외에 실어 올린다.
    """
    started = perf_counter()
    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = call()
        except Exception as exc:
            if is_transport_error(exc):
                last = exc
                continue
            if is_api_error(exc):
                raise LLMCallError(
                    stage=stage,
                    reason="api_error",
                    attempts=attempt,
                    cause=exc,
                    elapsed_ms=_elapsed_ms(started),
                ) from exc
            raise
        return result, attempt, _elapsed_ms(started)
    raise LLMCallError(
        stage=stage,
        reason="transport_error",
        attempts=MAX_ATTEMPTS,
        cause=last,
        elapsed_ms=_elapsed_ms(started),
    ) from last


def _parse_json_completion(
    *,
    stage: str,
    text: str,
    input_tokens: int,
    output_tokens: int,
    transport_attempts: int = 1,
    elapsed_ms: float = 0.0,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
) -> JsonCompletion:
    """구조화 출력 원문을 파싱한다 — 빈 응답·비 JSON 은 `LLMFormatError` (양 래퍼 공통).

    성공하든 형식 오류든 **실제 전송 수와 경과를 그대로 실어 보낸다** — 형식 루프를 도는
    호출자가 토큰과 같은 자격으로 누적한다.
    """
    if not text.strip():
        raise LLMFormatError(
            stage=stage,
            detail="빈 응답",
            raw_text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            transport_attempts=transport_attempts,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            elapsed_ms=elapsed_ms,
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
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            elapsed_ms=elapsed_ms,
        ) from exc
    return JsonCompletion(
        data=data,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        transport_attempts=transport_attempts,
        elapsed_ms=elapsed_ms,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
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
        api_key: SecretStr,
        model: str,
        timeout: float = 120.0,
        client: openai.OpenAI | None = None,
    ) -> None:
        # 재시도·타임아웃은 이 래퍼가 단독 통제한다 (모듈 docstring 참조). 값을 `or`
        # 우변에만 두면 주입 시 우회되므로 **관문을 지나게** 한다.
        self._client = _pin_transport_policy(
            client
            or openai.OpenAI(api_key=api_key.get_secret_value(), max_retries=0, timeout=timeout),
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

        response, sent, elapsed = _call_with_one_retry(
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
                # 거절도 시간을 썼다 — 토큰과 같은 자격으로 올려보낸다.
                elapsed_ms=elapsed,
            )

        text = response.output_text or ""
        return _parse_json_completion(
            stage=stage,
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            transport_attempts=sent,
            elapsed_ms=elapsed,
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
    """L2 판정용 Anthropic 호출 래퍼 — OpenAI 래퍼와 같은 실패 정책을 따른다.

    `prompt_caching` 은 **호출 구성**이지 지침 변경이 아니다: 켜면 `system`(고정 프리픽스)을
    `cache_control` 이 붙은 블록 하나로 보낸다. 문면은 한 글자도 바뀌지 않으므로 판정
    프롬프트 판(`judge_prompt_version`)도 따라 움직이지 않는다. 브레이크포인트를 질의(user)
    쪽에 두지 않는 이유는 그쪽이 호출마다 달라져 매번 새 프리픽스가 되기 때문이다.
    """

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str,
        timeout: float = 120.0,
        client: anthropic.Anthropic | None = None,
        prompt_caching: bool = False,
    ) -> None:
        # 재시도·타임아웃은 이 래퍼가 단독 통제한다 (모듈 docstring 참조). 값을 `or`
        # 우변에만 두면 주입 시 우회되므로 **관문을 지나게** 한다.
        self._client = _pin_transport_policy(
            client
            or anthropic.Anthropic(
                api_key=api_key.get_secret_value(), max_retries=0, timeout=timeout
            ),
            timeout=timeout,
        )
        self._model = model
        self._prompt_caching = prompt_caching

    @property
    def model(self) -> str:
        return self._model

    @property
    def prompt_caching(self) -> bool:
        """고정 프리픽스 캐싱이 켜져 있는가 — 실행 조건 지문이 읽는 값과 같은 스위치다."""
        return self._prompt_caching

    def _system_field(self, system: str) -> str | list[dict[str, Any]]:
        """`system` 을 어떤 모양으로 보낼지 — 캐싱이 꺼져 있으면 **문자열 그대로**.

        꺼짐 조건의 요청 모양을 바꾸지 않는 것이 중요하다: 기준선 실측과 조건이 갈리면
        전후 비교가 캐싱만의 효과를 분리하지 못한다.
        """
        if not self._prompt_caching:
            return system
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

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
            "system": self._system_field(system),
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }

        def _call() -> Message:
            # `**request` 로 넘기면 오버로드 추론이 풀려 Any 가 되므로 반환 타입을 고정한다.
            return cast(Message, self._client.messages.create(**request))

        response, sent, elapsed = _call_with_one_retry(
            stage=stage,
            call=_call,
            is_transport_error=_is_anthropic_transport_error,
            is_api_error=_is_anthropic_api_error,
        )

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        # **캐싱 켜짐 조건에서 `input_tokens` 는 캐시 적중분을 제외한 값이다.** 이 두 줄이
        # 없으면 적중이 "판정 입력 토큰 감소"로 보여 조용한 토큰 은폐가 된다 —
        # 절감 주장의 전제가 여기다.
        cache_creation = _optional_token_count(usage, "cache_creation_input_tokens")
        cache_read = _optional_token_count(usage, "cache_read_input_tokens")

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
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
                elapsed_ms=elapsed,
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
            elapsed_ms=elapsed,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        )


class OpenAIEmbeddingClient:
    """임베딩 호출 래퍼 (문의 임베딩·정책 청크 인덱싱 공통)."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str,
        dimensions: int,
        timeout: float = 60.0,
        client: openai.OpenAI | None = None,
    ) -> None:
        self._client = _pin_transport_policy(
            client
            or openai.OpenAI(api_key=api_key.get_secret_value(), max_retries=0, timeout=timeout),
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

        response, _sent, elapsed = _call_with_one_retry(
            stage=stage,
            call=_call,
            is_transport_error=_is_openai_transport_error,
            is_api_error=_is_openai_api_error,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return EmbeddingResult(
            vectors=[list(item.embedding) for item in ordered],
            total_tokens=response.usage.total_tokens,
            elapsed_ms=elapsed,
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
        started = perf_counter()
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
        # **시간은 0 이 아니다** — 과금이 없다는 것과 시간을 쓰지 않았다는 것은 다르다.
        return EmbeddingResult(vectors=vectors, total_tokens=0, elapsed_ms=_elapsed_ms(started))
