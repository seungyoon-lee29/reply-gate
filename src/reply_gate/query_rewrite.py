"""검색용 질의 재작성 — 정책 벡터 검색 **앞**에 서는 확률 층
(docs/architecture.md "대표 흐름" 3단계).

구어체·완곡 표현이 조항 어휘와 겹치지 않아 정답 조항이 임계값 아래로 떨어지는 경로를
겨냥한다([findings 11](../../docs/tracking/findings.md) 의 G17: `policy:support:4-1`
유사도 0.264 → 재작성 뒤 0.580).

**이 모듈이 소유한 프롬프트가 blind 재작성 픽스처를 만든 그 문장이다.**
`scripts/generate_blind_rewrites.py` 가 여기서 가져다 쓴다 — 두 벌로 복사하면 런타임과
픽스처의 생성 조건이 조용히 갈리고, 그 순간 오프라인 비교표(macro F1 0.734)가 런타임을
더 이상 예측하지 못한다. 같은 이유로 **모델·effort 도 같은 설정에서 온다**
(`Settings.generation_model` · `Settings.generation_effort`).

**커밋된 재작성 픽스처는 런타임이 절대 읽지 않는다.** 재작성은 매 문의마다 모델을 실제로
부른 결과이고, `data/` 의 blind·oracle 픽스처는 오프라인 비교 전용이다 — 읽는 순간 골든셋
30건 전용 치트가 된다. `tests/test_rewritten_queries.py` 의 구조 테스트가 패키지 디렉터리를
훑어 이 모듈까지 자동으로 검사한다(그 검사가 문자열 상수까지 보므로 여기에 픽스처 파일명을
적지 않는다 — 주석과 경로를 가릴 수 없는 것이 그 가드의 성질이고, 느슨하게 만들지 않는다).

**실패는 폴백이지 인계가 아니다**(docs/business-rules.md "검색 단계 실패").
fail-closed 원칙이 지키는 것은 "검증하지 못한 답변을 내보내지 않는다"이고, 검색은 검증이
아니라 재료 수집이다. 재작성을 얻지 못하면 원문 질의로 검색을 계속하며 — 그 결과는
사이클 2 까지의 제품 동작과 정확히 같고 뒤에 L1·L2 두 층이 그대로 서 있다. 다만
**조용히 지나가지 않는다**: 폴백 사유와 실비용 토큰이 호출자에게 그대로 올라가
처리 기록과 평가 리포트에 남는다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from reply_gate.llm import GenerationClient, LLMCallError, LLMFormatError

__all__ = [
    "QUERY_REWRITE_STAGE",
    "REWRITE_JSON_SCHEMA",
    "REWRITE_SYSTEM_PROMPT",
    "RewriteOutcome",
    "build_rewrite_user_prompt",
    "rewrite_query",
]

#: 처리 기록·비용 산출에 남는 단계 이름. 픽스처 제작 전용 이름
#: (`scripts.generate_blind_rewrites.FIXTURE_REWRITE_STAGE`)과 섞이지 않는다 —
#: 오프라인 제작이 런타임 단계 이름으로 기록에 들어오면 두 계열을 나중에 가를 수 없다.
QUERY_REWRITE_STAGE: Final = "query_rewrite"

REWRITE_SYSTEM_PROMPT: Final = """\
너는 한국어 이커머스 고객센터 문의를 정책 문서 검색용 질의로 다시 쓴다.
답변을 쓰지 말고 검색 질의만 만든다.

- 문의가 실제로 묻는 대상을 정책 문서에 쓰일 법한 표현으로 옮긴다.
- 원문에 없는 사실·수치·고유값을 지어내지 않는다.
- 너는 이 회사의 정책 문서를 본 적이 없다. 조항 번호나 문서 제목을 쓰지 않는다.
- 한 문장 또는 명사구 하나로 짧게 쓴다.

산출은 다음 형태의 JSON 이다:
{"rewritten": "..."}
"""

REWRITE_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {"rewritten": {"type": "string"}},
    "required": ["rewritten"],
    "additionalProperties": False,
}


def build_rewrite_user_prompt(*, inquiry: str) -> str:
    """재작성 프롬프트. **문의 원문 외에는 아무것도 싣지 않는다.**

    정책 문서도 검색 정답 라벨도 직전 결과도 들어가지 않는다 — 픽스처를 만든 조건과
    같아야 비교표가 런타임을 예측한다(`docs/tracking/decisions/0010`).
    """
    return f"[문의]\n{inquiry}"


@dataclass(frozen=True)
class RewriteOutcome:
    """재작성 1회의 결과와 관측 기록.

    `query` 가 `None` 이면 폴백이다 — 호출자는 원문 질의만으로 검색을 계속한다.
    `fallback_reason` 은 그때 비어 있지 않고, 실패한 호출이 쓴 토큰도 실비용이므로
    그대로 실려 온다(docs/contracts.md "토큰 집계 경계").
    """

    query: str | None
    input_tokens: int
    output_tokens: int
    fallback_reason: str | None
    #: 이 호출에 흐른 벽시계(ms) — **질의 재작성 구간의 시간**이다. 폴백으로 끝난 호출도
    #: 시간을 썼으므로 0 으로 접지 않는다(토큰과 같은 규칙). 재작성을 **시도하지 않은**
    #: 문의는 이 값을 만들지 않는다 — 그때 그 구간은 0 이 아니라 미측정이다.
    elapsed_ms: float = 0.0

    def __post_init__(self) -> None:
        if (self.query is None) != (self.fallback_reason is not None):
            raise ValueError("재작성 폴백은 사유와 짝을 이뤄야 한다 — 조용한 폴백을 만들지 않는다")

    @property
    def fell_back(self) -> bool:
        return self.query is None


def _extract_rewritten(data: object) -> str | None:
    if not isinstance(data, Mapping):
        return None
    value = data.get("rewritten")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _truncated(message: str, *, limit: int = 200) -> str:
    collapsed = " ".join(message.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "…"


def rewrite_query(
    *, client: GenerationClient, inquiry: str, effort: str | None = None
) -> RewriteOutcome:
    """검색용 질의를 1회 생성한다. **재시도하지 않고, 실패하면 폴백한다.**

    재시도하지 않는 것이 의도다: 의도 해석·SQL 생성의 1회 재시도는 그 산출이 없으면
    파이프라인이 진행하지 못해서 있는 것이고, 재작성은 없어도 원문으로 진행한다. 검색
    품질을 위해 지연과 토큰을 두 배로 쓰는 것은 이 층의 값어치를 넘는다.

    **잡는 예외는 LLM 래퍼가 선언한 실패면 두 가지뿐이다.** 넓게 잡으면 설정 오류
    (`MissingCredentialsError` — `LLMCallError` 를 상속하지 않는 것이 바로 이 때문이다)와
    프로그래밍 오류까지 "검색 폴백"으로 위장해, 키를 안 넣고 돌린 실행이 503 대신 정상
    처리로 집계된다.
    """
    try:
        completion = client.complete_json(
            stage=QUERY_REWRITE_STAGE,
            system=REWRITE_SYSTEM_PROMPT,
            user=build_rewrite_user_prompt(inquiry=inquiry),
            schema=REWRITE_JSON_SCHEMA,
            schema_name="rewrite",
            effort=effort,
        )
    except LLMFormatError as exc:
        return RewriteOutcome(
            query=None,
            input_tokens=exc.input_tokens,
            output_tokens=exc.output_tokens,
            fallback_reason=f"구조화 출력 형식 불일치: {_truncated(exc.detail)}",
            elapsed_ms=exc.elapsed_ms,
        )
    except LLMCallError as exc:
        return RewriteOutcome(
            query=None,
            input_tokens=exc.input_tokens,
            output_tokens=exc.output_tokens,
            fallback_reason=f"전송 오류: {_truncated(exc.reason)}",
            elapsed_ms=exc.elapsed_ms,
        )

    rewritten = _extract_rewritten(completion.data)
    if rewritten is None:
        return RewriteOutcome(
            query=None,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            fallback_reason=f"rewritten 이 비어 있지 않은 문자열이 아니다: {completion.data!r}",
            elapsed_ms=completion.elapsed_ms,
        )
    return RewriteOutcome(
        query=rewritten,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        fallback_reason=None,
        elapsed_ms=completion.elapsed_ms,
    )
