"""환경 변수 기반 설정.

비밀(API 키·DB 비밀번호)은 환경 변수 또는 gitignore 된 `.env` 에만 존재한다.
코드·문서·설정 파일에 평문으로 두지 않는다.

**자격 증명 필드는 비밀 전용 타입(`SecretStr`)이다.** 표시 억제(`Field(repr=False)`)만으로는
`model_dump()`·`model_dump_json()`·`dict(settings)` 같은 **덤프 경로**가 닫히지 않는다 —
그 경로들은 표시 설정을 보지 않고 값을 그대로 싣는다. 값이 필요한 곳은
`.get_secret_value()` 로 **명시적으로 꺼내 쓴다**: 꺼내는 자리가 코드에 드러나야 어디서
평문이 되는지 셀 수 있다. 전수 검사는 `tests/test_config_secrets.py` 가 한다.

**접속 문자열(`database_url`)은 이 규칙 밖이다.** 값을 조립한 새 문자열이라 비밀번호를
그대로 담고, 담지 않으면 DB 에 접속할 수 없다. 로그·오류 메시지용 설명은 비밀번호를
빼고 따로 만든다(`db.describe_target`).
"""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import quote

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from reply_gate.retrieval_strategies import AbstentionGate, AbstentionStatistic

#: τ 가 **실측으로 검증된** 임베딩 조건 — `(모델, 차원)` 쌍.
#:
#: 기권 게이트의 τ 는 임베딩 모델에 묶인 조건 종속 인자다. 무근거 문의군의 통계량 최댓값과
#: 정답이 있는 문의군의 최솟값 사이에 틈이 있어야 τ 를 끼울 수 있는데, `-3-large` 계열에서는
#: 그 틈이 **음수**라 어떤 τ 도 두 군을 가르지 못한다(후보 통계량 다섯 개 전부 그렇다).
#: 그래서 이 목록은 τ 값의 목록이 아니라 **조건의 목록**이다 — τ 를 만져서 조건 불일치를
#: 무마할 수 없다.
#:
#: **설정 필드가 아니라 모듈 상수인 것이 요점이다.** 설정 필드로 두면 `.env` 한 줄이 가드를
#: 통째로 끄고, 그 실행이 "게이트를 켜고 돌았다"로 집계된다. τ 를 재산출했을 때의 명시적
#: 경로는 **사람이 이 상수를 고치는 것**이고 코드에 우회 플래그를 두지 않는다 — 승격과
#: 같은 자격이다.
VALIDATED_ABSTENTION_EMBEDDINGS: frozenset[tuple[str, int]] = frozenset(
    {("text-embedding-3-small", 1536)}
)


class AbstentionGateWiringError(RuntimeError):
    """기권 게이트가 켜졌는데 τ 가 그 임베딩 조건에서 검증된 적이 없다 — **조립 시점 오류**다.

    `pipeline.PipelineWiringError`·`evidence.RetrievalWiringError` 와 같은 자격이다:
    기본값이 조용히 안전장치를 무력화하는 fail-open 배선을 금지한다. 다른 점은 무엇이
    빠졌는가가 아니라 **설정 조합이 어긋났는가**라는 것뿐이고, 둘 다 런타임 데이터 상태가
    아니라 조립 시점에 알 수 있는 문제다.

    **요청당 오류로 두지 않는 이유**: τ 가 엉뚱한 자리에 있으면 게이트는 계속 돌지만 무근거
    문의를 못 거르거나 정답 조항을 통째로 자른다. 어느 쪽이든 오류가 아니라 "검색 품질"로
    보이므로, 실행을 시작하기 전에 죽어야 조용한 무력화가 지표로 위장하지 않는다.

    **인계 사유가 아니다** — 업무 판정이 아니라 조립 오류이므로 `llm_call_failed` 로 삼키지
    않고 그대로 올려보낸다(자격 증명 부재와 같은 취급).
    """


class Settings(BaseSettings):
    """`.env` 또는 프로세스 환경에서 읽는 애플리케이션 설정."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 외부 API 키 ─────────────────────────────────────────────────────────
    #
    # **둘 다 `SecretStr` + `repr=False` 다.** 두 겹인 것은 닫는 경로가 다르기 때문이다:
    # `repr=False` 는 **표시**(repr·str·f-string·트레이스백)를, `SecretStr` 은 **덤프**
    # (`model_dump()`·`model_dump_json()`·`dict(settings)`·`vars()`)를 닫는다. 표시만 가렸을
    # 때 실제로 뚫려 있던 것이 덤프 쪽이었다 — 덤프는 표시 설정을 아예 보지 않는다.
    # 값을 읽는 쪽은 `.get_secret_value()` 로 **명시적으로 꺼낸다**(모듈 docstring).
    #
    # 생성(의도 해석·초안 생성·SQL 생성)·임베딩 공통 — docs/architecture.md "외부 의존".
    openai_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    #: L2 판정(Anthropic) 전용. 비밀 — 환경 변수 또는 .env 에만 둔다.
    anthropic_api_key: SecretStr = Field(default=SecretStr(""), repr=False)

    # ── Postgres 접속 ───────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "reply_gate"
    postgres_app_user: str = "reply_gate_app"
    postgres_app_password: SecretStr = Field(default=SecretStr(""), repr=False)
    postgres_ro_user: str = "reply_gate_ro"
    postgres_ro_password: SecretStr = Field(default=SecretStr(""), repr=False)

    # ── 모델 ────────────────────────────────────────────────────────────────
    # 생성 LLM 모델 등급은 조정 가능 기본값 (docs/operations.md "환경 변수").
    generation_model: str = "gpt-5.6-terra"
    #: 합성 데이터 1회 제작에만 쓰는 상위 모델. 런타임 경로에는 쓰지 않는다.
    bulk_generation_model: str = "gpt-5.6-sol"
    generation_effort: str | None = None
    #: **이 값을 바꾸면 `abstention_tau` 는 따라오지 않는다.** τ 는 임베딩 모델에 묶인
    #: 조건 종속 인자이고, `text-embedding-3-small` d1536 밖에서는 이전되지 않는다 —
    #: `-3-large` 1536·3072 에서는 채택 축 통계량이 기권군과 채택군을 **분리조차 못 한다**
    #: (여유 -0.0114 / -0.0160, `docs/tracking/decisions/0012`·`0014`). 모델을 바꿨다면
    #: τ 재산출은 **명시적 작업**이고, 재산출했다는 선언은 사람이
    #: `VALIDATED_ABSTENTION_EMBEDDINGS` 에 조건을 등재하는 것이다. 등재되지 않은 조건에서
    #: 게이트를 켜면 `abstention_gate()` 가 조립을 거부한다. 사후에 어긋남을 드러내는 장치는
    #: 그대로 남는다 — 실행 조건 지문의 `abstention_tau`↔`embedding_model` 짝
    #: (`regression_guard.PAIRED_FINGERPRINT_FIELDS`).
    embedding_model: str = "text-embedding-3-small"
    #: **`embedding_model` 과 함께 τ 검증 조건을 이룬다** — 같은 모델의 다른 차원은 다른
    #: 조건이다(`-3-large` d1536 과 d3072 의 분리 여유가 서로 다르다).
    embedding_dimensions: int = 1536
    #: 오프라인 검색 비교의 LLM 리랭크 모델. 다른 모델 기본값과 같은 자리에서 소유한다.
    rerank_model: str = "gpt-5.6-luna"

    # ── L2 판정 ─────────────────────────────────────────────────────────────
    #: L2 판정 스위치 — 기본 켜짐.
    l2_enabled: bool = True
    #: 판정 모델 등급은 조정 가능 기본값.
    judge_model: str = "claude-sonnet-5"
    #: 판정 effort — 지정했을 때만 요청에 실린다. 미지정이면 모델 기본값을 따른다.
    judge_effort: str | None = None
    #: 판정 max_tokens — thinking+응답 합산 상한이므로 여유 있게 둔다
    #: (thinking 설정 미전송 = adaptive thinking 켜짐이 모델 기본).
    judge_max_output_tokens: int = 16000
    #: 판정 **고정 프리픽스**(판정 지침) 프롬프트 캐싱 스위치. 캐싱은 호출 구성이지 지침
    #: 변경이 아니므로 프롬프트 문면·판(`judge_prompt_version`)은 따라 움직이지 않는다.
    #: **기본값은 꺼짐** — 실측이 정당화하지 않는 기본값을 남기지 않는다(사이클 4 T8).
    #: 이 값은 실행 조건 지문의 `judge_prompt_caching` 으로 그대로 실린다.
    judge_prompt_caching_enabled: bool = False

    # ── 검색 전략 ───────────────────────────────────────────────────────────
    #: 검색용 질의 재작성 스위치 — 기본 켜짐. **켜짐 + 재작성 클라이언트 미배선은 조립 시점
    #: 오류**다(`evidence.RetrievalWiringError`). 끄려면 `false` 로 명시한다.
    #: 하이브리드(BM25+RRF)·LLM 리랭크는 채택하지 않아 실행 경로에 스위치가 없다 —
    #: 조항 26개 코퍼스에서 벡터 단독이 이미 r@5 = 1.000 이라 재정렬은 풀린 문제를 공격한다
    #: (`docs/tracking/status.md` "검색 전략 비교").
    query_rewrite_enabled: bool = True

    # ── 조정 가능 기본값 (docs/operations.md "환경 변수" 절) ────────────────────────
    vector_top_k: int = 5
    #: 정책 근거 채택 컷. **사이클 3 T10 라이브 3회가 0.50 을 반증해 0.30 으로 되돌린 값이다**
    #: (`docs/tracking/status.md` "사이클 3 T10 라이브 재실측"). 0.50 은 무근거 4건(G21-G24)을
    #: 검색 단계에서 기권시켰지만 같은 컷이 G04 의 정답 조항(0.3571)과 G18 의 상충 조항
    #: (`policy:refund:2-4`, 0.4676)까지 잘라, 정상 문의가 인계되고 L2 의 모순 기각이
    #: 사라졌다. 두 조항 모두 0.30 위에 있다. **재작성은 유지한다** — G17 을 3/3 회 고쳤고
    #: 깎은 케이스가 없다(하드 게이트 9: 실측이 정당화하는 기본값만 남긴다).
    vector_similarity_threshold: float = 0.3
    #: 질의 단위 기권 게이트 스위치 — 기본 켜짐(`docs/tracking/decisions/0014`).
    #: 끄면 채택은 절대 하한 하나로 돌아간다. **원복은 이 한 줄이다** — 컷·`top_k`·응답
    #: 계약은 애초에 건드리지 않았으므로 축별 원복(결정 0011)의 범위가 게이트 하나로 남는다.
    abstention_gate_enabled: bool = True
    #: 게이트가 보는 통계량. 격자 165 구성 중 채택 규칙 다섯(케이스 하한 → 상충쌍 보존 →
    #: 기권 → 비악화 → 동률)을 **순서대로** 통과한 것은 이 하나뿐이다 — 상위 `top_k` 중
    #: **1위 빼기 `top_k`위 산포**. 동률이던 후보 넷 가운데 가장 단순하고(뺄셈 하나)
    #: 데이터 의존이 가장 적어(두 자리만 읽는다) 골랐다.
    abstention_gate_statistic: AbstentionStatistic = AbstentionStatistic.SPREAD
    #: 통계량이 이 값 **미만**이면 그 질의는 채택 0건으로 끝난다. 격자 눈금 위의 점이고
    #: (0.01 눈금), 채택 쪽 경계는 G15(0.0668, +0.0068) · 기권 쪽 경계는 G23(0.0521, -0.0079)이다.
    #: **설정값이지 데이터가 아니다** — 적재물도 캐시도 만들지 않는다. 조건이 바뀌면
    #: 자동으로 따라가지 않는다(위 `embedding_model` 주석).
    abstention_tau: float = 0.06
    sql_max_rows: int = 50
    #: text-to-SQL 실행 커넥션의 `statement_timeout`(ms). 0 이면 무제한이므로 0 을 두지 않는다 —
    #: 생성된 쿼리 하나가 워커를 몇 분씩 묶는 것을 코드가 막는 층이다.
    sql_statement_timeout_ms: int = 5000

    def abstention_gate(self) -> AbstentionGate | None:
        """실행 경로에 걸 기권 게이트. 스위치가 꺼져 있으면 `None`.

        게이트를 **어디서 켜고 끄는지가 한 자리**여야 런타임과 리포트의 조건 지문이 갈리지
        않는다. 그래서 수집기도 진입점도 이 메서드로만 게이트를 얻는다. τ 방어를 여기 거는
        것도 같은 이유다 — 조립자마다 검사를 복제하면 새 조립자가 생길 때 조용히 빠진다.

        **τ 가 검증되지 않은 임베딩 조건이면 게이트를 조립하지 않고 죽는다.** 조용히 끄지도
        않고 τ 를 추정하지도 않는다(`AbstentionGateWiringError`).

        **스위치가 꺼져 있으면 조건을 보지 않는다.** 이 가드가 막는 것은 *조용히 무력해진
        게이트*이고, 명시적으로 끈 실행에는 무력해질 게이트가 없다. 그 사실은 실행 조건
        지문에 `"꺼짐"` 으로 남아 조용하지도 않다. 반대로 꺼짐까지 죽이면 임베딩을 올리는
        사이클이 게이트를 끄고 실측하는 정상 경로가 막히고, 축별 원복(게이트 한 줄 끄기)이
        사라진다. 판단 근거는 결정 기록 0023 에 있다.

        **오프라인 비교·격자 도구는 이 메서드를 부르지 않는다.** 그 도구들은 τ 를 명시
        인자로 받아 여러 임베딩 조건을 일부러 훑는 것이 일이라, 방어를 거기 걸면 τ 재산출
        실측 자체가 막힌다.
        """
        if not self.abstention_gate_enabled:
            return None
        condition = (self.embedding_model, self.embedding_dimensions)
        if condition not in VALIDATED_ABSTENTION_EMBEDDINGS:
            validated = ", ".join(
                f"{model} d{dimensions}"
                for model, dimensions in sorted(VALIDATED_ABSTENTION_EMBEDDINGS)
            )
            raise AbstentionGateWiringError(
                f"기권 게이트가 켜져 있는데 τ 가 이 임베딩 조건에서 검증된 적이 없다: "
                f"{self.embedding_model} d{self.embedding_dimensions} "
                f"(검증된 조건: {validated}). τ 는 임베딩 모델에 묶인 조건 종속 인자이고 "
                "다른 계열에서는 기권군과 채택군의 분리 여유가 음수라 어떤 τ 도 두 군을 "
                "가르지 못한다. τ 를 재산출했다면 사람이 "
                "`config.VALIDATED_ABSTENTION_EMBEDDINGS` 에 그 조건을 등재하고, 게이트 없이 "
                "돌리려면 `ABSTENTION_GATE_ENABLED=false` 로 명시한다."
            )
        return AbstentionGate(statistic=self.abstention_gate_statistic, tau=self.abstention_tau)

    @property
    def database_url(self) -> str:
        """애플리케이션 계정(읽기/쓰기) 접속 URL.

        **여기가 비밀번호가 평문이 되는 자리 둘 중 하나다** — 접속 문자열은 값을 조립한 새
        문자열이라 필드의 표시·덤프 규칙 밖이다. 그래서 꺼내는 것을 이름 붙은 호출로 남긴다.
        """
        return self._dsn(self.postgres_app_user, self.postgres_app_password.get_secret_value())

    @property
    def readonly_database_url(self) -> str:
        """text-to-SQL 실행 전용 계정(SELECT 만) 접속 URL. 평문이 되는 자리 둘 중 하나다."""
        return self._dsn(self.postgres_ro_user, self.postgres_ro_password.get_secret_value())

    def _dsn(self, user: str, password: str) -> str:
        auth = quote(user, safe="")
        if password:
            auth = f"{auth}:{quote(password, safe='')}"
        return f"postgresql://{auth}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """프로세스당 1회만 로드되는 설정 싱글턴."""
    return Settings()
