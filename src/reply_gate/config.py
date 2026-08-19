"""환경 변수 기반 설정.

비밀(API 키·DB 비밀번호)은 환경 변수 또는 gitignore 된 `.env` 에만 존재한다.
코드·문서·설정 파일에 평문으로 두지 않는다.
"""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import quote

from pydantic_settings import BaseSettings, SettingsConfigDict

from reply_gate.retrieval_strategies import AbstentionGate, AbstentionStatistic


class Settings(BaseSettings):
    """`.env` 또는 프로세스 환경에서 읽는 애플리케이션 설정."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 외부 API 키 ─────────────────────────────────────────────────────────
    # 생성(의도 해석·초안 생성·SQL 생성)·임베딩 공통 — docs/architecture.md "외부 의존".
    openai_api_key: str = ""
    #: L2 판정(Anthropic) 전용. 비밀 — 환경 변수 또는 .env 에만 둔다.
    anthropic_api_key: str = ""

    # ── Postgres 접속 ───────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "reply_gate"
    postgres_app_user: str = "reply_gate_app"
    postgres_app_password: str = ""
    postgres_ro_user: str = "reply_gate_ro"
    postgres_ro_password: str = ""

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
    #: τ 재산출은 **명시적 작업**이고, 그전까지 어긋남을 드러내는 것은 실행 조건 지문의
    #: `abstention_tau`↔`embedding_model` 짝(`regression_guard.PAIRED_FINGERPRINT_FIELDS`)이다.
    embedding_model: str = "text-embedding-3-small"
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
    #: 깎은 케이스가 없다(하드 게이트 10: 실측이 정당화하는 기본값만 남긴다).
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
        않는다. 그래서 수집기도 진입점도 이 메서드로만 게이트를 얻는다.
        """
        if not self.abstention_gate_enabled:
            return None
        return AbstentionGate(statistic=self.abstention_gate_statistic, tau=self.abstention_tau)

    @property
    def database_url(self) -> str:
        """애플리케이션 계정(읽기/쓰기) 접속 URL."""
        return self._dsn(self.postgres_app_user, self.postgres_app_password)

    @property
    def readonly_database_url(self) -> str:
        """text-to-SQL 실행 전용 계정(SELECT 만) 접속 URL."""
        return self._dsn(self.postgres_ro_user, self.postgres_ro_password)

    def _dsn(self, user: str, password: str) -> str:
        auth = quote(user, safe="")
        if password:
            auth = f"{auth}:{quote(password, safe='')}"
        return f"postgresql://{auth}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """프로세스당 1회만 로드되는 설정 싱글턴."""
    return Settings()
