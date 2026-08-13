"""환경 변수 기반 설정.

비밀(API 키·DB 비밀번호)은 환경 변수 또는 gitignore 된 `.env` 에만 존재한다.
코드·문서·설정 파일에 평문으로 두지 않는다.
"""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import quote

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # ── 검색 전략 ───────────────────────────────────────────────────────────
    #: 검색용 질의 재작성 스위치 — 기본 켜짐. **켜짐 + 재작성 클라이언트 미배선은 조립 시점
    #: 오류**다(`evidence.RetrievalWiringError`). 끄려면 `false` 로 명시한다.
    #: 하이브리드(BM25+RRF)·LLM 리랭크는 채택하지 않아 실행 경로에 스위치가 없다 —
    #: 조항 26개 코퍼스에서 벡터 단독이 이미 r@5 = 1.000 이라 재정렬은 풀린 문제를 공격한다
    #: (`docs/tracking/status.md` "검색 전략 비교").
    query_rewrite_enabled: bool = True

    # ── 조정 가능 기본값 (docs/operations.md "환경 변수" 절) ────────────────────────
    vector_top_k: int = 5
    #: 정책 근거 채택 컷. **재작성 배선과 짝이다** — 재작성이 유사도를 밀어 올린 뒤의 값이라
    #: 재작성을 끄고 이 값만 유지하면 recall 이 0.880 → 0.640 으로 무너진다
    #: (`docs/tracking/status.md`). 0.50 은 무근거 4건(G21-G24)이 전부 기권하는 가장 낮은
    #: 컷이다 — F1 최적점 0.55 는 recall 을 0.920 → 0.840 으로 깎고 G17 마진을 0.03 으로 줄인다.
    vector_similarity_threshold: float = 0.5
    sql_max_rows: int = 50
    #: text-to-SQL 실행 커넥션의 `statement_timeout`(ms). 0 이면 무제한이므로 0 을 두지 않는다 —
    #: 생성된 쿼리 하나가 워커를 몇 분씩 묶는 것을 코드가 막는 층이다.
    sql_statement_timeout_ms: int = 5000

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
