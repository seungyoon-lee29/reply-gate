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
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # ── Postgres 접속 ───────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "reply_gate"
    postgres_app_user: str = "reply_gate_app"
    postgres_app_password: str = ""
    postgres_ro_user: str = "reply_gate_ro"
    postgres_ro_password: str = ""

    # ── 모델 ────────────────────────────────────────────────────────────────
    anthropic_model: str = "claude-opus-5"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ── 조정 가능 기본값 (spec "조정 가능 기본값" 절) ────────────────────────
    vector_top_k: int = 5
    vector_similarity_threshold: float = 0.3
    sql_max_rows: int = 50

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
