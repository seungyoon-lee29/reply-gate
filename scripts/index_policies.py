"""정책 문서를 임베딩해 pgvector 에 적재한다.

    uv run python -m scripts.index_policies

실제 임베딩 API 를 호출하므로 `OPENAI_API_KEY` 가 필요하다. 재실행해도 중복 적재되지
않고 갱신된다(문서에서 사라진 조항은 삭제).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reply_gate.config import get_settings
from reply_gate.db import connect
from reply_gate.llm import OpenAIEmbeddingClient
from reply_gate.policy_index import (
    DEFAULT_POLICY_DIR,
    index_policy_documents,
    load_policy_documents,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="정책 문서를 조항 단위로 임베딩해 적재한다")
    parser.add_argument(
        "--policy-dir", type=Path, default=DEFAULT_POLICY_DIR, help="정책 문서 디렉터리"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.openai_api_key:
        print(
            "OPENAI_API_KEY 가 비어 있다. .env 에 값을 채운 뒤 다시 실행한다.",
            file=sys.stderr,
        )
        return 2

    documents = load_policy_documents(args.policy_dir)
    embedder = OpenAIEmbeddingClient(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    with connect() as conn:
        result = index_policy_documents(conn=conn, documents=documents, embedder=embedder)

    print(
        f"정책 조항 적재 완료: {result.upserted}건 갱신, {result.deleted}건 삭제, "
        f"임베딩 토큰 {result.embedding_tokens}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
