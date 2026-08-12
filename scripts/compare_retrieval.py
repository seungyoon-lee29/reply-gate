"""DB 없는 정책 벡터 단독 비교 실행 진입점.

    uv run python -m scripts.compare_retrieval --stub-embedding
    uv run python -m scripts.compare_retrieval --live  # 실제 임베딩 호출, 과금 가능

기본 모드는 결정론 어휘 임베딩 대역이다. 실제 모델은 `--live`로만 호출한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reply_gate.evaluation import DEFAULT_GOLDEN_SET_PATH
from reply_gate.llm import BgeM3EmbeddingClient, OptionalEmbeddingDependencyError
from reply_gate.policy_index import DEFAULT_POLICY_DIR
from reply_gate.retrieval_eval import (
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_NGRAM_SIZE,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RETRIEVAL_REPORT_DIR,
    DEFAULT_RRF_CUTOFF,
    DEFAULT_RRF_K,
    RetrievalConfigurationError,
    run_retrieval_comparison,
)
from reply_gate.retrieval_labels import DEFAULT_RETRIEVAL_LABELS_PATH
from reply_gate.retrieval_strategies import RetrievalStage, StrategyDefinition

_VECTOR_ONLY = (StrategyDefinition("vector", (RetrievalStage.VECTOR,)),)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DB 없이 정책 조항과 골든셋 문의의 벡터 단독 검색을 평가한다. "
            "재작성 포함 사다리는 패키지 API에 재작성문을 주입해 실행한다."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--live",
        action="store_true",
        help="OpenAI 임베딩·리랭크를 실제 호출한다 (명시적 opt-in, 캐시 miss는 과금 가능)",
    )
    mode.add_argument(
        "--stub-embedding",
        action="store_true",
        help="결정론 임베딩·리랭크 대역을 쓴다 (기본값, 배관 검증 전용)",
    )
    mode.add_argument(
        "--bge-m3",
        action="store_true",
        help="선택 설치한 로컬 BGE-M3 1024차원 임베딩을 쓴다 (DB·외부 API 없음)",
    )
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_RETRIEVAL_LABELS_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RETRIEVAL_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_EMBEDDING_CACHE_DIR)
    parser.add_argument("--dimensions", type=int, default=None)
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="--live에서 비교할 OpenAI 임베딩 모델 (기본: 환경 설정값)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="채택 코사인 컷 (기본: 대역 0.10, 실제 모드 설정값)",
    )
    parser.add_argument("--sweep-start", type=float, default=0.10)
    parser.add_argument("--sweep-end", type=float, default=0.70)
    parser.add_argument("--sweep-step", type=float, default=0.05)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K, help="RRF 순위 완화 상수")
    parser.add_argument(
        "--rrf-cutoff",
        type=float,
        default=DEFAULT_RRF_CUTOFF,
        help="하이브리드 RRF 채택 컷",
    )
    parser.add_argument(
        "--ngram-size",
        type=int,
        default=DEFAULT_NGRAM_SIZE,
        help="BM25 한국어 문자 n-gram 크기",
    )
    parser.add_argument("--rerank-top-n", type=int, default=5, help="리랭크 뒤 최종 채택 순위 상한")
    parser.add_argument(
        "--rerank-model",
        default=DEFAULT_RERANK_MODEL,
        help="실제 모드의 OpenAI 리랭크 모델 (대역 모드에서는 호출하지 않음)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        local_embedder = BgeM3EmbeddingClient() if args.bge_m3 else None
        paths = run_retrieval_comparison(
            live=bool(args.live or args.bge_m3),
            embedding_model=(BgeM3EmbeddingClient.MODEL if args.bge_m3 else args.embedding_model),
            embedding_client=local_embedder,
            dimensions=(
                BgeM3EmbeddingClient.DIMENSIONS
                if args.bge_m3 and args.dimensions is None
                else args.dimensions
            ),
            top_k=args.top_k,
            cutoff=args.cutoff,
            sweep_start=args.sweep_start,
            sweep_end=args.sweep_end,
            sweep_step=args.sweep_step,
            policy_dir=args.policy_dir,
            golden_set_path=args.golden_set,
            labels_path=args.labels,
            output_dir=args.out_dir,
            cache_dir=args.cache_dir,
            rrf_k=args.rrf_k,
            rrf_cutoff=args.rrf_cutoff,
            ngram_size=args.ngram_size,
            rerank_top_n=args.rerank_top_n,
            rerank_model=args.rerank_model,
            strategies=_VECTOR_ONLY,
        )
    except (
        OptionalEmbeddingDependencyError,
        RetrievalConfigurationError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"검색 비교 실행 실패: {exc}", file=sys.stderr)
        return 2
    print(f"검색 비교 완료: {paths.markdown}")
    print(f"검색 비교 JSON: {paths.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
