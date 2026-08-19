"""DB 없는 정책 검색 전략 비교 실행 진입점.

    uv run python -m scripts.compare_retrieval --stub-embedding
    uv run python -m scripts.compare_retrieval --live  # 실제 임베딩 호출, 과금 가능

기본 모드는 결정론 어휘 임베딩 대역이다. 실제 모델은 `--live`로만 호출한다.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from reply_gate.config import get_settings
from reply_gate.evaluation import DEFAULT_GOLDEN_SET_PATH
from reply_gate.llm import BgeM3EmbeddingClient, OptionalEmbeddingDependencyError
from reply_gate.policy_index import DEFAULT_POLICY_DIR
from reply_gate.retrieval_eval import (
    DEFAULT_CHUNK_MIN_CONTAINMENT,
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_FUSION_POOL_SIZE,
    DEFAULT_NGRAM_SIZE,
    DEFAULT_ORACLE_REWRITTEN_QUERIES_PATH,
    DEFAULT_RETRIEVAL_REPORT_DIR,
    DEFAULT_REWRITTEN_QUERIES_PATH,
    DEFAULT_RRF_CUTOFF,
    DEFAULT_RRF_K,
    ReportPaths,
    RetrievalConfigurationError,
    RewriteCondition,
    run_chunking_comparison,
    run_embedding_model_axis,
    run_retrieval_comparison,
)
from reply_gate.retrieval_labels import DEFAULT_RETRIEVAL_LABELS_PATH
from reply_gate.retrieval_strategies import (
    RetrievalStage,
    StrategyDefinition,
    default_strategy_ladder,
)

_VECTOR_ONLY = (StrategyDefinition("vector", (RetrievalStage.VECTOR,)),)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DB 없이 정책 조항과 골든셋 문의의 네 단계 검색 전략을 비교한다. "
            "기본 재작성 입력은 정책·라벨을 본 적 없는 생성 모델이 원문만 보고 만든 "
            "독립 blind/deployable 픽스처다."
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
        help=(
            "선택 설치한 로컬 BGE-M3 1024차원 임베딩을 쓴다 (DB 없음, 임베딩은 로컬). "
            "리랭크 단은 --rerank-with-openai 없이는 미측정으로 남는다"
        ),
    )
    parser.add_argument(
        "--rerank-with-openai",
        action="store_true",
        help=(
            "로컬 임베딩 실행에서도 LLM 리랭크를 OpenAI로 실제 호출한다 "
            "(**과금**, 명시적 opt-in. --live 는 이미 포함한다)"
        ),
    )
    parser.add_argument(
        "--embedding-axis",
        action="store_true",
        help=(
            "임베딩 모델 축(3-small/3-large 1536·3072/BGE-M3)을 한 번에 비교한다. "
            "--live 필수이며 미설치 행은 사유와 함께 미측정으로 남는다 (**과금**)"
        ),
    )
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_RETRIEVAL_LABELS_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RETRIEVAL_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_EMBEDDING_CACHE_DIR)
    strategy = parser.add_mutually_exclusive_group()
    strategy.add_argument(
        "--vector-only",
        action="store_true",
        help="재작성·하이브리드·리랭크를 제외한 벡터 기준선만 실행한다",
    )
    strategy.add_argument(
        "--oracle-rewrite",
        action="store_true",
        help="정답 정책 어휘를 반영한 curated upper-bound 재작성 조건을 명시적으로 실행한다",
    )
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
        help=(
            "하이브리드 후보 풀의 RRF 하한 (기본 0.0 = 무필터). 순위 기반 점수라 절대 "
            "관련성을 뜻하지 않으며, 이 코퍼스에서 도달 불가능한 값은 거부된다"
        ),
    )
    parser.add_argument(
        "--fusion-pool",
        type=int,
        default=DEFAULT_FUSION_POOL_SIZE,
        help="융합 후보 풀 크기 (리랭크 최종 채택 상한보다 커야 한다)",
    )
    parser.add_argument(
        "--ngram-size",
        type=int,
        default=DEFAULT_NGRAM_SIZE,
        help="BM25 한국어 문자 n-gram 크기",
    )
    parser.add_argument("--rerank-top-n", type=int, default=5, help="리랭크 뒤 최종 채택 순위 상한")
    parser.add_argument(
        "--no-abstention-grid",
        action="store_true",
        help=(
            "질의 단위 기권 게이트 격자를 돌리지 않는다 (기본은 켜짐, 무과금 — 검색을 다시 "
            "돌리지 않고 채점만 얹는다). 끄면 리포트에 미실행 사유가 남는다"
        ),
    )
    parser.add_argument(
        "--chunking-grid",
        action="store_true",
        help=(
            "조항 단위 vs 고정 크기 청킹 비교 격자를 돌린다 (**측정 전용** — 제품에 아무것도 "
            "반영하지 않는다). 다른 전략 축과 함께 쓸 수 없다. --live 면 고정 크기 청크는 "
            "캐시에 없는 새 텍스트라 **과금**된다"
        ),
    )
    parser.add_argument(
        "--min-containment",
        type=float,
        default=DEFAULT_CHUNK_MIN_CONTAINMENT,
        help=(
            "청크가 조항을 적중했다고 셀 최소 본문 포함 비율 (0 초과 1 이하). "
            "리포트가 선언값과 민감도를 함께 인쇄한다"
        ),
    )
    parser.add_argument(
        "--rerank-model",
        default=None,
        help="실제 모드의 OpenAI 리랭크 모델 (기본: 환경 설정값, 대역 모드에서는 호출하지 않음)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chunking_grid:
        return _run_chunking(args)
    if args.embedding_axis and not args.live:
        print(
            "검색 비교 실행 실패: --embedding-axis 는 실제 모델을 호출하므로 --live 가 필요하다",
            file=sys.stderr,
        )
        return 2
    rewrite_condition = RewriteCondition.ORACLE if args.oracle_rewrite else RewriteCondition.BLIND
    strategies = _VECTOR_ONLY if args.vector_only else default_strategy_ladder()
    rewritten_queries_path = (
        DEFAULT_ORACLE_REWRITTEN_QUERIES_PATH
        if args.oracle_rewrite
        else DEFAULT_REWRITTEN_QUERIES_PATH
    )

    def compare(
        *,
        live: bool,
        embedding_model: str | None,
        embedding_client: object | None,
        dimensions: int | None,
        paid_rerank: bool,
    ) -> ReportPaths:
        return run_retrieval_comparison(
            live=live,
            embedding_model=embedding_model,
            embedding_client=cast(Any, embedding_client),
            dimensions=dimensions,
            top_k=args.top_k,
            cutoff=args.cutoff,
            sweep_start=args.sweep_start,
            sweep_end=args.sweep_end,
            sweep_step=args.sweep_step,
            policy_dir=args.policy_dir,
            golden_set_path=args.golden_set,
            labels_path=args.labels,
            rewritten_queries_path=rewritten_queries_path,
            output_dir=args.out_dir,
            cache_dir=args.cache_dir,
            rrf_k=args.rrf_k,
            rrf_cutoff=args.rrf_cutoff,
            fusion_pool_size=args.fusion_pool,
            ngram_size=args.ngram_size,
            rerank_top_n=args.rerank_top_n,
            rerank_model=args.rerank_model,
            paid_rerank=paid_rerank,
            rewrite_condition=rewrite_condition,
            strategies=strategies,
            abstention_grid=not args.no_abstention_grid,
        )

    try:
        if args.embedding_axis:
            # 축 경로도 비축 경로와 **같은 식**을 쓴다. 한 플래그가 자리에 따라 다른 뜻이면
            # `--rerank-with-openai` 의 "--live 는 이미 포함한다"가 거짓이 되고, 4단 사다리를
            # 기대한 실행이 조용히 3단으로 잘린다(실제로 그랬다).
            return _run_axis(
                compare=compare, paid_rerank=bool(args.live or args.rerank_with_openai)
            )
        local_embedder = BgeM3EmbeddingClient() if args.bge_m3 else None
        paths = compare(
            # 로컬 임베딩도 "실제 임베딩"이므로 live 경로다. 유료 리랭크는 별개 축으로
            # 명시해야 켜진다 — 무과금이라고 안내한 명령이 과금하지 않게 한다.
            live=bool(args.live or args.bge_m3),
            embedding_model=(BgeM3EmbeddingClient.MODEL if args.bge_m3 else args.embedding_model),
            embedding_client=local_embedder,
            dimensions=(
                BgeM3EmbeddingClient.DIMENSIONS
                if args.bge_m3 and args.dimensions is None
                else args.dimensions
            ),
            paid_rerank=bool(args.live or args.rerank_with_openai),
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


def _run_chunking(args: argparse.Namespace) -> int:
    """청킹 축은 다른 축과 섞이지 않는다 — 변수는 청킹 하나뿐이어야 비교가 성립한다."""
    conflicting = [
        name
        for name, chosen in (
            ("--bge-m3", args.bge_m3),
            ("--embedding-axis", args.embedding_axis),
            ("--vector-only", args.vector_only),
            ("--oracle-rewrite", args.oracle_rewrite),
            ("--rerank-with-openai", args.rerank_with_openai),
        )
        if chosen
    ]
    if conflicting:
        print(
            "검색 비교 실행 실패: --chunking-grid 는 "
            f"{' · '.join(conflicting)} 와 함께 쓸 수 없다 (청킹만이 변수여야 한다)",
            file=sys.stderr,
        )
        return 2
    try:
        paths = run_chunking_comparison(
            live=bool(args.live),
            embedding_model=args.embedding_model if args.live else None,
            dimensions=args.dimensions,
            top_k=args.top_k,
            cutoff=args.cutoff,
            min_containment=args.min_containment,
            policy_dir=args.policy_dir,
            golden_set_path=args.golden_set,
            labels_path=args.labels,
            output_dir=args.out_dir,
            cache_dir=args.cache_dir,
        )
    except (RetrievalConfigurationError, FileNotFoundError, ValueError) as exc:
        print(f"청킹 비교 실행 실패: {exc}", file=sys.stderr)
        return 2
    print(f"청킹 비교 완료: {paths.markdown}")
    print(f"청킹 비교 JSON: {paths.json}")
    return 0


def _run_axis(
    *,
    compare: Callable[..., ReportPaths],
    paid_rerank: bool,
) -> int:
    """임베딩 모델 축을 돌린다. 한 행의 미실행이 다른 행을 막지 않는다."""
    result = run_embedding_model_axis(
        api_key=get_settings().openai_api_key,
        evaluate=lambda candidate, client: compare(
            live=True,
            embedding_model=candidate.model,
            embedding_client=client,
            dimensions=candidate.dimensions,
            paid_rerank=paid_rerank,
        ),
    )
    for row in result.rows:
        if row.measured and row.reports is not None:
            print(f"{row.candidate.key}: {row.reports.markdown}")
        else:
            print(f"{row.candidate.key}: 미측정 — {row.reason}")
    if not any(row.measured for row in result.rows):
        print("임베딩 모델 축 전 행이 미측정이다", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
