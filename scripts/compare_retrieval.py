"""DB 없는 정책 검색 비교 실행 진입점.

    uv run python -m scripts.compare_retrieval --stub-embedding
    uv run python -m scripts.compare_retrieval --live  # 실제 임베딩 호출, 과금 가능

기본 모드는 결정론 어휘 임베딩 대역이다. 실제 모델은 `--live`로만 호출한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reply_gate.evaluation import DEFAULT_GOLDEN_SET_PATH
from reply_gate.policy_index import DEFAULT_POLICY_DIR
from reply_gate.retrieval_eval import (
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_RETRIEVAL_REPORT_DIR,
    RetrievalConfigurationError,
    run_retrieval_comparison,
)
from reply_gate.retrieval_labels import DEFAULT_RETRIEVAL_LABELS_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DB 없이 정책 조항과 골든셋 문의의 벡터 단독 검색 품질을 비교한다"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--live",
        action="store_true",
        help="OpenAI 임베딩을 실제 호출한다 (명시적 opt-in, 캐시 miss는 과금 가능)",
    )
    mode.add_argument(
        "--stub-embedding",
        action="store_true",
        help="결정론 어휘 임베딩 대역을 쓴다 (기본값, 배관 검증 전용)",
    )
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_RETRIEVAL_LABELS_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RETRIEVAL_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_EMBEDDING_CACHE_DIR)
    parser.add_argument("--dimensions", type=int, default=None)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = run_retrieval_comparison(
            live=bool(args.live),
            dimensions=args.dimensions,
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
        )
    except (RetrievalConfigurationError, FileNotFoundError, ValueError) as exc:
        print(f"검색 비교 실행 실패: {exc}", file=sys.stderr)
        return 2
    print(f"검색 비교 완료: {paths.markdown}")
    print(f"검색 비교 JSON: {paths.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
