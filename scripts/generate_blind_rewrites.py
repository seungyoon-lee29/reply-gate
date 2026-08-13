"""blind 재작성 질의 픽스처를 실제 생성 모델로 만든다.

    uv run python -m scripts.generate_blind_rewrites --live --out <경로>

**blind 조건은 이 스크립트의 입력 목록이 지킨다.** 모델에 들어가는 것은 골든셋 문의
원문뿐이고, 검색 정답 라벨·정책 문서·oracle 픽스처는 읽지 않는다 —
`tests/test_rewritten_queries.py` 의 구조 테스트가 이 파일을 검사한다
(`docs/standards.md` 의 "평가 입력 격리").

**프롬프트·스키마·모델·effort 를 런타임과 공유한다** — 문장은
`reply_gate.query_rewrite` 가 단독 소유하고 이 스크립트가 가져다 쓴다. 두 벌로 복사하면
픽스처의 생성 조건과 런타임의 호출 조건이 조용히 갈리고, 그 순간 오프라인 비교표가
런타임을 더 이상 예측하지 못한다. 이 픽스처가 재는 것은 "정답을 모르는 재작성기가 실제로
낼 수 있는 질의"이므로, 상위 모델이나 사람이 다듬은 문장을 쓰면 배포 가능한 이득이 아니라
상한을 재게 된다(`docs/tracking/decisions/0010`).

산출은 기존 픽스처를 덮어쓰지 않는다 — `--out` 을 반드시 받아 그 경로에만 쓴다.
검토한 뒤 사람이 `data/rewritten_queries.jsonl` 로 옮긴다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

from reply_gate.config import get_settings
from reply_gate.evaluation import DEFAULT_GOLDEN_SET_PATH, load_golden_set
from reply_gate.llm import GenerationClient, LLMCallError, LLMFormatError, OpenAIGenerationClient
from reply_gate.query_rewrite import (
    REWRITE_JSON_SCHEMA,
    REWRITE_SYSTEM_PROMPT,
    build_rewrite_user_prompt,
)

#: 처리 기록·비용 산출의 단계 이름과 섞이지 않는 픽스처 제작 전용 이름이다
#: (런타임은 `query_rewrite.QUERY_REWRITE_STAGE`). **단계 이름만 다르고 프롬프트·스키마·
#: 모델·effort 는 런타임과 같은 한 벌이다.**
FIXTURE_REWRITE_STAGE: Final = "fixture_query_rewrite"


def rewrite_one(*, client: GenerationClient, inquiry: str, effort: str | None = None) -> str:
    """문의 1건의 재작성 질의를 받는다. 형식 위반은 호출자에게 그대로 올린다."""
    completion = client.complete_json(
        stage=FIXTURE_REWRITE_STAGE,
        system=REWRITE_SYSTEM_PROMPT,
        user=build_rewrite_user_prompt(inquiry=inquiry),
        schema=REWRITE_JSON_SCHEMA,
        schema_name="rewrite",
        effort=effort,
    )
    data = completion.data
    if not isinstance(data, dict):
        raise LLMFormatError(stage=FIXTURE_REWRITE_STAGE, detail="산출이 JSON 객체가 아니다")
    rewritten = data.get("rewritten")
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise LLMFormatError(
            stage=FIXTURE_REWRITE_STAGE, detail="rewritten 이 비어 있지 않은 문자열이 아니다"
        )
    return rewritten.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="골든셋 문의 원문만 보고 blind 재작성 질의 픽스처를 생성한다"
    )
    parser.add_argument("--out", type=Path, required=True, help="산출 경로(JSONL)")
    parser.add_argument(
        "--golden-set", type=Path, default=DEFAULT_GOLDEN_SET_PATH, help="골든셋 경로"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="실제 생성 API 를 호출한다. 없으면 프롬프트만 출력하고 과금하지 않는다",
    )
    args = parser.parse_args(argv)

    if args.out.exists():
        print(f"산출 경로가 이미 있다: {args.out}", file=sys.stderr)
        return 2

    settings = get_settings()
    cases = load_golden_set(args.golden_set)

    if not args.live:
        print("--live 없음 — 호출하지 않는다. 첫 케이스 프롬프트만 보인다:\n")
        print(REWRITE_SYSTEM_PROMPT)
        print(build_rewrite_user_prompt(inquiry=cases[0].content))
        print(f"\n대상 {len(cases)}건 · 모델 {settings.generation_model}")
        return 0

    if not settings.openai_api_key:
        print("OPENAI_API_KEY 가 비어 있다. .env 에 값을 채운 뒤 다시 실행한다.", file=sys.stderr)
        return 2

    client = OpenAIGenerationClient(
        api_key=settings.openai_api_key, model=settings.generation_model
    )
    note = (
        f"생성 모델 {settings.generation_model} 이 문의 원문만 보고 만든 재작성이다 "
        "— 정책 문서·검색 정답 라벨을 입력으로 받지 않았다."
    )

    lines: list[str] = []
    for case in cases:
        try:
            rewritten = rewrite_one(
                client=client, inquiry=case.content, effort=settings.generation_effort
            )
        except (LLMCallError, LLMFormatError) as exc:
            print(f"{case.id} 재작성 실패: {exc}", file=sys.stderr)
            return 1
        lines.append(
            json.dumps(
                {
                    "id": case.id,
                    "original": case.content,
                    "rewritten": rewritten,
                    "note": note,
                },
                ensure_ascii=False,
            )
        )
        print(f"{case.id}: {rewritten}")

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{len(lines)}건 생성 완료 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
