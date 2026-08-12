"""blind 재작성 질의 픽스처를 실제 생성 모델로 만든다.

    uv run python -m scripts.generate_blind_rewrites --live --out <경로>

**blind 조건은 이 스크립트의 입력 목록이 지킨다.** 모델에 들어가는 것은 골든셋 문의
원문뿐이고, 검색 정답 라벨·정책 문서·oracle 픽스처는 읽지 않는다 —
`tests/test_rewritten_queries.py` 의 구조 테스트가 이 파일을 검사한다
(`docs/standards.md` 의 "평가 입력 격리").

모델·프롬프트는 런타임 의도 해석과 같은 계열을 쓴다. 이 픽스처가 재는 것은
"정답을 모르는 재작성기가 실제로 낼 수 있는 질의"이므로, 상위 모델이나 사람이 다듬은
문장을 쓰면 배포 가능한 이득이 아니라 상한을 재게 된다(`docs/tracking/decisions/0010`).

산출은 기존 픽스처를 덮어쓰지 않는다 — `--out` 을 반드시 받아 그 경로에만 쓴다.
검토한 뒤 사람이 `data/rewritten_queries.jsonl` 로 옮긴다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final

from reply_gate.config import get_settings
from reply_gate.evaluation import DEFAULT_GOLDEN_SET_PATH, load_golden_set
from reply_gate.llm import GenerationClient, LLMCallError, LLMFormatError, OpenAIGenerationClient

#: 처리 기록·비용 산출의 단계 이름과 섞이지 않는 픽스처 제작 전용 이름이다.
REWRITE_STAGE: Final = "fixture_query_rewrite"

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
    """재작성 프롬프트. 문의 원문 외에는 아무것도 싣지 않는다."""
    return f"[문의]\n{inquiry}"


def rewrite_one(*, client: GenerationClient, inquiry: str) -> str:
    """문의 1건의 재작성 질의를 받는다. 형식 위반은 호출자에게 그대로 올린다."""
    completion = client.complete_json(
        stage=REWRITE_STAGE,
        system=REWRITE_SYSTEM_PROMPT,
        user=build_rewrite_user_prompt(inquiry=inquiry),
        schema=REWRITE_JSON_SCHEMA,
        schema_name="rewrite",
    )
    data = completion.data
    if not isinstance(data, dict):
        raise LLMFormatError(stage=REWRITE_STAGE, detail="산출이 JSON 객체가 아니다")
    rewritten = data.get("rewritten")
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise LLMFormatError(
            stage=REWRITE_STAGE, detail="rewritten 이 비어 있지 않은 문자열이 아니다"
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
            rewritten = rewrite_one(client=client, inquiry=case.content)
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
