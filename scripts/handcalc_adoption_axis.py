"""채택 축 손계산을 산출물로 굽는다 — 무과금.

    uv run python -m scripts.handcalc_adoption_axis

입력은 저장소에 커밋된 검색 리포트(`reports/retrieval-strategies-live-*.json`)뿐이다.
새 임베딩도 새 API 호출도 없으므로 재실행이 공짜이고, 같은 입력에서 같은 산출을 낸다.
그래서 이 산출물은 **덮어쓴다** — 재생성에 과금이 드는 라이브 리포트와 이름 계열이 갈려
있고(`scripts/AGENTS.md` 불변식 7), 재생성이 공짜인 산출물이 라이브 이름을 쓰지 않는다.

계산 본체는 `reply_gate.adoption_axis` 가 갖는다. 이 파일은 인자를 받고 함수를 부르고
파일로 쓰는 껍데기다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reply_gate.adoption_axis import (
    DEFAULT_REPORT_DIR,
    default_conditions,
    render_markdown,
    run_hand_calculation,
    to_json,
)

#: 라이브 리포트 계열과 겹치지 않는 이름. 이 산출물은 실측이 아니라 재계산이다.
REPORT_STEM = "adoption-axis-handcalc"


def main() -> int:
    parser = argparse.ArgumentParser(description="채택 축 손계산 (무과금)")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="커밋된 검색 리포트를 읽고 산출물을 쓸 디렉터리",
    )
    args = parser.parse_args()

    result = run_hand_calculation(default_conditions(args.report_dir))
    markdown = render_markdown(result)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = args.report_dir / f"{REPORT_STEM}.md"
    json_path = args.report_dir / f"{REPORT_STEM}.json"
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(to_json(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(markdown)
    print(f"\n{markdown_path}\n{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
