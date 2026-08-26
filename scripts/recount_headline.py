"""헤드라인 수치를 커밋된 라이브 리포트에서 다시 센다 — **무과금·오프라인**.

**왜 있는가.** 미끼 기각 재현·`G06` 계보·`G30` 계보는 오래 **사람이 손으로 재집계한 값**
이었고, 그 수를 다시 계산하는 코드가 0줄이었다([미해결 36](../docs/tracking/findings.md)).
문서 넷이 동시에 틀려도 저장소는 초록이다.

**모집단의 정본은 이 파일이 아니라 문서다.** `docs/tracking/status.md` 의 "헤드라인 모집단을
규칙으로 고정한다" 절이 술어와 세는 규칙을 표로 들고 있고, 이 스크립트는 **그 표를 읽어**
적용한다. 순서가 반대면 — 구현이 목록을 들고 문서가 결과만 인용하면 — *"구현이 스스로
모집단을 정한다"* 가 된다. 미해결 36 이 미뤄진 이유가 정확히 그것이었다.

```bash
uv run python -m scripts.recount_headline          # 표의 값과 나란히 출력
uv run python -m scripts.recount_headline --json   # 기계가 읽는 형태
```

**리포트를 고르지 않는다** — `reports/evaluation-live*.json` 전수를 읽고 술어가 거른다.
고르는 순간 이 스크립트가 모집단의 두 번째 정의가 된다.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: 저장소 루트 — 이 파일 기준으로 유도한다.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: 모집단 규칙의 정본. **여기를 바꾸려면 문서를 바꿔야 한다.**
RULES_DOC: Path = REPO_ROOT / "docs" / "tracking" / "status.md"

#: 정본 표가 시작하는 자리. 문서 전체에서 표를 긁으면 옆 절의 판 표가 딸려 온다.
RULES_HEADING = "### 헤드라인 모집단을 규칙으로 고정한다"

#: 커밋된 라이브 리포트 전수.
REPORT_GLOB = "reports/evaluation-live*.json"

_CELL_NOISE = re.compile(r"[`*]")


@dataclass(frozen=True)
class Rule:
    """문서 표의 한 줄 — 헤드라인 하나의 모집단과 세는 방법."""

    #: 헤드라인 이름 (표의 첫 칸).
    name: str
    #: 모집단 술어 전수. 전부 참인 회차만 분모에 든다.
    predicates: tuple[str, ...]
    #: 세는 규칙 하나.
    counter: str
    #: 문서가 적어 둔 재집계값 `(분자, 분모)`.
    documented: tuple[int, int]


@dataclass(frozen=True)
class Recount:
    """규칙 하나를 리포트에 적용한 결과."""

    rule: Rule
    numerator: int
    denominator: int
    #: 분모에 든 리포트 스템 — 어긋났을 때 어느 회차가 들어왔는지 보려면 이게 필요하다.
    stems: tuple[str, ...]

    @property
    def matches(self) -> bool:
        return (self.numerator, self.denominator) == self.rule.documented


def _cell(text: str) -> str:
    return _CELL_NOISE.sub("", text).strip()


def parse_rules(doc: Path = RULES_DOC) -> tuple[Rule, ...]:
    """정본 표를 읽는다. 절이 없거나 표가 비면 **빈 튜플** — 검사가 그것을 잡는다."""
    text = doc.read_text(encoding="utf-8")
    if RULES_HEADING not in text:
        return ()
    section = text[text.index(RULES_HEADING) :]
    rules: list[Rule] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            if rules:
                break
            continue
        cells = [_cell(cell) for cell in line.strip("|").split("|")]
        if len(cells) != 4 or cells[0] in {"헤드라인", ""} or set(cells[1]) <= {"-", ":"}:
            continue
        share = re.fullmatch(r"(\d+)/(\d+)", cells[3])
        if share is None:
            continue
        rules.append(
            Rule(
                name=cells[0],
                predicates=tuple(part.strip() for part in cells[1].split("·") if part.strip()),
                counter=cells[2],
                documented=(int(share.group(1)), int(share.group(2))),
            )
        )
    return tuple(rules)


def _fingerprint(report: dict[str, Any]) -> dict[str, str]:
    conditions = report.get("conditions") or {}
    fingerprint = conditions.get("condition_fingerprint") or {}
    return {str(key): str(value) for key, value in fingerprint.items()}


def _agreement(report: dict[str, Any]) -> dict[str, Any]:
    return report.get("measurement_2_pipeline_agreement") or {}


def _outcome(report: dict[str, Any], case_id: str) -> dict[str, Any] | None:
    for outcome in _agreement(report).get("outcomes") or []:
        if outcome.get("case_id") == case_id:
            return dict(outcome)
    return None


def _holds(predicate: str, report: dict[str, Any]) -> bool:
    """술어 하나. **어휘를 모르면 참으로 넘기지 않고 예외다** — 조용히 모집단이 넓어진다."""
    if predicate == "measurement_2_executed":
        return bool(_agreement(report).get("executed"))
    field = predicate.removeprefix("fingerprint:")
    if field != predicate and "=" in field:
        name, _, value = field.partition("=")
        return _fingerprint(report).get(name) == value
    raise ValueError(
        f"모르는 모집단 술어다 — 어휘를 넓히려면 문서와 코드를 함께 고친다: {predicate!r}"
    )


def _count(counter: str, reports: list[dict[str, Any]]) -> tuple[int, int]:
    """세는 규칙 하나. 여기서도 모르는 어휘는 예외다."""
    if counter.startswith("sum:"):
        numerator_key, _, denominator_key = counter.removeprefix("sum:").partition("/")
        totals = [0, 0]
        for report in reports:
            agreement = _agreement(report)
            for index, key in enumerate((numerator_key, denominator_key)):
                value = agreement.get(key)
                if not isinstance(value, int):
                    raise ValueError(f"회차가 `{key}` 를 정수로 싣지 않는다 — 0 으로 채우지 않는다")
                totals[index] += value
        return totals[0], totals[1]
    if counter.startswith("case_status:"):
        case_id, _, status = counter.removeprefix("case_status:").partition("=")
        hits = sum(
            1 for report in reports if (_outcome(report, case_id) or {}).get("status") == status
        )
        return hits, len(reports)
    if counter.startswith("case_rejected:"):
        case_id = counter.removeprefix("case_rejected:")
        hits = sum(
            1 for report in reports if (_outcome(report, case_id) or {}).get("reject_reasons")
        )
        return hits, len(reports)
    raise ValueError(
        f"모르는 세는 규칙이다 — 어휘를 넓히려면 문서와 코드를 함께 고친다: {counter!r}"
    )


def reports(root: Path = REPO_ROOT) -> tuple[tuple[str, dict[str, Any]], ...]:
    """`(스템, 리포트)` 전수. 이름 순이 아니라 **번호 순**으로 든다."""
    paths = sorted(root.glob(REPORT_GLOB), key=lambda path: (len(path.name), path.name))
    return tuple(
        (path.name.removesuffix(".json"), json.loads(path.read_text(encoding="utf-8")))
        for path in paths
    )


def recount(root: Path = REPO_ROOT, doc: Path = RULES_DOC) -> tuple[Recount, ...]:
    """정본 표의 규칙 전수를 커밋된 리포트에 적용한다."""
    loaded = reports(root)
    results: list[Recount] = []
    for rule in parse_rules(doc):
        chosen = [
            (stem, report)
            for stem, report in loaded
            if all(_holds(item, report) for item in rule.predicates)
        ]
        numerator, denominator = _count(rule.counter, [report for _stem, report in chosen])
        results.append(
            Recount(
                rule=rule,
                numerator=numerator,
                denominator=denominator,
                stems=tuple(stem for stem, _report in chosen),
            )
        )
    return tuple(results)


def main(argv: list[str] | None = None, root: Path = REPO_ROOT, doc: Path = RULES_DOC) -> int:
    """종료 코드 `0`(전부 일치) 또는 `1`. 규칙이 0건이어도 `1` 이다 — 빈 대조는 통과가 아니다.

    **`root`·`doc` 을 인자로 받는 것은 진입점을 실제로 부를 수 있게 하기 위해서다** — 종료
    코드를 테스트 쪽에서 다시 계산하면 그 검사는 동어반복이 된다(같은 함정을 한 번 밟았다).
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", action="store_true", help="기계가 읽는 형태로 출력한다")
    args = parser.parse_args(argv)

    results = recount(root, doc)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "name": item.rule.name,
                        "predicates": list(item.rule.predicates),
                        "counter": item.rule.counter,
                        "documented": list(item.rule.documented),
                        "recounted": [item.numerator, item.denominator],
                        "stems": list(item.stems),
                    }
                    for item in results
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for item in results:
            문서 = "/".join(str(value) for value in item.rule.documented)
            표시 = "일치" if item.matches else "**어긋남**"
            잰_값 = f"{item.numerator}/{item.denominator}"
            print(f"{item.rule.name}: 재집계 {잰_값} · 문서 {문서} — {표시}")
            print(f"  모집단 {len(item.stems)}회: {', '.join(item.stems)}")
    if not results:
        print("정본 표에서 규칙을 하나도 읽지 못했다 — 빈 대조는 통과가 아니다")
        return 1
    return 0 if all(item.matches for item in results) else 1


if __name__ == "__main__":  # pragma: no cover - 진입점
    raise SystemExit(main())
