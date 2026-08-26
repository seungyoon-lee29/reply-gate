"""번호 계열 접기를 넓히기 **전/후**의 탐지 차이를 모집단을 명시해 센다 (무과금·오프라인).

```bash
uv run python -m scripts.contrast_pii_folding
```

**왜 있나.** 안전장치를 넓히는 수정은 미탐만 닫는 것이 아니라 오탐도 함께 연다. 그래서
채택 전에 **양방향**을 같은 대조에서 낸다 — 새로 잡히는 값(미탐이 닫힌 건수)과 사라지는
값(정상 에코가 기각될 건수)을 함께 센다. 오탐이 늘면 채택하지 않는다.

**전/후를 어떻게 재나.** 수정 전 접기는 지금도 모듈에 살아 있다 — 이메일 패턴이 쓰는
공통 접기(`gate.fold_for_detection`)가 그것이다. 그래서 이 대조는 옛 코드를 재현하지
않고, 같은 패턴 집합에 **접기만 갈아 끼워** 두 판정을 나란히 낸다. 병합 뒤에도 그대로
재현된다.

**모집단.**

- **정책 조항 전문** — 전수 (`data/policies/*.md` 의 모든 줄)
- **주문 테이블 값** — 전수 (`db/fixtures/orders.jsonl` 의 컬럼 이름과 값 전부.
  이 픽스처가 곧 테이블 내용이다)
- **L1 픽스처** — 전수 (초안 텍스트 · 근거 텍스트)
- **골든셋 문의** — 전수
- **실제 초안 텍스트** — **프로브**. 커밋된 리포트가 초안 문면을 담지 않아 자유 모집단이
  없다. 결정론 대역 생성기로 골든셋 문의 전건의 초안을 만들어 대신 잰다

마지막 줄이 이 대조의 한계다: **전수가 아니라 프로브다.** 대역 생성기는 실제 모델이 아니고
표기 변형을 스스로 만들어 내지도 않으므로, 여기서 "변화 0" 이 나왔다고 실제 모델의 초안이
안 바뀐다는 뜻은 아니다. 자유 모집단이 없는 축에서 낼 수 있는 최선의 관측이라는 뜻이다.

산출물은 표준 출력뿐이다 — 파일을 남기지 않는다. 재현이 무과금이라 재실행이 곧 근거다.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from reply_gate.contracts import Evidence, EvidenceSource
from reply_gate.draft import DraftGenerator
from reply_gate.evaluation import (
    DEFAULT_GOLDEN_SET_PATH,
    DEFAULT_L1_FIXTURES_PATH,
    StubGenerationClient,
)
from reply_gate.gate import DEFAULT_PII_PATTERNS, PiiPattern, fold_for_detection

_ROOT: Final = Path(__file__).resolve().parents[1]
_POLICY_DIR: Final = _ROOT / "data" / "policies"
_ORDER_FIXTURES: Final = _ROOT / "db" / "fixtures" / "orders.jsonl"

#: `(패턴 이름, 정규화된 값)` — 두 접기가 낸 판정을 비교하는 단위.
Detection = tuple[str, str]


@dataclass(frozen=True)
class TextUnit:
    """대조 단위 1건 — 어디서 왔는지를 함께 든다(차이가 나면 이름으로 찍어야 한다)."""

    origin: str
    text: str


@dataclass(frozen=True)
class Contrast:
    """모집단 하나의 대조 결과."""

    population: str
    census: bool
    units: int
    gained: tuple[tuple[str, Detection], ...]
    lost: tuple[tuple[str, Detection], ...]

    @property
    def headline(self) -> str:
        scope = "전수" if self.census else "프로브"
        return (
            f"{self.population} ({scope}, {self.units}건): "
            f"신규 탐지 {len(self.gained)}건 · 탐지 소실 {len(self.lost)}건"
        )


def _detect(text: str, *, fold_all_as_email: bool) -> set[Detection]:
    """`fold_all_as_email=True` 면 **수정 전** 동작이다 — 전 패턴이 공통 접기를 쓴다."""

    def fold(pattern: PiiPattern) -> str:
        return fold_for_detection(text) if fold_all_as_email else pattern.fold(text)

    return {
        (pattern.name, pattern.normalize(match))
        for pattern in DEFAULT_PII_PATTERNS
        for match in pattern.regex.findall(fold(pattern))
    }


def contrast(population: str, units: Sequence[TextUnit], *, census: bool) -> Contrast:
    gained: list[tuple[str, Detection]] = []
    lost: list[tuple[str, Detection]] = []
    for unit in units:
        before = _detect(unit.text, fold_all_as_email=True)
        after = _detect(unit.text, fold_all_as_email=False)
        gained.extend((unit.origin, item) for item in sorted(after - before))
        lost.extend((unit.origin, item) for item in sorted(before - after))
    return Contrast(
        population=population,
        census=census,
        units=len(units),
        gained=tuple(gained),
        lost=tuple(lost),
    )


# ── 모집단 ──────────────────────────────────────────────────────────────────


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed: dict[str, Any] = json.loads(line)
            yield parsed


def policy_units() -> list[TextUnit]:
    return [
        TextUnit(origin=f"{path.name}:{number}", text=line)
        for path in sorted(_POLICY_DIR.rglob("*.md"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if line.strip()
    ]


def order_units() -> list[TextUnit]:
    """주문 값 전수 — **컬럼 이름도 값과 같은 자격으로** 센다(근거 필터가 이름도 거른다)."""
    units: list[TextUnit] = []
    for row in _jsonl(_ORDER_FIXTURES):
        order_no = str(row.get("order_no", "?"))
        for key, value in row.items():
            units.append(TextUnit(origin=f"{order_no}.{key}(이름)", text=str(key)))
            units.append(TextUnit(origin=f"{order_no}.{key}", text=str(value)))
    return units


def l1_fixture_units() -> list[TextUnit]:
    units: list[TextUnit] = []
    for row in _jsonl(DEFAULT_L1_FIXTURES_PATH):
        fixture_id = str(row["id"])
        for index, evidence in enumerate(row["evidences"], start=1):
            text = str(evidence.get("evidence_text", evidence["content"]))
            units.append(TextUnit(origin=f"{fixture_id}.근거{index}", text=text))
        units.extend(
            TextUnit(origin=f"{fixture_id}.초안{index}", text=text)
            for index, text in enumerate(_draft_texts(row["raw_draft"]), start=1)
        )
    return units


def golden_units() -> list[TextUnit]:
    return [
        TextUnit(origin=str(row["id"]), text=str(row["content"]))
        for row in _jsonl(DEFAULT_GOLDEN_SET_PATH)
    ]


def _draft_texts(raw_draft: object) -> list[str]:
    if not isinstance(raw_draft, dict):
        return [str(raw_draft)]
    claims = raw_draft.get("claims")
    if not isinstance(claims, list):
        return [json.dumps(raw_draft, ensure_ascii=False)]
    return [
        str(claim["text"])
        for claim in claims
        if isinstance(claim, dict) and isinstance(claim.get("text"), str)
    ]


def _policy_evidence() -> tuple[Evidence, ...]:
    return tuple(
        Evidence(
            id=f"policy:{path.stem}:{number}",
            source=EvidenceSource.POLICY,
            content=line,
            evidence_text=line,
        )
        for path in sorted(_POLICY_DIR.rglob("*.md"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if line.strip()
    )


def stub_draft_units() -> list[TextUnit]:
    """**프로브** — 결정론 대역 생성기로 골든셋 문의 전건의 초안을 만든다 (외부 호출 0회).

    근거는 정책 조항 전문을 그대로 준다. 실제 실행은 검색이 고른 부분집합을 주지만, 이
    프로브가 재려는 것은 검색 품질이 아니라 **초안 문면이 접기 변경으로 달라지는가** 다.
    """
    evidence = _policy_evidence()
    drafter = DraftGenerator(client=StubGenerationClient())
    units: list[TextUnit] = []
    for row in _jsonl(DEFAULT_GOLDEN_SET_PATH):
        generation = drafter.generate(inquiry=str(row["content"]), evidence=evidence)
        units.extend(
            TextUnit(origin=f"{row['id']}.대역초안{index}", text=text)
            for index, text in enumerate(_draft_texts(generation.raw), start=1)
        )
    return units


@dataclass(frozen=True)
class VerdictContrast:
    """초안 1건의 `pii_detected` 판정이 접기로 뒤집히는지 — **양방향을 한 대조에서** 낸다."""

    population: str
    census: bool
    drafts: int
    #: 우회 라벨이 붙은 초안이 새로 기각됐다 — **미탐이 닫힌 건수**.
    미탐_닫힘: tuple[str, ...]
    #: 정상 라벨이 붙은 초안이 새로 기각됐다 — **새로 생긴 오탐**. 이 칸이 0 이 아니면
    #: 채택하지 않는다 — 정상 초안 오탐률은 이 제품의 헤드라인 지표 하나다.
    #: 라벨이 없으면 여기 담지 않는다.
    새_오탐: tuple[str, ...]
    #: 라벨이 없는 초안이 새로 기각됐다 — 방향을 단정할 수 없다.
    분류_불가: tuple[str, ...]
    #: 기각이던 것이 통과로 풀렸다. 정상 에코의 오기각이 풀린 자리가 여기다.
    기각_풀림: tuple[str, ...]

    @property
    def headline(self) -> str:
        scope = "전수" if self.census else "프로브"
        꼬리 = f" · 분류 불가 {len(self.분류_불가)}건" if self.분류_불가 else ""
        return (
            f"{self.population} ({scope}, 초안 {self.drafts}건): "
            f"미탐 닫힘 {len(self.미탐_닫힘)}건 · "
            f"새 오탐 {len(self.새_오탐)}건 · "
            f"기각 풀림 {len(self.기각_풀림)}건{꼬리}"
        )


def _unsourced(
    *, draft_texts: Sequence[str], evidence_texts: Sequence[str], fold_all_as_email: bool
) -> bool:
    """`gate._has_unsourced_pii` 와 같은 대조 — 접기만 갈아 끼운다."""
    found: set[Detection] = set()
    for text in draft_texts:
        found |= _detect(text, fold_all_as_email=fold_all_as_email)
    if not found:
        return False
    allowed: set[Detection] = set()
    for text in evidence_texts:
        allowed |= _detect(text, fold_all_as_email=fold_all_as_email)
    return bool({value for _name, value in found} - {value for _name, value in allowed})


def verdict_contrast(population: str, cases: Sequence[Case], *, census: bool) -> VerdictContrast:
    """**새로 기각된 건수를 라벨로 가른다.**

    가르지 않으면 우회를 닫은 건수와 정상 초안을 새로 기각한 건수가 한 칸에 들어가고,
    그 칸은 접기를 넓힐수록 커진다 — 오탐이 늘어도 성과처럼 읽힌다. 그러면
    *"오탐이 늘면 채택하지 않는다"* 를 이 대조로는 발동시킬 수 없다.
    """
    미탐_닫힘: list[str] = []
    새_오탐: list[str] = []
    분류_불가: list[str] = []
    기각_풀림: list[str] = []
    for case in cases:
        before = _unsourced(
            draft_texts=case.draft_texts,
            evidence_texts=case.evidence_texts,
            fold_all_as_email=True,
        )
        after = _unsourced(
            draft_texts=case.draft_texts,
            evidence_texts=case.evidence_texts,
            fold_all_as_email=False,
        )
        if after and not before:
            if case.우회인가 is None:
                분류_불가.append(case.case_id)
            elif case.우회인가:
                미탐_닫힘.append(case.case_id)
            else:
                새_오탐.append(case.case_id)
        if before and not after:
            기각_풀림.append(case.case_id)
    return VerdictContrast(
        population=population,
        census=census,
        drafts=len(cases),
        미탐_닫힘=tuple(미탐_닫힘),
        새_오탐=tuple(새_오탐),
        분류_불가=tuple(분류_불가),
        기각_풀림=tuple(기각_풀림),
    )


@dataclass(frozen=True)
class Case:
    """판정 대조 1건. **라벨이 있어야 방향을 가를 수 있다.**

    `우회인가` 는 이 초안이 **막아야 할 우회**인지(True), **통과해야 할 정상**인지(False),
    라벨이 없는지(None)를 담는다. 이것이 없으면 새로 기각된 건수가 "미탐 닫힘"과
    "새 오탐" 을 한 칸에 뭉쳐 담고, 그러면 *"오탐이 늘면 채택하지 않는다"* 가 발동할 수
    없다 — 오탐 증가가 성과 칸을 키우는 모양이 된다.
    """

    case_id: str
    draft_texts: Sequence[str]
    evidence_texts: Sequence[str]
    우회인가: bool | None


def _pii_우회_기대(row: dict[str, Any]) -> bool:
    """이 픽스처가 `pii_detected` 로 기각되기를 기대하는가."""
    expected = row.get("expected")
    if not isinstance(expected, dict):
        return False
    reasons = expected.get("reject_reasons")
    reasons = reasons if isinstance(reasons, list) else []
    return expected.get("verdict") == "reject" and "pii_detected" in reasons


def l1_fixture_cases() -> list[Case]:
    return [
        Case(
            case_id=str(row["id"]),
            draft_texts=_draft_texts(row["raw_draft"]),
            evidence_texts=[
                str(evidence.get("evidence_text", evidence["content"]))
                for evidence in row["evidences"]
            ],
            우회인가=_pii_우회_기대(row),
        )
        for row in _jsonl(DEFAULT_L1_FIXTURES_PATH)
    ]


def stub_draft_cases() -> list[Case]:
    """대역 초안에는 **라벨이 없다** — 우회인지 정상인지 이 프로브가 모른다.

    그래서 여기서 뒤집힌 판정은 방향을 단정하지 않고 `분류 불가` 로 따로 센다. 모르는
    것을 성과 칸에 넣지 않는 것이 이 대조의 목적이다.
    """
    evidence = _policy_evidence()
    drafter = DraftGenerator(client=StubGenerationClient())
    evidence_texts = [item.evidence_text for item in evidence]
    return [
        Case(
            case_id=str(row["id"]),
            draft_texts=_draft_texts(
                drafter.generate(inquiry=str(row["content"]), evidence=evidence).raw
            ),
            evidence_texts=evidence_texts,
            우회인가=None,
        )
        for row in _jsonl(DEFAULT_GOLDEN_SET_PATH)
    ]


def _report(contrasts: Iterable[Contrast]) -> None:
    for item in contrasts:
        print(item.headline)
        for origin, (pattern_name, value) in item.gained:
            print(f"    + {origin}: {pattern_name}={value}")
        for origin, (pattern_name, value) in item.lost:
            print(f"    - {origin}: {pattern_name}={value}")


def main() -> None:
    print("번호 계열 접기 전/후 대조 — 무과금·오프라인, 산출물 없음\n")
    print("[탐지값 차이] 텍스트 단위로 잡히는 값이 늘었는가 / 사라졌는가")
    _report(
        [
            contrast("정책 조항 전문", policy_units(), census=True),
            contrast("주문 테이블 값", order_units(), census=True),
            contrast("L1 픽스처", l1_fixture_units(), census=True),
            contrast("골든셋 문의", golden_units(), census=True),
            contrast("실제 초안 텍스트", stub_draft_units(), census=False),
        ]
    )

    print("\n[판정 차이] 초안 1건의 `pii_detected` 가 뒤집히는가 — 양방향을 한 대조에서")
    for item in (
        verdict_contrast("L1 픽스처", l1_fixture_cases(), census=True),
        verdict_contrast("실제 초안 텍스트", stub_draft_cases(), census=False),
    ):
        print(item.headline)
        for case_id in item.미탐_닫힘:
            print(f"    미탐 닫힘 (우회가 기각으로): {case_id}")
        for case_id in item.새_오탐:
            print(f"    ⚠ 새 오탐 (정상이 기각으로): {case_id}")
        for case_id in item.분류_불가:
            print(f"    분류 불가 (라벨 없는 초안이 기각으로): {case_id}")
        for case_id in item.기각_풀림:
            print(f"    기각 풀림 (기각이 통과로): {case_id}")

    print(
        "\n마지막 줄은 전수가 아니라 프로브다 — 커밋된 리포트가 초안 문면을 담지 않아 "
        "실제 초안의 자유 모집단이 없다."
    )
    print(
        "채택 조건은 **새 오탐 0건** 이다 — 오탐이 늘면 접기 확대를 채택하지 않는다. "
        "라벨이 없는 초안의 뒤집힘은 성과 칸에도 오탐 칸에도 넣지 않고 분류 불가로 따로 센다."
    )


if __name__ == "__main__":
    main()
