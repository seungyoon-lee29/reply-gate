"""표기 변형이 L1 판정을 비껴가지 못하게 하는 회귀 검사 — 그리고 반대 방향 구멍의 음성 대조.

사람이 같은 값으로 읽는 표기(대시 변종·중점·빗금·밑줄, 숫자 사이에 낀 결합 표식)로 쓴
번호가 초안 판정을 통과하면 지어낸 연락처가 그대로 답변에 실린다. 실제로 두 계열이
뚫려 있었다 — 무엇을 왜 이렇게 갈랐는지는 결정 기록 0021 이 든다
(`docs/tracking/decisions/0021-접기를-번호-계열-안에서만-넓힌다.md`).

**두 가지를 함께 못박는다.**

1. **미탐이 닫힌다** — 변형 표기로 지어낸 번호가 기각된다.
2. **오탐이 늘지 않는다** — 근거에 있는 값의 변형 표기 에코는 계속 통과하고, 이메일은
   밑줄·빗금을 접지 않아 `a_b@…` 와 `a-b@…` 가 서로 다른 값으로 남는다.

**배치가 판정을 가른다.** 결합 표식은 숫자열 **안쪽**에 끼워야 뚫리고 맨 끝에 붙이면
접기를 넓히기 전에도 기각된다 — 끝에 붙인 문자열로 검사하면 수정 전에도 초록이라
아무것도 지키지 못한다. 그래서 이 파일은 **자리 사이 배치**를 그대로 박는다.

눈으로 구분되지 않는 문자는 **이스케이프로 적는다**(`tests/test_gate.py` 의 표기 변형
케이스와 같은 규율) — 리터럴로 쓰면 무엇을 시험하는지 읽을 수 없고, 편집 중에 조용히
반각으로 바뀌어도 아무도 모른다.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from reply_gate.contracts import Evidence, EvidenceSource, RejectReason, Verdict
from reply_gate.evidence import _sql_evidence_texts
from reply_gate.gate import (
    DEFAULT_PII_PATTERNS,
    NUMERIC_SEPARATOR_VARIANTS,
    evaluate_draft,
    fold_for_detection,
    fold_numeric_for_detection,
    normalize_email,
    pii_shaped,
)

_ROOT = Path(__file__).resolve().parents[1]

POLICY_ID = "policy:support:4-1"
SQL_ID = "sql:inq_1:1"

#: ① 번호 계열이 **구분자로 받아야 하는** 표기 변종 12자.
#: 범주 이름으로 유도하지 않는다 — 이 12자는 네 범주에 흩어져 있다(아래 음성 대조 참조).
SEPARATOR_VARIANTS: tuple[str, ...] = (
    "\u2010",  # HYPHEN
    "\u2011",  # NON-BREAKING HYPHEN
    "\u2012",  # FIGURE DASH
    "\u2013",  # EN DASH
    "\u2014",  # EM DASH
    "\u2015",  # HORIZONTAL BAR
    "\u2212",  # MINUS SIGN
    "\u2027",  # HYPHENATION POINT
    "\u30fb",  # KATAKANA MIDDLE DOT
    "\uff65",  # HALFWIDTH KATAKANA MIDDLE DOT
    "/",  # SOLIDUS
    "_",  # LOW LINE
)

#: ② 번호 계열이 **무시해야 하는** 결합 표식의 회귀 검사 대표 2자.
IGNORABLE_REPRESENTATIVES: tuple[str, ...] = (
    "\u0301",  # COMBINING ACUTE ACCENT
    "\ufe0f",  # VARIATION SELECTOR-16
)

#: 대표 2자로 한정하면 같은 계열이 그대로 남는다는 것을 보이는 대조군.
#: 규칙 자체는 결합 표식 범주 **전체**(`Mn`·`Mc`·`Me`)이고, 변이선택자는 `Mn` 에 든다.
IGNORABLE_SAME_FAMILY: tuple[str, ...] = (
    "\u0300",  # COMBINING GRAVE ACCENT (Mn)
    "\ufe00",  # VARIATION SELECTOR-1 (Mn)
    "\U000e0100",  # VARIATION SELECTOR-17 (Mn)
    "\u0903",  # DEVANAGARI SIGN VISARGA (Mc — 간격을 차지하는 결합 표식)
    "\u20e3",  # COMBINING ENCLOSING KEYCAP (Me — 감싸는 결합 표식)
)

#: 결합 표식 계열이 갈리는 **세 범주**. `Mn` 하나만 보면 나머지 둘이 그대로 뚫린다.
COMBINING_CATEGORIES: tuple[str, ...] = ("Mn", "Mc", "Me")

#: 접기를 넓히기 전에도 이미 잡히던 계열 — 이번 변경으로 잃지 않았는지 함께 본다.
_FULLWIDTH_NUMBER = "\uff10\uff11\uff10-\uff19\uff19\uff19\uff19-\uff18\uff18\uff18\uff18"
_ZERO_WIDTH_NUMBER = "010-9999\u200b-8888"

_NO_PII_EVIDENCE = "고객센터 운영 시간은 평일 09시부터 18시까지입니다."


def _codepoint(char: str) -> str:
    return f"U+{ord(char):04X}"


def _evidence(
    evidence_id: str, text: str, *, source: EvidenceSource = EvidenceSource.POLICY
) -> Evidence:
    return Evidence(id=evidence_id, source=source, content=text, evidence_text=text)


def _draft(*, text: str, citation_ids: Sequence[str] = (POLICY_ID,)) -> dict[str, Any]:
    return {"claims": [{"text": text, "citation_ids": list(citation_ids)}]}


def _verdict(*, draft_text: str, evidence_text: str) -> tuple[Verdict, tuple[RejectReason, ...]]:
    result = evaluate_draft(
        raw_draft=_draft(text=draft_text, citation_ids=(SQL_ID,)),
        evidences=[_evidence(SQL_ID, evidence_text, source=EvidenceSource.SQL)],
    )
    return result.verdict, result.reject_reasons


# ── ① 구분자 변종 12자 — 미탐이 닫힌다 ──────────────────────────────────────

#: 번호 계열 네 패턴을 각각 (이름, 지어낸 값 서식, 근거에 둘 정상 표기) 으로 든다.
#: `{s}` 자리에 변종 구분자가 들어간다 — 원래 ASCII 하이픈이 서던 자리다.
NUMBER_FAMILIES: tuple[tuple[str, str], ...] = (
    ("mobile_phone", "010{s}9999{s}8888"),
    ("landline_phone", "02{s}123{s}4567"),
    ("service_phone", "1588{s}0000"),
    ("resident_registration_number", "900101{s}1234567"),
)


@pytest.mark.parametrize("separator", SEPARATOR_VARIANTS, ids=_codepoint)
def test_구분자_변종으로_지어낸_휴대전화는_기각된다(separator: str) -> None:
    """미탐 계열 ① — 12자 전부가 ASCII 하이픈과 같은 자격의 구분자여야 한다."""
    verdict, reasons = _verdict(
        draft_text=f"연락처는 010{separator}9999{separator}8888 로 연락 주세요.",
        evidence_text=_NO_PII_EVIDENCE,
    )

    assert verdict is Verdict.REJECT
    assert RejectReason.PII_DETECTED in reasons


@pytest.mark.parametrize(
    ("family", "template"),
    NUMBER_FAMILIES,
    ids=[family for family, _template in NUMBER_FAMILIES],
)
@pytest.mark.parametrize("separator", ("\u2013", "_"), ids=("U+2013", "U+005F"))
def test_번호_계열_네_패턴이_모두_변종_구분자를_받는다(
    family: str, template: str, separator: str
) -> None:
    """접기를 한 패턴에만 걸면 나머지 계열이 그대로 뚫린 채 문서만 닫힌다."""
    verdict, reasons = _verdict(
        draft_text=f"확인된 값은 {template.format(s=separator)} 입니다.",
        evidence_text=_NO_PII_EVIDENCE,
    )

    assert verdict is Verdict.REJECT, family
    assert RejectReason.PII_DETECTED in reasons, family


@pytest.mark.parametrize("separator", SEPARATOR_VARIANTS, ids=_codepoint)
def test_구분자_변종_에코는_근거에_있으면_통과한다(separator: str) -> None:
    """오탐 방향 — 접기가 **초안·근거 양쪽**에 걸려야 정상 에코가 살아남는다.

    근거를 일부러 ASCII 로 두고 초안만 변종으로 쓴다. 한쪽에만 접기를 걸면 같은 번호가
    다른 값이 되어 정상 답변이 `pii_detected` 로 기각된다 — 헤드라인 지표 오염이다.
    """
    verdict, reasons = _verdict(
        draft_text=f"등록된 연락처 010{separator}9999{separator}8888 로 안내드렸습니다.",
        evidence_text="1) customer_phone=010-9999-8888",
    )

    assert verdict is Verdict.PASS, reasons


# ── ② 결합 표식 — 자리 사이 배치 ────────────────────────────────────────────


@pytest.mark.parametrize("mark", IGNORABLE_REPRESENTATIVES + IGNORABLE_SAME_FAMILY, ids=_codepoint)
def test_숫자_자리_사이에_낀_결합_표식은_무시하고_기각한다(mark: str) -> None:
    """**자리 사이** 배치다 — 맨 끝에 붙이면 접기를 넓히기 전에도 기각된다.

    대표 2자(U+0301·U+FE0F)만이 아니라 같은 계열(U+0300·변이선택자)까지 함께 박는다.
    고정 2자로 구현하면 이 파라미터에서 무너진다 — ②는 범주로 덮어야 하는 계열이다.
    """
    verdict, reasons = _verdict(
        draft_text=f"연락처는 010-9{mark}999-8888 로 연락 주세요.",
        evidence_text=_NO_PII_EVIDENCE,
    )

    assert verdict is Verdict.REJECT
    assert RejectReason.PII_DETECTED in reasons


@pytest.mark.parametrize("mark", IGNORABLE_REPRESENTATIVES, ids=_codepoint)
def test_결합_표식을_맨_끝에_붙인_배치는_원래도_기각이었다(mark: str) -> None:
    """검사 배치의 근거를 코드로 남긴다 — 이 배치로는 회귀 검사가 성립하지 않는다.

    맨 끝에 붙이면 매치가 이미 끝난 뒤라 `(?![0-9])` 가 유지돼 접기와 무관하게 기각된다.
    자리 사이 배치를 쓰는 이유가 이것이다.
    """
    verdict, _reasons = _verdict(
        draft_text=f"연락처는 010-9999-8888{mark} 로 연락 주세요.",
        evidence_text=_NO_PII_EVIDENCE,
    )

    assert verdict is Verdict.REJECT


@pytest.mark.parametrize("mark", IGNORABLE_REPRESENTATIVES, ids=_codepoint)
def test_결합_표식이_낀_에코는_근거에_있으면_통과한다(mark: str) -> None:
    verdict, reasons = _verdict(
        draft_text=f"등록된 연락처 010-9{mark}999-8888 로 안내드렸습니다.",
        evidence_text="1) customer_phone=010-9999-8888",
    )

    assert verdict is Verdict.PASS, reasons


# ── 접기의 구현 형태를 못박는 음성 대조 ─────────────────────────────────────


def test_구분자_12자는_범주로_유도할_수_없는_열거_집합이다() -> None:
    """음성 대조 — `category == "Pd"` 로 짜면 절반이 그대로 뚫린다.

    12자 중 대시 범주에 드는 것은 U+2010~U+2015 여섯 자뿐이고, 나머지는 수학기호·
    구두점·연결선 세 범주에 흩어져 있다. 범주로 구현하면 열거된 절반이 통과한 채
    문서는 닫혔다고 적게 된다.
    """
    assert set(SEPARATOR_VARIANTS) == set(NUMERIC_SEPARATOR_VARIANTS)
    assert len(NUMERIC_SEPARATOR_VARIANTS) == 12

    dashes = {ch for ch in SEPARATOR_VARIANTS if unicodedata.category(ch) == "Pd"}
    assert len(dashes) == 6
    assert len(set(SEPARATOR_VARIANTS) - dashes) == 6
    assert {unicodedata.category(ch) for ch in SEPARATOR_VARIANTS} == {"Pd", "Sm", "Po", "Pc"}


def test_무시_문자는_결합_표식_범주로_덮는다() -> None:
    """음성 대조 — 고정 2자로 한정하면 같은 계열이 그대로 남는다.

    ⚠ **이 대조군은 `Mn` 하나로 채우면 안 된다.** 테스트가 고른 문자가 전부 `Mn` 이면
    `{category(ch)} == {"Mn"}` 는 동어반복이고, 접기를 `Mn` 하나로 좁혀도 초록이다 —
    실제로 그 상태였고 `U+0903`(Mc)·`U+20E3`(Me)가 숫자 자리 사이에서 통과했다.
    그래서 대조군에 세 범주를 **각각** 넣고, 아래 검사가 셋이 다 들어 있음을 못박는다.
    """
    family = IGNORABLE_REPRESENTATIVES + IGNORABLE_SAME_FAMILY
    범주 = {unicodedata.category(ch) for ch in family}

    assert 범주 == set(COMBINING_CATEGORIES), (
        "결합 표식 계열은 Mn·Mc·Me 셋이다 — 한 범주만 담으면 이 검사가 동어반복이 된다"
    )
    for mark in family:
        assert fold_numeric_for_detection(f"010-9{mark}999-8888") == "010-9999-8888"


@pytest.mark.parametrize("category", COMBINING_CATEGORIES)
def test_세_범주가_각각_대조군에_들어_있다(category: str) -> None:
    """검사 대상 목록이 비면 위 검사가 아무것도 지키지 않는다 — 범주별로 못박는다."""
    family = IGNORABLE_REPRESENTATIVES + IGNORABLE_SAME_FAMILY

    assert any(unicodedata.category(ch) == category for ch in family)


def test_Mn_으로_좁히면_두_범주가_뚫린다() -> None:
    """**고치기 전 동작을 그대로 재현한다** — 무엇이 열려 있었는지 검사가 기억한다.

    접기가 `Mn` 하나만 볼 때 `Mc`·`Me` 는 숫자 자리 사이에 남아 번호가 달라 보인다.
    이 검사는 규칙이 아니라 **유니코드 사실**을 확인하므로 구현을 따라 움직이지 않는다.
    """
    좁은_접기 = lambda text: "".join(  # noqa: E731
        ch for ch in text if unicodedata.category(ch) != "Mn"
    )

    for mark in ("\u0903", "\u20e3"):
        assert 좁은_접기(f"010-9{mark}999-8888") != "010-9999-8888"
        assert fold_numeric_for_detection(f"010-9{mark}999-8888") == "010-9999-8888"


def test_번호_접기는_공통_접기_위에_얹힌다() -> None:
    """전각 숫자·폭 없는 서식문자는 공통 접기가 이미 잡던 계열이다 — 잃지 않는다."""
    assert fold_numeric_for_detection(_FULLWIDTH_NUMBER) == "010-9999-8888"
    assert fold_numeric_for_detection(_ZERO_WIDTH_NUMBER) == "010-9999-8888"


def test_패턴마다_접기가_따로_붙어_있다() -> None:
    """접기는 패턴별이다 — 번호 계열 넷과 이메일이 서로 다른 접기를 든다."""
    folds = {pattern.name: pattern.fold for pattern in DEFAULT_PII_PATTERNS}

    assert folds["email"] is fold_for_detection
    assert {name for name, _template in NUMBER_FAMILIES} <= set(folds)
    for name, _template in NUMBER_FAMILIES:
        assert folds[name] is fold_numeric_for_detection


# ── 반대 방향 구멍 — 이메일은 지금의 접기를 유지한다 ────────────────────────


def test_밑줄만_다른_이메일은_서로_다른_주소로_남는다() -> None:
    """이메일에 번호 계열 접기를 걸면 **새 우회가 열린다.**

    밑줄과 빗금을 하이픈으로 접으면 `a_b@…` 와 `a-b@…` 가 같은 값이 되어, 근거에 있는
    주소를 한 글자 바꿔 지어낸 주소가 근거 유래로 통과한다. 우회 하나를 닫으면서
    반대 방향 우회를 여는 것이라, 넓히기는 번호 계열 안에서만 한다.
    """
    verdict, reasons = _verdict(
        draft_text="문의는 cs-team@shop.co.kr 로 보내주세요.",
        evidence_text="1) customer_email=cs_team@shop.co.kr",
    )

    assert verdict is Verdict.REJECT
    assert RejectReason.PII_DETECTED in reasons


def test_같은_주소의_이메일_에코는_통과한다() -> None:
    """양성 대조 — 위 검사가 이메일을 통째로 막아서 초록인 것이 아니다."""
    verdict, reasons = _verdict(
        draft_text="문의는 CS_Team@Shop.co.kr 로 보내주세요.",
        evidence_text="1) customer_email=cs_team@shop.co.kr",
    )

    assert verdict is Verdict.PASS, reasons


def test_번호_접기를_이메일에_걸면_다른_주소가_같은_값이_된다() -> None:
    """음성 대조 — 왜 이메일을 넓히지 않는지를 값으로 보인다."""
    underscored = "cs_team@shop.co.kr"
    hyphenated = "cs-team@shop.co.kr"

    assert normalize_email(fold_for_detection(underscored)) != normalize_email(
        fold_for_detection(hyphenated)
    )
    assert normalize_email(fold_numeric_for_detection(underscored)) == normalize_email(
        fold_numeric_for_detection(hyphenated)
    )


# ── 이메일 allowlist 전수 — 이 변경으로 충돌하는 값이 없다 ──────────────────

_EMAIL_PATTERN = next(pattern for pattern in DEFAULT_PII_PATTERNS if pattern.name == "email")


def _committed_texts() -> list[str]:
    """저장소에 커밋된 텍스트 모집단 — 정책 조항 전문 · 주문 픽스처 · 평가 픽스처."""
    texts = [
        path.read_text(encoding="utf-8")
        for path in sorted((_ROOT / "data" / "policies").rglob("*.md"))
    ]
    for path in (
        _ROOT / "db" / "fixtures" / "orders.jsonl",
        _ROOT / "data" / "l1_fixtures.jsonl",
        _ROOT / "data" / "golden_set.jsonl",
        _ROOT / "data" / "judge_fixtures.jsonl",
    ):
        texts.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return texts


def _committed_email_spans() -> list[str]:
    texts = _committed_texts()
    assert texts, "모집단이 비면 이 검사는 아무것도 지키지 않는다"
    spans = [
        match for text in texts for match in _EMAIL_PATTERN.regex.findall(_EMAIL_PATTERN.fold(text))
    ]
    assert spans, "이메일이 하나도 없으면 전수 검사가 성립하지 않는다"
    return spans


def test_커밋된_이메일_전수의_allowlist_값이_이_변경으로_움직이지_않는다() -> None:
    """전수 — 넓힌 접기가 이메일 쪽 allowlist 값을 한 건도 바꾸지 않는다."""
    spans = _committed_email_spans()

    moved = [
        span
        for span in spans
        if _EMAIL_PATTERN.normalize(_EMAIL_PATTERN.fold(span)) != span.lower()
    ]

    assert moved == []


def test_이메일까지_번호_접기를_걸면_전수에서_서로_다른_주소가_뭉개진다() -> None:
    """음성 대조(전수) — 왜 이메일을 넓히지 않는지를 실제 모집단으로 보인다.

    구성한 예시 하나가 아니라 저장소 전체에서, 번호 계열 접기를 이메일에 걸면 서로 다른
    주소의 개수가 줄어든다. 줄어든 만큼이 근거에 있는 주소를 한 글자 바꿔 통과시키는
    새 우회다.
    """
    spans = _committed_email_spans()

    distinct_now = {_EMAIL_PATTERN.normalize(_EMAIL_PATTERN.fold(span)) for span in spans}
    distinct_if_numeric = {normalize_email(fold_numeric_for_detection(span)) for span in spans}

    assert len(distinct_if_numeric) < len(distinct_now)


# ── 두 층의 기준이 같다 ─────────────────────────────────────────────────────

_LAYER_CORPUS: tuple[tuple[str, str], ...] = (
    *((f"sep-{_codepoint(sep)}", f"010{sep}9999{sep}8888") for sep in SEPARATOR_VARIANTS),
    *(
        (f"mark-{_codepoint(mark)}", f"010-9{mark}999-8888")
        for mark in IGNORABLE_REPRESENTATIVES + IGNORABLE_SAME_FAMILY
    ),
    ("fullwidth", _FULLWIDTH_NUMBER),
    ("zero-width", _ZERO_WIDTH_NUMBER),
    ("email", "cs_team@shop.co.kr"),
    ("status", "배송중"),
    ("date", "2026-02-08"),
    ("amount", "138000"),
    ("tracking", "320724559038"),
    ("order-no", "ORD-20260202-0001"),
)


@pytest.mark.parametrize(
    "value", [value for _label, value in _LAYER_CORPUS], ids=[label for label, _ in _LAYER_CORPUS]
)
def test_근거_필터와_게이트는_같은_판정을_낸다(value: str) -> None:
    """접기가 층마다 다르면 근거 필터가 게이트보다 헐거워져 allowlist 가 오염된다.

    계산 컬럼(승인 목록 밖) 값이 `evidence_text` 에서 빠지는지가 곧 근거 필터의 판정이다.
    """
    _content, evidence_text = _sql_evidence_texts(
        sql="SELECT '…' AS computed FROM orders",
        rows=({"computed": value},),
        pii_safe_output_columns=(),
    )
    dropped = "computed=" not in evidence_text

    assert dropped is pii_shaped(value)


def test_승인된_직접_컬럼_값은_접기를_넓혀도_문면_그대로_남는다() -> None:
    """정상 에코 보존 — 접기는 탐지에만 걸고 근거 문면을 갈아 끼우지 않는다."""
    value = "010\u20139999\u20138888"
    _content, evidence_text = _sql_evidence_texts(
        sql="SELECT customer_phone FROM orders",
        rows=({"customer_phone": value},),
        pii_safe_output_columns=("customer_phone",),
    )

    assert evidence_text == f"1) customer_phone={value}"


def test_변종_구분자로_쓰인_근거는_지어낸_반각_번호의_출처가_되지_않는다() -> None:
    """두 층을 이어 붙인 대조 — 근거 필터가 함께 넓어지지 않으면 여기서 통과가 난다."""
    dashed = "010\u20139999\u20138888"
    _content, evidence_text = _sql_evidence_texts(
        sql="SELECT '…' AS computed FROM orders",
        rows=({"computed": dashed},),
        pii_safe_output_columns=(),
    )
    result = evaluate_draft(
        raw_draft=_draft(text="연락처는 010-9999-8888 입니다", citation_ids=(SQL_ID,)),
        evidences=(
            Evidence(
                id=SQL_ID,
                source=EvidenceSource.SQL,
                content="표시용",
                evidence_text=evidence_text,
            ),
        ),
    )

    assert result.verdict is Verdict.REJECT
    assert RejectReason.PII_DETECTED in result.reject_reasons


# ── 채점표에 새 계열이 실제로 올라와 있다 ───────────────────────────────────

_L1_FIXTURES = _ROOT / "data" / "l1_fixtures.jsonl"

#: 사이클 4 까지의 L1 픽스처 27건. 이 사이클이 **추가만** 했다는 것을 숫자로 못박는다.
_PRESERVED_FIXTURE_COUNT = 27


def _fixtures() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in _L1_FIXTURES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_L1_픽스처는_지우지_않고_추가만_한다() -> None:
    """평가용으로 심은 장치는 한 건도 지우거나 고치지 않는다."""
    ids = [str(fixture["id"]) for fixture in _fixtures()]

    assert ids[:_PRESERVED_FIXTURE_COUNT] == [
        f"F{index:02d}" for index in range(1, _PRESERVED_FIXTURE_COUNT + 1)
    ]
    assert len(ids) > _PRESERVED_FIXTURE_COUNT, "새 우회 계열이 채점표에 올라와 있어야 한다"
    assert len(set(ids)) == len(ids)


def test_채점표가_구분자_변종과_결합_표식_계열을_모두_담는다() -> None:
    """넣지 않으면 새로 닫은 계열을 헤드라인 검출률이 한 번도 재지 않는다."""
    added = _fixtures()[_PRESERVED_FIXTURE_COUNT:]
    blob = json.dumps(added, ensure_ascii=False)

    missing = sorted(_codepoint(ch) for ch in SEPARATOR_VARIANTS if ch not in blob)
    assert missing == []
    assert all(mark in blob for mark in IGNORABLE_REPRESENTATIVES)
