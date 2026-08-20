"""L1 게이트 — 근거 없는 답변을 코드로 기각한다.

docs/business-rules.md "L1 게이트 판정 규칙" 을 그대로 옮긴 모듈이다. 제품 정체성이
실제로 구현되는 곳이므로 아래 두 성질은 어떤 편의로도 깨지 않는다.

* **LLM 호출 0회.** 이 모듈은 LLM·네트워크 라이브러리를 import 하지 않는다.
* **100% 결정론.** 시간·난수·환경에 의존하지 않는다 — 같은 입력은 항상 같은 판정과
  같은 사유 순서를 낸다("측정 1 — L1 게이트 단위 정확도(결정론)").

기각 사유는 **하나만 잡고 멈추지 않고 전부 수집한다.** 스키마가 깨져서 하위 검사가
불가능한 부분만 건너뛴다 — 예를 들어 claim 이 dict 가 아니면 그 claim 의 citation 검사는
할 수 없지만, 나머지 claim 의 citation 검사와 초안 전체의 PII 검사는 그대로 수행한다.

구조 검사를 `DRAFT_JSON_SCHEMA` 에 위임하지 않는다: 그건 생성 측 구조화 출력용이고,
생성 측 강제가 L1 검사를 대체하면 게이트가 실제로 무엇을 막는지 증명할 수 없다.

**패턴형 개인정보 정의의 단독 소유자가 이 모듈이다.** 정규식·접기·정규화를 한 벌
(`DEFAULT_PII_PATTERNS`)로 들고, 근거 렌더의 개인정보 필터와 조회 가드가 `pii_shaped`
로 가져다 쓴다 — 층마다 자기 정의를 두면 한쪽만 넓혀져 기준이 갈리고, 실제로 그렇게
뚫렸다(docs/engineering-notes.md "접기 기준이 층마다 달라 근거 필터가 게이트보다 헐거웠다").
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from reply_gate.contracts import Claim, Draft, Evidence, GateResult, RejectReason, Verdict

__all__ = [
    "DEFAULT_PII_PATTERNS",
    "NUMERIC_SEPARATOR_VARIANTS",
    "REASON_ORDER",
    "PiiPattern",
    "evaluate_draft",
    "fold_for_detection",
    "fold_numeric_for_detection",
    "normalize_digits",
    "normalize_email",
    "normalize_phone",
    "pii_shaped",
    "to_draft",
]

_CLAIMS_KEY = "claims"
_TEXT_KEY = "text"
_CITATION_IDS_KEY = "citation_ids"

#: 사유 목록의 고정 순서. 검출 순서에 의존하면 같은 입력이 다른 순서를 내 결정론이 깨진다.
#: 리포트의 사유별 내역도 이 순서를 그대로 쓴다 — 정의는 여기 한 곳뿐이다.
REASON_ORDER: tuple[RejectReason, ...] = (
    RejectReason.SCHEMA_VIOLATION,
    RejectReason.MISSING_CITATION,
    RejectReason.INVALID_CITATION,
    RejectReason.PII_DETECTED,
)

#: 원시 초안에서 텍스트를 긁어모을 때의 최대 중첩 깊이. 자기 참조 구조에서도 멈추게 한다.
_MAX_TEXT_DEPTH = 8

_NON_DIGIT = re.compile(r"[^0-9]")


# ── PII 접기 — 탐지 전 유니코드 정규화 ──────────────────────────────────────

#: 번호 계열이 **구분자로 받는** 표기 변종 12자. **열거된 고정 집합**이고 범주 이름으로
#: 유도하지 않는다 — 이 12자는 네 범주에 흩어져 있어 어떤 범주도 절반 이상을 덮지 못한다:
#: 대시 범주(`Pd`)에 드는 것은 U+2010~U+2015 여섯 자뿐이고, U+2212 는 수학기호(`Sm`),
#: U+2027·U+30FB·U+FF65·`/` 는 구두점(`Po`), `_` 는 연결선(`Pc`) 이다.
#: `category == "Pd"` 로 짜면 **열거된 절반이 그대로 뚫린 채** 문서는 닫혔다고 적게 된다.
#:
#: 눈으로 구분되지 않는 문자는 **이스케이프로 적는다** — 리터럴로 쓰면 무엇이 들어 있는지
#: 읽을 수 없고, 편집 중에 조용히 ASCII 하이픈으로 바뀌어도 아무도 모른다.
NUMERIC_SEPARATOR_VARIANTS: frozenset[str] = frozenset(
    (
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
)

#: 번호 계열이 **무시하는** 결합 표식의 유니코드 범주. 변이선택자도 전부 여기 든다.
#: ①의 12자와 반대로 **범주로 구현한다** — 이 계열은 범주 하나가 정확히 덮으므로,
#: 대표 문자 몇 개로 한정하면 U+0300 같은 동일 계열이 그대로 남는다.
_COMBINING_MARK_CATEGORY = "Mn"

#: 폭 없는 서식 문자(U+200B·U+2060 등)의 범주. 공통 접기가 통째로 지운다.
_FORMAT_CATEGORY = "Cf"


def fold_for_detection(text: str) -> str:
    """탐지 전 **공통** 접기 — 사람 눈에 같은 값이 패턴을 비껴가지 못하게 한다.

    전각 숫자(U+FF10~U+FF19)로 쓴 전화번호는 사람이 읽으면 전화번호지만 ASCII 숫자
    정규식에 걸리지 않는다. NFKC 로 호환 문자를 반각으로 접고, 폭 없는 서식 문자
    (`Cf` 범주 — U+200B 등)를 지운다.

    **이메일 패턴이 쓰는 접기가 이것이고, 여기서 더 넓히지 않는다.** 밑줄과 빗금을
    하이픈으로 접으면 `a_b@…` 와 `a-b@…` 가 **같은 값이 되어**, 근거에 있는 주소를 한
    글자 바꿔 지어낸 주소가 근거 유래로 통과한다 — 우회 하나를 닫으면서 반대 방향 우회를
    여는 것이다. 밑줄은 이 저장소의 사유 코드·근거 ID·컬럼명에 흔해서, 전역으로 접으면
    번호와 무관한 문자열까지 함께 움직인다.

    **초안과 근거 양쪽에 같은 접기를 적용한다.** 한쪽에만 걸면 근거의 같은 값이 다른
    문자열이 되어 정상 에코가 오기각되고, 반대로 근거 쪽만 헐거우면 접어야 PII 가 되는
    값이 allowlist 근거로 살아남는다.
    """
    folded = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in folded if unicodedata.category(ch) != _FORMAT_CATEGORY)


def fold_numeric_for_detection(text: str) -> str:
    """번호 계열 접기 — 공통 접기 위에 구분자 변종과 결합 표식을 얹는다.

    NFKC 는 대시류를 ASCII 하이픈으로 접지 않고, 결합 표식은 `Cf` 가 아니라 지워지지도
    않는다. 그래서 en dash 로 쓴 `010-9999-8888` 도, 숫자 **자리 사이**에 U+0301 을 끼운
    `010-9999-8888` 도 구분자 클래스 `[-. ()]` 를 그대로 비껴갔다 — 사람 눈에는 둘 다
    같은 번호로 읽힌다. 실제 우회 문자열은 `tests/test_pii_folding.py` 가 들고 있다.

    **배치가 판정을 가른다** — 결합 표식은 숫자열 안쪽에 끼워야 뚫리고 끝에 붙이면 매치가
    이미 끝난 뒤라 원래도 기각된다. 회귀 검사가 자리 사이 배치를 박는 이유다.

    접기는 **탐지에만** 쓴다. 근거 문면을 갈아 끼우지 않는다 — 바꾸면 정상 에코 계약과
    L2 근거가 함께 흔들린다.
    """
    folded = fold_for_detection(text)
    return "".join(
        "-" if ch in NUMERIC_SEPARATOR_VARIANTS else ch
        for ch in folded
        if unicodedata.category(ch) != _COMBINING_MARK_CATEGORY
    )


# ── PII 정규화 ──────────────────────────────────────────────────────────────


def normalize_digits(value: str) -> str:
    """숫자형 PII 정규화: 구분자·공백을 지우고 숫자열만 남긴다."""
    return _NON_DIGIT.sub("", value)


def normalize_phone(value: str) -> str:
    """전화번호 정규화: 구분자를 지우고 한국 국가번호를 국내 `0` 표기로 바꾼다."""
    digits = normalize_digits(value)
    if digits.startswith("0082"):
        return f"0{digits[4:]}"
    if digits.startswith("82"):
        return f"0{digits[2:]}"
    return digits


def normalize_email(value: str) -> str:
    """이메일 정규화: 앞뒤 공백 제거 후 소문자화."""
    return value.strip().lower()


@dataclass(frozen=True)
class PiiPattern:
    """패턴형 PII 1종 — 탐지 정규식 · 탐지 전 접기 · 대조용 정규화를 함께 가진다.

    **접기가 패턴별인 것이 이 자료형의 요점이다.** 번호 계열은 구분자 변종을 하이픈으로
    접고 결합 표식을 무시해야 하지만, 이메일에 같은 접기를 걸면 서로 다른 주소가 같은
    값이 되어 반대 방향 우회가 열린다. 접기를 전역 함수 하나로 두면 그 둘을 가를 자리가
    없어서, 기본값 없이 **패턴마다 명시**하게 한다.
    """

    name: str
    regex: re.Pattern[str]
    fold: Callable[[str], str]
    normalize: Callable[[str], str]


# 숫자 경계 lookaround 를 붙이는 이유:
#   1) 초안 쪽 — 주문번호처럼 긴 숫자열 한가운데의 일부가 전화번호로 오탐되는 것을 막는다.
#   2) 근거 쪽 — 긴 숫자열의 조각이 allowlist 에 올라 지어낸 번호를 통과시키는 것을 막는다.
#
# **이 집합이 저장소의 패턴형 개인정보 정의 정본이고, 소유자는 이 모듈 하나다.** 근거 렌더의
# 개인정보 필터(`evidence._sql_evidence_texts`)와 조회 가드가 여기서 가져다 쓴다 — 층마다
# 자기 정의를 두면 한쪽만 넓혀져 기준이 갈리고, 접기가 층마다 달라 터진 것이 정확히 그
# 모양이었다. 층별 주입점을 두지 않는 것도 같은 이유다(구조 테스트가 검사한다).
DEFAULT_PII_PATTERNS: tuple[PiiPattern, ...] = (
    PiiPattern(
        name="mobile_phone",
        regex=re.compile(
            r"(?<![0-9])(?:(?:\+82|0082)[-. ()]*10|01[016789])"
            r"[-. ()]*[0-9]{3,4}[-. ()]*[0-9]{4}(?![0-9])"
        ),
        fold=fold_numeric_for_detection,
        normalize=normalize_phone,
    ),
    PiiPattern(
        name="landline_phone",
        regex=re.compile(
            r"(?<![0-9])(?:(?:\+82|0082)[-. ()]*(?:2|[3-9][0-9]?)|0[2-9][0-9]?)"
            r"[-. ()]*[0-9]{3,4}[-. ()]*[0-9]{4}(?![0-9])"
        ),
        fold=fold_numeric_for_detection,
        normalize=normalize_phone,
    ),
    # 15xx/16xx/17xx/18xx 대표번호. 개인 연락처는 아니지만 정책 문서의 미끼 조항이
    # 겨냥하는 값이 바로 이것이다(docs/engineering-notes.md "대표번호(15xx~18xx)를
    # PII 패턴에 넣은 이유") — 고객센터 번호를 비워 둔 조항을 근거로 받고 모델이
    # 일반 지식으로 번호를 채우면 `pii_detected` 로 걸려야 한다. 빼면 미끼가 무력해진다.
    PiiPattern(
        name="service_phone",
        regex=re.compile(r"(?<![0-9])1[5-8][0-9]{2}[-. ]?[0-9]{4}(?![0-9])"),
        fold=fold_numeric_for_detection,
        normalize=normalize_digits,
    ),
    PiiPattern(
        name="resident_registration_number",
        regex=re.compile(r"(?<![0-9])[0-9]{6}[-. ]?[1-4][0-9]{6}(?![0-9])"),
        fold=fold_numeric_for_detection,
        normalize=normalize_digits,
    ),
    # 이메일만 **공통 접기**를 든다. 번호 계열 접기를 걸면 밑줄·빗금이 하이픈으로 접혀
    # `a_b@…` 와 `a-b@…` 가 같은 allowlist 값이 되고, 근거에 있는 주소를 한 글자 바꿔
    # 지어낸 주소가 통과한다 — 넓히기는 번호 계열 안에서만 한다.
    PiiPattern(
        name="email",
        regex=re.compile(r"[A-Za-z0-9._%+-]+@(?:[^\W_]|-)+(?:\.(?:[^\W_]|-)+)+"),
        fold=fold_for_detection,
        normalize=normalize_email,
    ),
)


def pii_shaped(text: str) -> bool:
    """텍스트에 패턴형 PII 가 한 건이라도 있으면 True — **탐지 전용 공개 표면**.

    값 대조 없이 "개인정보 모양인가"만 묻는 층이 쓴다: 근거 렌더의 개인정보 필터가
    계산 컬럼의 이름·값을 `evidence_text` 에서 걸러낼 때, 그리고 조회 가드가 고정값을
    승인할지 볼 때. **그 층들이 자기 정규식을 따로 두면 저장소에 개인정보 정의가 셋이
    되고, 한쪽만 넓혀지는 순간 기준이 갈린다** — 접기가 층마다 달라 뚫렸던 경로가
    정확히 그 모양이었다(docs/engineering-notes.md "접기 기준이 층마다 달라 근거
    필터가 게이트보다 헐거웠다").

    패턴별 접기를 그대로 태운다. 접기는 판정에만 쓰고 값을 갈아 끼우지 않는다.
    """
    return any(pattern.regex.search(pattern.fold(text)) for pattern in DEFAULT_PII_PATTERNS)


# ── 진입점 ──────────────────────────────────────────────────────────────────


def evaluate_draft(*, raw_draft: object, evidences: Sequence[Evidence]) -> GateResult:
    """원시 초안을 이번 문의의 수집 근거와 대조해 `pass | reject + 사유 목록` 을 낸다.

    `raw_draft` 는 **무엇이든 들어올 수 있다** — 초안 생성이 형식 불일치 시 원문 문자열을
    그대로 넘기기 때문이다. dict 가 아닌 값에도 예외를 던지지 않고 `schema_violation`
    으로 판정한다.

    **패턴 집합은 인자로 받지 않는다** — 정본은 `DEFAULT_PII_PATTERNS` 하나이고 근거
    필터·조회 가드가 같은 것을 쓴다.
    """
    reasons: set[RejectReason] = set()

    inspection = _inspect_schema(raw_draft)
    if not inspection.ok:
        reasons.add(RejectReason.SCHEMA_VIOLATION)

    known_ids = {evidence.id for evidence in evidences}
    for citation_ids in inspection.citation_lists:
        if not citation_ids:
            reasons.add(RejectReason.MISSING_CITATION)
        # 문자열이 아닌 원소는 이미 schema_violation 이다 — 참조 무결성은 문자열만 본다.
        if any(isinstance(cid, str) and cid not in known_ids for cid in citation_ids):
            reasons.add(RejectReason.INVALID_CITATION)

    # PII 검사는 텍스트만 있으면 할 수 있으므로 스키마가 깨져도 가능한 한 수행한다.
    if _has_unsourced_pii(
        draft_texts=_answer_texts(inspection=inspection, raw_draft=raw_draft),
        evidence_texts=[evidence.evidence_text for evidence in evidences],
    ):
        reasons.add(RejectReason.PII_DETECTED)

    ordered = tuple(reason for reason in REASON_ORDER if reason in reasons)
    return GateResult(
        verdict=Verdict.REJECT if ordered else Verdict.PASS,
        reject_reasons=ordered,
    )


def to_draft(raw_draft: object) -> Draft:
    """L1 스키마 검사를 통과한 원시 초안을 `Draft` 로 바꾼다.

    구조가 깨진 초안은 `ValueError` 다 — 판정은 `evaluate_draft` 가 하고, 이 함수는
    **통과한 초안에만** 쓴다. citation_ids 가 비어 있는 것은 구조 오류가 아니므로
    (그건 `missing_citation` 전담) 여기서 막지 않는다.
    """
    inspection = _inspect_schema(raw_draft)
    if inspection.draft is None:
        raise ValueError("L1 스키마 검사를 통과하지 못한 초안은 Draft 로 바꿀 수 없다")
    return inspection.draft


# ── 스키마 검사 ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _SchemaInspection:
    """구조 검사 결과 + 하위 검사가 가능한 부분.

    `citation_lists` 는 citation 검사를 할 수 있는 claim 들의 원시 citation_ids 목록이다
    (구조가 깨진 claim 은 빠진다). `draft` 는 구조가 완전할 때만 채워진다.

    `answer_texts` 는 **답변으로 나갈 수 있는 텍스트**다 — claim 의 `text` 중 문자열인 것
    전부. `draft` 와 달리 구조가 깨져도 채워진다(깨진 claim 옆의 멀쩡한 claim 은 여전히
    답변 후보다). PII 검사 대상이 이것이다.
    """

    ok: bool
    citation_lists: tuple[tuple[object, ...], ...]
    answer_texts: tuple[str, ...]
    draft: Draft | None


def _inspect_schema(raw_draft: object) -> _SchemaInspection:
    """구조(필수 키·타입·claims 비어 있음)까지만 본다.

    **`citation_ids` 최소 개수 제약을 여기에 넣지 않는다.** 넣으면 `missing_citation` 이
    영원히 발화하지 않아 사유 분리가 무너진다(docs/business-rules.md "L1 게이트 판정 규칙").

    스키마에 없는 추가 키는 위반으로 보지 않는다 — 답변 계약은 L2 가 그대로 이어받는
    계약이고, 추가 키는 claim 이 근거를 딛고 섰는지와 무관하기 때문이다.
    """
    if not isinstance(raw_draft, Mapping):
        return _SchemaInspection(ok=False, citation_lists=(), answer_texts=(), draft=None)

    claims = raw_draft.get(_CLAIMS_KEY)
    if not isinstance(claims, list | tuple) or not claims:
        return _SchemaInspection(ok=False, citation_lists=(), answer_texts=(), draft=None)

    ok = True
    citation_lists: list[tuple[object, ...]] = []
    answer_texts: list[str] = []
    parsed: list[Claim] = []
    for claim in claims:
        if not isinstance(claim, Mapping):
            ok = False
            continue

        text = claim.get(_TEXT_KEY)
        if isinstance(text, str):
            answer_texts.append(text)
        if not isinstance(text, str) or not text.strip():
            ok = False

        citation_ids = claim.get(_CITATION_IDS_KEY)
        if not isinstance(citation_ids, list | tuple):
            ok = False
            continue

        citation_lists.append(tuple(citation_ids))
        if not all(isinstance(cid, str) for cid in citation_ids):
            ok = False
            continue

        if isinstance(text, str):
            parsed.append(Claim(text=text, citation_ids=tuple(citation_ids)))

    return _SchemaInspection(
        ok=ok,
        citation_lists=tuple(citation_lists),
        answer_texts=tuple(answer_texts),
        draft=Draft(claims=tuple(parsed)) if ok else None,
    )


# ── PII 검사 ────────────────────────────────────────────────────────────────


def _answer_texts(*, inspection: _SchemaInspection, raw_draft: object) -> Sequence[str]:
    """PII 검사 대상을 고른다 — **답변으로 나갈 수 있는 텍스트만.**

    `to_draft` 가 살려내는 것은 claim 의 `text` 뿐이고 답변은 그것들의 연결이다
    (`Draft.answer_text`). 초안의 나머지 키(최상위 `debug`, claim 안의 `note` 등)는
    답변에 실리지 않으므로 검사하면 **답변에 없는 값으로 정상 초안을 기각하게 된다.**
    초안 JSON 은 LLM 산출이라 이런 키가 언제든 늘어날 수 있고, 헤드라인 지표가
    "정상 초안 오탐률"이므로 그 오탐이 곧 지표 오염이다
    (`src/reply_gate/AGENTS.md` 불변식 7 · docs/business-rules.md "PII 규칙").

    **답변 후보 텍스트를 하나도 식별할 수 없을 때만** 초안 전체를 긁는다 — 원문 문자열이
    통째로 넘어온 경우처럼 무엇이 답변인지 코드가 모르는 상태다. 그런 초안은 어차피
    `schema_violation` 으로 기각되므로 답변이 나가지는 않지만, 게이트는 모르는 쪽에서
    검사하는 방향으로 보수적으로 간다.
    """
    if inspection.answer_texts:
        return inspection.answer_texts
    return _collect_texts(raw_draft)


def _collect_texts(value: object, *, depth: int = 0) -> list[str]:
    """원시 초안에서 문자열을 전부 긁어온다 (dict 순서대로 — 결정론).

    답변 후보를 식별할 수 없을 때의 폴백 전용이다(`_answer_texts`).

    **`citation_ids` 는 제외한다.** docs/business-rules.md "PII 규칙" 의 검사 대상은
    "초안 텍스트" — 최종 사용자에게
    보여줄 답변이다. 근거 ID 는 답변 문장이 아니라 식별자이고, `sql:<문의 UUID>:<순번>`
    형식이라 UUID 의 16진 숫자 구간이 전화번호 패턴에 우연히 걸린다. 근거 ID 는
    `evidence_text` 에 들어가지 않으므로 allowlist 에도 오르지 않아, 그대로 두면
    **PII 가 전혀 없는 정상 초안이 기각된다** (무작위 UUID 20,000개 중 168개 = 0.84%).
    이 프로젝트의 헤드라인 지표가 "정상 초안 오탐률"이므로 그 오탐이 곧 지표 오염이다.
    참조 무결성은 `invalid_citation` 이 이미 전담하므로 여기서 볼 이유도 없다.
    """
    if depth > _MAX_TEXT_DEPTH:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [
            text
            for key, item in value.items()
            if key != _CITATION_IDS_KEY
            for text in _collect_texts(item, depth=depth + 1)
        ]
    if isinstance(value, list | tuple):
        return [text for item in value for text in _collect_texts(item, depth=depth + 1)]
    return []


def _normalized_matches(*, texts: Sequence[str], pattern: PiiPattern) -> set[str]:
    """패턴 하나가 텍스트들에서 뽑은 **정규화된 값** 집합. 접기는 그 패턴의 것을 쓴다."""
    return {
        pattern.normalize(match)
        for text in texts
        for match in pattern.regex.findall(pattern.fold(text))
    }


def _has_unsourced_pii(*, draft_texts: Sequence[str], evidence_texts: Sequence[str]) -> bool:
    """초안의 패턴형 PII 중 **근거에서 유래하지 않은 값**이 있으면 True.

    대조 방식 — 근거 텍스트에도 **같은 패턴·같은 정규화**를 적용해 값 집합을 뽑고,
    정규화된 값끼리 **완전 일치**로 비교한다. 덕분에 근거의 `010-1234-5678` 과 초안의
    `01012345678` 은 통과하고, 지어낸 번호는 기각된다.

    정규화한 숫자열을 근거 텍스트에 '부분 문자열 포함' 으로 대조하지 않는 이유:
    근거에 주문번호 같은 긴 숫자열이 있으면 짧은 전화번호가 우연히 그 안에 들어가
    지어낸 번호가 통과해 버린다(= PII 유출). 완전 일치는 이 오탐 경로를 통째로 없앤다.
    국내형과 한국 국가번호(+82/0082), 공백·괄호 구분자는 같은 값으로 접는다. 그래도 패턴이
    모르는 표기는 정상 에코라도 기각될 수 있는데, L1 실패는 인계(escalation)로 흘러 안전한
    쪽이므로 이 방향의 보수성을 택한다.

    비패턴형 개인정보(이름·주소)는 정규식으로 잡을 수 없어 **L1 의 검사 대상이 아니다** —
    L2 사이클의 claim 단위 근거 대조로 이월한다. 커버리지를 과장하지 않는다.

    **패턴 집합은 주입받지 않는다.** 이 저장소의 정본은 `DEFAULT_PII_PATTERNS` 하나이고,
    근거 필터·조회 가드도 같은 집합을 가져다 쓴다 — 층마다 다른 집합을 받을 수 있으면
    한쪽만 넓혀져 기준이 갈린다.
    """
    found: set[str] = set()
    for pattern in DEFAULT_PII_PATTERNS:
        found |= _normalized_matches(texts=draft_texts, pattern=pattern)
    if not found:
        return False

    # allowlist 는 패턴별로 나누지 않고 하나로 합친다 — 근거에서 어느 패턴이 뽑았든
    # 근거에 있었다는 사실은 같기 때문이다(숫자열과 이메일은 정규화 결과가 겹치지 않는다).
    allowed: set[str] = set()
    for pattern in DEFAULT_PII_PATTERNS:
        allowed |= _normalized_matches(texts=evidence_texts, pattern=pattern)

    return bool(found - allowed)
