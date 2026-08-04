"""L1 게이트 — 근거 없는 답변을 코드로 기각한다.

spec "L1 게이트 검사 규칙" 을 그대로 옮긴 모듈이다. 제품 정체성이 실제로 구현되는 곳이므로
아래 두 성질은 어떤 편의로도 깨지 않는다.

* **LLM 호출 0회.** 이 모듈은 LLM·네트워크 라이브러리를 import 하지 않는다.
* **100% 결정론.** 시간·난수·환경에 의존하지 않는다 — 같은 입력은 항상 같은 판정과
  같은 사유 순서를 낸다("측정 1 — L1 게이트 단위 정확도(결정론)").

기각 사유는 **하나만 잡고 멈추지 않고 전부 수집한다.** 스키마가 깨져서 하위 검사가
불가능한 부분만 건너뛴다 — 예를 들어 claim 이 dict 가 아니면 그 claim 의 citation 검사는
할 수 없지만, 나머지 claim 의 citation 검사와 초안 전체의 PII 검사는 그대로 수행한다.

구조 검사를 `DRAFT_JSON_SCHEMA` 에 위임하지 않는다: 그건 생성 측 구조화 출력용이고,
생성 측 강제가 L1 검사를 대체하면 게이트가 실제로 무엇을 막는지 증명할 수 없다.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from reply_gate.contracts import Claim, Draft, Evidence, GateResult, RejectReason, Verdict

__all__ = [
    "DEFAULT_PII_PATTERNS",
    "PiiPattern",
    "evaluate_draft",
    "normalize_digits",
    "normalize_email",
    "to_draft",
]

_CLAIMS_KEY = "claims"
_TEXT_KEY = "text"
_CITATION_IDS_KEY = "citation_ids"

#: 사유 목록의 고정 순서 (spec 의 사유 표 순서). 검출 순서에 의존하면 결정론이 깨진다.
_REASON_ORDER: tuple[RejectReason, ...] = (
    RejectReason.SCHEMA_VIOLATION,
    RejectReason.MISSING_CITATION,
    RejectReason.INVALID_CITATION,
    RejectReason.PII_DETECTED,
)

#: 원시 초안에서 텍스트를 긁어모을 때의 최대 중첩 깊이. 자기 참조 구조에서도 멈추게 한다.
_MAX_TEXT_DEPTH = 8

_NON_DIGIT = re.compile(r"[^0-9]")


# ── PII 정규화 — 조정 가능 기본값 ────────────────────────────────────────────


def normalize_digits(value: str) -> str:
    """숫자형 PII 정규화: 구분자·공백을 지우고 숫자열만 남긴다."""
    return _NON_DIGIT.sub("", value)


def normalize_email(value: str) -> str:
    """이메일 정규화: 앞뒤 공백 제거 후 소문자화."""
    return value.strip().lower()


@dataclass(frozen=True)
class PiiPattern:
    """패턴형 PII 1종 — 탐지 정규식과 대조용 정규화 규칙을 함께 가진다."""

    name: str
    regex: re.Pattern[str]
    normalize: Callable[[str], str]


# 숫자 경계 lookaround 를 붙이는 이유:
#   1) 초안 쪽 — 주문번호처럼 긴 숫자열 한가운데의 일부가 전화번호로 오탐되는 것을 막는다.
#   2) 근거 쪽 — 긴 숫자열의 조각이 allowlist 에 올라 지어낸 번호를 통과시키는 것을 막는다.
# 패턴 집합·정규화 규칙은 "조정 가능 기본값" 이다. 호출자가 `pii_patterns` 로 바꿔 넣을 수 있다.
DEFAULT_PII_PATTERNS: tuple[PiiPattern, ...] = (
    PiiPattern(
        name="mobile_phone",
        regex=re.compile(r"(?<![0-9])01[016789][-. ]?[0-9]{3,4}[-. ]?[0-9]{4}(?![0-9])"),
        normalize=normalize_digits,
    ),
    PiiPattern(
        name="landline_phone",
        regex=re.compile(r"(?<![0-9])0[2-9][0-9]?[-. ]?[0-9]{3,4}[-. ]?[0-9]{4}(?![0-9])"),
        normalize=normalize_digits,
    ),
    PiiPattern(
        name="resident_registration_number",
        regex=re.compile(r"(?<![0-9])[0-9]{6}-?[1-4][0-9]{6}(?![0-9])"),
        normalize=normalize_digits,
    ),
    PiiPattern(
        name="email",
        regex=re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        normalize=normalize_email,
    ),
)


# ── 진입점 ──────────────────────────────────────────────────────────────────


def evaluate_draft(
    *,
    raw_draft: object,
    evidences: Sequence[Evidence],
    pii_patterns: Sequence[PiiPattern] = DEFAULT_PII_PATTERNS,
) -> GateResult:
    """원시 초안을 이번 문의의 수집 근거와 대조해 `pass | reject + 사유 목록` 을 낸다.

    `raw_draft` 는 **무엇이든 들어올 수 있다** — 초안 생성이 형식 불일치 시 원문 문자열을
    그대로 넘기기 때문이다. dict 가 아닌 값에도 예외를 던지지 않고 `schema_violation`
    으로 판정한다.
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
        draft_texts=_collect_texts(raw_draft),
        evidence_texts=[evidence.evidence_text for evidence in evidences],
        patterns=pii_patterns,
    ):
        reasons.add(RejectReason.PII_DETECTED)

    ordered = tuple(reason for reason in _REASON_ORDER if reason in reasons)
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
    """

    ok: bool
    citation_lists: tuple[tuple[object, ...], ...]
    draft: Draft | None


def _inspect_schema(raw_draft: object) -> _SchemaInspection:
    """구조(필수 키·타입·claims 비어 있음)까지만 본다.

    **`citation_ids` 최소 개수 제약을 여기에 넣지 않는다.** 넣으면 `missing_citation` 이
    영원히 발화하지 않아 사유 분리가 무너진다(spec L1 절 각주).

    스키마에 없는 추가 키는 위반으로 보지 않는다 — 답변 계약은 다음 사이클 L2 가 이어받아
    확장할 계약이고, 추가 키는 claim 이 근거를 딛고 섰는지와 무관하기 때문이다.
    """
    if not isinstance(raw_draft, Mapping):
        return _SchemaInspection(ok=False, citation_lists=(), draft=None)

    claims = raw_draft.get(_CLAIMS_KEY)
    if not isinstance(claims, list | tuple) or not claims:
        return _SchemaInspection(ok=False, citation_lists=(), draft=None)

    ok = True
    citation_lists: list[tuple[object, ...]] = []
    parsed: list[Claim] = []
    for claim in claims:
        if not isinstance(claim, Mapping):
            ok = False
            continue

        text = claim.get(_TEXT_KEY)
        if not isinstance(text, str):
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
        draft=Draft(claims=tuple(parsed)) if ok else None,
    )


# ── PII 검사 ────────────────────────────────────────────────────────────────


def _collect_texts(value: object, *, depth: int = 0) -> list[str]:
    """원시 초안에서 검사 대상 문자열을 전부 긁어온다 (dict 순서대로 — 결정론).

    claim 의 text 만 보지 않는 이유: 형식이 깨진 초안(원문 문자열, 엉뚱한 키)에서도
    PII 는 새어 나갈 수 있고, L1 은 그때도 검사해야 한다.
    """
    if depth > _MAX_TEXT_DEPTH:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for item in value.values() for text in _collect_texts(item, depth=depth + 1)]
    if isinstance(value, list | tuple):
        return [text for item in value for text in _collect_texts(item, depth=depth + 1)]
    return []


def _normalized_matches(*, texts: Sequence[str], pattern: PiiPattern) -> set[str]:
    return {pattern.normalize(match) for text in texts for match in pattern.regex.findall(text)}


def _has_unsourced_pii(
    *,
    draft_texts: Sequence[str],
    evidence_texts: Sequence[str],
    patterns: Sequence[PiiPattern],
) -> bool:
    """초안의 패턴형 PII 중 **근거에서 유래하지 않은 값**이 있으면 True.

    대조 방식 — 근거 텍스트에도 **같은 패턴·같은 정규화**를 적용해 값 집합을 뽑고,
    정규화된 값끼리 **완전 일치**로 비교한다. 덕분에 근거의 `010-1234-5678` 과 초안의
    `01012345678` 은 통과하고, 지어낸 번호는 기각된다.

    정규화한 숫자열을 근거 텍스트에 '부분 문자열 포함' 으로 대조하지 않는 이유:
    근거에 주문번호 같은 긴 숫자열이 있으면 짧은 전화번호가 우연히 그 안에 들어가
    지어낸 번호가 통과해 버린다(= PII 유출). 완전 일치는 이 오탐 경로를 통째로 없앤다.
    대신 근거가 패턴이 모르는 표기(예: 국가번호 접두)를 쓰면 정상 에코도 기각될 수 있는데,
    L1 의 실패는 인계(escalation)로 흘러 안전한 쪽이므로 이 방향의 보수성을 택한다.

    비패턴형 개인정보(이름·주소)는 정규식으로 잡을 수 없어 **L1 의 검사 대상이 아니다** —
    L2 사이클의 claim 단위 근거 대조로 이월한다. 커버리지를 과장하지 않는다.
    """
    if not patterns:
        return False

    found: set[str] = set()
    for pattern in patterns:
        found |= _normalized_matches(texts=draft_texts, pattern=pattern)
    if not found:
        return False

    # allowlist 는 패턴별로 나누지 않고 하나로 합친다 — 근거에서 어느 패턴이 뽑았든
    # 근거에 있었다는 사실은 같기 때문이다(숫자열과 이메일은 정규화 결과가 겹치지 않는다).
    allowed: set[str] = set()
    for pattern in patterns:
        allowed |= _normalized_matches(texts=evidence_texts, pattern=pattern)

    return bool(found - allowed)
