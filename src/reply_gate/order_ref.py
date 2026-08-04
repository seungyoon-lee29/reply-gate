"""주문번호 형식의 **단독 소유 모듈**.

정규식·검증·생성 규칙이 여기 한 곳에만 있다. 데이터 시딩(`scripts.seed_orders`)과
접수 검증이 같은 정의를 import 해서 쓰고, 어디서도 형식을 다시 정의하지 않는다
(spec "접수 [코드]" 절 — "주문번호 형식의 정의는 코드 한 곳에 두고 데이터 시딩과
접수 검증이 같은 정의를 공유한다").

형식: ``ORD-YYYYMMDD-NNNN``

- ``YYYYMMDD`` — 주문 일자. **실재하는 달력 날짜**여야 한다.
- ``NNNN`` — 그 날짜 안에서의 주문 순번. ``0001``..``9999``.

`ORDER_NO_REGEX` 는 `db/schema.sql` 의 CHECK 제약과 **같은 문자열**을 쓴다
(`tests/test_db_schema.py` 가 두 곳이 어긋나지 않는지 검사한다). DB 제약은 형태만 보고,
달력 날짜 유효성·순번 범위까지 보는 것은 이 모듈이다 — 형태만 맞고 날짜가 없는
`ORD-20260230-0001` 같은 값은 `is_valid_order_no` 가 걸러낸다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

__all__ = [
    "ORDER_NO_MAX_SEQUENCE",
    "ORDER_NO_MIN_SEQUENCE",
    "ORDER_NO_PREFIX",
    "ORDER_NO_REGEX",
    "InvalidOrderNoError",
    "OrderRef",
    "format_order_no",
    "is_valid_order_no",
    "normalize_order_no",
    "parse_order_no",
]

#: 주문번호 접두어.
ORDER_NO_PREFIX = "ORD"

#: 주문번호의 **형태** 정규식. Postgres CHECK 제약(`db/schema.sql`)이 이 문자열을 그대로 쓴다.
#: 캡처 그룹을 두지 않는다 — Postgres 정규식과 문자 단위로 동일해야 하기 때문이다.
ORDER_NO_REGEX = r"^ORD-\d{8}-\d{4}$"

#: 하루 안에서 허용되는 순번 범위 (`0000` 은 주문이 아니므로 제외).
ORDER_NO_MIN_SEQUENCE = 1
ORDER_NO_MAX_SEQUENCE = 9999

_ORDER_NO_PATTERN = re.compile(ORDER_NO_REGEX)

_DATE_SLICE = slice(4, 12)
_SEQUENCE_SLICE = slice(13, 17)


class InvalidOrderNoError(ValueError):
    """주문번호가 형식에 맞지 않는다.

    접수 단계는 이 예외를 잡아 "형식 오류"로 응답하고, 시딩은 절대 발생시키지 않는다.
    """


@dataclass(frozen=True)
class OrderRef:
    """주문번호를 뜯어본 결과 — 주문 일자와 그 날의 순번."""

    ordered_on: date
    sequence: int

    def __str__(self) -> str:
        return format_order_no(ordered_on=self.ordered_on, sequence=self.sequence)


def normalize_order_no(value: str) -> str:
    """접수 입력을 비교 가능한 형태로 정규화한다 — 앞뒤 공백 제거 + 대문자화.

    **검증하지 않는다.** 정규화 결과를 `parse_order_no`/`is_valid_order_no` 에 넘겨라.
    사용자가 `ord-20260315-0001` 이나 앞뒤 공백을 붙여 넣는 경우를 흡수하는 것이 전부다.
    """
    return value.strip().upper()


def parse_order_no(value: str) -> OrderRef:
    """주문번호를 검증하고 뜯어본다. 형식 위반이면 `InvalidOrderNoError`.

    정규화는 하지 않는다 — 호출자가 필요하면 `normalize_order_no` 를 먼저 부른다.
    """
    if not _ORDER_NO_PATTERN.fullmatch(value):
        raise InvalidOrderNoError(
            f"주문번호 형식이 아니다: {value!r} (기대 형식: ORD-YYYYMMDD-NNNN)"
        )

    raw_date = value[_DATE_SLICE]
    try:
        ordered_on = date(int(raw_date[:4]), int(raw_date[4:6]), int(raw_date[6:8]))
    except ValueError as exc:
        raise InvalidOrderNoError(f"주문번호의 날짜가 달력에 없다: {value!r}") from exc

    sequence = int(value[_SEQUENCE_SLICE])
    if not ORDER_NO_MIN_SEQUENCE <= sequence <= ORDER_NO_MAX_SEQUENCE:
        raise InvalidOrderNoError(
            f"주문번호 순번이 범위 밖이다: {value!r} "
            f"(허용 {ORDER_NO_MIN_SEQUENCE}..{ORDER_NO_MAX_SEQUENCE})"
        )

    return OrderRef(ordered_on=ordered_on, sequence=sequence)


def is_valid_order_no(value: str) -> bool:
    """주문번호가 유효한 형식인지. 접수 단계의 형식 검증이 이 함수를 쓴다."""
    try:
        parse_order_no(value)
    except InvalidOrderNoError:
        return False
    return True


def format_order_no(*, ordered_on: date, sequence: int) -> str:
    """생성 규칙 — 주문 일자와 그 날의 순번으로 주문번호를 만든다.

    시딩이 유일한 호출자다. 범위 밖 순번은 조용히 자르지 않고 예외를 낸다.
    """
    if not ORDER_NO_MIN_SEQUENCE <= sequence <= ORDER_NO_MAX_SEQUENCE:
        raise InvalidOrderNoError(
            f"순번이 범위 밖이다: {sequence} "
            f"(허용 {ORDER_NO_MIN_SEQUENCE}..{ORDER_NO_MAX_SEQUENCE})"
        )
    return f"{ORDER_NO_PREFIX}-{ordered_on:%Y%m%d}-{sequence:04d}"
