"""합성 주문 500건 시딩 — `uv run python -m scripts.seed_orders`.

시딩 경로는 **저장소에 커밋된 픽스처**(`db/fixtures/orders.jsonl`)를 읽어 넣는 것뿐이다
(docs/standards.md "데이터 재현성" — 합성 주문은 저장소에 커밋된 픽스처를 로드하고
시딩 경로에 LLM 생성을 넣지 않는다). 시딩 중에 데이터를 새로 만들지 않으므로
몇 번을 돌려도 같은 500건이다.

픽스처 자체는 `--regenerate` 로 다시 만든다. 생성기는 손으로 고른 한국어 어휘 목록을
고정 시드로 조합하는 결정론 함수라 같은 코드에서 같은 파일이 나온다.

멱등 전략: 주문번호 기준 upsert + 픽스처에 없는 행 삭제. 따라서 두 번 돌려도, 중간에
누가 행을 끼워 넣었어도 결과는 정확히 픽스처와 같다.

데이터 성격: 고객명·배송지·상품명은 한국어, 연락처는 한국 전화번호 형식이다
(L1 의 패턴형 PII 정책 시연용). 실존 인물의 데이터를 쓰지 않는다 — 이름은 흔한 성씨와
이름의 기계적 조합이고, 이메일은 예약 도메인 `example.com`, 전화번호 가운데 자리는
010 번호로 배정되지 않는 `0xxx` 대역이다.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

import psycopg
from psycopg.rows import DictRow

from reply_gate.db import apply_schema, connect
from reply_gate.order_ref import format_order_no

__all__ = [
    "FIXTURE_PATH",
    "ORDER_COUNT",
    "OrderRecord",
    "build_orders",
    "load_fixture",
    "seed_orders",
    "write_fixture",
]

#: 합성 주문 픽스처의 기대 건수 — `db/fixtures/orders.jsonl` 의 줄 수와 같아야 한다.
ORDER_COUNT: Final = 500

#: 커밋된 픽스처. 시딩은 이 파일만 읽는다.
FIXTURE_PATH: Final = Path(__file__).resolve().parents[1] / "db" / "fixtures" / "orders.jsonl"

#: 생성기 고정 시드 — 바꾸면 픽스처가 통째로 달라진다.
FIXTURE_SEED: Final = 20260201

#: 한국 표준시. 한국 이커머스 데이터이므로 저장은 timestamptz + KST 오프셋.
KST: Final = timezone(timedelta(hours=9), "KST")

#: 주문 일자 범위와 "지금" 기준점. 생성 시점에 의존하지 않도록 상수로 고정한다.
FIRST_ORDER_DATE: Final = date(2026, 2, 2)
REFERENCE_DATE: Final = date(2026, 8, 1)
ORDER_DATE_SPAN_DAYS: Final = (REFERENCE_DATE - FIRST_ORDER_DATE).days  # 180

#: 만들어 두는 고객 수. 500건보다 적어 한 고객이 여러 번 주문한다 —
#: "이 고객의 다른 주문" 류 text-to-SQL 질의가 성립한다.
CUSTOMER_COUNT: Final = 200

#: `db/schema.sql` 의 orders_status_enum 과 같은 집합이어야 한다.
ORDER_STATUSES: Final = (
    "결제완료",
    "상품준비중",
    "배송중",
    "배송완료",
    "취소",
    "반품접수",
    "환불완료",
    "교환접수",
)

_SURNAMES: Final = (
    ("김", "kim"),
    ("이", "lee"),
    ("박", "park"),
    ("최", "choi"),
    ("정", "jung"),
    ("강", "kang"),
    ("조", "cho"),
    ("윤", "yoon"),
    ("장", "jang"),
    ("임", "lim"),
    ("한", "han"),
    ("오", "oh"),
    ("서", "seo"),
    ("신", "shin"),
    ("권", "kwon"),
    ("황", "hwang"),
    ("안", "ahn"),
    ("송", "song"),
    ("류", "ryu"),
    ("홍", "hong"),
)

_GIVEN_NAMES: Final = (
    ("지훈", "jihun"),
    ("서연", "seoyeon"),
    ("민준", "minjun"),
    ("하은", "haeun"),
    ("도윤", "doyun"),
    ("지우", "jiwoo"),
    ("예준", "yejun"),
    ("수아", "sua"),
    ("시우", "siwoo"),
    ("지민", "jimin"),
    ("준서", "junseo"),
    ("하윤", "hayun"),
    ("건우", "gunwoo"),
    ("유진", "yujin"),
    ("태윤", "taeyun"),
    ("서준", "seojun"),
    ("나윤", "nayun"),
    ("은우", "eunwoo"),
    ("채원", "chaewon"),
    ("다인", "dain"),
    ("현우", "hyunwoo"),
    ("소율", "soyul"),
    ("재원", "jaewon"),
    ("가온", "gaon"),
)

_REGIONS: Final = (
    "서울특별시 강남구",
    "서울특별시 마포구",
    "서울특별시 송파구",
    "서울특별시 성동구",
    "서울특별시 은평구",
    "경기도 성남시 분당구",
    "경기도 고양시 일산동구",
    "경기도 수원시 영통구",
    "경기도 용인시 기흥구",
    "인천광역시 연수구",
    "부산광역시 해운대구",
    "부산광역시 수영구",
    "대구광역시 수성구",
    "대전광역시 유성구",
    "광주광역시 서구",
    "울산광역시 남구",
    "세종특별자치시",
    "강원특별자치도 춘천시",
    "충청북도 청주시 흥덕구",
    "전북특별자치도 전주시 덕진구",
    "경상남도 창원시 성산구",
    "제주특별자치도 제주시",
)

_ROADS: Final = (
    "테헤란로",
    "월드컵북로",
    "올림픽로",
    "성수이로",
    "은평로",
    "판교역로",
    "중앙로",
    "광교중앙로",
    "해운대해변로",
    "동백로",
    "달구벌대로",
    "대학로",
    "상무중앙로",
    "삼산로",
    "한밭대로",
    "첨단과기로",
    "번영로",
    "무역로",
)

_BUILDINGS: Final = ("한빛아파트", "그린빌", "리버뷰타워", "센트럴파크", "행복주택", "하늘채")

#: (상품명, 옵션 목록, 최저가, 최고가) — 가격은 1,000원 단위로 반올림한다.
_PRODUCTS: Final = (
    ("무선 블루투스 이어폰", ("화이트", "블랙", "미드나잇블루"), 79000, 189000),
    ("스탠드 조명", ("웜화이트", "쿨화이트"), 34000, 89000),
    ("스테인리스 텀블러 500ml", ("실버", "매트블랙", "아이보리"), 19000, 42000),
    ("오버핏 반팔 티셔츠", ("S", "M", "L", "XL"), 15000, 39000),
    ("경량 패딩 조끼", ("S", "M", "L"), 59000, 129000),
    ("원목 식탁 의자", ("내추럴", "월넛"), 89000, 210000),
    ("기계식 키보드", ("적축", "청축", "갈축"), 79000, 189000),
    ("게이밍 마우스", ("블랙", "화이트"), 39000, 99000),
    ("휴대용 보조배터리 20000mAh", ("블랙", "라벤더"), 29000, 69000),
    ("논슬립 요가매트", ("6mm", "8mm", "10mm"), 24000, 58000),
    ("캠핑용 접이식 체어", ("카키", "베이지"), 45000, 120000),
    ("에어프라이어 5L", ("화이트", "그레이"), 89000, 179000),
    ("원목 도마 세트", ("소", "중", "대"), 18000, 45000),
    ("주방용 실리콘 집게", ("레드", "그레이"), 6000, 15000),
    ("이유식 보관 용기 세트", ("4구", "6구"), 21000, 49000),
    ("반려동물 자동 급수기", ("화이트", "핑크"), 32000, 78000),
    ("크로스 숄더백", ("블랙", "탄"), 59000, 149000),
    ("러닝화", ("250", "260", "270", "280"), 69000, 159000),
    ("전동 칫솔", ("화이트", "블랙"), 45000, 119000),
    ("무선 청소기", ("실버", "차콜"), 149000, 389000),
)

_COURIERS: Final = ("CJ대한통운", "롯데택배", "한진택배", "우체국택배", "로젠택배")

#: 주문 경과일 구간별 상태 가중치. 최근 주문일수록 초기 단계에 머문다.
_STATUS_WEIGHTS: Final = (
    (2, (("결제완료", 45), ("상품준비중", 30), ("취소", 8))),
    (6, (("상품준비중", 25), ("배송중", 45), ("결제완료", 5), ("취소", 6))),
    (12, (("배송중", 20), ("배송완료", 50), ("취소", 5), ("교환접수", 4))),
    (
        ORDER_DATE_SPAN_DAYS,
        (("배송완료", 60), ("반품접수", 8), ("환불완료", 9), ("교환접수", 7), ("취소", 6)),
    ),
)

#: 발송 이후 단계에 이른 상태 — 송장·발송일시가 있다.
_SHIPPED_STATUSES: Final = frozenset({"배송중", "배송완료", "반품접수", "환불완료", "교환접수"})
#: 수령까지 끝난 상태 — 배송완료일시가 있다.
_DELIVERED_STATUSES: Final = frozenset({"배송완료", "반품접수", "환불완료", "교환접수"})


@dataclass(frozen=True)
class OrderRecord:
    """주문 1건. 필드 이름은 `orders` 테이블 컬럼과 1:1 이다."""

    order_no: str
    customer_name: str
    customer_phone: str
    customer_email: str
    shipping_address: str
    product_name: str
    product_option: str | None
    quantity: int
    unit_price_krw: int
    total_price_krw: int
    status: str
    ordered_at: datetime
    shipped_at: datetime | None
    delivered_at: datetime | None
    courier: str | None
    tracking_no: str | None

    def to_json(self) -> dict[str, Any]:
        """픽스처에 쓸 JSON 표현 — 일시는 ISO 8601 문자열."""
        return {
            "order_no": self.order_no,
            "customer_name": self.customer_name,
            "customer_phone": self.customer_phone,
            "customer_email": self.customer_email,
            "shipping_address": self.shipping_address,
            "product_name": self.product_name,
            "product_option": self.product_option,
            "quantity": self.quantity,
            "unit_price_krw": self.unit_price_krw,
            "total_price_krw": self.total_price_krw,
            "status": self.status,
            "ordered_at": self.ordered_at.isoformat(),
            "shipped_at": None if self.shipped_at is None else self.shipped_at.isoformat(),
            "delivered_at": None if self.delivered_at is None else self.delivered_at.isoformat(),
            "courier": self.courier,
            "tracking_no": self.tracking_no,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> OrderRecord:
        """픽스처 한 줄을 레코드로. 일시는 파이썬 datetime 으로 되돌린다."""

        def _dt(key: str) -> datetime | None:
            raw = payload[key]
            return None if raw is None else datetime.fromisoformat(str(raw))

        ordered_at = _dt("ordered_at")
        if ordered_at is None:
            raise ValueError(f"ordered_at 이 비어 있다: {payload.get('order_no')!r}")
        return cls(
            order_no=str(payload["order_no"]),
            customer_name=str(payload["customer_name"]),
            customer_phone=str(payload["customer_phone"]),
            customer_email=str(payload["customer_email"]),
            shipping_address=str(payload["shipping_address"]),
            product_name=str(payload["product_name"]),
            product_option=_opt_str(payload["product_option"]),
            quantity=int(payload["quantity"]),
            unit_price_krw=int(payload["unit_price_krw"]),
            total_price_krw=int(payload["total_price_krw"]),
            status=str(payload["status"]),
            ordered_at=ordered_at,
            shipped_at=_dt("shipped_at"),
            delivered_at=_dt("delivered_at"),
            courier=_opt_str(payload["courier"]),
            tracking_no=_opt_str(payload["tracking_no"]),
        )


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


# ── 픽스처 생성 (시딩 경로가 아니다 — `--regenerate` 로만 돈다) ────────────────


@dataclass(frozen=True)
class _Customer:
    name: str
    phone: str
    email: str
    address: str


def _build_customers(rng: random.Random, count: int) -> list[_Customer]:
    customers: list[_Customer] = []
    for index in range(count):
        surname, surname_roman = rng.choice(_SURNAMES)
        given, given_roman = rng.choice(_GIVEN_NAMES)
        # 가운데 자리 0xxx 는 010 번호로 배정되지 않는 대역이다 — 형식은 진짜지만
        # 실제 가입자와 겹치지 않는다.
        phone = f"010-0{rng.randrange(0, 1000):03d}-{rng.randrange(0, 10000):04d}"
        email = f"{surname_roman}.{given_roman}{index + 1:03d}@example.com"
        address = (
            f"{rng.choice(_REGIONS)} {rng.choice(_ROADS)} {rng.randrange(1, 400)} "
            f"{rng.choice(_BUILDINGS)} {rng.randrange(101, 130)}동 {rng.randrange(101, 2005)}호"
        )
        customers.append(
            _Customer(name=f"{surname}{given}", phone=phone, email=email, address=address)
        )
    return customers


def _pick_status(rng: random.Random, days_ago: int) -> str:
    for upper_bound, weights in _STATUS_WEIGHTS:
        if days_ago <= upper_bound:
            pool = [status for status, weight in weights for _ in range(weight)]
            return rng.choice(pool)
    raise AssertionError(f"상태 가중치 구간을 벗어났다: days_ago={days_ago}")


def build_orders(*, count: int = ORDER_COUNT, seed: int = FIXTURE_SEED) -> list[OrderRecord]:
    """합성 주문을 결정론적으로 만든다. 같은 인자면 항상 같은 결과다."""
    rng = random.Random(seed)
    customers = _build_customers(rng, CUSTOMER_COUNT)

    # 최근 주문이 더 많도록 경과일을 앞쪽으로 몰아준다(지수 1.8).
    days_ago_list = sorted(
        (1 + int(ORDER_DATE_SPAN_DAYS * (rng.random() ** 1.8)) for _ in range(count)),
        reverse=True,
    )

    per_day_sequence: dict[date, int] = {}
    records: list[OrderRecord] = []
    for days_ago in days_ago_list:
        ordered_on = REFERENCE_DATE - timedelta(days=days_ago)
        sequence = per_day_sequence.get(ordered_on, 0) + 1
        per_day_sequence[ordered_on] = sequence

        customer = rng.choice(customers)
        product_name, options, low, high = rng.choice(_PRODUCTS)
        unit_price = rng.randrange(low // 1000, high // 1000 + 1) * 1000
        quantity = rng.choice((1, 1, 1, 1, 2, 2, 3))
        status = _pick_status(rng, days_ago)

        ordered_at = datetime.combine(
            ordered_on,
            datetime.min.time().replace(
                hour=rng.randrange(8, 24),
                minute=rng.randrange(0, 60),
                second=rng.randrange(0, 60),
            ),
            tzinfo=KST,
        )

        shipped_at: datetime | None = None
        delivered_at: datetime | None = None
        courier: str | None = None
        tracking_no: str | None = None
        if status in _SHIPPED_STATUSES:
            ship_days = min(rng.choice((1, 1, 2, 2, 3)), max(days_ago - 1, 1))
            shipped_at = ordered_at + timedelta(days=ship_days, hours=rng.randrange(0, 12))
            courier = rng.choice(_COURIERS)
            tracking_no = f"{rng.randrange(10**11, 10**12)}"
            if status in _DELIVERED_STATUSES:
                deliver_days = min(rng.choice((1, 1, 2, 3)), max(days_ago - ship_days, 1))
                delivered_at = shipped_at + timedelta(days=deliver_days, hours=rng.randrange(0, 10))

        records.append(
            OrderRecord(
                order_no=format_order_no(ordered_on=ordered_on, sequence=sequence),
                customer_name=customer.name,
                customer_phone=customer.phone,
                customer_email=customer.email,
                shipping_address=customer.address,
                product_name=product_name,
                product_option=rng.choice(options),
                quantity=quantity,
                unit_price_krw=unit_price,
                total_price_krw=unit_price * quantity,
                status=status,
                ordered_at=ordered_at,
                shipped_at=shipped_at,
                delivered_at=delivered_at,
                courier=courier,
                tracking_no=tracking_no,
            )
        )

    records.sort(key=lambda record: record.order_no)
    _assert_generated_invariants(records, count=count)
    return records


def _assert_generated_invariants(records: list[OrderRecord], *, count: int) -> None:
    """생성기가 조용히 망가지지 않게 하는 자체 점검 — 픽스처 제작 시점에만 돈다."""
    if len(records) != count:
        raise AssertionError(f"주문 건수가 {count} 이 아니다: {len(records)}")
    order_nos = {record.order_no for record in records}
    if len(order_nos) != count:
        raise AssertionError(f"주문번호가 중복됐다: 고유 {len(order_nos)} / 전체 {count}")
    missing = [status for status in ORDER_STATUSES if sum(r.status == status for r in records) < 5]
    if missing:
        raise AssertionError(f"데모에 쓰기엔 표본이 너무 적은 상태가 있다: {missing}")


def write_fixture(records: list[OrderRecord], path: Path = FIXTURE_PATH) -> Path:
    """픽스처를 JSONL 로 쓴다 — 한 줄 한 주문이라 diff 가 읽힌다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_json(), ensure_ascii=False, sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_fixture(path: Path = FIXTURE_PATH) -> list[OrderRecord]:
    """커밋된 픽스처를 읽는다. 시딩 경로가 데이터를 만나는 유일한 지점이다."""
    if not path.exists():
        raise FileNotFoundError(
            f"주문 픽스처가 없다: {path}. "
            "`uv run python -m scripts.seed_orders --regenerate` 로 만든다."
        )
    with path.open(encoding="utf-8") as handle:
        return [OrderRecord.from_json(json.loads(line)) for line in handle if line.strip()]


# ── 시딩 ─────────────────────────────────────────────────────────────────────

_INSERT_SQL = """
INSERT INTO orders (
    order_no, customer_name, customer_phone, customer_email, shipping_address,
    product_name, product_option, quantity, unit_price_krw, total_price_krw,
    status, ordered_at, shipped_at, delivered_at, courier, tracking_no
) VALUES (
    %(order_no)s, %(customer_name)s, %(customer_phone)s, %(customer_email)s, %(shipping_address)s,
    %(product_name)s, %(product_option)s, %(quantity)s, %(unit_price_krw)s, %(total_price_krw)s,
    %(status)s, %(ordered_at)s, %(shipped_at)s, %(delivered_at)s, %(courier)s, %(tracking_no)s
)
ON CONFLICT (order_no) DO UPDATE SET
    customer_name    = EXCLUDED.customer_name,
    customer_phone   = EXCLUDED.customer_phone,
    customer_email   = EXCLUDED.customer_email,
    shipping_address = EXCLUDED.shipping_address,
    product_name     = EXCLUDED.product_name,
    product_option   = EXCLUDED.product_option,
    quantity         = EXCLUDED.quantity,
    unit_price_krw   = EXCLUDED.unit_price_krw,
    total_price_krw  = EXCLUDED.total_price_krw,
    status           = EXCLUDED.status,
    ordered_at       = EXCLUDED.ordered_at,
    shipped_at       = EXCLUDED.shipped_at,
    delivered_at     = EXCLUDED.delivered_at,
    courier          = EXCLUDED.courier,
    tracking_no      = EXCLUDED.tracking_no
"""

_PRUNE_SQL = "DELETE FROM orders WHERE order_no <> ALL(%(order_nos)s)"


def seed_orders(
    *,
    conn: psycopg.Connection[DictRow] | None = None,
    fixture_path: Path = FIXTURE_PATH,
    ensure_schema: bool = True,
) -> int:
    """픽스처를 주문 테이블에 반영하고 최종 건수를 돌려준다.

    멱등하다: upsert 로 픽스처 내용을 덮어쓰고, 픽스처에 없는 주문번호는 지운다.
    """
    records = load_fixture(fixture_path)
    if conn is not None:
        return _seed_with_connection(conn, records, ensure_schema=ensure_schema)
    with connect() as owned:
        return _seed_with_connection(owned, records, ensure_schema=ensure_schema)


def _seed_with_connection(
    conn: psycopg.Connection[DictRow], records: list[OrderRecord], *, ensure_schema: bool
) -> int:
    if ensure_schema:
        apply_schema(conn)
    payloads = [record.to_json() | _datetime_overrides(record) for record in records]
    with conn.cursor() as cursor:
        cursor.executemany(_INSERT_SQL, payloads)
        cursor.execute(_PRUNE_SQL, {"order_nos": [record.order_no for record in records]})
        cursor.execute("SELECT count(*) AS total FROM orders")
        row = cursor.fetchone()
    conn.commit()
    return 0 if row is None else int(row["total"])


def _datetime_overrides(record: OrderRecord) -> dict[str, Any]:
    """JSON 표현의 ISO 문자열을 timestamptz 파라미터용 datetime 으로 되돌린다."""
    return {
        "ordered_at": record.ordered_at,
        "shipped_at": record.shipped_at,
        "delivered_at": record.delivered_at,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="합성 주문 픽스처를 Postgres 에 시딩한다.")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="시딩 전에 픽스처를 다시 만든다(결정론 생성기). 평소에는 쓰지 않는다.",
    )
    parser.add_argument(
        "--fixture-only",
        action="store_true",
        help="DB 에 붙지 않고 픽스처만 다시 만든다.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=FIXTURE_PATH,
        help=f"픽스처 경로 (기본: {FIXTURE_PATH})",
    )
    args = parser.parse_args(argv)

    if args.regenerate or args.fixture_only:
        path = write_fixture(build_orders(), args.fixture)
        print(f"픽스처를 다시 만들었다: {path} ({ORDER_COUNT}건)")
        if args.fixture_only:
            return 0

    total = seed_orders(fixture_path=args.fixture)
    print(f"주문 시딩 완료: {total}건")
    return 0 if total == ORDER_COUNT else 1


if __name__ == "__main__":  # pragma: no cover - CLI 진입점
    raise SystemExit(main())
