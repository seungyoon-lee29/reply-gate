"""조회 가드의 두 수정 **전/후** 판정 차이를 모집단을 명시해 센다 (무과금·오프라인).

```bash
uv run python -m scripts.contrast_sql_guard
```

**무엇을 재나.** 두 수정이 각각 어느 방향으로 몇 건을 움직이는지다.

1. **PII 허용 출처의 유래 추적** — 임시 테이블(CTE)·파생 테이블·빈칸 채우기를 거친 직접
   컬럼이 허용 출처가 된다. 넓히는 수정이라 **반대 방향(합성된 값이 승인되는 것)** 을 같은
   대조에서 함께 센다.
2. **형변환 대상 타입 허용 목록** — 목록 밖 타입이 거부된다. 좁히는 수정이라 **정상 조회가
   새로 거부되는 것**을 함께 센다.

**전/후를 어떻게 재나.** 수정 전 규칙은 둘 다 한 자리에 있었다 — 유래 판정은 "최상위
projection 이 기저 테이블의 직접 컬럼인가" 하나였고, 캐스트 대상 타입 검사는 **아예 없었다.**
그래서 이 대조는 옛 코드를 통째로 되살리지 않고 그 두 자리만 갈아 끼워 두 판정을 나란히 낸다
(`_legacy_guard`). 병합 뒤에도 그대로 재현된다.

**모집단.**

- **손으로 만든 조회문 코퍼스** — 전수. 정상 조회 · 합성 공격 · 카탈로그 훑기 · 기존 우회를
  범주별로 적어 둔 목록이고, 이 파일이 그 목록의 전부다.
- **실제 생성된 SQL** — **프로브**. 커밋된 리포트가 SQL 문면을 담지 않아 자유 모집단이 없다.
  결정론 대역 생성기로 골든셋 전건의 조회문을 만들어 대신 잰다.

마지막 줄이 이 대조의 한계다: **전수가 아니라 프로브다.** 대역 생성기는 실제 모델이 아니고
임시 테이블·빈칸 채우기·형변환을 스스로 만들어 내지 않으므로, 여기서 "변화 0"이 나왔다고
실제 모델의 조회문이 안 바뀐다는 뜻은 아니다. 자유 모집단이 없는 축에서 낼 수 있는 최선의
관측이라는 뜻이다.

산출물은 표준 출력뿐이다 — 파일을 남기지 않는다. 재현이 무과금이라 재실행이 곧 근거다.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

from sqlglot import exp

from reply_gate import sql_guard
from reply_gate.evaluation import StubGenerationClient, load_golden_set
from reply_gate.evidence import build_sql_user_prompt
from reply_gate.sql_guard import SqlGuardRejection, validate_sql

ORDER_NO: Final = "ORD-20260201-0001"
OTHER_ORDER_NO: Final = "ORD-20260201-0002"
MAX_ROWS: Final = 50
SCOPE: Final = f"WHERE order_no = '{ORDER_NO}'"


# ── 수정 전 규칙 재현 ───────────────────────────────────────────────────────


def _legacy_pii_safe_output_columns(scope: sql_guard._Scope) -> tuple[str, ...]:
    """**수정 전** 유래 판정 — 최상위 projection 이 기저 테이블의 직접 컬럼일 때만 승인.

    임시 테이블·파생 테이블은 `base_table` 이 `None` 이라 경로 전체가 불승인이었고,
    빈칸 채우기는 `exp.Column` 이 아니라 그 자리에서 걸러졌다.
    """
    safe: list[str] = []
    for projection in scope.select.expressions:
        if isinstance(projection, exp.Star):
            for source in scope.sources.values():
                if source.base_table is not None:
                    safe.extend(sorted(source.columns))
            continue

        value = projection.this if isinstance(projection, exp.Alias) else projection
        if isinstance(value, exp.Column) and isinstance(value.this, exp.Star):
            qualified = scope.sources.get(value.table.lower())
            if qualified is not None and qualified.base_table is not None:
                safe.extend(sorted(qualified.columns))
            continue
        if not isinstance(value, exp.Column):
            continue

        if value.table:
            qualified = scope.sources.get(value.table.lower())
            from_base_table = qualified is not None and qualified.base_table is not None
        else:
            from_base_table = any(
                source.base_table is not None and value.name.lower() in source.columns
                for source in scope.sources.values()
            )
        if from_base_table and projection.output_name:
            safe.append(sql_guard._database_output_name(projection))

    return tuple(dict.fromkeys(safe))


def _legacy_check_cast_types(statement: exp.Select) -> None:
    """**수정 전** 캐스트 검사 — 없었다. 대상 타입을 한 번도 읽지 않았다."""
    del statement


@contextmanager
def _legacy_guard() -> Iterator[None]:
    """두 자리만 수정 전 규칙으로 갈아 끼운다. 나머지 검증은 그대로 돈다."""
    new_provenance = sql_guard._pii_safe_output_columns
    new_cast_check = sql_guard._check_cast_types
    sql_guard._pii_safe_output_columns = _legacy_pii_safe_output_columns
    sql_guard._check_cast_types = _legacy_check_cast_types
    try:
        yield
    finally:
        sql_guard._pii_safe_output_columns = new_provenance
        sql_guard._check_cast_types = new_cast_check


# ── 판정 단위 ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Verdict:
    """조회문 1건의 판정 — 거부 사유 코드(통과면 `None`)와 허용 출처 목록."""

    rule: str | None
    pii_safe: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"거부[{self.rule}]" if self.rule else f"통과{list(self.pii_safe)}"


def judge(sql: str, *, order_no: str = ORDER_NO) -> Verdict:
    try:
        validated = validate_sql(sql, order_no=order_no, max_rows=MAX_ROWS)
    except SqlGuardRejection as rejection:
        return Verdict(rule=rejection.rule.value, pii_safe=())
    return Verdict(rule=None, pii_safe=validated.pii_safe_output_columns)


@dataclass(frozen=True)
class Case:
    category: str
    label: str
    sql: str


# ── 손으로 만든 조회문 코퍼스 (전수) ────────────────────────────────────────

CORPUS: Final[tuple[Case, ...]] = (
    # ① 승인한 정상 조회 — 수정이 노리는 회복 대상.
    Case("정상", "직접 컬럼", f"SELECT customer_phone, order_no FROM orders {SCOPE}"),
    Case("정상", "직접 컬럼 별칭", f"SELECT customer_phone AS phone FROM orders {SCOPE}"),
    Case("정상", "별표", f"SELECT * FROM orders {SCOPE}"),
    Case(
        "정상",
        "임시 테이블 경유",
        f"WITH t AS (SELECT customer_phone, order_no FROM orders {SCOPE}) "
        "SELECT customer_phone FROM t",
    ),
    Case(
        "정상",
        "임시 테이블 별표",
        f"WITH t AS (SELECT * FROM orders {SCOPE}) SELECT customer_phone, customer_email FROM t",
    ),
    Case(
        "정상",
        "임시 테이블 두 겹",
        f"WITH a AS (SELECT customer_phone AS p, order_no FROM orders {SCOPE}), "
        "b AS (SELECT p, order_no FROM a) SELECT p FROM b",
    ),
    Case(
        "정상",
        "파생 테이블 경유",
        f"SELECT d.customer_phone FROM (SELECT customer_phone, order_no FROM orders {SCOPE}) d",
    ),
    Case(
        "정상",
        "파생 테이블 한정 별표",
        f"SELECT d.* FROM (SELECT customer_phone, order_no FROM orders {SCOPE}) d",
    ),
    Case(
        "정상",
        "빈칸 채우기 연락처",
        f"SELECT coalesce(customer_phone, '미정') AS customer_phone FROM orders {SCOPE}",
    ),
    Case(
        "정상",
        "빈칸 채우기 송장",
        f"SELECT coalesce(tracking_no, '미정') AS tracking_no, order_no FROM orders {SCOPE}",
    ),
    Case(
        "정상",
        "빈칸 채우기 컬럼 둘",
        f"SELECT coalesce(shipped_at, delivered_at) AS shipped FROM orders {SCOPE}",
    ),
    Case(
        "정상",
        "빈칸 채우기 nullif",
        f"SELECT nullif(customer_phone, '') AS customer_phone FROM orders {SCOPE}",
    ),
    Case(
        "정상",
        "임시 테이블 안의 빈칸 채우기",
        f"WITH t AS (SELECT coalesce(customer_phone, '미정') AS p, order_no FROM orders {SCOPE}) "
        "SELECT p FROM t",
    ),
    Case("정상", "허용 캐스트 문자열", f"SELECT quantity::text AS q FROM orders {SCOPE}"),
    Case(
        "정상",
        "허용 캐스트 일자",
        f"SELECT cast(ordered_at AS date) AS d, order_no FROM orders {SCOPE}",
    ),
    Case(
        "정상",
        "허용 캐스트 숫자",
        f"SELECT cast(total_price_krw AS numeric) AS p, order_no FROM orders {SCOPE}",
    ),
    Case(
        "정상",
        "집계와 조건 분기",
        f"SELECT count(*) AS n, max(ordered_at) AS last FROM orders {SCOPE}",
    ),
    Case(
        "정상",
        "자기 조인",
        "SELECT o1.customer_phone FROM orders o1 JOIN orders o2 ON o1.order_no = o2.order_no "
        f"WHERE o1.order_no = '{ORDER_NO}' AND o2.order_no = '{ORDER_NO}'",
    ),
    # ② 합성 공격 — 넓힌 규칙이 열지 말아야 할 것.
    Case(
        "공격",
        "빈칸 채우기 지어낸 번호",
        f"SELECT coalesce(customer_phone, '010-0000-0000') AS customer_phone FROM orders {SCOPE}",
    ),
    Case(
        "공격",
        "빈칸 채우기 지어낸 이메일",
        "SELECT coalesce(customer_email, 'nobody@example.com') AS customer_email "
        f"FROM orders {SCOPE}",
    ),
    Case(
        "공격",
        "빈칸 채우기 대표번호",
        f"SELECT coalesce(courier, '1588-1234') AS courier FROM orders {SCOPE}",
    ),
    Case(
        "공격",
        "빈칸 채우기 전각 번호",
        "SELECT coalesce(customer_phone, '\uff10\uff11\uff10-9999-9999') AS customer_phone "
        f"FROM orders {SCOPE}",
    ),
    Case(
        "공격",
        "빈칸 채우기 en dash 번호",
        "SELECT coalesce(customer_phone, '010\u20139999\u20139999') AS customer_phone "
        f"FROM orders {SCOPE}",
    ),
    Case(
        "공격",
        "빈칸 채우기 escape 문자열",
        r"SELECT coalesce(customer_phone, E'010\0559999\0559999') AS customer_phone "
        f"FROM orders {SCOPE}",
    ),
    Case(
        "공격",
        "빈칸 채우기 안 조각 합성",
        "SELECT coalesce(customer_phone, concat('010', '-9999-', '9999')) AS customer_phone "
        f"FROM orders {SCOPE}",
    ),
    Case(
        "공격",
        "조각 합성 이어 붙이기",
        "SELECT coalesce(NULL, '01') || coalesce(NULL, '0-9999-9999') AS customer_phone "
        f"FROM orders {SCOPE}",
    ),
    Case(
        "공격",
        "유래 없는 순수 고정값",
        f"SELECT coalesce(NULL, '미정') AS customer_phone FROM orders {SCOPE}",
    ),
    Case(
        "공격",
        "이어 붙이기 컬럼 둘",
        f"SELECT concat(customer_phone, customer_email) AS mixed FROM orders {SCOPE}",
    ),
    Case(
        "공격",
        "임시 테이블 안의 조각 합성",
        "WITH t AS (SELECT concat('010', '-9999-', '9999') AS customer_phone, order_no "
        f"FROM orders {SCOPE}) SELECT customer_phone FROM t",
    ),
    Case(
        "공격",
        "임시 테이블 이름 중복",
        "WITH t AS (SELECT customer_phone AS p, concat('010', '-9999-', '9999') AS p, order_no "
        f"FROM orders {SCOPE}) SELECT p FROM t",
    ),
    Case(
        "공격",
        "파생 테이블 컬럼 별칭 목록",
        f"SELECT x.a FROM (SELECT customer_phone, order_no FROM orders {SCOPE}) x(a, b)",
    ),
    Case(
        "공격",
        "부질의 경유",
        f"SELECT (SELECT customer_phone FROM orders o2 WHERE o2.order_no = '{ORDER_NO}') "
        f"AS customer_phone FROM orders {SCOPE}",
    ),
    Case(
        "공격",
        "조건 분기 경유",
        "SELECT CASE WHEN status IS NULL THEN '미정' ELSE customer_phone END AS customer_phone "
        f"FROM orders {SCOPE}",
    ),
    # ③ 카탈로그 훑기 — 캐스트 대상 타입 검사가 노리는 것.
    Case(
        "카탈로그",
        "regclass 존재 확인",
        f"SELECT cast('inquiries' AS regclass) AS x, order_no FROM orders {SCOPE}",
    ),
    Case(
        "카탈로그",
        "oid→regclass 연쇄",
        f"SELECT quantity::oid::regclass AS x, order_no FROM orders {SCOPE}",
    ),
    Case(
        "카탈로그",
        "regtype",
        f"SELECT cast(quantity AS regtype) AS x, order_no FROM orders {SCOPE}",
    ),
    Case(
        "카탈로그",
        "regproc",
        f"SELECT cast(quantity AS regproc) AS x, order_no FROM orders {SCOPE}",
    ),
    Case(
        "카탈로그",
        "name 타입",
        f"SELECT cast(product_name AS name) AS x, order_no FROM orders {SCOPE}",
    ),
    Case(
        "카탈로그",
        "스키마 한정 regclass",
        f"SELECT cast(product_name AS pg_catalog.regclass) AS x, order_no FROM orders {SCOPE}",
    ),
    Case(
        "카탈로그",
        "WHERE 절 경유",
        f"SELECT order_no FROM orders {SCOPE} AND cast('policy_chunks' AS regclass) IS NOT NULL",
    ),
    Case(
        "카탈로그",
        "임시 테이블 경유",
        f"WITH t AS (SELECT cast('inquiries' AS regclass) AS x, order_no FROM orders {SCOPE}) "
        "SELECT x FROM t",
    ),
    Case(
        "카탈로그",
        "빈칸 채우기 경유",
        "SELECT coalesce(product_name, cast('inquiries' AS regclass)) AS x, order_no "
        f"FROM orders {SCOPE}",
    ),
    Case(
        "카탈로그",
        "ORDER BY 경유",
        f"SELECT order_no FROM orders {SCOPE} ORDER BY cast('orders' AS regclass)",
    ),
    # ④ 실제로 뚫렸던 우회 — 그대로 거부돼야 한다.
    Case("보존", "주문 범위 없음", "SELECT order_no, customer_phone FROM orders LIMIT 3"),
    Case(
        "보존",
        "다른 주문번호",
        f"SELECT customer_phone FROM orders WHERE order_no = '{OTHER_ORDER_NO}'",
    ),
    Case(
        "보존",
        "OR 우회",
        f"SELECT customer_phone FROM orders WHERE order_no = '{ORDER_NO}' OR 1 = 1",
    ),
    Case(
        "보존",
        "외부 조인 우회",
        f"SELECT o.* FROM orders o LEFT JOIN (SELECT 1 AS x) d ON o.order_no = '{ORDER_NO}'",
    ),
    Case(
        "보존",
        "임시 테이블 안쪽 미한정",
        "WITH t AS (SELECT customer_phone FROM orders) SELECT customer_phone FROM t",
    ),
    Case(
        "보존",
        "파생 테이블 안쪽 미한정",
        "SELECT d.customer_phone FROM (SELECT customer_phone FROM orders) d",
    ),
    Case("보존", "시스템 컬럼", f"SELECT ctid, order_no FROM orders {SCOPE}"),
    Case("보존", "별칭 우회", f"SELECT 1 AS ctid, ctid, order_no FROM orders {SCOPE}"),
    Case(
        "보존",
        "화이트리스트 밖 테이블",
        f"WITH t AS (SELECT * FROM inquiries) SELECT * FROM t, orders {SCOPE}",
    ),
    Case("보존", "다중문", f"SELECT order_no FROM orders {SCOPE}; DROP TABLE orders"),
    Case("보존", "주석 우회", f"SELECT order_no FROM orders {SCOPE} -- ; DROP TABLE orders"),
    Case("보존", "행 잠금", f"SELECT order_no FROM orders {SCOPE} FOR UPDATE"),
    Case(
        "보존",
        "쓰기 CTE",
        "WITH x AS (INSERT INTO orders (order_no) VALUES ('X') RETURNING order_no) "
        "SELECT order_no FROM x",
    ),
    Case(
        "보존",
        "허용 목록 밖 함수",
        f"SELECT order_no FROM orders {SCOPE} AND pg_sleep(2) IS NULL",
    ),
    Case(
        "보존",
        "직접 상수 projection",
        f"SELECT '010-9999-9999' AS customer_phone, order_no FROM orders {SCOPE}",
    ),
    Case(
        "보존",
        "출력 이름 중복",
        f"SELECT customer_phone AS x, concat('010', '-9999-', '9999') AS x FROM orders {SCOPE}",
    ),
    Case(
        "보존",
        "별표와 계산식 이름 충돌",
        f"SELECT *, concat('010', '-9999-', '9999') AS customer_phone FROM orders {SCOPE}",
    ),
    Case("보존", "소스 별칭 중복", f"SELECT o.order_no FROM orders o, orders o {SCOPE}"),
)


# ── 대조 ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Diff:
    case: Case
    before: Verdict
    after: Verdict

    @property
    def changed(self) -> bool:
        return self.before != self.after


def contrast(cases: tuple[Case, ...]) -> tuple[Diff, ...]:
    with _legacy_guard():
        before = [judge(case.sql) for case in cases]
    after = [judge(case.sql) for case in cases]
    return tuple(
        Diff(case=case, before=old, after=new)
        for case, old, new in zip(cases, before, after, strict=True)
    )


def _print_corpus(diffs: tuple[Diff, ...]) -> None:
    categories: dict[str, list[Diff]] = {}
    for diff in diffs:
        categories.setdefault(diff.case.category, []).append(diff)

    print("## 손으로 만든 조회문 코퍼스 — 전수\n")
    print("| 범주 | 건수 | 판정이 바뀐 건수 | 허용 출처가 는 건수 | 허용 출처가 준 건수 |")
    print("|---|---|---|---|---|")
    for category, group in categories.items():
        widened = sum(1 for d in group if set(d.after.pii_safe) > set(d.before.pii_safe))
        narrowed = sum(1 for d in group if set(d.after.pii_safe) < set(d.before.pii_safe))
        print(
            f"| {category} | {len(group)} | {sum(1 for d in group if d.changed)} "
            f"| {widened} | {narrowed} |"
        )
    print(f"| **합계** | **{len(diffs)}** | **{sum(1 for d in diffs if d.changed)}** | | |")

    print("\n### 바뀐 건 전부\n")
    print("| 범주 | 조회문 | 수정 전 | 수정 후 |")
    print("|---|---|---|---|")
    for diff in diffs:
        if diff.changed:
            print(
                f"| {diff.case.category} | {diff.case.label} "
                f"| {diff.before.label} | {diff.after.label} |"
            )

    unchanged_attacks = [d for d in diffs if d.case.category in {"공격", "보존"} and not d.changed]
    print(
        f"\n공격·보존 {sum(1 for d in diffs if d.case.category in {'공격', '보존'})}건 중 "
        f"**판정이 그대로인 것 {len(unchanged_attacks)}건**."
    )
    leaked = [d for d in diffs if d.case.category == "공격" and d.after.pii_safe]
    print(f"공격 중 허용 출처를 **얻은 것 {len(leaked)}건**" + (f": {leaked}" if leaked else "."))


def _stub_generated_sql() -> tuple[tuple[str, str], ...]:
    """결정론 대역 생성기가 골든셋 전건에 대해 만드는 조회문 — **프로브**."""
    client = StubGenerationClient()
    generated: list[tuple[str, str]] = []
    for case in load_golden_set():
        if case.order_no is None:
            continue
        prompt = build_sql_user_prompt(
            inquiry=case.content, order_no=case.order_no, max_rows=MAX_ROWS
        )
        completion = client.complete_json(
            stage="sql_generation",
            system="",
            user=prompt,
            schema={},
        )
        generated.append((case.id, str(completion.data["sql"])))
    return tuple(generated)


def _print_probe() -> None:
    print("\n## 실제 생성된 SQL — **프로브(전수 아님)**\n")
    generated = _stub_generated_sql()
    if not generated:
        raise SystemExit("대역이 조회문을 하나도 만들지 않았다 — 프로브가 빈 검사가 된다")

    changed: list[tuple[str, Verdict, Verdict]] = []
    for case_id, sql in generated:
        order_no = sql.split("order_no = '", 1)[1].split("'", 1)[0]
        with _legacy_guard():
            before = judge(sql, order_no=order_no)
        after = judge(sql, order_no=order_no)
        if before != after:
            changed.append((case_id, before, after))

    print(f"- 대역이 만든 조회문: **{len(generated)}건** (골든셋 중 주문번호가 있는 문의 전건)")
    print(f"- 판정이 바뀐 건수: **{len(changed)}건**")
    for case_id, before, after in changed:
        print(f"  - {case_id}: {before.label} → {after.label}")
    print(
        "\n**이 줄은 전수가 아니라 프로브다.** 커밋된 리포트가 SQL 문면을 담지 않아 실제 "
        "생성된 조회문에는 자유 모집단이 없다. 대역 생성기는 임시 테이블·빈칸 채우기·형변환을 "
        '스스로 만들어 내지 않으므로, 여기서 "변화 0"이 실제 모델의 조회문도 안 바뀐다는 '
        "뜻은 아니다."
    )


def main() -> None:
    print("# 조회 가드 수정 전/후 대조 (무과금·오프라인)\n")
    _print_corpus(contrast(CORPUS))
    _print_probe()


if __name__ == "__main__":
    main()
