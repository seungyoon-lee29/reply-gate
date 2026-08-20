"""조회 가드의 PII 허용 출처 — **유래를 안쪽 스코프까지 따라간다**.

승인 대상이 `orders` 직접 컬럼 projection 하나뿐이라, 의미가 같고 가드를 전부 통과하는
조회인데 **임시 테이블(CTE)·파생 테이블·빈칸 채우기 함수**를 거치면 진짜 고객 연락처가
`evidence_text` 에서 사라져 **정상 답변이 `pii_detected` 로 기각**됐다. 빈칸 채우기
(`coalesce`·`nullif`)는 조회 생성 안내가 허용 함수로 직접 광고하는 것이고, 허용한 이유
자체가 "미배송 주문의 빈 컬럼을 근거 문장에 그대로 싣지 않기 위해"다.

**넓히는 수정이므로 세 겹으로 검사한다**(`tests/AGENTS.md` "양성 대조를 반드시 함께 둔다"):

* **양성** — 임시 테이블·파생 테이블·빈칸 채우기를 거친 직접 컬럼이 허용 출처가 되고,
  정상 에코가 실제로 게이트를 통과한다.
* **음성·공격** — 개인정보 모양 고정값·이어 붙이기 합성·조각 합성·부질의 경유가 허용
  출처를 **얻지 못한다.**
* **보존** — 실제로 뚫렸던 우회 목록(주문 1건 한정·화이트리스트·별칭·외부 조인·출력 이름
  중복)이 전부 그대로 거부된다. 특히 **출력 이름 중복 거부는 이 수정이 재작업하는 바로 그
  자리의 기존 규칙**이라 회귀 가드로 둔다.

"개인정보 모양"의 정의는 여기서 만들지 않는다 — 게이트 모듈이 단독 소유하고
(`gate.pii_shaped`) 조회 가드는 가져다 쓴다. `tests/test_pii_pattern_ownership.py` 가
그 형태를 AST 로 지킨다.
"""

from __future__ import annotations

import pytest

from reply_gate.contracts import Evidence, EvidenceSource, Verdict
from reply_gate.evidence import _sql_evidence_texts
from reply_gate.gate import evaluate_draft
from reply_gate.sql_guard import SqlGuardRejection, validate_sql

MAX_ROWS = 50
ORDER_NO = "ORD-20260201-0001"
SCOPE = f"WHERE order_no = '{ORDER_NO}'"

#: 근거에 실제로 있는 연락처 — 정상 에코가 살아나는지 재는 값이다.
STORED_PHONE = "010-2345-6789"


def _safe(sql: str) -> tuple[str, ...]:
    return validate_sql(sql, order_no=ORDER_NO, max_rows=MAX_ROWS).pii_safe_output_columns


def _reject(sql: str) -> SqlGuardRejection:
    with pytest.raises(SqlGuardRejection) as excinfo:
        validate_sql(sql, order_no=ORDER_NO, max_rows=MAX_ROWS)
    return excinfo.value


# ── 양성 — 승인한 정상 조회가 허용 출처를 얻는다 ────────────────────────────


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        pytest.param(
            f"WITH t AS (SELECT customer_phone, order_no FROM orders {SCOPE}) "
            "SELECT customer_phone FROM t",
            ("customer_phone",),
            id="임시테이블_경유",
        ),
        pytest.param(
            f"WITH t AS (SELECT * FROM orders {SCOPE}) "
            "SELECT customer_phone, customer_email FROM t",
            ("customer_phone", "customer_email"),
            id="임시테이블_별표",
        ),
        pytest.param(
            f"WITH t AS (SELECT customer_phone AS p, order_no FROM orders {SCOPE}) "
            "SELECT p AS phone FROM t",
            ("phone",),
            id="임시테이블_이중_별칭",
        ),
        pytest.param(
            f"SELECT d.customer_phone FROM (SELECT customer_phone, order_no FROM orders {SCOPE}) d",
            ("customer_phone",),
            id="파생테이블_경유",
        ),
        pytest.param(
            f"SELECT d.* FROM (SELECT customer_phone, order_no FROM orders {SCOPE}) d",
            ("customer_phone", "order_no"),
            id="파생테이블_한정_별표",
        ),
        pytest.param(
            f"SELECT coalesce(customer_phone, '미정') AS customer_phone FROM orders {SCOPE}",
            ("customer_phone",),
            id="빈칸채우기_전화",
        ),
        pytest.param(
            f"SELECT coalesce(tracking_no, '미정') AS tracking_no FROM orders {SCOPE}",
            ("tracking_no",),
            id="빈칸채우기_송장",
        ),
        pytest.param(
            f"SELECT coalesce(shipped_at, delivered_at) AS shipped FROM orders {SCOPE}",
            ("shipped",),
            id="빈칸채우기_컬럼둘",
        ),
        pytest.param(
            f"SELECT nullif(customer_phone, '') AS customer_phone FROM orders {SCOPE}",
            ("customer_phone",),
            id="빈칸채우기_nullif",
        ),
        pytest.param(
            "WITH t AS (SELECT coalesce(customer_phone, '미정') AS p, order_no "
            f"FROM orders {SCOPE}) SELECT p FROM t",
            ("p",),
            id="임시테이블_안의_빈칸채우기",
        ),
        pytest.param(
            f"SELECT (customer_phone) AS phone FROM orders {SCOPE}",
            ("phone",),
            id="괄호로_감싼_직접컬럼",
        ),
    ],
)
def test_유래가_증명된_출력은_허용_출처가_된다(sql: str, expected: tuple[str, ...]) -> None:
    """유래 추적이 안쪽 스코프까지 내려가지 않으면 이 목록이 전부 빈 튜플이 된다."""
    assert _safe(sql) == expected


def test_임시테이블을_거친_정상_에코가_게이트를_통과한다() -> None:
    """가드 → 근거 렌더 → L1 을 이어 붙인 대조.

    유래 추적이 안쪽 스코프에서 멈추면 진짜 연락처가 `evidence_text` 에서 빠지고,
    그 값을 그대로 옮겨 적은 **정상 답변이 `pii_detected` 로 기각**된다. 재생성도 같은
    근거를 쓰므로 반드시 다시 실패해 `rejected_twice` 인계가 된다.
    """
    validated = validate_sql(
        f"WITH t AS (SELECT customer_phone, order_no FROM orders {SCOPE}) "
        "SELECT customer_phone FROM t",
        order_no=ORDER_NO,
        max_rows=MAX_ROWS,
    )
    _content, evidence_text = _sql_evidence_texts(
        sql=validated.sql,
        rows=({"customer_phone": STORED_PHONE},),
        pii_safe_output_columns=validated.pii_safe_output_columns,
    )
    evidence = Evidence(
        id="sql:00000000-0000-0000-0000-000000000001:1",
        source=EvidenceSource.SQL,
        content="표시용",
        evidence_text=evidence_text,
    )

    result = evaluate_draft(
        raw_draft={
            "claims": [
                {"text": f"등록된 연락처는 {STORED_PHONE} 입니다.", "citation_ids": [evidence.id]}
            ]
        },
        evidences=(evidence,),
    )

    assert f"customer_phone={STORED_PHONE}" in evidence_text
    assert result.verdict is Verdict.PASS
    assert result.reject_reasons == ()


# ── 음성·공격 — 합성된 값은 허용 출처를 얻지 못한다 ─────────────────────────


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            f"SELECT coalesce(customer_phone, '010-0000-0000') AS customer_phone FROM orders {SCOPE}",
            id="빈칸채우기_지어낸_휴대전화",
        ),
        pytest.param(
            f"SELECT coalesce(courier, '1588-1234') AS courier FROM orders {SCOPE}",
            id="빈칸채우기_지어낸_대표번호",
        ),
        pytest.param(
            "SELECT coalesce(customer_email, 'nobody@example.com') AS customer_email "
            f"FROM orders {SCOPE}",
            id="빈칸채우기_지어낸_이메일",
        ),
        pytest.param(
            "SELECT coalesce(customer_phone, '02-9999-9999') AS customer_phone "
            f"FROM orders {SCOPE}",
            id="빈칸채우기_지어낸_일반전화",
        ),
        pytest.param(
            "SELECT coalesce(customer_phone, '900101-1234567') AS customer_phone "
            f"FROM orders {SCOPE}",
            id="빈칸채우기_지어낸_주민번호",
        ),
        pytest.param(
            "SELECT coalesce(customer_phone, '\uff10\uff11\uff10-9999-9999') AS customer_phone "
            f"FROM orders {SCOPE}",
            id="빈칸채우기_전각_휴대전화",
        ),
        pytest.param(
            "SELECT coalesce(customer_phone, '010\u20139999\u20139999') AS customer_phone "
            f"FROM orders {SCOPE}",
            id="빈칸채우기_en_dash_휴대전화",
        ),
        pytest.param(
            r"SELECT coalesce(customer_phone, E'010\0559999\0559999') AS customer_phone "
            f"FROM orders {SCOPE}",
            id="빈칸채우기_escape_문자열",
        ),
        pytest.param(
            r"SELECT coalesce(customer_phone, U&'010\002D9999\002D9999') AS customer_phone "
            f"FROM orders {SCOPE}",
            id="빈칸채우기_unicode_문자열",
        ),
        pytest.param(
            "SELECT coalesce(customer_phone, $$010-9999-9999$$) AS customer_phone "
            f"FROM orders {SCOPE}",
            id="빈칸채우기_dollar_quoted",
        ),
        pytest.param(
            "SELECT coalesce(customer_phone, N'010-9999-9999') AS customer_phone "
            f"FROM orders {SCOPE}",
            id="빈칸채우기_national_문자열",
        ),
        pytest.param(
            "SELECT coalesce(customer_phone, cast(1099999999 AS text)) AS customer_phone "
            f"FROM orders {SCOPE}",
            id="빈칸채우기_숫자cast",
        ),
        pytest.param(
            "SELECT coalesce(customer_phone, concat('010', '-9999-', '9999')) AS customer_phone "
            f"FROM orders {SCOPE}",
            id="빈칸채우기_조각_concat",
        ),
        pytest.param(
            f"SELECT coalesce(NULL, '미정') AS customer_phone FROM orders {SCOPE}",
            id="유래_없는_순수_고정값",
        ),
        pytest.param(
            "SELECT coalesce(NULL, '01') || coalesce(NULL, '0-9999-9999') AS customer_phone "
            f"FROM orders {SCOPE}",
            id="조각_합성_이어붙이기",
        ),
        pytest.param(
            "SELECT concat(coalesce(customer_phone, '01'), '0-9999-9999') AS customer_phone "
            f"FROM orders {SCOPE}",
            id="조각_합성_concat",
        ),
        pytest.param(
            f"SELECT customer_phone || '' AS customer_phone FROM orders {SCOPE}",
            id="이어붙이기_빈문자열",
        ),
        pytest.param(
            f"SELECT concat(customer_phone, customer_email) AS mixed FROM orders {SCOPE}",
            id="이어붙이기_컬럼둘",
        ),
        pytest.param(
            f"SELECT (SELECT customer_phone FROM orders o2 WHERE o2.order_no = '{ORDER_NO}') "
            f"AS customer_phone FROM orders {SCOPE}",
            id="부질의_경유",
        ),
        pytest.param(
            "WITH t AS (SELECT concat('010', '-9999-', '9999') AS customer_phone, order_no "
            f"FROM orders {SCOPE}) SELECT customer_phone FROM t",
            id="임시테이블_안의_조각_합성",
        ),
        pytest.param(
            r"WITH t AS (SELECT E'010\0559999\0559999' AS p, order_no "
            f"FROM orders {SCOPE}) SELECT p FROM t",
            id="임시테이블_안의_escape_문자열",
        ),
        pytest.param(
            f"SELECT x.a FROM (SELECT customer_phone, order_no FROM orders {SCOPE}) x(a, b)",
            id="파생테이블_컬럼_별칭_목록",
        ),
        pytest.param(
            f"WITH t(a, b) AS (SELECT customer_phone, order_no FROM orders {SCOPE}) "
            "SELECT a FROM t",
            id="임시테이블_컬럼_별칭_목록",
        ),
        pytest.param(
            "SELECT CASE WHEN status IS NULL THEN '미정' ELSE customer_phone END AS customer_phone "
            f"FROM orders {SCOPE}",
            id="조건분기는_빈칸채우기가_아니다",
        ),
        pytest.param(
            f"SELECT upper(customer_phone) AS customer_phone FROM orders {SCOPE}",
            id="문자열_함수",
        ),
        pytest.param(
            f"SELECT cast(customer_phone AS text) AS customer_phone FROM orders {SCOPE}",
            id="형변환",
        ),
    ],
)
def test_합성될_수_있는_출력은_허용_출처가_되지_않는다(sql: str) -> None:
    """검증기가 값의 유래를 증명하지 못한 출력은 **자동으로 불승인**이다(fail-closed)."""
    assert _safe(sql) == ()


def test_지어낸_번호는_임시테이블을_거쳐도_allowlist_근거가_되지_않는다() -> None:
    """공격의 끝단 — 승인되지 않은 출력의 값은 `evidence_text` 에서 떨어져 나간다."""
    fabricated = "010-9999-9999"
    validated = validate_sql(
        f"WITH t AS (SELECT concat('010', '-9999-', '9999') AS customer_phone, order_no "
        f"FROM orders {SCOPE}) SELECT customer_phone FROM t",
        order_no=ORDER_NO,
        max_rows=MAX_ROWS,
    )
    _content, evidence_text = _sql_evidence_texts(
        sql=validated.sql,
        rows=({"customer_phone": fabricated},),
        pii_safe_output_columns=validated.pii_safe_output_columns,
    )
    evidence = Evidence(
        id="sql:00000000-0000-0000-0000-000000000002:1",
        source=EvidenceSource.SQL,
        content="표시용",
        evidence_text=evidence_text,
    )

    result = evaluate_draft(
        raw_draft={
            "claims": [
                {"text": f"등록된 연락처는 {fabricated} 입니다.", "citation_ids": [evidence.id]}
            ]
        },
        evidences=(evidence,),
    )

    assert fabricated not in evidence_text
    assert result.verdict is Verdict.REJECT


# ── 보존 — 실제로 뚫렸던 우회는 그대로 거부된다 ─────────────────────────────


@pytest.mark.parametrize(
    ("sql", "rule"),
    [
        pytest.param(
            "SELECT order_no, customer_name, customer_phone FROM orders LIMIT 3",
            "order_scope",
            id="주문_범위_없음",
        ),
        pytest.param(
            f"SELECT o.* FROM orders o LEFT JOIN (SELECT 1 AS x) d ON o.order_no = '{ORDER_NO}'",
            "unsupported_join",
            id="외부_조인",
        ),
        pytest.param(
            "WITH t AS (SELECT customer_phone FROM orders) SELECT customer_phone FROM t",
            "order_scope",
            id="임시테이블_안쪽_미한정",
        ),
        pytest.param(
            "SELECT d.customer_phone FROM (SELECT customer_phone FROM orders) d",
            "order_scope",
            id="파생테이블_안쪽_미한정",
        ),
        pytest.param(
            f"SELECT ctid, order_no FROM orders {SCOPE}",
            "unknown_column",
            id="시스템_컬럼",
        ),
        pytest.param(
            f"SELECT 1 AS ctid, ctid, order_no FROM orders {SCOPE}",
            "unknown_column",
            id="별칭_우회",
        ),
        pytest.param(
            f"WITH t AS (SELECT * FROM inquiries) SELECT * FROM t, orders {SCOPE}",
            "unknown_table",
            id="화이트리스트_밖_테이블",
        ),
        pytest.param(
            f"SELECT '010-9999-9999' AS customer_phone, order_no FROM orders {SCOPE}",
            "unsupported_projection",
            id="직접_상수_projection",
        ),
        pytest.param(
            f"SELECT customer_phone AS x, concat('010', '-9999-', '9999') AS x FROM orders {SCOPE}",
            "unsupported_projection",
            id="출력_이름_중복",
        ),
        pytest.param(
            f"SELECT *, concat('010', '-9999-', '9999') AS customer_phone FROM orders {SCOPE}",
            "unsupported_projection",
            id="별표와_계산식_이름_충돌",
        ),
        pytest.param(
            "SELECT t.customer_phone, concat('010', '-9999-', '9999') AS customer_phone "
            f"FROM (SELECT customer_phone, order_no FROM orders {SCOPE}) t",
            "unsupported_projection",
            id="파생테이블_출력_이름_중복",
        ),
    ],
)
def test_기존_우회는_유래_확대_뒤에도_그대로_거부된다(sql: str, rule: str) -> None:
    """넓히는 수정이 옛 구멍을 다시 여는지 재는 회귀 가드다."""
    assert _reject(sql).rule == rule


def test_임시테이블_출력_이름이_겹치면_허용_출처가_되지_않는다() -> None:
    """안쪽에서 계산값이 직접 컬럼 이름을 덮으면 유래가 증명되지 않는다.

    바깥 스코프의 중복 거부(`_check_unique_output_names`)는 안쪽 SELECT 를 보지 않으므로,
    유래 추적 자체가 "같은 이름의 projection 이 전부 직접 컬럼일 때만" 승인해야 한다.
    """
    sql = (
        "WITH t AS (SELECT customer_phone AS p, concat('010', '-9999-', '9999') AS p, order_no "
        f"FROM orders {SCOPE}) SELECT p FROM t"
    )

    assert _safe(sql) == ()
