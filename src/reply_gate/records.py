"""처리 기록 저장·조회 — `db/schema.sql` 의 테이블 4개와 `ProcessedInquiry` 를 잇는다.

처리 기록은 **평가 지표 산출의 원천**이다. 그래서 이 모듈의 계약은 하나다:
저장한 `ProcessedInquiry` 를 다시 읽으면 **같은 값**이 나온다. `GET /inquiries/{id}` 는
메모리 캐시가 아니라 이 조회를 통해 응답을 재구성한다.

왕복 대상은 종합 판정만이 아니라 **층별 내역까지**다: 시도별 L1 판정과 L2 판정(사유 ·
claim 단위 판정 · 근거쌍 모순), 그리고 판정 토큰. L2 미실행(L1 reject · 스위치 꺼짐 ·
L2 호출 실패)은 컬럼이 전부 NULL 이고 복원값도 `None` 이다 — "판정이 없었다"와 "판정이
비었다"를 저장이 접지 않는다.

왕복 동등성에 남는 단 하나의 예외는 `draft` 다. L1 이 검사한 **원시** 초안을 jsonb 로
그대로 남기므로, JSON 이 구분하지 못하는 파이썬 값(예: 튜플)은 리스트로 돌아온다.
실행 경로의 초안은 `json.loads` 산출이라 이 예외에 걸리지 않는다.

토큰은 생성 계열(`input_tokens`/`output_tokens`)·임베딩(`embedding_tokens`)·판정
(`judge_input_tokens`/`judge_output_tokens`)을 **분리해서** 쓴다 — 건당 비용을 계열별로
산출해야 하기 때문이다.

커밋은 하지 않는다. 트랜잭션 경계는 호출자(요청 핸들러·테스트 픽스처)가 정한다.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import psycopg
from psycopg.rows import DictRow
from psycopg.types.json import Jsonb

from reply_gate.contracts import (
    Claim,
    ClaimJudgment,
    EscalationReason,
    Evidence,
    EvidenceContradiction,
    EvidenceSource,
    GateResult,
    InquiryStatus,
    IntentSource,
    JudgeResult,
    RejectReason,
    Verdict,
)
from reply_gate.evidence import SqlEvidenceSnapshot, SqlFailure, SqlFailureKind
from reply_gate.pipeline import AttemptRecord, ProcessedInquiry

__all__ = ["load_inquiry", "save_inquiry"]


# ── 저장 ────────────────────────────────────────────────────────────────────


def save_inquiry(*, conn: psycopg.Connection[DictRow], processed: ProcessedInquiry) -> None:
    """문의 1건 + 시도 + 근거 스냅샷 + SQL 실패 내역을 한 트랜잭션 안에서 넣는다.

    `id` 를 DB 기본값에 맡기지 않고 **명시적으로** 넣는다: SQL 근거 ID 가 이미 이 값을
    품고 있고 `inquiry_evidence` 의 CHECK 가 둘의 일치를 강제하기 때문이다.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO inquiries
                (id, order_no, content, intent_source, status, answer, claims,
                 escalation_reason, failed_stage, latency_ms,
                 input_tokens, output_tokens, embedding_tokens,
                 judge_input_tokens, judge_output_tokens)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                processed.inquiry_id,
                processed.order_no,
                processed.content,
                _enum_value(processed.intent),
                processed.status.value,
                processed.answer,
                Jsonb(_claims_json(processed.claims)),
                _enum_value(processed.escalation_reason),
                processed.failed_stage,
                processed.latency_ms,
                processed.input_tokens,
                processed.output_tokens,
                processed.embedding_tokens,
                processed.judge_input_tokens,
                processed.judge_output_tokens,
            ),
        )

        for attempt in processed.attempts:
            cur.execute(
                """
                INSERT INTO inquiry_attempts
                    (inquiry_id, attempt_no, verdict, reject_reasons, draft,
                     l1_verdict, l1_reject_reasons, l2_verdict, l2_reject_reasons,
                     claim_verdicts, evidence_contradictions)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    processed.inquiry_id,
                    attempt.attempt_no,
                    attempt.verdict.value,
                    [reason.value for reason in attempt.reject_reasons],
                    Jsonb(attempt.draft),
                    *_l1_columns(attempt.l1_result),
                    *_l2_columns(attempt.l2_result),
                ),
            )

        snapshots = {snapshot.evidence_id: snapshot for snapshot in processed.sql_snapshots}
        for item in processed.evidence:
            snapshot = snapshots.get(item.id)
            cur.execute(
                """
                INSERT INTO inquiry_evidence
                    (inquiry_id, evidence_id, source, sequence, content, evidence_text,
                     query_sql, result_rows)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    processed.inquiry_id,
                    item.id,
                    item.source.value,
                    snapshot.sequence if snapshot else None,
                    item.content,
                    item.evidence_text,
                    snapshot.query_sql if snapshot else None,
                    Jsonb(list(snapshot.result_rows)) if snapshot else None,
                ),
            )

        for failure in processed.sql_failures:
            cur.execute(
                """
                INSERT INTO inquiry_sql_failures
                    (inquiry_id, attempt_no, failure_kind, query_sql, error)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    processed.inquiry_id,
                    failure.attempt_no,
                    failure.kind.value,
                    failure.query_sql,
                    failure.error,
                ),
            )


# ── 조회 ────────────────────────────────────────────────────────────────────


def load_inquiry(*, conn: psycopg.Connection[DictRow], inquiry_id: str) -> ProcessedInquiry | None:
    """저장된 처리 기록에서 `ProcessedInquiry` 를 재구성한다. 없으면 `None`.

    uuid 로 해석되지 않는 문자열도 "없음"으로 끝난다 — 경로 파라미터에 무엇이 들어와도
    조회는 404 로 수렴하고 DB 타입 오류가 새어 나가지 않는다.
    """
    try:
        canonical = str(uuid.UUID(inquiry_id))
    except ValueError:
        return None

    row = conn.execute("SELECT * FROM inquiries WHERE id = %s", (canonical,)).fetchone()
    if row is None:
        return None

    evidence, snapshots = _load_evidence(conn=conn, inquiry_id=canonical)
    return ProcessedInquiry(
        inquiry_id=str(row["id"]),
        order_no=row["order_no"],
        content=row["content"],
        intent=IntentSource(row["intent_source"]) if row["intent_source"] else None,
        status=InquiryStatus(row["status"]),
        answer=row["answer"],
        claims=_claims_from_json(row["claims"]),
        escalation_reason=(
            EscalationReason(row["escalation_reason"]) if row["escalation_reason"] else None
        ),
        failed_stage=row["failed_stage"],
        evidence=evidence,
        sql_snapshots=snapshots,
        sql_failures=_load_sql_failures(conn=conn, inquiry_id=canonical),
        attempts=_load_attempts(conn=conn, inquiry_id=canonical),
        latency_ms=row["latency_ms"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        embedding_tokens=row["embedding_tokens"],
        judge_input_tokens=row["judge_input_tokens"],
        judge_output_tokens=row["judge_output_tokens"],
    )


def _load_attempts(
    *, conn: psycopg.Connection[DictRow], inquiry_id: str
) -> tuple[AttemptRecord, ...]:
    """종합 판정 + 층별 내역을 함께 되살린다. 층별 컬럼이 NULL 인 행은 `None` 으로 남는다."""
    rows = conn.execute(
        "SELECT attempt_no, verdict, reject_reasons, draft,"
        " l1_verdict, l1_reject_reasons, l2_verdict, l2_reject_reasons,"
        " claim_verdicts, evidence_contradictions FROM inquiry_attempts"
        " WHERE inquiry_id = %s ORDER BY attempt_no",
        (inquiry_id,),
    ).fetchall()
    return tuple(
        AttemptRecord(
            attempt_no=row["attempt_no"],
            verdict=Verdict(row["verdict"]),
            reject_reasons=tuple(RejectReason(value) for value in row["reject_reasons"]),
            draft=row["draft"],
            l1_result=_gate_result_from_row(row),
            l2_result=_judge_result_from_row(row),
        )
        for row in rows
    )


def _gate_result_from_row(row: DictRow) -> GateResult | None:
    if row["l1_verdict"] is None:
        return None
    return GateResult(
        verdict=Verdict(row["l1_verdict"]),
        reject_reasons=_reasons_from_column(row["l1_reject_reasons"]),
    )


def _judge_result_from_row(row: DictRow) -> JudgeResult | None:
    if row["l2_verdict"] is None:
        return None
    return JudgeResult(
        verdict=Verdict(row["l2_verdict"]),
        reject_reasons=_reasons_from_column(row["l2_reject_reasons"]),
        claim_judgments=_judgments_from_json(row["claim_verdicts"]),
        contradictions=_contradictions_from_json(row["evidence_contradictions"]),
    )


def _load_evidence(
    *, conn: psycopg.Connection[DictRow], inquiry_id: str
) -> tuple[tuple[Evidence, ...], tuple[SqlEvidenceSnapshot, ...]]:
    """근거는 **저장 순서(정책 → SQL)** 그대로 돌려준다 — API 응답의 citations 순서다."""
    rows = conn.execute(
        "SELECT evidence_id, source, sequence, content, evidence_text, query_sql, result_rows"
        " FROM inquiry_evidence WHERE inquiry_id = %s ORDER BY id",
        (inquiry_id,),
    ).fetchall()

    evidence: list[Evidence] = []
    snapshots: list[SqlEvidenceSnapshot] = []
    for row in rows:
        source = EvidenceSource(row["source"])
        evidence.append(
            Evidence(
                id=row["evidence_id"],
                source=source,
                content=row["content"],
                evidence_text=row["evidence_text"],
            )
        )
        if source is EvidenceSource.SQL:
            snapshots.append(
                SqlEvidenceSnapshot(
                    evidence_id=row["evidence_id"],
                    sequence=row["sequence"],
                    query_sql=row["query_sql"],
                    result_rows=tuple(row["result_rows"]),
                )
            )
    return tuple(evidence), tuple(snapshots)


def _load_sql_failures(
    *, conn: psycopg.Connection[DictRow], inquiry_id: str
) -> tuple[SqlFailure, ...]:
    rows = conn.execute(
        "SELECT attempt_no, failure_kind, query_sql, error FROM inquiry_sql_failures"
        " WHERE inquiry_id = %s ORDER BY attempt_no, id",
        (inquiry_id,),
    ).fetchall()
    return tuple(
        SqlFailure(
            attempt_no=row["attempt_no"],
            kind=SqlFailureKind(row["failure_kind"]),
            query_sql=row["query_sql"],
            error=row["error"],
        )
        for row in rows
    )


# ── 변환 ────────────────────────────────────────────────────────────────────


def _enum_value(value: IntentSource | EscalationReason | None) -> str | None:
    return None if value is None else value.value


def _claims_json(claims: Sequence[Claim]) -> list[dict[str, Any]]:
    return [{"text": claim.text, "citation_ids": list(claim.citation_ids)} for claim in claims]


def _claims_from_json(value: object) -> tuple[Claim, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        Claim(text=item["text"], citation_ids=tuple(item["citation_ids"]))
        for item in value
        if isinstance(item, Mapping)
    )


def _reason_values(reasons: Sequence[RejectReason]) -> list[str]:
    """층별 사유 배열. 순서는 계약(`COMBINED_REASON_ORDER`)이 이미 정규화해 넘긴 그대로다 —
    여기서 다시 정렬하면 종합 사유 = L1 ∥ L2 를 검사하는 DB CHECK 와 어긋날 수 있다."""
    return [reason.value for reason in reasons]


def _l1_columns(result: GateResult | None) -> tuple[str | None, list[str] | None]:
    """`(l1_verdict, l1_reject_reasons)`. 층별 내역이 없으면 둘 다 NULL 이다."""
    if result is None:
        return (None, None)
    return (result.verdict.value, _reason_values(result.reject_reasons))


def _l2_columns(
    result: JudgeResult | None,
) -> tuple[str | None, list[str] | None, Jsonb | None, Jsonb | None]:
    """`(l2_verdict, l2_reject_reasons, claim_verdicts, evidence_contradictions)`.

    L2 미실행이면 **부속까지 전부 NULL** 이고(스키마 CHECK 가 같은 것을 요구한다), 실행됐으면
    claim 판정·모순쌍이 0건이라도 빈 배열로 남긴다 — NULL 은 "미실행"의 자리라서
    "실행했고 결과가 0건"과 섞으면 복원이 두 상태를 하나로 접는다.
    """
    if result is None:
        return (None, None, None, None)
    return (
        result.verdict.value,
        _reason_values(result.reject_reasons),
        Jsonb(_judgments_json(result.claim_judgments)),
        Jsonb(_contradictions_json(result.contradictions)),
    )


def _reasons_from_column(value: Sequence[str] | None) -> tuple[RejectReason, ...]:
    """`l1_verdict`/`l2_verdict` 가 있으면 사유 배열도 NOT NULL 이다(CHECK). `None` 방어는
    층별 컬럼이 없던 시절에 적재된 행을 만나도 복원이 죽지 않게 하기 위한 것이다."""
    if value is None:
        return ()
    return tuple(RejectReason(item) for item in value)


def _judgments_json(judgments: Sequence[ClaimJudgment]) -> list[dict[str, Any]]:
    return [
        {
            "claim_text": judgment.claim_text,
            "verdict": judgment.verdict.value,
            "explanation": judgment.explanation,
        }
        for judgment in judgments
    ]


def _judgments_from_json(value: object) -> tuple[ClaimJudgment, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        ClaimJudgment(
            claim_text=item["claim_text"],
            verdict=Verdict(item["verdict"]),
            explanation=item["explanation"],
        )
        for item in value
        if isinstance(item, Mapping)
    )


def _contradictions_json(
    contradictions: Sequence[EvidenceContradiction],
) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id_a": contradiction.evidence_id_a,
            "evidence_id_b": contradiction.evidence_id_b,
            "explanation": contradiction.explanation,
        }
        for contradiction in contradictions
    ]


def _contradictions_from_json(value: object) -> tuple[EvidenceContradiction, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        EvidenceContradiction(
            evidence_id_a=item["evidence_id_a"],
            evidence_id_b=item["evidence_id_b"],
            explanation=item["explanation"],
        )
        for item in value
        if isinstance(item, Mapping)
    )
