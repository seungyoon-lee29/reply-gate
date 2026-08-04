"""처리 기록 저장·조회 — `db/schema.sql` 의 테이블 4개와 `ProcessedInquiry` 를 잇는다.

처리 기록은 **평가 지표 산출의 원천**이다(spec "저장"). 그래서 이 모듈의 계약은 하나다:
저장한 `ProcessedInquiry` 를 다시 읽으면 **같은 값**이 나온다. `GET /inquiries/{id}` 는
메모리 캐시가 아니라 이 조회를 통해 응답을 재구성한다.

토큰은 생성 계열(`input_tokens`/`output_tokens`)과 임베딩(`embedding_tokens`)을 **분리해서**
쓴다 — 건당 비용을 계열별로 산출해야 하기 때문이다.

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
    EscalationReason,
    Evidence,
    EvidenceSource,
    InquiryStatus,
    IntentSource,
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
                 input_tokens, output_tokens, embedding_tokens)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            ),
        )

        for attempt in processed.attempts:
            cur.execute(
                """
                INSERT INTO inquiry_attempts
                    (inquiry_id, attempt_no, verdict, reject_reasons, draft)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    processed.inquiry_id,
                    attempt.attempt_no,
                    attempt.verdict.value,
                    [reason.value for reason in attempt.reject_reasons],
                    Jsonb(attempt.draft),
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
    )


def _load_attempts(
    *, conn: psycopg.Connection[DictRow], inquiry_id: str
) -> tuple[AttemptRecord, ...]:
    rows = conn.execute(
        "SELECT attempt_no, verdict, reject_reasons, draft FROM inquiry_attempts"
        " WHERE inquiry_id = %s ORDER BY attempt_no",
        (inquiry_id,),
    ).fetchall()
    return tuple(
        AttemptRecord(
            attempt_no=row["attempt_no"],
            verdict=Verdict(row["verdict"]),
            reject_reasons=tuple(RejectReason(value) for value in row["reject_reasons"]),
            draft=row["draft"],
        )
        for row in rows
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
