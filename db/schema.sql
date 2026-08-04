-- Reply-Gate 스키마 — 주문 / 정책 청크(+벡터) / 처리 기록이 한 DB 에 산다 (spec "저장" 절).
--
-- 적용 주체는 **애플리케이션 계정**이다 (`reply_gate.db.apply_schema`). 그래야 테이블 소유자가
-- 앱 계정이 되고, 앱 계정이 read-only 그룹에 SELECT 를 줄 수 있다.
-- 재실행 안전: 모든 문장이 IF NOT EXISTS / 멱등 GRANT 다.
--
-- 주의: 이미 존재하는 테이블의 컬럼을 바꾸지는 못한다(IF NOT EXISTS 는 조용히 넘어간다).
-- 스키마를 바꿨으면 `docker compose down -v && docker compose up -d` 로 볼륨째 재생성한다.

CREATE EXTENSION IF NOT EXISTS vector;

-- ── 주문 ─────────────────────────────────────────────────────────────────────
-- text-to-SQL 의 유일한 조회 대상이자, PII 정책(전화번호·이메일) 시연의 원천이다.
CREATE TABLE IF NOT EXISTS orders (
    order_no          text        NOT NULL,
    customer_name     text        NOT NULL,
    customer_phone    text        NOT NULL,
    customer_email    text        NOT NULL,
    shipping_address  text        NOT NULL,
    product_name      text        NOT NULL,
    product_option    text,
    quantity          integer     NOT NULL,
    unit_price_krw    integer     NOT NULL,
    total_price_krw   integer     NOT NULL,
    status            text        NOT NULL,
    ordered_at        timestamptz NOT NULL,
    shipped_at        timestamptz,
    delivered_at      timestamptz,
    courier           text,
    tracking_no       text,

    CONSTRAINT orders_pkey PRIMARY KEY (order_no),
    -- 이 정규식은 `reply_gate.order_ref.ORDER_NO_REGEX` 와 **같은 문자열**이어야 한다.
    -- tests/test_db_schema.py 가 두 곳이 어긋나면 실패한다.
    CONSTRAINT orders_order_no_format CHECK (order_no ~ '^ORD-\d{8}-\d{4}$'),
    CONSTRAINT orders_quantity_positive CHECK (quantity > 0),
    CONSTRAINT orders_unit_price_nonneg CHECK (unit_price_krw >= 0),
    CONSTRAINT orders_total_price_nonneg CHECK (total_price_krw >= 0),
    CONSTRAINT orders_status_enum CHECK (
        status IN ('결제완료', '상품준비중', '배송중', '배송완료', '취소', '반품접수', '환불완료', '교환접수')
    ),
    CONSTRAINT orders_shipped_after_ordered CHECK (shipped_at IS NULL OR shipped_at >= ordered_at),
    CONSTRAINT orders_delivered_after_shipped CHECK (
        delivered_at IS NULL OR (shipped_at IS NOT NULL AND delivered_at >= shipped_at)
    )
);

CREATE INDEX IF NOT EXISTS orders_status_idx ON orders (status);
CREATE INDEX IF NOT EXISTS orders_ordered_at_idx ON orders (ordered_at);
CREATE INDEX IF NOT EXISTS orders_customer_phone_idx ON orders (customer_phone);

-- ── 정책 조항 청크 + 벡터 ────────────────────────────────────────────────────
-- 청킹 단위는 **조항**이다 (spec "데이터" 절). 임베딩은 scripts.index_policies 가 채운다 — 여기서는
-- 컬럼과 인덱스만 만든다. 차원 1536 은 `Settings.embedding_dimensions` 와 같아야 한다.
CREATE TABLE IF NOT EXISTS policy_chunks (
    id              bigint GENERATED ALWAYS AS IDENTITY,
    evidence_id     text   NOT NULL,
    document_slug   text   NOT NULL,
    document_title  text   NOT NULL,
    article         text   NOT NULL,
    article_title   text,
    content         text   NOT NULL,
    embedding       vector(1536),
    created_at      timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT policy_chunks_pkey PRIMARY KEY (id),
    CONSTRAINT policy_chunks_evidence_id_key UNIQUE (evidence_id),
    CONSTRAINT policy_chunks_document_article_key UNIQUE (document_slug, article),
    -- `reply_gate.contracts.policy_evidence_id()` 의 정의를 DB 가 강제한다.
    CONSTRAINT policy_chunks_evidence_id_shape CHECK (
        evidence_id = 'policy:' || document_slug || ':' || article
    )
);

CREATE INDEX IF NOT EXISTS policy_chunks_embedding_idx
    ON policy_chunks USING hnsw (embedding vector_cosine_ops);

-- ── 처리 기록 1 — 문의 ───────────────────────────────────────────────────────
-- 평가 지표(p50/p95 지연, 건당 토큰 비용)의 원천이다 (spec "저장"·"평가" 절).
-- 생성 계열 토큰과 임베딩 토큰을 **분리해서** 기록한다.
CREATE TABLE IF NOT EXISTS inquiries (
    id                 uuid        NOT NULL DEFAULT gen_random_uuid(),
    -- 주문 테이블로의 FK 를 두지 않는다: 존재하지 않는 주문번호도 접수되어야
    -- `order_not_found` 인계 경로가 처리 기록에 남는다 (spec "초안 전 인계" 절).
    order_no           text,
    content            text        NOT NULL,
    intent_source      text,
    status             text        NOT NULL,
    answer             text,
    claims             jsonb       NOT NULL DEFAULT '[]'::jsonb,
    escalation_reason  text,
    -- `llm_call_failed` 일 때 실패한 단계 이름 (spec "LLM 호출 공통 실패 정책").
    failed_stage       text,
    latency_ms         integer     NOT NULL,
    input_tokens       integer     NOT NULL DEFAULT 0,
    output_tokens      integer     NOT NULL DEFAULT 0,
    embedding_tokens   integer     NOT NULL DEFAULT 0,
    created_at         timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT inquiries_pkey PRIMARY KEY (id),
    CONSTRAINT inquiries_order_no_format CHECK (
        order_no IS NULL OR order_no ~ '^ORD-\d{8}-\d{4}$'
    ),
    -- `reply_gate.contracts.IntentSource`
    CONSTRAINT inquiries_intent_source_enum CHECK (
        intent_source IS NULL OR intent_source IN ('policy', 'order', 'both')
    ),
    -- `reply_gate.contracts.InquiryStatus`
    CONSTRAINT inquiries_status_enum CHECK (status IN ('answered', 'escalated')),
    -- `reply_gate.contracts.EscalationReason`
    CONSTRAINT inquiries_escalation_reason_enum CHECK (
        escalation_reason IS NULL OR escalation_reason IN (
            'no_evidence', 'missing_order_ref', 'order_not_found',
            'sql_failed', 'llm_call_failed', 'rejected_twice'
        )
    ),
    -- 종결 상태의 불변식: answered 는 답변이, escalated 는 인계 사유가 반드시 있다.
    CONSTRAINT inquiries_terminal_shape CHECK (
        (status = 'answered' AND answer IS NOT NULL AND escalation_reason IS NULL)
        OR (status = 'escalated' AND answer IS NULL AND escalation_reason IS NOT NULL)
    ),
    CONSTRAINT inquiries_latency_nonneg CHECK (latency_ms >= 0),
    CONSTRAINT inquiries_tokens_nonneg CHECK (
        input_tokens >= 0 AND output_tokens >= 0 AND embedding_tokens >= 0
    )
);

CREATE INDEX IF NOT EXISTS inquiries_created_at_idx ON inquiries (created_at);
CREATE INDEX IF NOT EXISTS inquiries_status_idx ON inquiries (status);

-- ── 처리 기록 2 — 시도 (초안 + 재생성 1회, 최대 2건) ─────────────────────────
CREATE TABLE IF NOT EXISTS inquiry_attempts (
    id              bigint      GENERATED ALWAYS AS IDENTITY,
    inquiry_id      uuid        NOT NULL,
    attempt_no      smallint    NOT NULL,
    verdict         text        NOT NULL,
    reject_reasons  text[]      NOT NULL DEFAULT '{}',
    -- L1 이 검사한 초안 원문. 재현·감사의 근거다.
    draft           jsonb       NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT inquiry_attempts_pkey PRIMARY KEY (id),
    CONSTRAINT inquiry_attempts_inquiry_fkey
        FOREIGN KEY (inquiry_id) REFERENCES inquiries (id) ON DELETE CASCADE,
    CONSTRAINT inquiry_attempts_inquiry_attempt_key UNIQUE (inquiry_id, attempt_no),
    -- 루프 상한(재생성 1회)을 DB 도 강제한다 (spec "종결 [코드]").
    CONSTRAINT inquiry_attempts_attempt_no_range CHECK (attempt_no BETWEEN 1 AND 2),
    -- `reply_gate.contracts.Verdict`
    CONSTRAINT inquiry_attempts_verdict_enum CHECK (verdict IN ('pass', 'reject')),
    -- `reply_gate.contracts.RejectReason`
    CONSTRAINT inquiry_attempts_reject_reasons_enum CHECK (
        reject_reasons <@ ARRAY[
            'schema_violation', 'missing_citation', 'invalid_citation', 'pii_detected'
        ]::text[]
    ),
    CONSTRAINT inquiry_attempts_reasons_match_verdict CHECK (
        (verdict = 'pass' AND cardinality(reject_reasons) = 0)
        OR (verdict = 'reject' AND cardinality(reject_reasons) > 0)
    )
);

CREATE INDEX IF NOT EXISTS inquiry_attempts_inquiry_idx ON inquiry_attempts (inquiry_id);

-- ── 처리 기록 3 — 근거 스냅샷 ────────────────────────────────────────────────
-- SQL 근거는 **실행된 쿼리문과 결과 행 전체**를 영속화한다 (spec "근거 ID 체계").
-- 감사·재현·차기 L2 claim 대조가 모두 이 스냅샷 위에 선다.
CREATE TABLE IF NOT EXISTS inquiry_evidence (
    id             bigint      GENERATED ALWAYS AS IDENTITY,
    inquiry_id     uuid        NOT NULL,
    evidence_id    text        NOT NULL,
    source         text        NOT NULL,
    -- SQL 근거의 실행 순번. 채택된 쿼리에만 매긴다 (정책 근거는 NULL).
    sequence       smallint,
    -- API 응답·화면에 실리는 표시용 요약 (`contracts.Evidence.content`).
    content        text        NOT NULL,
    -- PII allowlist 대조에 쓰는 원문 전체 (`contracts.Evidence.evidence_text`).
    evidence_text  text        NOT NULL,
    -- SQL 근거 전용: 실행된 쿼리문과 결과 행 전체.
    query_sql      text,
    result_rows    jsonb,
    created_at     timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT inquiry_evidence_pkey PRIMARY KEY (id),
    CONSTRAINT inquiry_evidence_inquiry_fkey
        FOREIGN KEY (inquiry_id) REFERENCES inquiries (id) ON DELETE CASCADE,
    CONSTRAINT inquiry_evidence_inquiry_evidence_key UNIQUE (inquiry_id, evidence_id),
    -- `reply_gate.contracts.EvidenceSource`
    CONSTRAINT inquiry_evidence_source_enum CHECK (source IN ('policy', 'sql')),
    -- 근거 ID 체계(`contracts.policy_evidence_id` / `sql_evidence_id`)와 스냅샷 필수 여부를
    -- 소스별로 함께 강제한다.
    CONSTRAINT inquiry_evidence_snapshot_shape CHECK (
        (
            source = 'sql'
            AND sequence IS NOT NULL
            AND query_sql IS NOT NULL
            AND result_rows IS NOT NULL
            AND evidence_id = 'sql:' || inquiry_id::text || ':' || sequence::text
        )
        OR (
            source = 'policy'
            AND sequence IS NULL
            AND query_sql IS NULL
            AND result_rows IS NULL
            AND evidence_id LIKE 'policy:%:%'
        )
    )
);

CREATE INDEX IF NOT EXISTS inquiry_evidence_inquiry_idx ON inquiry_evidence (inquiry_id);

-- ── 처리 기록 4 — SQL 실패 내역 ──────────────────────────────────────────────
-- 안전장치에 거부되거나 실행에 실패한 쿼리는 **근거 ID 없이** 쿼리문·오류만 남긴다
-- (spec "근거 ID 체계" / "SQL 실패 경로").
CREATE TABLE IF NOT EXISTS inquiry_sql_failures (
    id            bigint      GENERATED ALWAYS AS IDENTITY,
    inquiry_id    uuid        NOT NULL,
    -- SQL 생성은 1회 재시도한다 → 1 또는 2.
    attempt_no    smallint    NOT NULL,
    failure_kind  text        NOT NULL,
    query_sql     text,
    error         text        NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT inquiry_sql_failures_pkey PRIMARY KEY (id),
    CONSTRAINT inquiry_sql_failures_inquiry_fkey
        FOREIGN KEY (inquiry_id) REFERENCES inquiries (id) ON DELETE CASCADE,
    CONSTRAINT inquiry_sql_failures_attempt_no_range CHECK (attempt_no BETWEEN 1 AND 2),
    CONSTRAINT inquiry_sql_failures_kind_enum CHECK (
        failure_kind IN ('guard_rejected', 'execution_error', 'generation_failed')
    ),
    -- 유효 SQL 생성 자체가 실패한 경우에만 쿼리문이 없을 수 있다.
    CONSTRAINT inquiry_sql_failures_query_present CHECK (
        failure_kind = 'generation_failed' OR query_sql IS NOT NULL
    )
);

CREATE INDEX IF NOT EXISTS inquiry_sql_failures_inquiry_idx ON inquiry_sql_failures (inquiry_id);

-- ── read-only 권한 (text-to-SQL 안전장치 1) ─────────────────────────────────
-- `reply_gate_readers` 는 `db/init/01_roles.sh` 가 만드는 NOLOGIN 그룹이고,
-- read-only 로그인 계정이 이 그룹의 멤버다. 역할 이름을 고정해 둔 덕분에 계정 이름이
-- 환경 변수여도 이 정적 DDL 이 권한을 부여할 수 있다.
--
-- **orders 에만** SELECT 를 준다. text-to-SQL 의 조회 대상은 주문뿐이고
-- (spec "근거 수집" 절), 처리 기록·정책 청크는 앱 계정만 읽는다 — 안전장치 2(스키마
-- 화이트리스트)와 겹치는 방어층을 하나 더 두는 것이다.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'reply_gate_readers') THEN
        GRANT SELECT ON TABLE public.orders TO reply_gate_readers;
    END IF;
END
$$;
