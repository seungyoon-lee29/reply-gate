-- Reply-Gate 스키마 — 주문 / 정책 청크(+벡터) / 처리 기록이 한 DB 에 산다
-- (docs/architecture.md "전체 구성").
--
-- 적용 주체는 **애플리케이션 계정**이다 (`reply_gate.db.apply_schema`). 그래야 테이블 소유자가
-- 앱 계정이 되고, 앱 계정이 read-only 그룹에 SELECT 를 줄 수 있다.
-- 재실행 안전: 모든 문장이 IF NOT EXISTS / 멱등 GRANT / 조건부 ALTER 다.
--
-- 주의: `CREATE TABLE IF NOT EXISTS` 는 **이미 있는 테이블의 컬럼을 바꾸지 않는다**
-- (조용히 넘어간다). 컬럼을 바꾸는 변경은 `docker compose down -v && docker compose up -d`
-- 로 볼륨째 재생성하는 것이 기본이다.
--
-- **예외는 컬럼 추가뿐이다.** 추가는 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` 로 살아 있는
-- 볼륨에 그대로 붙일 수 있고(PG11+ 에서 테이블 재작성 없는 메타데이터 연산), 그래야
-- 보존하기로 한 처리 기록과 이미 과금한 정책 인덱스를 날리지 않는다. 새 볼륨에서는 위쪽
-- CREATE TABLE 이 이미 만든 뒤라 아무 일도 하지 않는다 — **두 자리에 같은 정의를 적는
-- 것까지가 이 방식의 값이다.** 컬럼 변경·삭제는 여전히 볼륨 재생성이다.

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
-- 청킹 단위는 **조항**이다 (docs/architecture.md "구성요소 지도" 의 policy_index 행).
-- 임베딩은 scripts.index_policies 가 채운다 — 여기서는
-- 컬럼과 인덱스만 만든다. 차원 1536 은 `Settings.embedding_dimensions` 와 같아야 한다.
--
-- **벡터는 자기 출처를 함께 들고 있어야 한다.** 차원이 다르면 pgvector 가 거부하지만
-- (`different vector dimensions`), **같은 차원의 다른 모델**은 아무도 막지 않는다 —
-- `text-embedding-3-small`·`3-large` 는 둘 다 1536 을 낼 수 있다. 모델만 바꾸고 재색인하지
-- 않으면 서로 다른 공간의 벡터를 코사인으로 비교하고, 그 결과는 오류가 아니라 **근거 없음**
-- 으로 위장된다. `embedding_model`·`embedding_dimensions` 가 그 판정의 근거다
-- (`policy_index.search_policy_chunks`).
CREATE TABLE IF NOT EXISTS policy_chunks (
    id                   bigint GENERATED ALWAYS AS IDENTITY,
    evidence_id          text   NOT NULL,
    document_slug        text   NOT NULL,
    document_title       text   NOT NULL,
    article              text   NOT NULL,
    article_title        text,
    content              text   NOT NULL,
    embedding            vector(1536),
    embedding_model      text   NOT NULL,
    embedding_dimensions integer NOT NULL,
    created_at           timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT policy_chunks_pkey PRIMARY KEY (id),
    CONSTRAINT policy_chunks_evidence_id_key UNIQUE (evidence_id),
    CONSTRAINT policy_chunks_document_article_key UNIQUE (document_slug, article),
    CONSTRAINT policy_chunks_embedding_model_nonempty CHECK (embedding_model <> ''),
    CONSTRAINT policy_chunks_embedding_dimensions_positive CHECK (embedding_dimensions > 0),
    -- `reply_gate.contracts.policy_evidence_id()` 의 정의를 DB 가 강제한다.
    CONSTRAINT policy_chunks_evidence_id_shape CHECK (
        evidence_id = 'policy:' || document_slug || ':' || article
    )
);

-- 벡터 인덱스를 **두지 않는다.** HNSW 는 근사 인덱스라 `LIMIT k` 를 요청해도 k 보다 적게,
-- 그리고 최상위가 빠진 채로 돌려줄 수 있다. 조항이 26개인 코퍼스에서 그것은 순수한 손해다:
-- 정확 스캔이 0.1 ms 도 안 걸리는데 검색 품질을 확률적으로 깎는다. 실측으로 확인한 모양은
-- `top_k=5` · 26행 · 임계값 0.0 인데 **1~2행만, 그것도 1위가 빠진 채** 돌아오는 것이었다
-- (docs/engineering-notes.md "근사 인덱스가 검색을 조용히 잘라먹었다").
--
-- 이 프로젝트의 유일한 주장이 신뢰성 지표라서 더 그렇다 — 검색이 조용히 짧아지면
-- `no_evidence` 인계가 늘고, 그 수치가 "검색 품질"로 리포트에 실린다.
DROP INDEX IF EXISTS policy_chunks_embedding_idx;

-- ── 처리 기록 1 — 문의 ───────────────────────────────────────────────────────
-- 평가 지표(p50/p95 지연, 건당 토큰 비용)의 원천이다 (docs/business-rules.md "엔티티와 관계").
-- 생성 계열 토큰·임베딩 토큰·판정(judge) 토큰을 **분리해서** 기록한다.
CREATE TABLE IF NOT EXISTS inquiries (
    id                 uuid        NOT NULL DEFAULT gen_random_uuid(),
    -- 주문 테이블로의 FK 를 두지 않는다: 존재하지 않는 주문번호도 접수되어야
    -- `order_not_found` 인계 경로가 처리 기록에 남는다 (docs/business-rules.md "엔티티와 관계").
    order_no           text,
    content            text        NOT NULL,
    intent_source      text,
    status             text        NOT NULL,
    answer             text,
    claims             jsonb       NOT NULL DEFAULT '[]'::jsonb,
    escalation_reason  text,
    -- `llm_call_failed` 일 때 실패한 단계 이름 (docs/business-rules.md "인계 사유 6종").
    failed_stage       text,
    latency_ms         integer     NOT NULL,
    input_tokens       integer     NOT NULL DEFAULT 0,
    output_tokens      integer     NOT NULL DEFAULT 0,
    embedding_tokens   integer     NOT NULL DEFAULT 0,
    -- L2 판정(judge) 호출 토큰. 생성 토큰(input/output)과 섞으면 건당 비용에서 게이트
    -- 비용을 분리해 말할 수 없으므로 별도 컬럼이다. 적재 코드가 채우기 전까지는 0 이다.
    judge_input_tokens  integer    NOT NULL DEFAULT 0,
    judge_output_tokens integer    NOT NULL DEFAULT 0,
    -- 검색 단계(질의 재작성) 생성 호출 토큰. 생성 토큰과 섞으면 초안을 만들지도 않은
    -- 문의가 초안 생성 토큰을 쓴 것으로 찍힌다 (docs/contracts.md "토큰 집계 경계").
    retrieval_input_tokens  integer NOT NULL DEFAULT 0,
    retrieval_output_tokens integer NOT NULL DEFAULT 0,
    -- 검색 단계가 폴백한 사유. NULL 은 "폴백하지 않았다"이고 인계 사유가 아니다 —
    -- 조용한 폴백을 금지하는 것이 이 컬럼의 존재 이유다
    -- (docs/business-rules.md "검색 단계 실패 — 폴백이지 인계가 아니다").
    retrieval_fallback_reason text,
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
    ),
    CONSTRAINT inquiries_judge_tokens_nonneg CHECK (
        judge_input_tokens >= 0 AND judge_output_tokens >= 0
    ),
    CONSTRAINT inquiries_retrieval_tokens_nonneg CHECK (
        retrieval_input_tokens >= 0 AND retrieval_output_tokens >= 0
    )
);

-- 이미 있는 볼륨에도 검색 계열 컬럼을 붙인다.
--
-- `CREATE TABLE IF NOT EXISTS` 는 기존 테이블의 컬럼을 바꾸지 못한다. 그래서 지금까지
-- 스키마 변경은 `docker compose down -v` 로 볼륨째 재생성했지만, 이 세 컬럼은 그럴 수
-- 없다 — 볼륨을 지우면 보존하기로 한 처리 기록 2건(오탐 버그 시절 기록·데모용
-- 기각→재생성→통과)이 사라지고, 정책 인덱스 재색인이 다시 과금된다.
--
-- 아래 문장은 이 파일의 기존 성질을 지킨다: **재실행 안전**(IF NOT EXISTS)이고,
-- 새 볼륨에서는 위 CREATE TABLE 이 이미 만든 뒤라 아무 일도 하지 않는다. PG11+ 에서
-- 기본값이 있는 컬럼 추가는 테이블 재작성 없는 메타데이터 연산이다.
-- CHECK 제약은 이름이 겹치면 실패하므로 조건부로 붙인다.
ALTER TABLE inquiries ADD COLUMN IF NOT EXISTS retrieval_input_tokens  integer NOT NULL DEFAULT 0;
ALTER TABLE inquiries ADD COLUMN IF NOT EXISTS retrieval_output_tokens integer NOT NULL DEFAULT 0;
ALTER TABLE inquiries ADD COLUMN IF NOT EXISTS retrieval_fallback_reason text;

DO $$
BEGIN
    ALTER TABLE inquiries ADD CONSTRAINT inquiries_retrieval_tokens_nonneg CHECK (
        retrieval_input_tokens >= 0 AND retrieval_output_tokens >= 0
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END
$$;

CREATE INDEX IF NOT EXISTS inquiries_created_at_idx ON inquiries (created_at);
CREATE INDEX IF NOT EXISTS inquiries_status_idx ON inquiries (status);

-- ── 처리 기록 2 — 시도 (초안 + 재생성 1회, 최대 2건) ─────────────────────────
-- verdict/reject_reasons 는 층 통합 결과(종합)이고, l1_*/l2_* 가 층별 내역이다.
CREATE TABLE IF NOT EXISTS inquiry_attempts (
    id              bigint      GENERATED ALWAYS AS IDENTITY,
    inquiry_id      uuid        NOT NULL,
    attempt_no      smallint    NOT NULL,
    verdict         text        NOT NULL,
    reject_reasons  text[]      NOT NULL DEFAULT '{}',
    -- L1 이 검사한 초안 원문. 재현·감사의 근거다.
    draft           jsonb       NOT NULL,
    -- ── 층별 판정 — L1(규칙 게이트) / L2(판정 LLM) ──
    -- L2 는 L1 통과분에만 실행된다. L2 미실행이면 l2_verdict 와 부속
    -- (l2_reject_reasons·claim_verdicts·evidence_contradictions)은 전부 NULL 이다.
    -- 적재 코드가 층별 컬럼을 채우기 전까지는 NULL 로 남으므로 전부 NULL 허용이다.
    l1_verdict               text,
    l1_reject_reasons        text[],
    l2_verdict               text,
    l2_reject_reasons        text[],
    -- L2 의 claim 단위 판정 원문 (claim 별 지지 여부와 대조 근거).
    claim_verdicts           jsonb,
    -- L2 가 발견한 근거쌍 모순 목록.
    evidence_contradictions  jsonb,
    created_at      timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT inquiry_attempts_pkey PRIMARY KEY (id),
    CONSTRAINT inquiry_attempts_inquiry_fkey
        FOREIGN KEY (inquiry_id) REFERENCES inquiries (id) ON DELETE CASCADE,
    CONSTRAINT inquiry_attempts_inquiry_attempt_key UNIQUE (inquiry_id, attempt_no),
    -- 루프 상한(재생성 1회)을 DB 도 강제한다 (docs/standards.md "재시도 상한").
    CONSTRAINT inquiry_attempts_attempt_no_range CHECK (attempt_no BETWEEN 1 AND 2),
    -- `reply_gate.contracts.Verdict`
    CONSTRAINT inquiry_attempts_verdict_enum CHECK (verdict IN ('pass', 'reject')),
    -- `reply_gate.contracts.RejectReason` — L1 사유 4종 + L2 사유 2종.
    CONSTRAINT inquiry_attempts_reject_reasons_enum CHECK (
        reject_reasons <@ ARRAY[
            'schema_violation', 'missing_citation', 'invalid_citation', 'pii_detected',
            'unsupported_claim', 'contradictory_evidence'
        ]::text[]
    ),
    CONSTRAINT inquiry_attempts_reasons_match_verdict CHECK (
        (verdict = 'pass' AND cardinality(reject_reasons) = 0)
        OR (verdict = 'reject' AND cardinality(reject_reasons) > 0)
    ),
    -- ── 층별 판정 제약 ──
    -- 주의: Postgres CHECK 는 결과가 NULL 이면 통과시킨다. 층별 컬럼은 적재 코드가 채우기
    -- 전까지 NULL 이므로, 아래 제약들은 NULL 스코프를 명시한다 — 미기록(NULL) 행은 걸리지
    -- 않고, 기록된 행은 정확히 걸린다.
    CONSTRAINT inquiry_attempts_l1_verdict_enum CHECK (
        l1_verdict IS NULL OR l1_verdict IN ('pass', 'reject')
    ),
    CONSTRAINT inquiry_attempts_l2_verdict_enum CHECK (
        l2_verdict IS NULL OR l2_verdict IN ('pass', 'reject')
    ),
    -- 사유는 층을 넘지 못한다: L1 배열엔 L1 사유만, L2 배열엔 L2 사유만.
    CONSTRAINT inquiry_attempts_l1_reject_reasons_enum CHECK (
        l1_reject_reasons IS NULL OR l1_reject_reasons <@ ARRAY[
            'schema_violation', 'missing_citation', 'invalid_citation', 'pii_detected'
        ]::text[]
    ),
    CONSTRAINT inquiry_attempts_l2_reject_reasons_enum CHECK (
        l2_reject_reasons IS NULL OR l2_reject_reasons <@ ARRAY[
            'unsupported_claim', 'contradictory_evidence'
        ]::text[]
    ),
    -- 층별 pass ⟺ 그 층 사유 0건. verdict 가 기록됐으면 사유 배열도 반드시 함께 있다
    -- (IS NOT NULL 이 없으면 cardinality(NULL)=NULL 로 검사가 통과해 버린다).
    CONSTRAINT inquiry_attempts_l1_reasons_match_verdict CHECK (
        (l1_verdict IS NULL AND l1_reject_reasons IS NULL)
        OR (l1_verdict = 'pass' AND l1_reject_reasons IS NOT NULL
            AND cardinality(l1_reject_reasons) = 0)
        OR (l1_verdict = 'reject' AND l1_reject_reasons IS NOT NULL
            AND cardinality(l1_reject_reasons) > 0)
    ),
    CONSTRAINT inquiry_attempts_l2_reasons_match_verdict CHECK (
        (l2_verdict IS NULL AND l2_reject_reasons IS NULL)
        OR (l2_verdict = 'pass' AND l2_reject_reasons IS NOT NULL
            AND cardinality(l2_reject_reasons) = 0)
        OR (l2_verdict = 'reject' AND l2_reject_reasons IS NOT NULL
            AND cardinality(l2_reject_reasons) > 0)
    ),
    -- L2 는 L1 통과분에만 실행된다 → L2 판정이 있으면 L1 은 반드시 pass 다.
    -- "L1 reject 인데 L2 존재"와 "L1 미기록인데 L2 존재"를 함께 막는다.
    CONSTRAINT inquiry_attempts_l2_requires_l1_pass CHECK (
        l2_verdict IS NULL OR (l1_verdict IS NOT NULL AND l1_verdict = 'pass')
    ),
    -- L2 미실행(NULL)이면 부속도 전부 없어야 한다: 사유·claim 판정·모순쌍.
    CONSTRAINT inquiry_attempts_l2_null_means_no_artifacts CHECK (
        l2_verdict IS NOT NULL
        OR (l2_reject_reasons IS NULL AND claim_verdicts IS NULL
            AND evidence_contradictions IS NULL)
    ),
    -- 종합 pass 인데 어느 층이 reject 인 모순 상태 불가.
    CONSTRAINT inquiry_attempts_pass_needs_no_layer_reject CHECK (
        verdict <> 'pass'
        OR (l1_verdict IS DISTINCT FROM 'reject' AND l2_verdict IS DISTINCT FROM 'reject')
    ),
    -- 종합 사유 = L1 사유 ∥ L2 사유. 층별 기록이 있으면 종합과 어긋난 상태가
    -- 양방향으로 닫힌다 (예: 두 층 모두 pass 인데 종합 reject 인 행은 존재할 수 없다).
    CONSTRAINT inquiry_attempts_reasons_compose_layers CHECK (
        l1_reject_reasons IS NULL
        OR reject_reasons = l1_reject_reasons || COALESCE(l2_reject_reasons, '{}'::text[])
    )
);

CREATE INDEX IF NOT EXISTS inquiry_attempts_inquiry_idx ON inquiry_attempts (inquiry_id);

-- ── 처리 기록 3 — 근거 스냅샷 ────────────────────────────────────────────────
-- SQL 근거는 **실행된 쿼리문과 결과 행 전체**를 영속화한다 (docs/contracts.md "답변 계약").
-- 감사·재현·L2 claim 대조가 모두 이 스냅샷 위에 선다.
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
-- (docs/contracts.md "답변 계약" / docs/business-rules.md "인계 사유 6종").
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
-- (docs/security.md "text-to-SQL 안전장치"), 처리 기록·정책 청크는 앱 계정만 읽는다 — 안전장치 2(스키마
-- 화이트리스트)와 겹치는 방어층을 하나 더 두는 것이다.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'reply_gate_readers') THEN
        GRANT SELECT ON TABLE public.orders TO reply_gate_readers;
    END IF;
END
$$;
