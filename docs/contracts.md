# 외부 인터페이스 계약

HTTP 엔드포인트 4개가 외부 표면의 전부다. 인증은 없다.

## 공통 규약

- 요청·응답 모두 `application/json` (웹 폼 `GET /` 만 `text/html`).
- **응답 스키마의 모든 키는 항상 존재한다.** 해당 없는 값은 `null`(스칼라) 또는 `[]`(배열)이며,
  키가 빠지지 않는다. 소비자는 키 존재 여부로 분기하지 않고 값으로 분기한다.
- 오류 본문은 FastAPI 기본 형식 `{"detail": "<한국어 설명>"}` 이다.
- 문의 ID 는 소문자 하이픈 UUID 문자열이다.

## `POST /inquiries` — 문의 접수 + 동기 처리

요청 전체가 처리될 때까지 응답하지 않는다. 실측 지연 중앙값은 약 4.5초, 95백분위 약 9초다.

**요청**

```json
{ "order_no": "ORD-20260202-0001", "content": "제 주문 배송 상태가 어떻게 되나요?" }
```

| 필드 | 필수 | 규칙 |
|---|---|---|
| `content` | ✅ | 빈 문자열·공백만은 거부 |
| `order_no` | — | 주면 `ORD-YYYYMMDD-NNNN` 형식이어야 하고 날짜가 실재해야 한다. 빈 문자열·공백은 **미입력으로 취급**(HTML 폼이 빈 필드를 항상 보내기 때문) |

**응답 200**

```json
{
  "inquiry_id": "58276c6f-870f-46ea-9675-467dace9f115",
  "status": "answered",
  "answer": "확정 답변 텍스트 — escalated 면 null",
  "claims": [ { "text": "답변 문장 1개", "citation_ids": ["policy:support:4-1"] } ],
  "citations": [ { "id": "policy:support:4-1", "source": "policy", "content": "조항 텍스트 또는 쿼리+결과 요약" } ],
  "attempts": [ { "verdict": "reject", "reject_reasons": ["missing_citation"] },
                { "verdict": "pass",   "reject_reasons": [] } ],
  "escalation_reason": null,
  "metrics": { "latency_ms": 4048, "tokens": { "input": 1032, "output": 96 } }
}
```

- `status` — `answered` | `escalated`
- `attempts` — 최대 2건. 초안 전 인계면 `[]`
- `escalation_reason` — `no_evidence` | `missing_order_ref` | `order_not_found` | `sql_failed`
  | `llm_call_failed` | `rejected_twice`. `answered` 면 `null`
- `reject_reasons` — `schema_violation` | `missing_citation` | `invalid_citation` | `pii_detected`
- `citations[].source` — `policy` | `sql`
- **초안 전 인계라도 그 시점까지 수집된 근거는 `citations` 에 담긴다**(감사 목적)
- `metrics.tokens` — **생성 LLM 호출의 합산이다. 임베딩 토큰은 포함하지 않는다.**
  임베딩 토큰은 처리 기록(`inquiries.embedding_tokens`)에만 별도로 남는다
- `metrics.latency_ms` — 파이프라인 처리의 벽시계 시간. 처리 기록 저장은 포함하지 않는다

**오류**

| 코드 | 조건 |
|---|---|
| 422 | `content` 누락/빈 값, `order_no` 형식 오류. **파이프라인에 진입하지 않는다** — 형식 오류는 인계가 아니다 |
| 503 | `OPENAI_API_KEY` 미설정. 설정 오류이므로 `llm_call_failed` 로 집계하지 않고 처리 기록도 남기지 않는다 |
| 500 | DB 오류. 인계 사유로 변환하지 않고 전파한다 |

## `GET /inquiries/{id}` — 처리 기록 조회

**저장된 기록에서** `POST /inquiries` 와 같은 골격을 재구성한다(메모리 캐시가 아니다).
시도 이력·근거 스냅샷·판정이 그대로 남아 있어, 기각이 일어난 문의를 나중에 다시 열어
판정 과정을 확인할 수 있다.

| 코드 | 조건 |
|---|---|
| 200 | 같은 응답 스키마 |
| 404 | 없는 id, 또는 UUID 형식이 아닌 id — `{"detail": "그런 문의가 없다"}` |

## `GET /` — 웹 폼

한 장짜리 HTML. 대시보드가 아니다. **판정 과정이 답변보다 먼저·크게** 보이도록 구성돼 있다:
최종 상태 → 시도별 `pass`/`reject` 배지와 기각 사유 코드 → 인계 사유 → 확정 답변 → 근거 목록
→ 지표 → 원본 JSON.

문의 ID 로 저장된 기록을 다시 그리는 조회 입력이 함께 있다.

새 라우트를 만들지 않고 인라인 스크립트로 `POST /inquiries` 를 호출한다.
**외부 CDN·폰트·스크립트를 불러오지 않는다** — 오프라인에서 열려야 한다.

## `GET /health`

```json
{ "status": "ok" }
```

**DB 를 확인하지 않는다.** 프로세스가 살아 있는지만 본다 — 의존성 헬스가 필요하면 확장 대상이다.

## 답변 계약 (내부 → 다음 사이클 입력)

초안 생성 LLM 의 구조화 출력이자 L1 검사의 대상이며, **의미 수준 검증(L2)이 그대로 이어받는
계약**이다. `claims` / `citation_ids` 구조와 근거 ID 체계를 편의로 바꾸지 않는다.

```json
{ "claims": [ { "text": "<답변 문장 1개>", "citation_ids": ["<근거 ID>"] } ] }
```

근거 ID 체계:

| 종류 | 형식 | 안정성 |
|---|---|---|
| 정책 조항 청크 | `policy:<문서 slug>:<조항 번호>` | 문서 기반 안정 식별자 |
| SQL 조회 결과 | `sql:<문의 ID>:<실행 순번>` | 요청별 생성, 스냅샷 영속화 |

실행 순번은 **채택된 쿼리에만** 매긴다. 안전장치에 거부되거나 실행에 실패한 쿼리, 그리고
존재성 선검사 쿼리는 근거 ID 를 받지 않는다.
