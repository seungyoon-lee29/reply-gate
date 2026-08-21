# 외부 인터페이스 계약

HTTP 엔드포인트 4개가 외부 표면의 전부다. 인증은 없다.

## 공통 규약

- 요청·응답 모두 `application/json` (웹 폼 `GET /` 만 `text/html`).
- **응답 스키마의 모든 키는 항상 존재한다.** 해당 없는 값은 `null`(스칼라) 또는 `[]`(배열)이며,
  키가 빠지지 않는다. 소비자는 키 존재 여부로 분기하지 않고 값으로 분기한다.
- 오류 본문은 FastAPI 기본 형식 `{"detail": "<한국어 설명>"}` 이다.
- 문의 ID 는 소문자 하이픈 UUID 문자열이다.

## `POST /inquiries` — 문의 접수 + 동기 처리

요청 전체가 처리될 때까지 응답하지 않는다. `latency_ms` 가 판정 호출까지 포함하므로
**실측 지연은 L2 켜짐/꺼짐 조건을 반드시 병기해서 읽어야 한다**(골든셋 30건 × 3회 평균):

| 조건 | 중앙값(p50) | 95백분위(p95) | 근거 |
|---|---|---|---|
| **L2 켜짐** (기본값) | 약 **11.9초** | 약 **35.1초** | `reports/evaluation-live-l2-{1,2,3}.json` |
| L2 꺼짐 (기준선) | 약 4.1초 | 약 8.1초 | `reports/evaluation-live-{1,2,3}.json` |

차이는 회귀가 아니라 **층이 하나 더 붙은 값**이다 — 판정 호출이 그 안에 들어 있다.

**요청**

```json
{ "order_no": "ORD-20260202-0001", "content": "제 주문 배송 상태가 어떻게 되나요?" }
```

| 필드 | 필수 | 규칙 |
|---|---|---|
| `content` | ✅ | 빈 문자열·공백만은 거부 |
| `order_no` | — | 주면 `ORD-YYYYMMDD-NNNN` 형식이어야 하고 날짜가 실재해야 한다. 빈 문자열·공백은 **미입력으로 취급**(HTML 폼이 빈 필드를 항상 보내기 때문) |

**응답 200** — 아래는 **L1 기각 → 재생성 → L2 기각**으로 인계된 장면이다(`answered` 면
`answer`/`claims` 가 채워지고 `escalation_reason` 이 `null` 이다).

```json
{
  "inquiry_id": "58276c6f-870f-46ea-9675-467dace9f115",
  "status": "escalated",
  "answer": null,
  "claims": [],
  "citations": [ { "id": "policy:support:4-1", "source": "policy", "content": "조항 텍스트 또는 쿼리+결과 요약" },
                 { "id": "policy:shipping:1-3", "source": "policy", "content": "조항 텍스트" },
                 { "id": "policy:shipping:1-4", "source": "policy", "content": "조항 텍스트" } ],
  "attempts": [
    { "verdict": "reject", "reject_reasons": ["missing_citation"],
      "l1": { "verdict": "reject", "reject_reasons": ["missing_citation"] },
      "l2": null },
    { "verdict": "reject", "reject_reasons": ["unsupported_claim"],
      "l1": { "verdict": "pass", "reject_reasons": [] },
      "l2": { "verdict": "reject",
              "reject_reasons": ["unsupported_claim"],
              "claim_judgments": [ { "claim_text": "답변 문장 1개", "verdict": "reject",
                                     "explanation": "인용 조항이 이 주제를 다루지 않는다" } ],
              "contradictions": [ { "evidence_id_a": "policy:shipping:1-3",
                                    "evidence_id_b": "policy:shipping:1-4",
                                    "explanation": "같은 사안에 다른 기준을 말한다" } ] } }
  ],
  "escalation_reason": "rejected_twice",
  "metrics": { "latency_ms": 8123,
               "tokens": { "input": 1032, "output": 96,
                           "judge_input": 2104, "judge_output": 188 } }
}
```

- `status` — `answered` | `escalated`
- `answer` / `claims` — `answered` 면 확정 답변 텍스트와 claim 배열
  (`{text, citation_ids}` — 아래 "답변 계약" 절과 같은 모양)이 실리고, `escalated` 면
  각각 `null` 과 `[]` 이다. **이 셋은 함께 움직인다** — `answered` ⟺ `answer != null` ⟺
  `escalation_reason == null` 이고, DB 도 같은 불변식을 CHECK 로 건다
  (`inquiries_terminal_shape`)
- `attempts` — 최대 2건. 초안 전 인계면 `[]`
- `escalation_reason` — `no_evidence` | `missing_order_ref` | `order_not_found` | `sql_failed`
  | `llm_call_failed` | `rejected_twice`. `answered` 면 `null`
- `reject_reasons` — L1 4종 `schema_violation` | `missing_citation` | `invalid_citation`
  | `pii_detected` + L2 2종 `unsupported_claim` | `contradictory_evidence`.
  최상위 `attempts[].reject_reasons` 는 **종합**(두 층 사유의 합집합)이고 순서는 고정이다
  (L1 4종 먼저, L2 2종 뒤 — `contracts.COMBINED_REASON_ORDER`)
- `attempts[].l1` / `attempts[].l2` — **층별 내역**. 종합(`verdict`/`reject_reasons`)과 함께
  실린다. 자세한 규칙은 아래 "층별 판정 키" 절
- `citations[].source` — `policy` | `sql`
- **초안 전 인계라도 그 시점까지 수집된 근거는 `citations` 에 담긴다**(감사 목적)
- `metrics.tokens` — 계열별로 분리한다. 규칙은 아래 "토큰 집계 경계" 절
- `metrics.latency_ms` — 파이프라인 처리의 벽시계 시간. **L2 판정 호출과 검색 단계 재작성
  호출을 포함**하고 처리 기록 저장은 포함하지 않는다
- `metrics.retrieval_fallback_reason` — 검색 단계가 폴백한 사유. 기본은 `null` 이고
  **인계 사유가 아니다**: 재작성을 얻지 못해 원문 질의로 검색했다는 뜻이며, 그 문의도
  그대로 답변될 수 있다. 값이 있는데 `escalation_reason` 이 `null` 인 조합이 정상이다.
  키는 미실행이어도 사라지지 않는다(공통 규약)

### 층별 판정 키 — `attempts[].l1` / `attempts[].l2`

`verdict`/`reject_reasons` 는 기존 키이고 의미도 그대로 **종합**이다: 종합 `pass` ⟺ L1 이
pass 이고 **L2 가 실행됐다면** L2 도 pass. 층별 내역은 두 키로 따로 실린다.

| 키 | 모양 | `null` 이 되는 때 |
|---|---|---|
| `l1` | `{verdict, reject_reasons}` | 층별 컬럼이 없던 시절의 처리 기록을 복원한 경우뿐이다 — 현재 파이프라인이 만드는 시도에는 항상 있다 |
| `l2` | `{verdict, reject_reasons, claim_judgments, contradictions}` | **셋** — ① L1 이 기각(L2 는 L1 통과분에만 돈다) ② L2 스위치 꺼짐 ③ L2 판정 호출 실패 |

- **키는 사라지지 않는다.** 미실행은 키 부재가 아니라 `null` 이다(공통 규약).
- **`l2: null` 은 "통과"가 아니라 "판정이 없었다"** 이다. 특히 ③ 판정 호출 실패 시도는
  층 결합 정의상 **종합 `verdict` 가 `pass` 인데 문의는 인계된다** — 그 시도의 진실은
  `escalation_reason: "llm_call_failed"` 가 들고 있다. 종합 verdict 만 보고 통과로 읽으면 안 된다.
  **이 규칙은 소비자에게 맡기지 않고 코드가 지킨다** — 평가 하네스는 `escalation_reason` +
  `failed_stage` 로 이 상태를 판별해(`GoldenOutcome.gate_never_ran`) 게이트 품질 지표의
  분모에서 빼고 `bait_unmeasured` 로 따로 센다. 문서로만 두면 "verdict 가 pass 니까 통과"로
  집계하는 소비자가 언제든 다시 생긴다.
- `l2.claim_judgments` 는 **그 시도 초안의** claim **전부**(통과한 claim 포함)와 **1:1 로
  대응한다**. 각 항목은 `{claim_text, verdict, explanation}` 이고, `claim_text` 는 답변
  계약의 claim 에 별도 ID 가 없어 text 로 가리키는 참조다.
  **짝짓기는 `claim_text` 로 한다 — 배열 위치는 계약이 아니다.** fail-closed 검증기가
  강제하는 것은 초안 claim 과의 **개수까지 포함한** 완전 대응이고(`judge._parse_claim_judgments`),
  배열 순서는 프롬프트가 요청할 뿐 거부 사유가 아니다. 위치로 짝지으면 "어느 문장이 왜
  기각됐는지"가 다른 claim 에 붙을 수 있다.
  **개수까지 보는 이유**: 초안은 LLM 산출이라 claim 의 text 가 유일하다는 보장이 없고, 같은
  문장이 두 번 들어오면 두 claim 의 `citation_ids` 는 다를 수 있다. 집합으로만 대조하면
  판정 1건이 claim 2건을 덮어도 통과하고(판정받지 못한 문장이 답변에 실린다), 반대로 판정을
  정직하게 2건 낸 옳은 산출이 "중복"으로 거부된다. 그래서 **같은 문장이 N번이면 판정도 N번**
  이고, 프롬프트가 이 규칙을 명시한다.
- 인계된 문의는 최상위 `claims` 가 `[]` 여도 이 배열은 **기각된 초안의** claim 을 담는다 —
  두 배열을 서로 짝지으면 안 된다.
- `l2.contradictions` 는 **근거쌍 단위** 기록 `{evidence_id_a, evidence_id_b, explanation}` 이다.
  **여기 나오는 ID 는 반드시 같은 응답의 `citations` 안에 있다** — `citations` 는 인용된 근거가
  아니라 **수집 근거 전체**이고, fail-closed 파싱이 모순 쌍의 ID 를 그 집합으로 검증한다
  (`judge._parse_contradictions`). 초안이 인용하지 않은 근거의 모순도 잡히므로, 그 ID 가
  `claims[].citation_ids` 에는 없을 수 있다 — `citations` 밖일 수는 없다.
  **기록됐다고 곧 기각은 아니다** — 초안이 모순을 명시하고 두 기준을 모두 안내했으면
  `reject_reasons` 에 오르지 않고 기록만 남는다.
- `claim_judgments`/`contradictions` 가 **빈 배열**인 것과 `l2` 자체가 `null` 인 것은 다른
  상태다. 전자는 "판정했고 해당 없음", 후자는 "판정이 없었다"이다.

### 토큰 집계 경계

토큰은 **계열별로 분리**한다 — provider 와 단가가 달라 섞으면 건당 비용 지표가 무너진다.

| 키 | 무엇을 세나 | 경계 |
|---|---|---|
| `metrics.tokens.input` / `.output` | **생성 LLM 합산** — 의도 해석 + SQL 생성 + 초안 생성 | 임베딩·판정·검색 토큰을 여기 섞지 않는다 |
| `metrics.tokens.judge_input` / `.judge_output` | **L2 판정 모델 토큰** | 생성 합산과 별도 키 쌍이다. **판정 모델을 부르지 않았으면 0** — 불러서 실패한 경우는 0 이 아닐 수 있다(아래 "실행됐으나 실패한 호출") |
| `metrics.tokens.retrieval_input` / `.retrieval_output` | **검색 단계 생성 호출 토큰** — 질의 재작성 | 생성 합산과 별도 키 쌍이다. 재작성을 쓰지 않은 문의는 0 |
| 임베딩 토큰 | 문의 임베딩·정책 인덱싱 | **응답에 싣지 않는다.** 처리 기록(`inquiries.embedding_tokens`)에만 남는다 |
| 캐시 계열 write/read — **세 쌍** (생성·검색·판정) | 각 계열 호출의 캐시 기록분·적중분 | **응답에도 처리 기록에도 싣지 않는다.** 평가 리포트에만 있다. 계열마다 한 쌍이고 **계열 수를 늘리지 않는다** |

- **검색 계열을 생성 합산에서 가른 이유는 특히 뾰족하다.** 무근거 문의가 검색 단계에서
  걸러져 **초안 없이** 끝나는 것이 이 제품의 장면인데, 재작성 토큰이 생성 칸에 들어가면
  초안을 만들지도 않은 문의가 초안 생성 토큰을 쓴 것으로 찍힌다 — 리포트가 거짓말을 한다.
  **생성 토큰의 정의는 바뀌지 않았다**(의도 해석 + SQL 생성 + 초안 생성 합산). 의도 해석은
  검색보다 먼저·무조건 돌므로 생성 토큰은 원래 0 이 되지 않는다.

- **판정 프롬프트 캐싱이 켜진 조건에서 `judge_input` 이 담는 값은 "비캐시 입력"이다.**
  Anthropic 의 `usage.input_tokens` 가 캐시 적중분을 **제외한** 값이라, 캐싱을 켜면 이 키가
  총 프롬프트 토큰이 아니라 정가로 과금된 부분만 센다(실측: 회당 30,042 → 6,832, 나머지
  23,210 은 캐시 read 로 옮겨갔다 — [결정 0019](tracking/decisions/0019-판정-프롬프트-캐싱을-실측하고-채택하지-않는다.md)).
  **캐시 계열(write/read)은 API 응답이 아니라 평가 리포트에서 본다**
  (`tokens.cache_creation_total`·`tokens.cache_read_total`). **키·스키마·정의는 바뀌지 않았다** —
  이 문단은 서술이고 응답 계약의 확장이 아니다. 스위치(`JUDGE_PROMPT_CACHING_ENABLED`)의
  기본값은 **꺼짐**이므로 기본 실행에서 이 키는 총 프롬프트 토큰과 같다.
- **생성 계열과 검색 계열에도 같은 캐시 쌍이 있고, 경계도 같다** — 응답이 아니라 평가
  리포트에서만 본다(`tokens.generation_cache_creation_total`·`generation_cache_read_total` ·
  `retrieval_cache_creation_total`·`retrieval_cache_read_total`). **응답 계약은 확장되지
  않는다** — `metrics.tokens` 에 새 키가 생기지 않았고 계열 수도 그대로 넷이다. 늘어난 것은
  **리포트 안에서** 계열마다의 칸이다.
  **다만 판정 계열과 포함 관계가 반대다.** OpenAI 의 두 값은 `usage.input_tokens` 의
  **내역**이라 `metrics.tokens.input`·`retrieval_input` 안에 이미 들어 있고, Anthropic 의
  캐시 계열은 `judge_input` **밖**이다. 그래서 생성·검색 쪽 캐시 칸은 입력 칸에서 빼지도
  더하지도 않는다 — 빼면 옛 산출물과 정의가 갈려 대조가 끊기고, 더하면 같은 토큰을 두 번
  센다(환산식은 [단가 문서](tracking/pricing.md) 1절).
  **없는 값은 0 이 아니라 미측정(`None`)이다** — 판정 계열이 이미 쓰는 규칙과 같다.
- **실행됐으나 실패한 호출의 토큰도 그대로 집계한다.** 전송 오류로 죽기 전에 200 으로
  돌아온 호출, 안전 분류기 거절(HTTP 200), 형식 불일치로 버려진 산출 — 전부 실비용이므로
  0 으로 접지 않는다. 판정 호출이 실패해 `l2` 가 `null` 인 시도에도 `judge_input`/
  `judge_output` 은 0 이 아닐 수 있다.
- `judge_*` 가 0 인 것은 "판정이 공짜였다"가 아니라 **판정을 부르지 않았다**는 뜻이다.

**오류**

| 코드 | 조건 |
|---|---|
| 422 | `content` 누락/빈 값, `order_no` 형식 오류. **파이프라인에 진입하지 않는다** — 형식 오류는 인계가 아니다 |
| 503 | `OPENAI_API_KEY` 미설정, 또는 **L2 가 켜져 있는데 `ANTHROPIC_API_KEY` 미설정**. 설정 오류이므로 `llm_call_failed` 로 집계하지 않고 처리 기록도 남기지 않는다 |
| 503 | **정책 인덱스의 임베딩 출처가 질의와 다르다** — 모델을 바꾸고 재색인하지 않은 상태. 같은 계열의 설정 오류이며 `no_evidence` 인계로 바꾸지 않는다(그러면 낡은 인덱스가 검색 품질 지표로 위장한다). 처리 기록은 커넥션 롤백으로 남지 않는다 |
| 500 | DB 오류. 인계 사유로 변환하지 않고 전파한다 |

**422·503 은 DB 상태와 무관하다.** 요청 오류와 설정 오류는 DB 가 떠 있지 않아도 각자의
코드로 끝난다 — 커넥션은 그 선검사를 통과한 뒤에야 열린다. DB 가 **진짜** 필요한
경로(조회·처리)만 500 이다.

판정 키 선검사는 **`POST /inquiries` 경로 전용**이다. 처리에 진입하자마자 보므로 생성
토큰을 태우기 전에 503 이 난다 — 검증하지 못할 답변을 만들기 시작하지 않는 것이
fail-closed 다. `GET` 라우트와 접수 거부 422 는 판정 키가 없어도 그대로 산다(선검사가
`Depends` 가 아니라 POST 핸들러 안에 있다).

## `GET /inquiries/{id}` — 처리 기록 조회

**저장된 기록에서** `POST /inquiries` 와 같은 골격을 재구성한다(메모리 캐시가 아니다).
시도 이력·근거 스냅샷·판정이 그대로 남아 있어, 기각이 일어난 문의를 나중에 다시 열어
판정 과정을 확인할 수 있다.

| 코드 | 조건 |
|---|---|
| 200 | 같은 응답 스키마 |
| 404 | 없는 id, 또는 UUID 형식이 아닌 id — `{"detail": "그런 문의가 없다"}` |

## `GET /` — 웹 폼

한 장짜리 HTML. 대시보드가 아니다. **판정 과정이 답변보다 먼저·크게** 보이도록 구성돼 있고,
**어느 층이** 무엇을 왜 기각했는지가 화면의 주인공이다. 표시 순서:

최종 상태 → 시도별 **종합** `pass`/`reject` 배지 + **층별 배지**(`L1 pass` / `L2 reject` /
`L2 미실행`) → 기각 사유 코드 → **L2 의 claim 단위 판정**(claim 문장별 pass/reject + 설명)
→ **근거쌍 모순**(모순 쌍 ID + 설명, 기각이 아니어도 기록되면 표시) → 인계 사유 →
확정 답변 → 근거 목록 → 지표(지연은 L2 포함, 토큰은 생성·판정 분리 표기) → 원본 JSON.

`L2 미실행` 배지에는 이유가 함께 붙는다(L1 기각 · 판정 층 꺼짐 · 판정 호출 실패).
**응답에 실패 단계 필드가 없어 화면이 관측값으로 추론한다** — 한계는
`docs/tracking/findings.md` 에 적어 두었다.

문의 ID 로 저장된 기록을 다시 그리는 조회 입력이 함께 있다.

새 라우트를 만들지 않고 인라인 스크립트로 `POST /inquiries` 를 호출한다.
**외부 CDN·폰트·스크립트를 불러오지 않는다** — 오프라인에서 열려야 한다.

## `GET /health`

```json
{ "status": "ok" }
```

**DB 를 확인하지 않는다.** 프로세스가 살아 있는지만 본다 — 의존성 헬스가 필요하면 확장 대상이다.

## 답변 계약 (내부)

초안 생성 LLM 의 구조화 출력이자 L1 검사의 대상이며, **의미 수준 검증(L2)이 그대로 이어받는
계약**이다 — L2 는 이 `claims` 배열을 그대로 판정 단위로 쓰고, `citation_ids` 가 가리키는
근거 원문이 뒷받침 판정의 입력이 된다. `claims` / `citation_ids` 구조와 근거 ID 체계를
편의로 바꾸지 않는다.

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
