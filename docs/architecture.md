# 시스템 구성

## 전체 구성

단일 FastAPI 프로세스가 문의 1건을 **동기로** 끝까지 처리하고, 로컬 Docker Postgres 하나가
주문 데이터·정책 벡터·처리 기록을 모두 담는다. 외부 의존은 **LLM provider 두 곳**이다 —
OpenAI(생성·임베딩 공통)와 Anthropic(L2 판정 전용). 계열을 가른 것은 편의가 아니라
self-judging bias 를 구조로 막기 위해서다(`tracking/decisions/0004`).

```
웹 폼 / HTTP 클라이언트
        │  POST /inquiries          │  GET /inquiries/{id}
        ▼                           ▼
  ┌──────────────────── api.py (FastAPI) ────────────────────┐
  │  커넥션 2개(앱 계정 · RO 계정)를 열어 아래로 넘긴다        │
  └───────┬──────────────────────────────────┬───────────────┘
          ▼                                  ▼
    pipeline.py ── 루프 제어             records.py ── 처리 기록 저장/복원
      │  │   │   │                            │
      │  │   │   └── gate.py (L1) ── 코드 검사 │ (pipeline·evidence 의 자료형을 읽는다)
      │  │   └────── judge.py (L2) ─┐          │
      │  └────────── draft.py ──────┤          │
      └───────────── evidence.py ───┤          │
                     │    │          │         │
            policy_index.py        llm.py      │
                     │           ┌───┴───┐     │
                     ▼           ▼       ▼     ▼
             Postgres+pgvector  OpenAI Anthropic Postgres
```

**DB 커넥션은 위에서 아래로 인자로 전달된다.** `evidence.py`·`policy_index.py`·`records.py` 는
`db.py` 를 부르지 않고 이미 열린 커넥션을 받는다 — 그래서 테스트가 롤백 커넥션을 끼워 넣을 수
있고, text-to-SQL 이 실수로 앱 계정 커넥션을 집을 수 없다. 커넥션을 여는 곳은 `api.py` 와
실행 스크립트뿐이다.

`gate.py` 는 `contracts.py` 만 부르는 잎 노드이고, `contracts.py`·`llm.py`·`order_ref.py`·
`config.py` 는 이 패키지의 어떤 모듈도 부르지 않는다. `judge.py` 는 `contracts.py`·`llm.py`
만 부르고 **`gate.py` 를 부르지 않는다** — 두 층은 서로를 모르고, 층을 결합하는 것은
`pipeline.py` 뿐이다.

## 구성요소 지도

"의존" 열은 이 패키지 안에서 실제로 import 하는 모듈이다(외부 라이브러리 제외).

| 모듈 | 역할 | 의존 |
|---|---|---|
| `api.py` | HTTP 표면 4개, 응답 스키마 조립, 웹 폼 서빙, **커넥션 2개 개방** | config, contracts, db, llm, pipeline, records |
| `pipeline.py` | 접수 검증 → 근거 수집 → 초안 → L1 → L2 → 종결. **재생성 상한과 층 결합을 강제하는 곳** | config, contracts, draft, evidence, gate, judge, llm, order_ref |
| `evidence.py` | 의도 분류(LLM) · 질의 재작성 조율 · 정책 검색 · **근거 채택(기권 게이트 → 절대 하한)** · 주문 존재성 선검사 · text-to-SQL 조율 | config, contracts, gate, llm, order_ref, policy_index, query_rewrite, retrieval_strategies, sql_guard |
| `sql_guard.py` | 생성된 SQL 을 실행 전에 파싱·검증. 화이트리스트·주문 범위·함수·잠금절 | order_ref |
| `gate.py` | L1 판정. **LLM·네트워크 라이브러리를 import 하지 않는다** | contracts |
| `judge.py` | L2 판정(LLM). 초안 1개 + 근거를 **시도당 1회 배치 판정**. 제어·저장은 하지 않는다 | contracts, llm |
| `draft.py` | 근거만 컨텍스트로 답변 계약 JSON 생성. 검증하지 않는다 | contracts, llm |
| `policy_index.py` | 정책 문서 조항 단위 청킹·임베딩 적재·코사인 검색 | contracts, llm |
| `records.py` | 처리 기록 4테이블 저장/복원(층별 판정·판정 토큰 포함) | contracts, evidence, pipeline |
| `llm.py` | OpenAI 호출 래퍼 + Anthropic 판정 클라이언트. 전송 오류 1회 재시도를 **단독 통제** | (없음) |
| `db.py` | 앱 계정/RO 계정 커넥션 분리 | config |
| `order_ref.py` | 주문번호 형식의 **단독 소유자** | (없음) |
| `contracts.py` | 답변 계약·근거 ID 체계·판정/인계 enum | (없음) |
| `query_rewrite.py` | 검색용 질의 재작성 — 별도 LLM 호출, 산출은 검색 **입력**일 뿐이다 | llm |
| `retrieval_strategies.py` | 순수 검색 전략 연산 — 재작성 합집합·BM25·RRF·리랭크, **기권 게이트 판정 연산** | llm |
| `retrieval_labels.py` | 검색 정답 라벨 로드·교차 검증 | evaluation, policy_index |
| `retrieval_eval.py` | 오프라인 검색 비교 하네스 — 전략 사다리·컷 스윕·기권 게이트 격자·청킹 격자 | adoption_axis, config, evaluation, llm, policy_index, retrieval_labels, retrieval_strategies, testing |
| `adoption_axis.py` | 채택 축 손계산 — 커밋된 검색 산출물만 읽는 오프라인 채점자 | (없음) |
| `regression_guard.py` | 회귀 판정 — 이중 기준선 대조(승격=구속 / 직전 라이브=경보), 조건 지문, 비악화 두 겹 | (없음 — 입력은 리포트 JSON 그대로다) |
| `evaluation.py` | 측정 1(픽스처)·측정 2(골든셋)·측정 3(판정 픽스처) 산출, 리포트 생성 | contracts, gate, judge, llm, pipeline, regression_guard |
| `testing.py` | 결정론 대역 — 어휘 임베딩·판정(`StubJudge`)·대역 파이프라인 조립기 | config, contracts, draft, evidence, judge, llm, pipeline |
| `config.py` | 환경 변수 설정, DSN 조립, **채택 축 구성(컷·`top_k`·기권 게이트)** 소유 | (없음) |

`records.py` 가 `pipeline`·`evidence` 를 부르는 것은 **자료형 때문**이다 — 저장 대상인
`ProcessedInquiry`·근거 스냅샷의 정의가 거기 있다. 반대 방향(파이프라인이 저장을 부르는 것)은
없다: 저장 시점을 정하는 것은 `api.py` 다.

`testing.py` 는 결정론 대역 모음이다 — 어휘 기반 임베딩 대역과 판정 대역(`StubJudge`), 그리고
둘을 끼운 대역 파이프라인 조립기. 외부 키 없이 벡터 검색·L2 배관을 끝까지 돌리기 위한 것이고
실행 경로에서는 쓰지 않는다. **대역으로 낸 판정 수치는 판정 모델의 정확도가 아니다.**

## 대표 흐름 — 주문 문의 1건

1. `POST /inquiries {order_no, content}` → `api.py` 가 `pipeline.accept_inquiry` 로 넘긴다.
   주문번호 형식이 틀리면 여기서 **422 로 끝난다** — 파이프라인에 들어가지 않는다.
2. 문의 ID(UUID)를 **근거 수집 전에** 만든다. SQL 근거 ID 가 문의 ID 를 품고 DB CHECK 가
   그 형태를 강제하므로 순서를 바꿀 수 없다.
3. `evidence.collect` — 의도 분류(LLM, 구조화 출력) → 분류 결과에 따라 코드가 조회를 실행한다.
   - 정책: **질의 재작성(LLM)** → 원문·재작성 두 질의 임베딩 → pgvector 코사인 검색 ×2 →
     **합집합(같은 조항은 더 큰 유사도)** → 상위 top_k → **기권 게이트(질의 축)** →
     임계값 미달 폐기(항목 축).
     재작성을 얻지 못하면 **원문 질의로 폴백**한다(인계가 아니다). 벡터 인덱스는 두지 않는다 —
     근사 인덱스가 `LIMIT k` 를 조용히 잘라먹는다(`docs/engineering-notes.md`)
   - 주문: 고정 파라미터 쿼리로 존재 확인 → 통과분만 text-to-SQL(LLM 이 문자열 생성) →
     `sql_guard.validate_sql` 통과분만 **RO 커넥션**에서 실행
4. 근거 0건이거나 구조적 실패면 **초안 생성에 진입하지 않고** 인계 사유를 들고 돌아온다.
   기권 게이트가 발동한 질의도 여기로 온다(`no_evidence`).
5. `draft.generate` — 근거만 컨텍스트로 `{claims:[{text, citation_ids}]}` 생성.
6. **게이트 2층.** 먼저 `gate.evaluate_draft`(L1) — LLM 호출 0회로 사유 4종을 **전부 수집**해
   pass/reject. **L1 을 통과한 초안만** `judge.judge`(L2)로 넘어가 claim 단위 뒷받침과
   근거쌍 모순을 **1회 배치 판정**하고, 사유 2종을 낸다. 두 층을 결합해 **종합 판정**을
   만든다(종합 pass ⟺ L1 pass 이고 L2 가 실행됐다면 L2 도 pass).
   - L2 스위치가 꺼져 있으면 이 단계에서 L1 만 돈다.
   - **L2 호출이 실패하면**(전송 오류·형식 불일치 소진) 검증하지 못한 답변을 내보내지 않고
     `llm_call_failed`(실패 단계 `l2_judge`)로 인계한다 — 7단계 재생성으로 가지 않는다.
7. 종합 reject 면 사유 전체를(L2 기각이면 claim 단위 상세까지) 피드백으로 붙여 **같은 근거로**
   1회 재생성 → 6단계를 다시 거치고, 재기각이면 인계(`rejected_twice`, **층 무관**).
8. `records.save_inquiry` — 문의·시도·근거 스냅샷·SQL 실패를 4테이블에 남긴다.
9. 응답 조립. `GET /inquiries/{id}` 는 메모리가 아니라 **저장된 기록에서** 같은 골격을 재구성한다.

## 근거 채택은 축이 둘이다 — 질의 축(기권 게이트) → 항목 축(절대 하한)

3단계의 정책 검색과 5단계의 초안 생성 **사이**에 축이 하나 더 있다. 결정론이고 LLM 호출이
없다([결정 0014](tracking/decisions/0014-기권-게이트를-1위-top_k위-산포-축으로-채택한다.md)).

| 순서 | 축 | 무엇을 보나 | 결과 |
|---|---|---|---|
| ① | **질의 축 — 기권 게이트** | 상위 `top_k` 코사인의 **1위 − `top_k`위 산포**가 τ(=0.06) 미만인가 | 미만이면 그 질의의 채택 집합이 **통째로** 빈다 → 4단계 `no_evidence` |
| ② | **항목 축 — 절대 하한** | 각 후보의 코사인 ≥ 컷(=0.30) | 게이트가 발동하지 않았을 때만 돈다 |

세 가지가 이 순서의 계약이다.

- **통계량 입력은 컷 전 슬라이스다.** `policy_index.search_policy_chunks` 의 SQL `LIMIT top_k`
  가 자른 뒤 파이썬이 임계값을 거는 순서를 그대로 따른다 — 오프라인 격자와 라이브가 같은 수를
  내려면 이 절단이 같은 자리에 있어야 한다. **조정 가능한 인자가 아니라 행동 계약이다.**
  컷 **뒤** 슬라이스로 재면 컷 위 후보가 둘뿐인 케이스(G04)의 산포가 τ 아래로 떨어져 정답
  조항을 잃는다.
- **게이트는 항목을 고르지 않는다.** 발동하면 전부 버리고, 발동하지 않으면 아무것도 건드리지
  않는다. 그래서 항목 간 비교는 여전히 코사인 절대 축 하나뿐이고
  ([결정 0009](tracking/decisions/0009-채택-판정을-절대-축으로-통일한다.md)), 상대 축이
  항목 축으로 새지 않는다.
- **통계량이 미정의면(측정된 후보 2건 미만) 게이트는 열린 채로 남는다.** 미정의를 0 으로
  채우면 모든 양수 τ 에서 기권이 되어, 후보가 하나뿐인 질의가 근거 없이 인계된다.

소유는 셋으로 갈린다: 게이트 구성(통계량·τ)은 `config.py`, 순수 판정 연산은
`retrieval_strategies.py`, 순서를 거는 곳은 `evidence.adopt_policy_hits` 하나다.
**τ 는 임베딩 모델에 묶인 조건 종속 인자다** — 모델을 바꾸면 따라오지 않는다
(`docs/engineering-notes.md` "τ 는 임베딩 모델에 묶인다").

## 경계에서 무엇이 오가는가

- **LLM ↔ 시스템**: LLM 이 만드는 것은 다섯이고, **각각이 무엇으로 끝나는지는 코드가 정한다.**
  - (a) 의도 분류 라벨 3지선다 — 분기를 고르는 것은 코드다.
  - (b) SQL **문자열** — 실행 여부는 `sql_guard` 가, 실행 계정은 `db` 가 정한다.
  - (c) 답변 초안 JSON — 그 초안이 나가는지는 게이트 두 층이 정한다.
  - (d) **L2 판정 JSON** — 판정 산출을 재생성·인계·종결로 옮기는 것도, 해석되지 않는
    산출을 거부하는 것도(fail-closed) 코드다.
  - (e) **검색용 재작성 질의** — 검색 **입력**일 뿐이다. 무엇이 근거가 되는지는
    **`top_k` 선절단 → 기권 게이트(질의 축) → 임계값(항목 축)** 이 순서대로 자르고
    (셋 다 결정론이고 LLM 호출이 없다 — 아래 "근거 채택은 축이 둘이다"), 재작성을 얻지
    못하면 코드가 원문으로 폴백한다. 재작성이 주제를 옮겨도 원문 질의가 함께 검색되므로
    근거 집합이 재작성에 종속되지 않는다.

  조회 실행·**판정의 사용**·**근거 채택**·루프 종료는 넘어가지 않는다.
  **L1 판정은 여전히 LLM 을 거치지 않는다.**
- **애플리케이션 ↔ DB**: 두 계정으로 갈라진다. text-to-SQL 이 만든 쿼리는 `orders` SELECT
  권한만 가진 계정으로만 나간다. 처리 기록·정책 청크는 그 계정에서 아예 보이지 않는다.
- **저장 경계**: 처리 기록은 API 응답을 재구성할 수 있을 만큼 완결적이다 — 시도별 종합 판정과
  층별 판정(L2 의 claim 단위 판정·근거쌍 모순 포함), 근거 스냅샷(쿼리문 + 결과 행 전체),
  지연, 생성 토큰, 판정 토큰, 임베딩 토큰, **검색 토큰과 검색 단계 폴백 사유**가 모두 남는다.
  **토큰은 네 계열로 갈라 저장한다** — provider 와 단가가 다르고, 검색 단계에서 끝난 문의가
  초안 생성 토큰을 쓴 것처럼 찍히면 리포트가 거짓말을 한다(`contracts.md` "토큰 집계 경계").

## 외부 의존

- **OpenAI** — 생성(`gpt-5.6-terra`)과 임베딩(`text-embedding-3-small`). 키는
  `OPENAI_API_KEY` 하나로 둘 다 쓴다. 키가 없으면 `POST /inquiries` 는 503 으로 끝나고
  처리 기록을 남기지 않는다.
- **Anthropic** — L2 판정(`claude-sonnet-5`). 키는 `ANTHROPIC_API_KEY`.
  **L2 가 켜져 있는데 이 키가 없으면 `POST /inquiries` 는 503** 이고, 선검사가 처리 진입
  시점에 돌아 생성 토큰을 태우기 전에 끝난다. 스위치(`L2_ENABLED=false`)로 끄면 이 의존은
  사라진다 — 그때는 L1 단독 동작이다. 모델 등급·effort·출력 상한은 설정이 정하고
  코드에 하드코딩하지 않는다.
- **Postgres 17 + pgvector** — `docker-compose.yml` 이 컨테이너 이름 `reply-gate-postgres`,
  포트 5433 으로 고정한다. 이름을 고정한 이유는 컨테이너가 새어도 찾아서 정리할 수 있게 하기 위함이다.

두 provider 를 가른 이유는 비용도 가용성도 아니다: **생성과 같은 계열로 판정하면
self-judging bias 로 검출률이 오염된다**(`tracking/decisions/0004`·`0007`).
