# 시스템 구성

## 전체 구성

단일 FastAPI 프로세스가 문의 1건을 **동기로** 끝까지 처리하고, 로컬 Docker Postgres 하나가
주문 데이터·정책 벡터·처리 기록을 모두 담는다. 외부 의존은 OpenAI 한 곳(생성·임베딩 공통)뿐이다.

```
웹 폼 / HTTP 클라이언트
        │  POST /inquiries
        ▼
  api.py (FastAPI)  ──────────────┐
        │                          │ GET /inquiries/{id}
        ▼                          ▼
  pipeline.py  ← 루프 제어      records.py ── 처리 기록 저장/복원
   │   │   │                          │
   │   │   └── gate.py (L1)           │
   │   └────── draft.py ──┐           │
   └────────── evidence.py ┤          │
                 │   │     │          │
        policy_index.py  llm.py       │
                 │         │          │
                 ▼         ▼          ▼
        Postgres+pgvector   OpenAI    Postgres
        (앱 계정 / RO 계정)
```

의존 방향은 위에서 아래로만 흐른다. `gate.py` 는 아무것도 부르지 않는 잎 노드이고,
`contracts.py` 는 모두가 부르지만 아무것도 부르지 않는다.

## 구성요소 지도

| 모듈 | 역할 | 의존 방향 |
|---|---|---|
| `api.py` | HTTP 표면 4개, 응답 스키마 조립, 웹 폼 서빙 | → pipeline, records, config |
| `pipeline.py` | 접수 검증 → 근거 수집 → 초안 → L1 → 종결. **재생성 상한을 강제하는 곳** | → evidence, draft, gate, contracts |
| `evidence.py` | 의도 분류(LLM) · 정책 검색 · 주문 존재성 선검사 · text-to-SQL 조율 | → llm, policy_index, sql_guard, db, order_ref |
| `sql_guard.py` | 생성된 SQL 을 실행 전에 파싱·검증. 화이트리스트·주문 범위·함수·잠금절 | → (없음) |
| `gate.py` | L1 판정. **LLM·네트워크 라이브러리를 import 하지 않는다** | → contracts |
| `draft.py` | 근거만 컨텍스트로 답변 계약 JSON 생성. 검증하지 않는다 | → llm, contracts |
| `policy_index.py` | 정책 문서 조항 단위 청킹·임베딩 적재·코사인 검색 | → llm, contracts, db |
| `records.py` | 처리 기록 4테이블 저장/복원 | → contracts, db |
| `llm.py` | OpenAI 호출 래퍼. 전송 오류 1회 재시도를 **단독 통제** | → (openai SDK) |
| `db.py` | 앱 계정/RO 계정 커넥션 분리 | → config |
| `order_ref.py` | 주문번호 형식의 **단독 소유자** | → (없음) |
| `contracts.py` | 답변 계약·근거 ID 체계·판정/인계 enum | → (없음) |
| `evaluation.py` | 측정 1(픽스처)·측정 2(골든셋) 산출, 리포트 생성 | → gate, pipeline, policy_index |
| `config.py` | 환경 변수 설정, DSN 조립 | → (없음) |

`testing.py` 는 어휘 기반 결정론 임베딩 대역이다. 외부 키 없이 벡터 검색 배관을 끝까지 돌리기
위한 것이고 실행 경로에서는 쓰지 않는다.

## 대표 흐름 — 주문 문의 1건

1. `POST /inquiries {order_no, content}` → `api.py` 가 `pipeline.accept_inquiry` 로 넘긴다.
   주문번호 형식이 틀리면 여기서 **422 로 끝난다** — 파이프라인에 들어가지 않는다.
2. 문의 ID(UUID)를 **근거 수집 전에** 만든다. SQL 근거 ID 가 문의 ID 를 품고 DB CHECK 가
   그 형태를 강제하므로 순서를 바꿀 수 없다.
3. `evidence.collect` — 의도 분류(LLM, 구조화 출력) → 분류 결과에 따라 코드가 조회를 실행한다.
   - 정책: 문의 임베딩 → pgvector 코사인 검색 → 임계값 미달 폐기
   - 주문: 고정 파라미터 쿼리로 존재 확인 → 통과분만 text-to-SQL(LLM 이 문자열 생성) →
     `sql_guard.validate_sql` 통과분만 **RO 커넥션**에서 실행
4. 근거 0건이거나 구조적 실패면 **초안 생성에 진입하지 않고** 인계 사유를 들고 돌아온다.
5. `draft.generate` — 근거만 컨텍스트로 `{claims:[{text, citation_ids}]}` 생성.
6. `gate.evaluate_draft` — 사유 4종을 **전부 수집**해 pass/reject.
7. reject 면 사유 전체를 피드백으로 붙여 **같은 근거로** 1회 재생성 → 재기각이면 인계.
8. `records.save_inquiry` — 문의·시도·근거 스냅샷·SQL 실패를 4테이블에 남긴다.
9. 응답 조립. `GET /inquiries/{id}` 는 메모리가 아니라 **저장된 기록에서** 같은 골격을 재구성한다.

## 경계에서 무엇이 오가는가

- **LLM ↔ 시스템**: LLM 이 만드는 것은 (a) 의도 분류 라벨 3지선다, (b) SQL **문자열**,
  (c) 답변 초안 JSON — 셋뿐이다. 조회 실행, 판정, 루프 종료는 넘어가지 않는다.
- **애플리케이션 ↔ DB**: 두 계정으로 갈라진다. text-to-SQL 이 만든 쿼리는 `orders` SELECT
  권한만 가진 계정으로만 나간다. 처리 기록·정책 청크는 그 계정에서 아예 보이지 않는다.
- **저장 경계**: 처리 기록은 API 응답을 재구성할 수 있을 만큼 완결적이다 — 시도별 판정,
  근거 스냅샷(쿼리문 + 결과 행 전체), 지연, 생성 토큰, 임베딩 토큰이 모두 남는다.

## 외부 의존

- **OpenAI** — 생성(`gpt-5.6-terra`)과 임베딩(`text-embedding-3-small`). 키 1개.
  키가 없으면 `POST /inquiries` 는 503 으로 끝나고 처리 기록을 남기지 않는다.
- **Postgres 17 + pgvector** — `docker-compose.yml` 이 컨테이너 이름 `reply-gate-postgres`,
  포트 5433 으로 고정한다. 이름을 고정한 이유는 컨테이너가 새어도 찾아서 정리할 수 있게 하기 위함이다.
