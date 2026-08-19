# Reply-Gate

근거 없는 답변을 스스로 기각하는 이커머스 CS 답변 에이전트. 초안을 잘 쓰는 것이 아니라
**틀린 초안을 걸러내는 것**이 이 제품의 존재 이유다.

단일 개발자·합성 데이터·로컬 실행 규모다. 실사용자도 실운영도 없고 목적은 시연과 설명이다.
그래서 운영급 인프라는 필요 없지만, **게이트 검증 로직과 평가 지표는 실제 제품 수준으로
촘촘해야 한다** — 신뢰성 지표가 이 프로젝트의 유일한 주장이기 때문이다. 데이터와 문의는 한국어다.

## 프로젝트 구조

```
.
├── CLAUDE.md / AGENTS.md          → 이 파일 (두 파일 내용 동일)
├── README.md                      → 외부 공개용 전시 문서
├── docs/
│   ├── problem.md                 → 문제정의: 무엇을 왜 만드는가, 성공 판정 기준, 범위
│   ├── architecture.md            → 구성요소와 데이터 흐름
│   ├── business-rules.md          → 도메인 규칙·상태 전이·판정 규칙
│   ├── security.md                → text-to-SQL 안전장치 3층, PII 정책, 비밀 관리
│   ├── standards.md               → 검증 게이트, 소유권, 하드 게이트, 재시도 상한
│   ├── engineering-notes.md       → 실제로 뚫렸던 경로와 걸렸던 함정
│   ├── operations.md              → 설치부터 실행·검증·지표 산출까지의 순서
│   ├── contracts.md               → HTTP 표면 4개와 답변 계약
│   └── tracking/
│       ├── status.md              → 지금 어디까지 왔는지 + 첫 실측값
│       ├── pricing.md             → 단가 표(기준일 명시)와 계열별 비용 분해 — 달러 수치의 정본
│       ├── decisions/             → 트레이드오프가 있었던 결정 기록
│       └── findings.md            → 지금 해결할 수 없는 문제
├── src/reply_gate/
│   └── AGENTS.md                  → 파일별 경계·불변식·테스트 지침
├── scripts/
│   └── AGENTS.md                  → 실행 진입점의 범위와 불변식
├── tests/
│   └── AGENTS.md                  → 테스트 네 종류·구조 검사 규율·음성 대조 규칙
├── db/                            → 스키마 DDL, 컨테이너 초기화, 주문 픽스처
├── data/                          → 정책 문서, 골든셋, 검색 정답 라벨·재작성 질의,
│                                     L1·L2 판정 픽스처, 승격 기준선 참조
└── reports/                       → 산출물. **라이브 실측 리포트만 추적한다** —
                                      재생성에 과금이 들어 문서가 인용하는 근거가 여기 있다
```

## 절대 깨지 않는 것

1. **L1 게이트는 LLM 을 호출하지 않는다.** `gate.py` 는 LLM·네트워크 라이브러리를 import 하지
   않고, 구조 테스트가 이를 검사한다. 게이트가 확률적이 되는 순간 검출률·오탐률을 재현 가능한
   숫자로 말할 수 없다.
2. **text-to-SQL 안전장치 3종은 생략 불가.** read-only 계정 · 스키마 화이트리스트 · 쿼리 검증.
   조회 범위는 프롬프트가 아니라 **코드가 AST 로 강제한다.**
3. **루프 종료·데이터 접근·권한·근거 채택은 코드가 통제한다.** LLM 은 의도 분류 라벨·SQL
   문자열·초안 JSON·**L2 판정 JSON**·**검색용 재작성 질의** 까지만 만든다. 전부 산출물일
   뿐이다 — 판정 JSON 을 재생성·인계·종결로 옮기는 것도, 해석되지 않는 산출을 거부하는
   것도(fail-closed) 코드다. **재작성 질의는 검색 입력일 뿐이고 무엇이 근거가 되는지는
   `top_k`·기권 게이트·임계값이 자른다** — 셋 다 결정론이고 LLM 호출이 없다. 재작성 호출이
   실패하면 코드가 원문으로 폴백한다(인계가 아니다).
4. **비밀은 어디에도 평문 금지.**
5. **평가용으로 심은 장치를 제거하지 않는다.** 정책 문서의 미끼·모호·상충 조항, 기각 유발
   골든셋 문의, L1 픽스처, **L2 판정 픽스처**(`data/judge_fixtures.jsonl` — 측정 3 의 유일한
   입력), **검색 정답 라벨**(`data/retrieval_labels.jsonl` — 검색 recall·precision 의 유일한
   정답 입력), **blind 재작성 질의 픽스처**(`data/rewritten_queries.jsonl` — 기본 비교 입력),
   **oracle upper-bound 픽스처**(`data/rewritten_queries_oracle.jsonl` — 정책·라벨 기반 상한
   대조군). 제거하면 기각 장면과 검색 품질 측정이 재현되지 않는다.
6. **승격은 사람만 한다.** 회귀 판정의 구속 기준선인 `data/promoted_baseline.json` 을 사람이
   바꾸는 것이 승격이고, 판정 층(프롬프트·픽스처·캐싱)이 바뀐 뒤의 **재등재**도 같은 자격이다.
   저장소 어디에도 그 파일에 **쓰는** 경로가 없고 구조 테스트가 그것을 검사한다 — 코드가
   구속 기준선을 갱신하면 판정이 자기추인이 된다.
7. **커밋된 라이브 리포트는 사후 편집하지 않는다.** 재생성에 과금이 들고 확률 층이라 같은 값도
   나오지 않는다. 산출물이 틀렸으면 **문서 쪽에 정정을 병기**한다.

전체 규칙은 `docs/standards.md` 에 있다.

## 작업 전에 읽을 것

기본: `docs/standards.md` → `docs/engineering-notes.md` → 손댈 모듈의 `AGENTS.md`.

**이 프로젝트가 처음이거나, 어떤 변경이 제품의 주장과 어긋나는지 판단이 서지 않으면
`docs/problem.md` 를 먼저 읽는다.** 무엇을 왜 만드는지, 성공을 무엇으로 판정하는지,
무엇이 명시적으로 범위 밖인지가 거기 있다.

작업 종류별로 먼저 볼 곳:

| 손대려는 것 | 먼저 읽을 것 |
|---|---|
| 범위·목표·측정 기준 자체 | `docs/problem.md` + `docs/tracking/decisions/0001`·`0006` |
| SQL 검증기(`sql_guard.py`) | `docs/engineering-notes.md` 의 "실제로 뚫렸던 경로" 3건 + `docs/tracking/decisions/0005` |
| L1 게이트(`gate.py`) | `docs/business-rules.md` 의 PII 규칙 + `docs/engineering-notes.md` 의 오탐 사례 |
| L2 판정(`judge.py`) | `docs/business-rules.md` 의 "L2 판정 규칙" + `docs/tracking/decisions/0007` |
| 인계 사유·상태 전이 | `docs/business-rules.md` 의 사유 6종 표와 `both` 우선순위 규칙 |
| 응답 스키마·API | `docs/contracts.md` 의 "공통 규약"(키 존재 규칙)·"층별 판정 키"·"토큰 집계 경계" |
| 스키마 변경 | `docs/engineering-notes.md` 의 "볼륨째 지워야 한다" + `docs/operations.md` 4단계 |
| 평가·지표 | `docs/tracking/status.md` 의 첫 실측값 + `scripts/AGENTS.md` 의 리포트 불변식 |
| 벡터 검색·임베딩 | `docs/engineering-notes.md` 의 pgvector 캐스트 + **"τ 는 임베딩 모델에 묶인다"** + 대역 수치 주의 |
| 근거 채택 축·기권 게이트(τ) | `docs/architecture.md` 의 "근거 채택은 축이 둘이다" + `docs/tracking/decisions/0009`·`0012`·`0014` |
| 회귀 가드·승격 기준선 | `scripts/AGENTS.md` 불변식 16~18 + `src/reply_gate/AGENTS.md` 의 `regression_guard.py` 행 |
| 테스트·구조 검사 | `tests/AGENTS.md` (음성 대조·검사 대상 유도·`declared_settings()` 규칙) |
| 비용·토큰·달러 | `docs/tracking/pricing.md` (**기준일을 반드시 함께 인용한다**) + `docs/contracts.md` 의 "토큰 집계 경계" |

## 문제를 발견했을 때

**즉시 사용자에게 보고할 것** (이 프로젝트에서 치명적인 실패):

- text-to-SQL 이 선검사를 통과한 주문 **밖의** 행을 돌려주는 경로 — 남의 개인정보가 근거가 되고
  그 근거가 곧 PII 허용 목록이 된다
- L1 이 정상 초안을 기각하거나(오탐) 구조적 결함을 통과시키는 경로 — 헤드라인 지표가 곧 오염된다
- read-only 계정으로 쓰기가 되거나 화이트리스트 밖 테이블이 읽히는 경로
- 자격 증명이 저장소·로그·오류 메시지에 평문으로 남는 경우

그 외에는 `docs/tracking/findings.md` 에 기록한다 — **단, 먼저 고쳐보고** 고칠 수 있으면
고친 뒤 얻은 지식을 `docs/engineering-notes.md` 로 옮긴다. findings 는 "지금 해결할 수 없는
것"의 자리이지 "귀찮아서 미룬 것"의 자리가 아니다.

## 검증

```bash
docker compose up -d --wait      # DB 통합 테스트의 전제
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run pytest -m db              # skip 0 이어야 한다
```

DB 없이 돌리면 `db` 마커 테스트가 사유를 담아 skip 된다. **전체 녹색을 주장하려면
`pytest -m db` 로 skip 0 을 따로 확인해야 한다.**
