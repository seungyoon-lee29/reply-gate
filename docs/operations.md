# 운영

로컬 실행이 전부다. 배포는 없다.

## 사전 준비

- Docker (Compose v2 포함)
- [uv](https://docs.astral.sh/uv/)
- **API 키 2개**
  - OpenAI — 생성과 임베딩에 공통으로 쓴다
  - Anthropic — L2 판정 전용. **L2 는 기본으로 켜져 있으므로 이 키가 없으면
    `POST /inquiries` 가 503 이다.** 판정 없이 돌려보려면 `L2_ENABLED=false` 로 **명시**한다
    (그 실행은 L1 단독 = 꺼짐 기준선이다)

## 1. 환경 변수

```bash
cp .env.example .env
```

`.env` 를 열어 채운다. `.env` 는 gitignore 되고, `.env.example` 에는 키 이름만 있다.

| 변수 | 역할 | 비우면 |
|---|---|---|
| `OPENAI_API_KEY` | 생성·임베딩 공통 | `POST /inquiries` 가 503, 정책 인덱싱 불가 |
| `ANTHROPIC_API_KEY` | **L2 판정 전용** | L2 가 켜져 있으면 `POST /inquiries` 가 503, `--live` 평가가 측정 2·3 을 시작하지 않는다 |
| `POSTGRES_HOST` / `POSTGRES_PORT` | 접속 대상 (기본 `localhost` / `5433`) | 기본값 사용 |
| `POSTGRES_DB` | 데이터베이스 이름 (기본 `reply_gate`) | 기본값 사용 |
| `POSTGRES_SUPERUSER` / `POSTGRES_SUPERUSER_PASSWORD` | 컨테이너 초기화(확장 설치·계정 생성)에만 쓴다 | 컨테이너가 뜨지 않는다 |
| `POSTGRES_APP_USER` / `POSTGRES_APP_PASSWORD` | 애플리케이션 계정(읽기/쓰기) | 접속 실패 |
| `POSTGRES_RO_USER` / `POSTGRES_RO_PASSWORD` | text-to-SQL 실행 전용(SELECT 만) | 주문 조회 경로 실패 |

주석 처리된 조정 가능 기본값(`GENERATION_MODEL`, `VECTOR_TOP_K`,
`VECTOR_SIMILARITY_THRESHOLD`, `SQL_MAX_ROWS`, `SQL_STATEMENT_TIMEOUT_MS` 등)은
비워두면 코드 기본값을 쓴다.

**검색 전략 스위치**도 같은 자리에 있다.

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `QUERY_REWRITE_ENABLED` | `true` (**기본 켜짐**) | 문의를 정책 문서 어휘의 검색 질의로 다시 써 **원문과 함께** 검색한다. 호출이 실패하면 원문 질의로 폴백하고 그 사실이 처리 기록·리포트에 남는다(인계가 아니다). **끄려면 명시해야 한다** — 켜진 채 클라이언트가 배선되지 않은 조립은 시작 시점에 죽는다 |
| `VECTOR_TOP_K` | `5` | 검색 후보의 **선절단** 개수. 기권 게이트의 통계량도 이 슬라이스에서 계산한다 — 조정 가능 인자가 아니라 **행동 계약**이다 |
| `VECTOR_SIMILARITY_THRESHOLD` | `0.3` | **사이클 3 T10 이 0.50 을 반증해 되돌린 값이다.** 0.50 은 무근거 4건을 검색 단계에서 기권시키지만 같은 컷이 G04 의 정답 조항(0.3571)과 G18 의 상충 조항(0.4676)까지 잘라 정상 문의가 인계되고 L2 모순 기각이 사라진다(`docs/tracking/status.md` "사이클 3 T10 라이브 재실측") |
| `ABSTENTION_GATE_ENABLED` | `true` (**기본 켜짐**) | 질의 단위 기권 게이트. 상위 `top_k` 코사인의 통계량이 τ 미만이면 그 질의는 **채택 0건**이다(→ `no_evidence`). 끄면 채택이 컷 하나로 돌아간다 — **원복은 이 한 줄이다** |
| `ABSTENTION_GATE_STATISTIC` | `rank1_minus_rank_k_spread` | 게이트가 보는 통계량. 격자 165 구성 중 채택 규칙 다섯을 순서대로 통과한 넷 가운데 **가장 단순한 것**(뺄셈 하나)을 골랐다(`tracking/decisions/0014`) |
| `ABSTENTION_TAU` | `0.06` | 통계량이 이 값 **미만**이면 기권. **`EMBEDDING_MODEL` 을 바꿔도 이 값은 따라오지 않는다** — τ 는 조건 종속 인자이고 `-3-large` 계열에서는 이 축이 기권군과 채택군을 **분리조차 못 한다**(`docs/engineering-notes.md` "τ 는 임베딩 모델에 묶인다"). 재산출은 명시적 작업이다 |

**L2 스위치와 판정 모델**도 같은 자리에 있다.

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `L2_ENABLED` | `true` (**기본 켜짐**) | 끄면 L1 단독 동작 = 사이클 1 과 같다. 켜진 채 판정이 실패하면 통과가 아니라 인계다(fail-closed) |
| `JUDGE_MODEL` | `claude-sonnet-5` | 판정 모델 등급. 조정 가능 기본값. 3종 비교표는 `tracking/decisions/0018` — **비교만 했고 바꾸지 않았다** |
| `JUDGE_EFFORT` | (미지정) | 지정했을 때만 요청에 실린다 |
| `JUDGE_MAX_OUTPUT_TOKENS` | `16000` | thinking + 응답 **합산** 상한이라 여유 있게 둔다 |
| `JUDGE_PROMPT_CACHING_ENABLED` | `false` (**기본 꺼짐**) | 판정 **고정 프리픽스**(판정 지침)에 Anthropic 프롬프트 캐싱을 건다. 실측했고 **채택하지 않았다**(`tracking/decisions/0019`). **켜면 리포트의 판정 입력 토큰이 "비캐시 입력"이 된다** — 캐시 계열은 리포트의 별도 칸에서 본다(`docs/contracts.md` "토큰 집계 경계") |

## 2. 의존성

```bash
uv sync
```

## 3. DB 기동

```bash
docker compose up -d --wait
```

컨테이너 이름은 `reply-gate-postgres` 로 고정돼 있다. `--wait` 는 healthcheck 통과까지
기다린다 — 이걸 빼면 다음 단계가 접속 실패로 끝난다.

## 4. 스키마 + 주문 시딩 — **API 키 불필요**

```bash
uv run python -m scripts.seed_orders
```

스키마를 적용하고 커밋된 픽스처(`db/fixtures/orders.jsonl`) 500건을 적재한다.
재실행해도 500건 그대로다.

> 스키마를 고친 뒤에는 볼륨째 재생성해야 한다:
> `docker compose down -v && docker compose up -d --wait` → 이 단계를 다시 실행.
> DDL 이 전부 `CREATE TABLE IF NOT EXISTS` 라 기존 테이블의 컬럼은 바뀌지 않는다.
> **단 컬럼 추가는 예외다** — `db/schema.sql` 은 `ALTER TABLE … ADD COLUMN IF NOT EXISTS` 로
> 살아 있는 볼륨에 컬럼을 붙인다(검색 계열 3개가 이 방식으로 들어갔다). 컬럼 **변경·삭제**만
> 볼륨 재생성이 필요하다 — 볼륨을 지우면 보존 기록과 **유료 정책 재색인**이 함께 딸려온다
> (`docs/engineering-notes.md` "스키마를 바꿨으면 볼륨째 지워야 한다 — 컬럼 추가만 예외다").
>
> **L2 도입으로 스키마가 바뀌었다** — `inquiry_attempts` 에 층별 판정 컬럼
> (`l1_verdict`·`l1_reject_reasons`·`l2_verdict`·`l2_reject_reasons`·`claim_verdicts`·
> `evidence_contradictions`)이, `inquiries` 에 판정 토큰 컬럼
> (`judge_input_tokens`·`judge_output_tokens`)이 추가됐다. **이전 사이클의 볼륨을 그대로 쓰면
> 이 컬럼들이 없어 저장이 실패한다.** 위 볼륨 재생성을 반드시 한 번 실행한 뒤 5단계(정책
> 인덱싱)를 다시 돌린다 — 볼륨을 지우면 정책 청크도 함께 사라진다.
>
> **이번 사이클에도 스키마가 바뀌었다** — `policy_chunks` 에 임베딩 출처 컬럼
> (`embedding_model`·`embedding_dimensions`, 둘 다 NOT NULL)이 추가됐다. 이전 볼륨을 그대로
> 쓰면 정책 적재가 NOT NULL 위반으로 실패한다. 같은 볼륨 재생성 절차를 한 번 더 돌린다.

## 5. 정책 인덱싱 — **API 키 필요**

```bash
uv run python -m scripts.index_policies
```

`data/policies/` 의 문서 4개를 조항 단위로 쪼개 임베딩하고 `policy_chunks` 에 적재한다(26건).
재실행하면 중복 없이 갱신되고, 문서에서 사라진 조항은 삭제된다.
**이 단계를 건너뛰면 정책 근거 검색이 항상 0건이라 모든 문의가 `no_evidence` 로 인계된다.**

**라이브 실측 전에는 색인 출처를 확인한다.** DB 는 워크트리 사이에 공유되므로(컨테이너 이름과
볼륨이 고정) 다른 작업이 같은 `policy_chunks` 를 건드릴 수 있다. 기대와 다르면 이 단계를 다시
돌린다 — 26 청크 · 임베딩 토큰 2,577 · $0.0001 미만이다
(`docs/engineering-notes.md` "병렬 작업이 정책 색인을 덮어쓸 수 있다").

```bash
uv run python -c "
from reply_gate.db import connect
with connect() as conn, conn.cursor() as cur:
    cur.execute('select embedding_model, embedding_dimensions, count(*) from policy_chunks group by 1,2')
    print(cur.fetchall())
"
# [{'embedding_model': 'text-embedding-3-small', 'embedding_dimensions': 1536, 'count': 26}]
```

**`EMBEDDING_MODEL`·`EMBEDDING_DIMENSIONS` 를 바꿨으면 이 단계를 반드시 다시 돌린다.**
적재된 벡터는 자기 출처를 함께 들고 있고, 질의 임베딩의 출처와 다르면 검색이 유사도를 내지
않고 **503** 으로 거부한다. 재색인 없이 모델만 바꾸면 서로 다른 공간을 비교하게 되는데,
그 결과는 오류가 아니라 "근거 없음"처럼 보인다
(docs/engineering-notes.md "같은 차원의 다른 모델은 아무도 막지 않는다").

## 6. 서버 기동

```bash
uv run uvicorn reply_gate.api:app --reload
```

- 웹 폼: `http://127.0.0.1:8000/`
- 헬스 체크: `GET /health` (DB 를 확인하지 않는다 — 항상 200)

## 7. 검증 — **API 키 불필요**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run pytest -m db        # skip 0 이어야 한다
```

`pytest` 는 DB 기동을 전제한다. DB 가 없으면 `db` 마커 테스트가 사유를 담아 skip 되므로,
전체 녹색을 주장하려면 `pytest -m db` 로 skip 0 을 따로 확인한다.

**문서 링크·앵커 검사와 검증 건수 대조는 이 스위트 안에 있다** — 별도 명령이 아니다.
링크 쪽은 따로도 돌릴 수 있다:

```bash
uv run python -m scripts.check_links        # 문서 46개 · 링크 400개 · 깨짐 0건
```

**건수 대조는 전체 스위트에서만 성립한다.** 경로·`-k`·`-m` 을 주면 그 세션이 스위트가 아니므로
사유를 담아 skip 된다 — 부분 실행에서 대조하면 거짓 빨강이 나고, 거짓 빨강은 검사를 끄게 만든다.

**그리고 테스트 건수 쪽은 DB 를 띄운 실행에서만 성립한다.** 문서가 인용하는 것은 `N passed`
인데 세션이 아는 것은 수집 건수이고, 둘은 **skip 이 0 일 때만** 같다. DB 가 없으면 `db` 마커
178건이 skip 되므로 그 대조는 **사유를 담아 skip** 된다(도구 쪽 — `ruff format`·`mypy`·링크
검사 — 은 skip 과 무관해서 계속 대조한다). 이 갈래가 없던 첫 판은 `1,310 passed · 178 skipped`
인 실행에서도 `1,488 passed` 인용을 초록으로 통과시켰다([findings 37](tracking/findings.md)).

**사이클 5 종료 시점의 실행값**(2026-08-21): `ruff check` 0 · `ruff format --check` 182 파일 ·
`mypy` 75 파일 0 · `pytest` **1,431 통과** · `pytest -m db` **176 통과 / skip 0**.

**종료 뒤 감사 수정을 반영한 실행값**(2026-08-22): `ruff check` 0 · `ruff format --check`
183 파일 · `mypy` 76 파일 0 · `pytest` **1,452 통과** · `pytest -m db` **176 통과 / skip 0**.

**사이클 2 적대 리뷰 수정을 반영한 실행값**(2026-08-23): `ruff check` 0 · `ruff format --check`
**184 파일** · `mypy` 76 파일 0 · `pytest` **1,454 통과** · `pytest -m db` **178 통과 / skip 0**.
`pytest` 가 2건 는 것은 층별 사유 CHECK 두 가지의 음성 대조를 더한 결과다. 파일 수는 위
줄에서 늘어난 것이 아니라 **위 줄이 처음부터 1 적게 적혀 있었다** — 이 명령은 마크다운까지
센다.

**사이클 1~5 적대 리뷰 반영을 마친 실행값**(2026-08-25): `ruff check` 0 ·
`ruff format --check` **187 파일** · `mypy` **79 파일** 0 · `pytest` **1,488 통과** ·
`pytest -m db` **178 통과 / skip 0**. 늘어난 34건은 조회 가드 분기 넷의 격리 음성 대조,
링크·앵커 검사와 그 음성 대조, 검증 건수 대조, 비밀 이름 규칙의 예외 등재 검사,
캐스트 허용 목록의 문서↔코드 대조, 접속 문자열 마스킹 검사다.
**이 줄이 마지막으로 손으로 옮겨 적는 값이다** — 이제 문서가 인용한 건수를 스위트가 직접
대조한다(위 "건수 대조").

**판정 수단을 되짚어 고친 뒤의 실행값**(2026-08-26): `ruff check` 0 · `ruff format --check`
**187 파일** · `mypy` **79 파일** 0 · `pytest` **1,497 통과** ·
`pytest -m db` **178 통과 / skip 0**. 앞 줄의 *"마지막으로 손으로 옮겨 적는 값"* 은
**틀렸다** — 대조가 있어도 건수를 **움직이는 변경**은 그 인용을 같은 커밋에서 함께 고쳐야
하고, 대조는 그것을 잊었을 때 RED 로 알려 줄 뿐이다. 늘어난 9건은 건수 대조를 도구·테스트 둘로 가른 것,
"수집 = 통과" 전제의 음성·양성 대조 셋, 부분 실행 판별의 음성 대조, 링크 수 훑기, 루트
모집단 검사, 밑줄 슬러그 두 건이다([findings 37](tracking/findings.md)).

## 8. 평가 지표 산출

```bash
uv run python -m scripts.evaluate            # 측정 1 만 — API 키 불필요
uv run python -m scripts.evaluate --live     # 측정 2·3 포함 — API 키 필요, 과금
uv run python -m scripts.evaluate --stub-llm # 배관 검증용 — 실제 수치가 아니다

# 측정 3 단독 실측 — DB 도 생성 키도 필요 없다(입력이 판정 픽스처다)
uv run python -m scripts.evaluate --live --measurements 3

# 판정 모델을 실행 인자로만 덮는다 — config.py 의 기본값은 무접촉이다
uv run python -m scripts.evaluate --live --measurements 3 --judge-model claude-opus-5 \
  --declare-experiment judge_model
```

**`--declare-experiment` 는 "이번에 의도적으로 바꾼 축"의 선언이다.** 선언된 항목은 회귀
가드가 대조를 **진행하며 차이를 병기**하고, 선언되지 않은 지문 불일치만 "대조 불가"가 된다.
선언은 그 실행의 예외 처리이지 승격 등재의 정정이 아니다.

측정은 셋이다.

| 측정 | 무엇을 재나 | 언제 도나 |
|---|---|---|
| 측정 1 | L1 게이트 단위 정확도 (L1 픽스처 **37건**) | **항상** — LLM 호출 0회, 키도 DB 도 불필요 |
| 측정 2 | 파이프라인 판정 일치율 (골든셋 30건 end-to-end) | `--live`(실측·과금) 또는 `--stub-llm`(대역·배관 검증) |
| 측정 3 | **L2 판정 단위 정확도** (판정 픽스처 11건) | **`--live` + L2 켜짐에서만.** 확률 층이고 과금된다 |

**측정 3 은 `--live` 전용이다.** 판정 모델을 실제로 부르는 확률 층이라 대역으로는 낼 수 없는
수치이고, `--stub-llm` 에서는 아예 돌지 않는다(미실행 사유가 리포트에 남는다). L2 꺼짐 실행에서도
돌지 않는다 — 꺼짐 기준선이 판정 비용을 쓰면 안 되기 때문이다. **목표치는 두지 않는다**
(무목표 관측 — `tracking/decisions/0006`). 그래서 달성·미달 판정이 붙지 않는다.

**측정 선택은 `--measurements` 이고, 고르지 않은 측정은 "미실행 + 사유"다.** 측정 3 이 실제로
요구하는 것은 **판정 키 + L2 켜짐**뿐이다 — DB 도 생성 키도 골든셋도 아니다. 그래서
`--live --measurements 3` 은 **DB 없이 도는 측정 3 단독 실측**이고, 라이브 이름을 그대로 받는다
(리포트 이름의 자격은 "측정 2 실측 여부"가 아니라 **"과금 실행 여부"** 다).

남은 결합은 둘뿐이다: ① **측정 2 를 함께 고른** 실행에서는 측정 2 의 선검사 사유를 물려받는다
(요청되지 않은 부분 구매를 막는다), ② 측정 2 가 **중단되면** 측정 3 도 잇지 않는다.

**키 선검사는 측정 시작 전이다.** `--live` + L2 켜짐이면 `OPENAI_API_KEY` 와
`ANTHROPIC_API_KEY` 를 **둘 다** 본다. 판정 키가 없으면 측정 2·3 을 **모두** 건너뛴다 —
L2 를 꺼서 측정 2 만 돌리는 **강등 실행은 하지 않는다**(요청은 켜짐인데 값은 꺼짐 기준선이라
둘 다 오염된다).

**종료 코드**가 1 이 되는 경우는 둘뿐이다.

1. **측정 3 이 돌 조건이었는데 미실행** — `--live` 로 선검사를 통과해 측정 2 가 실측으로 돌
   조건이었고 **L2 도 켜져 있던** 실행에서 판정 수치가 안 나온 경우다. 골든셋 30건을 사고
   판정 수치는 못 낸 실행이 래퍼·CI 에 "성공"으로 읽히면 안 된다.
2. **사용자 중단(Ctrl-C)** — 과금 여부와 무관하다. `--stub-llm`·기본 실행에서 중단해도 1 이다.
   리포트는 그대로 쓰지만(과금분 보존) 중단 사실은 종료 코드가 들고 간다.

그 밖은 0 이다. `--live` 없는 평범한 실행에서 측정 3 이 "`--live` 아님" 사유로 미실행인 것과,
**L2 꺼짐 기준선 실행에서 측정 3 이 설계상 안 도는 것**은 정상이므로 0 이다.

리포트는 `.md`(사람용)와 `.json`(기계용)으로 나온다. **이름 규칙은 두 겹이고 둘 다 양방향이며,
기본 이름과 `--report-stem` 명시 스템 모두에 걸린다.**

1. **라이브 이름 ⇔ 과금 실행** — 과금이 일어난 실행만 `reports/evaluation-live*` 에 쓸 수
   있고, 과금 실행은 그 이름에만 쓴다. **자격 판정 근거는 "측정 2 실측 여부"가 아니다**
   (위 `--measurements` 절) — 측정 2 를 건너뛴 **측정 3 단독 실측도 과금 실행이므로 라이브
   이름을 받는다.** 코드가 보는 값도 그대로 `측정 2 실측 ∨ 측정 3 실행`이다
   (`evaluation.resolve_report_stem` 의 `billed`).
2. **L2 켜짐 실측 ⇔ `evaluation-live-l2` 접두** — L2 켜짐 실측은 l2 계열에만, L2 꺼짐 실측은
   l2 계열이 **아닌** `evaluation-live*` 에만 쓴다. 계열이 뒤섞이면 산출물만 보고는 어느 쪽이
   L2 를 포함해 잰 값인지 알 수 없다.

L2 켜짐 실측의 기본 이름은 코드가 `evaluation-live-l2-<n>` 으로 **자동 넘버링**한다(재실측
프로토콜이 3회 반복이라 기본 이름 충돌로 죽으면 안 된다). 이미 있는 라이브 리포트는 덮어쓸 수
없고 빈 번호를 제안받는다. 그 밖의 실행 — 기본·대역, 그리고 `--live` 를 줬어도 선검사에
걸려 **과금 측정이 하나도 돌지 않은** 경우 — 은 `reports/evaluation.*` 에 쓴다. 측정 2·3 을
함께 고른 실행에서 DB 가 없으면 위 "남은 결합" ①로 **측정 3 도 함께 미실행**이라 여기 해당하고,
`--measurements 3` 은 DB 없이도 과금되므로 라이브 이름이다. 라이브 실측 리포트만 저장소가
추적하고 나머지는 gitignore 된다. 규칙 위반은 **측정 시작 전에** 거부된다 — 문서가 인용하는
근거를 덮어쓰지 않기 위해서다(engineering-notes 의 "라이브 리포트를 기본 실행이 덮어썼다").

`--live` 는 골든셋 30건을 실제 생성 모델에, 판정 픽스처 11건을 실제 판정 모델에 흘린다
(과금·비결정론). 실행 전에 5번 단계가 최신인지 확인한다 — 하네스는 이미 적재된 임베딩을
그대로 읽는다.

**실행마다 회귀 가드가 두 줄을 콘솔과 리포트에 낸다** — 승격 기준선(**구속**)과 직전
라이브(**경보**). 두 줄이 상반되면 승격이 이긴다. 판정은 `reply_gate.regression_guard` 가 하고
스크립트는 결과를 찍기만 한다. **승격·재등재는 사람이 `data/promoted_baseline.json` 을 바꾸는
것뿐이고 하네스에 자동 경로가 없다**(구조 테스트가 쓰기 호출을 막는다). 승격 참조가 없으면
리포트는 **"기준선 미등재"** 라고 적는다 — 0 이나 "통과"가 아니다.
**세트가 안 찼거나 검사가 못 돌았으면 헤드라인은 "보류"** 이고, 제외된 산출물과 사유가 함께
실린다.

`--stub-llm` 은 생성·임베딩·**판정**을 전부 결정론 대역으로 갈아 끼워 **외부 호출 0회**로
돈다. 정책 청크를 어휘 임베딩 대역으로 재적재하되 끝나면 **롤백**하므로 공유 DB 의 실제
임베딩을 덮어쓰지 않는다. 대역으로 낸 수치는 실제 수치가 아니고, 리포트가 실행 조건과 함께
그 사실을 찍는다.

## 9. 검색 전략 비교 — **DB 불필요**

정책 검색만 따로 재는 오프라인 하네스다. `scripts.evaluate` 와 다른 진입점이고 DB 를 쓰지 않는다.

```bash
uv run python -m scripts.compare_retrieval --stub-embedding   # 기본 — 외부 호출 0회, 배관 검증
uv run python -m scripts.compare_retrieval --live             # 실제 임베딩 + 리랭크 — 과금
uv run python -m scripts.compare_retrieval --bge-m3           # 로컬 임베딩 — 리랭크 단은 미측정
uv run python -m scripts.compare_retrieval --live --embedding-axis  # 모델 축 4행 — 과금
```

산출물은 `reports/retrieval-strategies-<모드>-<모델>-d<차원>-<재작성조건>-k<top_k>-c<컷>.{md,json}`
이다. 이름에 재작성 조건(blind/oracle)과 컷·top_k 가 들어가므로 결정 기록이 인용할 파일을
이름으로 식별할 수 있다. 기존 산출물은 덮어쓰지 않는다.

**과금 축은 둘이다.** 임베딩을 실제로 부르는지(`--live`/`--bge-m3`)와 LLM 리랭크를 실제로
부르는지(`--live` 또는 `--rerank-with-openai`)가 별개다. `--bge-m3` 는 로컬 임베딩만 쓰고,
리랭크 과금을 승인하지 않으면 그 단을 **0 이 아니라 "미측정 + 사유"** 로 리포트에 남긴다.
BGE-M3 는 선택 의존성이므로 `uv sync --extra rag-local` 이 필요하고, 미설치는 `--embedding-axis`
에서 그 행만 미측정으로 남는다.

**채택 판정은 전 전략이 코사인 유사도 하나를 쓴다**(결정 0009). 리포트는 전략마다 컷 스윕과
자기 최적 컷을 함께 싣고, precision·recall 분모를 값과 같은 표에 인쇄한다. 컷 스윕은 검색을
다시 돌리지 않으므로 추가 과금이 없다.

`--policy-dir`·`--golden-set`·`--labels` 로 입력을 바꿀 수 있고, 검색 정답 라벨은 **그 실행이
실제로 쓸** 코퍼스·골든셋 기준으로 교차검증된다 — 없는 조항이 정답으로 남아 recall 이 조용히
0 으로 계상되지 않는다.

### blind 재작성 픽스처 재생성 — 상시 실행 경로가 아니다

`data/rewritten_queries.jsonl` 은 **정책·라벨을 본 적 없는 생성 모델**이 문의 원문만 보고
만든 것을 한 번 생성해 커밋한 입력이다(결정 0010). 다시 만들 일이 생기면:

```bash
uv run python -m scripts.generate_blind_rewrites --out /tmp/rewrites.jsonl        # 프롬프트만 — 무과금
uv run python -m scripts.generate_blind_rewrites --live --out /tmp/rewrites.jsonl # 30건 생성 — 과금
```

**기존 픽스처를 스크립트가 직접 덮어쓰지 않는다.** `--out` 경로에만 쓰고, 사람이 산출을 읽은
뒤 옮긴다. 옮기면 구조 테스트가 G17 문장에 걸려 빨개지므로, 교체는 **의도했을 때만** 통과한다.
생성 모델은 런타임 의도 해석과 같은 등급(`generation_model`)이어야 한다 — 상위 모델은
배포 가능한 이득이 아니라 또 하나의 상한이다.

## 10. 채택 축 손계산 — **무과금 · DB 불필요 · 외부 호출 0회**

```bash
uv run python -m scripts.handcalc_adoption_axis
```

커밋된 검색 산출물(`reports/retrieval-strategies-live-*.json`)만 읽어 채택 축 후보 공간을
전수 계산한다. 새 임베딩도 새 API 호출도 없으므로 **재실행이 공짜이고 같은 입력에서 같은
산출을 낸다** — 그래서 이 산출물(`reports/adoption-axis-handcalc.{md,json}`)은 **덮어쓴다.**
라이브 리포트 계열과 이름이 갈려 있어 불변식 7(라이브 이름 ⇔ 과금 실행)과 충돌하지 않는다.
**저장소는 이 산출물을 추적하지 않는다**(`.gitignore` 의 `reports/*`) — 문서가 이 값을
인용하면 읽는 사람은 위 명령을 한 번 돌려야 파일을 본다.

내는 것 셋: ① 절대 하한 × 적응 컷 비율 × 상대 마진 전수 격자에서 세 방향 동시 만족 구성 수,
② 기권 게이트 통계량 5종의 **조건별 분리 여유**(임베딩 모델·재작성 조건별), ③ 상충쌍 보존
규칙이 무엇을 고치고 무엇을 늘리는지. **τ 를 재산출해야 할 때 여기가 출발점이다.**

## 11. 시연 — 세 장면을 실제로 흘린다  (문의 단위 소액 과금)

**아래는 사이클 5 가 실제로 밟은 순서다.** 세 장면이 각각 "어느 층이 무엇을 잡는가"를
보여 준다 — ① L1 구조 기각 → 재생성 → 통과, ② L1 통과 → **L2 기각**, ③ 검색 기권 → 시도 0건.
평가 하네스의 라이브(풀셋 실행)가 아니라 **문의 단위 과금**이다. 실제 실행은 문의 6건에
**$0.102877** 이었다(처리 기록의 토큰에서 재집계 · 단가 기준일 2026-08-19). 상한은 $0.5 이고,
넘게 되면 멈추고 승인을 받는다.

**1. DB 를 띄운다** — 이미 떠 있으면 건너뛴다(3단계와 같다).

```bash
docker compose up -d --wait      # 컨테이너 reply-gate-postgres
```

**2. 정책 색인의 출처와 건수를 확인한다** (5단계의 확인 명령 그대로).
기대값은 `text-embedding-3-small` / 1536 / **26 청크**다. 다르면
`uv run python -m scripts.index_policies` 로 다시 만든다 — DB 가 워크트리 사이에 공유되므로
다른 작업이 같은 `policy_chunks` 를 덮어썼을 수 있다.

**3. 서버를 띄우고 헬스 체크를 본다.**

```bash
uv run uvicorn reply_gate.api:app --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/health      # {"status":"ok"}
```

**4. 세 장면을 흘린다** — 웹 폼(`http://127.0.0.1:8000/`) 또는 `POST /inquiries`.
**문의 문구는 골든셋 원문 그대로 넣는다.** 장면 2 는 **주문 문의라 주문번호를 함께 넣어야
한다** — 빼면 다른 경로로 간다.

| 장면 | 문의 (골든셋 원문) | 주문번호 | 무엇을 보나 |
|---|---|---|---|
| 1 | `고객센터 전화번호 알려주세요.` (G16) | 없음 | L1 `missing_citation` 기각 → 재생성 → 통과 |
| 2 | `이 주문 결제한 카드 번호 뒷자리를 알려주세요.` (G30) | `ORD-20260202-0001` | L1 통과 → **L2 `unsupported_claim` 기각** |
| 3 | `해외 배송도 되나요? 미국으로 보내고 싶어요.` (G21) | 없음 | 검색 기권 → **시도 0건** → `no_evidence` |

**5. 문의 ID 로 다시 연다** — `GET /inquiries/{id}` 가 저장된 기록에서 같은 골격을
재구성한다. 시연의 증거는 별도 산출물 파일이 아니라 **이 ID 목록**이다
(추적되는 산출물 디렉터리는 라이브 실측 리포트만 담는다).

**6. 서버를 내리고 좀비가 없는지 확인한다.**

```bash
pgrep -fl "uvicorn reply_gate"    # 출력이 비어야 한다
```

### 재현 확률을 먼저 적는다 — 시도 상한도 함께

**확률 층이 낀 장면을 "돌리면 나온다"로 적으면 안 된다.** 실제 시도 회차는 아래와 같았고,
전시 문서의 데모 절이 같은 회차를 인용한다.

| 장면 | 시도 상한 | 실제 재현 | 사전 기대 |
|---|---|---|---|
| 1 | **10회** | 3회차에 재현 (1·2회차는 1시도 통과) | **낮다** — 직전 라이브 15 케이스-실행 중 1회 |
| 2 | 3회 | 2회차에 재현 (1회차는 1시도 통과) | **계보 4/7** — 사이클 4 세 회차는 3/3 이지만 **현 기본값 세 회차는 0/3**, 확인 라이브 1/1. 재기각 인계까지 간 것은 한 번뿐 |
| 3 | 3회 | 1회차에 재현 | 12/12 |

**실행 기록 (2026-08-21) — 전부 `GET /inquiries/{id}` 로 다시 열린다.**
**재현되지 않은 회차도 전부 적는다.**

| 장면 | 회차 | 문의 ID | 관측 |
|---|---|---|---|
| 1 | 1 | `784099c2-5014-4bf5-bb87-f1c3dc5b7e58` | 1시도 통과 — **미재현** |
| 1 | 2 | `2e2ec6a5-38e2-4c19-81d0-34b719f8222f` | 1시도 통과 — **미재현** |
| 1 | **3** | `335b7c05-553c-450c-9de0-0ebf9da8f727` | 시도1 L1 `missing_citation`(L2 미실행) → 시도2 통과 |
| 2 | 1 | `2448b031-143c-4a12-b258-0ea65f22ba81` | 1시도 통과 — **미재현** |
| 2 | **2** | `5791baee-0627-4c88-b029-aea39a33d02e` | 시도1 L1 pass · **L2 `unsupported_claim`** → 시도2 통과 |
| 3 | **1** | `81713976-7c2a-4f50-9b70-193ef3ce1d89` | 검색 기권 → **시도 0건** → `no_evidence` 인계 |

**시도 상한을 실행자가 스스로 올리지 않는다** — 상향은 사용자 지시로만 한다.
**상한 안에 재현되지 않으면 그 사실과 시도 회차를 그대로 적는다.** 시도를 늘리는 것이 아니라
**실패 회차를 숨기는 것**이 "잘 나온 것만 실었다"이고, 그것이 이 프로젝트의 유일한 주장을
스스로 깎는다.

## 정리

```bash
docker compose down          # 컨테이너만 내린다 (데이터 유지)
docker compose down -v       # 볼륨까지 지운다 (데이터 삭제)
```

서버를 백그라운드로 띄웠다면 종료를 확인한다: `pgrep -fl "uvicorn reply_gate"`.
