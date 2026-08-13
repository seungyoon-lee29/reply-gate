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

**L2 스위치와 판정 모델**도 같은 자리에 있다.

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `L2_ENABLED` | `true` (**기본 켜짐**) | 끄면 L1 단독 동작 = 사이클 1 과 같다. 켜진 채 판정이 실패하면 통과가 아니라 인계다(fail-closed) |
| `JUDGE_MODEL` | `claude-sonnet-5` | 판정 모델 등급. 조정 가능 기본값 |
| `JUDGE_EFFORT` | (미지정) | 지정했을 때만 요청에 실린다 |
| `JUDGE_MAX_OUTPUT_TOKENS` | `16000` | thinking + 응답 **합산** 상한이라 여유 있게 둔다 |

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

## 8. 평가 지표 산출

```bash
uv run python -m scripts.evaluate            # 측정 1 만 — API 키 불필요
uv run python -m scripts.evaluate --live     # 측정 2·3 포함 — API 키 필요, 과금
uv run python -m scripts.evaluate --stub-llm # 배관 검증용 — 실제 수치가 아니다
```

측정은 셋이다.

| 측정 | 무엇을 재나 | 언제 도나 |
|---|---|---|
| 측정 1 | L1 게이트 단위 정확도 (L1 픽스처 27건) | **항상** — LLM 호출 0회, 키도 DB 도 불필요 |
| 측정 2 | 파이프라인 판정 일치율 (골든셋 30건 end-to-end) | `--live`(실측·과금) 또는 `--stub-llm`(대역·배관 검증) |
| 측정 3 | **L2 판정 단위 정확도** (판정 픽스처 11건) | **`--live` + L2 켜짐에서만.** 확률 층이고 과금된다 |

**측정 3 은 `--live` 전용이다.** 판정 모델을 실제로 부르는 확률 층이라 대역으로는 낼 수 없는
수치이고, `--stub-llm` 에서는 아예 돌지 않는다(미실행 사유가 리포트에 남는다). L2 꺼짐 실행에서도
돌지 않는다 — 꺼짐 기준선이 판정 비용을 쓰면 안 되기 때문이다. **목표치는 두지 않는다**
(무목표 관측 — `tracking/decisions/0006`). 그래서 달성·미달 판정이 붙지 않는다.

**측정 3 은 측정 2 와 실측 여부를 공유한다.** 측정 3 자체는 DB 가 필요 없지만, 두 측정의
실측 여부가 갈리면 과금된 판정 수치가 덮어쓸 수 있는 이름으로 새기 때문이다.

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

1. **라이브 이름 ⇔ 실측** — 실측(`--live` 성공) 실행만 `reports/evaluation-live*` 에 쓸 수
   있고, 실측은 그 이름에만 쓴다.
2. **L2 켜짐 실측 ⇔ `evaluation-live-l2` 접두** — L2 켜짐 실측은 l2 계열에만, L2 꺼짐 실측은
   l2 계열이 **아닌** `evaluation-live*` 에만 쓴다. 계열이 뒤섞이면 산출물만 보고는 어느 쪽이
   L2 를 포함해 잰 값인지 알 수 없다.

L2 켜짐 실측의 기본 이름은 코드가 `evaluation-live-l2-<n>` 으로 **자동 넘버링**한다(재실측
프로토콜이 3회 반복이라 기본 이름 충돌로 죽으면 안 된다). 이미 있는 라이브 리포트는 덮어쓸 수
없고 빈 번호를 제안받는다. 그 밖의 실행 — 기본·대역, 그리고 `--live` 를 줬어도 키·DB 문제로
측정 2 가 미실행인 경우 — 은 `reports/evaluation.*` 에 쓴다. 라이브 실측 리포트만 저장소가
추적하고 나머지는 gitignore 된다. 규칙 위반은 **측정 시작 전에** 거부된다 — 문서가 인용하는
근거를 덮어쓰지 않기 위해서다(engineering-notes 의 "라이브 리포트를 기본 실행이 덮어썼다").

`--live` 는 골든셋 30건을 실제 생성 모델에, 판정 픽스처 11건을 실제 판정 모델에 흘린다
(과금·비결정론). 실행 전에 5번 단계가 최신인지 확인한다 — 하네스는 이미 적재된 임베딩을
그대로 읽는다.

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

## 정리

```bash
docker compose down          # 컨테이너만 내린다 (데이터 유지)
docker compose down -v       # 볼륨까지 지운다 (데이터 삭제)
```

서버를 백그라운드로 띄웠다면 종료를 확인한다: `pgrep -fl "uvicorn reply_gate"`.
