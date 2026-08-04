# 운영

로컬 실행이 전부다. 배포는 없다.

## 사전 준비

- Docker (Compose v2 포함)
- [uv](https://docs.astral.sh/uv/)
- OpenAI API 키 1개 — 생성과 임베딩에 공통으로 쓴다

## 1. 환경 변수

```bash
cp .env.example .env
```

`.env` 를 열어 채운다. `.env` 는 gitignore 되고, `.env.example` 에는 키 이름만 있다.

| 변수 | 역할 | 비우면 |
|---|---|---|
| `OPENAI_API_KEY` | 생성·임베딩 공통 | `POST /inquiries` 가 503, 정책 인덱싱 불가 |
| `POSTGRES_HOST` / `POSTGRES_PORT` | 접속 대상 (기본 `localhost` / `5433`) | 기본값 사용 |
| `POSTGRES_DB` | 데이터베이스 이름 (기본 `reply_gate`) | 기본값 사용 |
| `POSTGRES_SUPERUSER` / `POSTGRES_SUPERUSER_PASSWORD` | 컨테이너 초기화(확장 설치·계정 생성)에만 쓴다 | 컨테이너가 뜨지 않는다 |
| `POSTGRES_APP_USER` / `POSTGRES_APP_PASSWORD` | 애플리케이션 계정(읽기/쓰기) | 접속 실패 |
| `POSTGRES_RO_USER` / `POSTGRES_RO_PASSWORD` | text-to-SQL 실행 전용(SELECT 만) | 주문 조회 경로 실패 |

주석 처리된 조정 가능 기본값(`GENERATION_MODEL`, `VECTOR_TOP_K`,
`VECTOR_SIMILARITY_THRESHOLD`, `SQL_MAX_ROWS`, `SQL_STATEMENT_TIMEOUT_MS` 등)은
비워두면 코드 기본값을 쓴다.

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

## 5. 정책 인덱싱 — **API 키 필요**

```bash
uv run python -m scripts.index_policies
```

`data/policies/` 의 문서 4개를 조항 단위로 쪼개 임베딩하고 `policy_chunks` 에 적재한다(26건).
재실행하면 중복 없이 갱신되고, 문서에서 사라진 조항은 삭제된다.
**이 단계를 건너뛰면 정책 근거 검색이 항상 0건이라 모든 문의가 `no_evidence` 로 인계된다.**

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
uv run python -m scripts.evaluate --live     # 측정 2 포함 — API 키 필요, 과금
uv run python -m scripts.evaluate --stub-llm # 배관 검증용 — 실제 수치가 아니다
```

리포트는 `reports/evaluation.md`(사람용)와 `reports/evaluation.json`(기계용)으로 나온다.
`reports/` 는 gitignore 된다.

`--live` 는 골든셋 30건을 실제 모델에 흘린다(과금·비결정론). 실행 전에 5번 단계가
최신인지 확인한다 — 하네스는 이미 적재된 임베딩을 그대로 읽는다.

## 정리

```bash
docker compose down          # 컨테이너만 내린다 (데이터 유지)
docker compose down -v       # 볼륨까지 지운다 (데이터 삭제)
```

서버를 백그라운드로 띄웠다면 종료를 확인한다: `pgrep -fl "uvicorn reply_gate"`.
