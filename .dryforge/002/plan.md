# 사이클 2 플랜 — L2 의미 수준 검증

spec 을 실현하는 잠정 청사진이다. 어긋나면 spec 이 이긴다.

---

## 단계 서사 (사람용)

기반 셋이 서로 독립이다: **계약 확장**(기각 사유 2종 + L2 자료형), **Anthropic 판정
클라이언트**(래퍼 규칙은 OpenAI 래퍼와 동일), **스키마 확장**(층별 판정 컬럼 + CHECK).
그 위에 **판정 모듈**이 서고, 판정 모듈을 **파이프라인**이 배선한다(스위치·루프·피드백).
파이프라인 결과 형태가 확정되면 **저장/복원**과 **API·웹 폼**이 따라온다. 평가는
**픽스처·라벨**(데이터)과 **하네스 확장**(측정 3·대역·리포트)으로 갈라 병렬을 살린다.
**문서 갱신**이 마지막이다 — 문서는 최종 동작을 서술하므로 코드가 먼저 확정돼야 한다.

## 태스크

### T1 — 기각 사유·L2 자료형 계약 확장 (`src/reply_gate/contracts.py`)

- **목표**: `RejectReason` 에 `unsupported_claim`·`contradictory_evidence` 추가.
  L2 판정 자료형 신설: claim 단위 판정(참조·verdict·설명), 근거쌍 모순(`근거 ID A/B·설명`),
  L2 판정 결과(verdict + 사유 + 위 둘). **종합 사유 고정 순서 상수를 contracts.py 에
  신설**하고(L1 4종 먼저, L2 2종 뒤 — 결정론 순서가 계약), `gate.REASON_ORDER` 와의
  접두 정합을 테스트로 고정한다. 주의: 기존 `REASON_ORDER` 는 `gate.py` 소유이고
  gate.py 는 무변경 하드 게이트다 — gate 를 고치거나 contracts→gate import(순환)를
  만들지 말 것. 참고: 한 시도 안에서 두 층 사유는 공존하지 않는다(L2 는 L1 pass 시에만
  실행) — 교차층 순서의 실효는 시도 간 평탄화·집계의 결정론이니, 혼합 시도를 기대하는
  테스트를 쓰지 말 것. 겸사: `contracts.py:5` 모듈 docstring 의 "다음 사이클의 L2"
  시제를 현재형으로 정정한다.
- **불변**: 답변 계약(`claims`/`citation_ids`)·근거 ID 체계·인계 사유 6종 무변경.
  기존 L1 4종의 의미·순서 무변경.
- **검증**: 계약 테스트 갱신(enum 집합·순서), 기존 테스트 녹색.
- 공유 파일 주의: `contracts.py` 는 이 태스크만 쓴다.

### T2 — Anthropic 판정 클라이언트 + 설정 (`src/reply_gate/llm.py`, `config.py`, `.env.example`)

- **목표**: `GenerationClient` 프로토콜을 만족하는 Anthropic 클라이언트를 `llm.py` 에
  추가(구조화 출력 JSON, 전송 오류 1회 재시도, SDK 자동 재시도 차단, 샘플링 파라미터
  미전송, thinking 설정 미전송). `config.py` 에 `ANTHROPIC_API_KEY`(비밀, 기본 빈 값),
  `L2_ENABLED`(기본 켜짐), 판정 모델(기본 `claude-sonnet-5`)·effort(조정 가능 기본값) 추가.
  `.env.example` 에 키 이름만 추가. **`anthropic` SDK 의존성을 `pyproject.toml`(+`uv.lock`)
  에 추가**(의존성 선언은 pyproject 단독 소유 — standards 소유권 표).
- **생각-근거**: 래퍼를 `llm.py` 에 두는 이유 — 재시도 정책·자동 재시도 차단 검사를
  한 파일에서 유지한다(파일 경계 문서도 T10 에서 "LLM 호출 래퍼"로 갱신).
  Anthropic SDK 의 현행 규약은 구현 시점에 SDK 문서로 재확인한다 — 기억으로 쓰지 않는다.
  레퍼런스로 선확인된 사실: 구조화 출력은 `output_config={"format": {"type":
  "json_schema", ...}}` (톱레벨 `output_format` 은 deprecated), 스키마는
  `additionalProperties: false`+`required` 필수·`minItems` 류 미지원,
  `max_retries=0` 지원(기본 2회 자동 재시도 — 차단 필요성 실재), Sonnet 5 는 기본값
  아닌 샘플링 파라미터 전송 시 400(기본값·미전송은 허용), thinking 미전송 = adaptive
  켜짐이 기본(spec 3 — `max_tokens` 여유 필요), `stop_reason == "refusal"` 처리
  필요(OpenAI 래퍼의 refusal 처리와 동수준).
- **검증**: 자동 재시도 차단 검사·전송 오류 1회 재시도·샘플링 미전송 테스트를
  OpenAI 래퍼와 같은 수준으로. 외부 호출은 전부 목.
- 공유 파일 주의: `config.py` 는 이 태스크만 쓴다.

### T3 — 스키마 확장 (`db/schema.sql`)

- **목표**: `inquiry_attempts` 에 층별 판정 컬럼(L1 verdict·L1 사유, L2 verdict·L2 사유
  — L2 미실행 null, claim 단위 판정·근거쌍 모순 jsonb), `inquiries` 에 판정 토큰
  분리 컬럼(입력/출력). CHECK: 기존 사유 enum 에 L2 사유 2종 추가, "L1 reject 인데
  L2 판정 존재" 불가, "종합 pass 인데 어느 층 reject" 불가, 층별 pass ⟺ 그 층 사유 0건.
- **불변**: `attempt_no BETWEEN 1 AND 2`, 기존 테이블·근거 스냅샷 CHECK, read-only 권한.
- **주의**: 새 컬럼은 **null 허용(또는 default)** 이어야 한다 — T3 시점의 `records.py` 는
  새 컬럼을 모르고, 볼륨 재생성은 1회뿐이다. 채움 강제는 T6 의 코드·테스트 몫.
  선택: "종합 사유 = L1 사유 ∥ L2 사유" 형 CHECK 를 추가하면 모순 상태가 양방향으로 닫힌다.
  필수 CHECK 하나 더(spec 5-1): L2 판정이 null 이면 L2 부속(사유·claim 판정·모순쌍)도
  null/빈 값. 작성 주의: Postgres CHECK 는 NULL 결과를 통과시키므로 "층별 pass ⟺ 그 층
  사유 0건"은 null 스코프를 명시해 쓴다. 겸사: `db/schema.sql:168` 의 "차기 L2" 주석
  시제 정정.
- **검증**: 볼륨 재생성 후 스키마 적용이 성공하고 기존 db 테스트가 새 스키마에서 녹색.
- 공유 파일 주의: `db/schema.sql` 은 이 태스크만 쓴다.

### T4 — 판정 모듈 (`src/reply_gate/judge.py` 신규)

- **목표**: 시도(초안)당 1회 배치 판정(spec 4-1 — 재생성 초안은 다시 판정하므로 문의당
  최대 2회). 입력 조립(뒷받침=claim별 인용 근거 `evidence_text`,
  모순=수집 근거 전체), 한국어 판정 프롬프트(spec 4-4·4-5 의미 정책을 지시로 옮김),
  구조화 출력 파싱, 형식 불일치 1회 재시도, L2 판정 결과(T1 자료형) 산출.
  전송 오류(`LLMCallError`)는 그대로 위로(호출자가 `llm_call_failed` 매핑).
- **경계**: 이 모듈은 판정만 한다 — 파이프라인 제어·저장·재생성은 하지 않는다.
  `gate.py` 를 import 하지 않는다. `unsupported_claim` 은 **근거-주장 정합**만 본다 —
  문의-답변 관련성은 판정 범위 밖이고, 프롬프트로 범위를 넓혀 "고치지" 않는다
  (L2 사유 2종이 전부라는 계약 — 관련 리스크는 handoff 의 '주제 바꿔치기' 항목).
- **검증**: 의미 정책 4분면 각각(주제 인접 기각 / 정면 "없다" 통과 / 모순 비명시 기각 /
  모순 명시 통과)의 파싱·판정 단위 테스트(목), 형식 불일치 재시도·재실패 경로,
  미인용 근거만 있는 모순 검출. 양성 대조(정상 통과) 필수.

### T5 — 파이프라인 배선 + 재생성 피드백 확장 (`src/reply_gate/pipeline.py`, `draft.py`, `api.py` 의 503 조건)

- **목표**: L1 pass 후 스위치가 켜져 있으면 T4 판정 호출. L2 기각 → 사유 + claim 단위
  상세를 피드백으로 재생성 1회(같은 근거) → L1 → L2 → 재기각 시
  `escalated(rejected_twice)`. `AttemptRecord` 를 층별 판정 구조로 확장 — **새 필드는
  전부 기본값(None) 필수**: `records._load_attempts` 가 이 생성자를 직접 부르므로
  기본값이 없으면 T6 전까지 records 가 깨진다. `draft.py` 재생성 인터페이스를 L2 상세를
  받도록 확장(SQL 검증기 선례의 "코드+안내" 형태). L2 실패는 `llm_call_failed` + 실패
  단계 `l2_judge` — 단 **키 부재(`MissingCredentialsError` 계열)는 이 매핑에 삼키지 않고
  전파**한다. `ANTHROPIC_API_KEY` 부재 + 스위치 켜짐 → 503 은 **eager**: POST 처리 진입
  시 선검사(spec 4-7 — POST 경로 한정, 조회·422 경로 무관). **구현 함정**: 이 검사는
  POST 핸들러 안에서 키 존재만 본다 — `get_service`(Depends)는 GET 라우트와 공유라
  (`api.py:287-303`) Depends·조립 시점에 넣으면 키 없는 환경에서 조회가 죽는다. 판정
  클라이언트 조립 자체는 `_LazyGenerationClient` 선례(`api.py:232-236`)대로 lazy 유지.
  **배선 규칙(fail-open 차단, spec 2)**: `InquiryPipeline` 은 스위치 켜짐 + 판정자
  미배선이면 **조립 시점 명시적 오류**다 — 기본값 None 이 조용히 L2 를 끄는 구현을
  금지한다(테스트가 덜 깨지는 쉬운 구현이 정확히 이 함정이다). 하위호환은
  `build_pipeline` 이 진다: 설정 기반으로 lazy 판정자를 항상 배선하므로 api·
  `scripts/evaluate.py` 호출은 T9 전에도 안 깨진다(전체 녹색 게이트가 태스크마다 돈다).
  **테스트 파급은 e2e 2건이 아니라 전반이다**: `InquiryPipeline` 을 직접 조립하며 L1
  pass 에 도달하는 기존 파이프라인·API 테스트 전부가 L2-off 설정 명시 주입 또는 판정
  대역 주입을 요구한다. 그중 대역 e2e 2건(`tests/test_evaluation.py` 의 골든셋 30건·
  미끼 기각 재현)은 **L2 꺼짐 설정을 명시 주입**해 사이클 1 동작으로 녹색을 유지한다 —
  L2-on 대역 커버리지는 T9 이 판정 대역과 함께 복원한다. T5~T9 사이 `--stub-llm` CLI 는
  L2 기본 켜짐 + 판정 대역 부재로 **명시적 오류로 죽는** 알려진 일시 공백이다(조용히
  사이클 1 동작으로 돌지 않는다 — 게이트는 이 경로를 돌리지 않고, T9 이 닫는다).
- **불변**: 시도 상한은 `for` 루프 범위, 재생성은 같은 근거, 초안 전 인계는 초안 미진입,
  인프라 오류는 인계로 바꾸지 않음.
- **검증**: L2 기각→재생성→통과, 재기각→인계, 스위치 꺼짐(사이클 1 동작 동일),
  L1 reject 시 L2 미호출, L2 전송 실패→`llm_call_failed`, 형식 불일치 재시도 소진→인계,
  503 경로(**키 부재가 `escalated(llm_call_failed)` 로 집계되지 않고 처리 기록도 남지
  않음을 함께 검사**). 토큰 집계(판정 토큰이 생성 합산에 섞이지 않음). **기본 배선이
  Anthropic 계열임을 검사하는 구조 테스트**(하드 게이트 2)와 **스위치 켜짐 + 판정자
  미배선 = 명시적 오류** 검사를 여기서 고정한다.

### T6 — 처리 기록 저장/복원 확장 (`src/reply_gate/records.py`)

- **목표**: 층별 판정·claim 단위 판정·근거쌍 모순·판정 토큰의 저장→복원 왕복.
  L2 미실행 null 왕복 포함.
- **검증**: 왕복 동등성 테스트(`db` 마커), 새 CHECK 위반이 실제로 거부되는 음성 케이스.

### T7 — API 응답·웹 폼 (`src/reply_gate/api.py`, `src/reply_gate/templates/form.html`)

- **목표**: `attempts[]` 에 층 구분·L2 상세 키(미실행 null, 키 항상 존재),
  `metrics.tokens` 에 판정 토큰 분리 키 신설(기존 생성 합산 의미 불변).
  웹 폼에 층 배지·claim 단위 판정·근거쌍 모순 표시(판정 과정 먼저 원칙, 외부 CDN 금지).
- **검증**: 응답 스키마 테스트(키 존재 규칙), `GET /inquiries/{id}` 재구성 동등성,
  웹 폼 렌더 확인.

### T8 — 판정 픽스처 + 골든셋 라벨 (`data/judge_fixtures.jsonl` 신규, `data/golden_set.jsonl`)

- **목표**: 판정 픽스처 셋 신설 — 단위는 claim 집합(배치 호출과 동일). spec 6-1 의
  4분면(주제 인접 기각 기대 / 정면 "없다" 통과 기대 / 모순 비명시 기각 기대 / 모순 명시
  통과 기대)과 정상 통과·혼합 케이스 포함, 사유별 양성·음성 짝. 골든셋 라벨 변경은
  정확히 **케이스 3건**(G17·G01·G02 — G01·G02 는 로더 검증 때문에 `statuses` 와
  `escalation_reasons` 를 함께 고치고, 두 케이스의 note 문구(사이클 1 시제)도 현재형으로
  정정하므로 케이스당 필드 diff 는 3개일 수 있다 — spec 6-2):
  G17 허용 인계 사유에 `no_evidence` 추가, G01·G02 허용 결과에
  `escalated(rejected_twice)` 추가. **다른 라벨·조항·픽스처는 건드리지 않는다.**
- **검증**: **자체 형식 검사**(JSON 파싱·필수 키·4분면 커버) — 로더는 T9(세 파도 뒤)
  산출물이라 T8 시점에 소비 확인이 불가능하다; 로더 소비 확인은 T9 검증으로 미룬다.
  라벨 diff 가 3건뿐임을 확인.

### T9 — 평가 하네스 확장 (`src/reply_gate/evaluation.py`, `scripts/evaluate.py`, `src/reply_gate/testing.py`)

- **목표**: **측정 3** 신설 — 판정 픽스처를 실제 판정 모델로 흘려 검출률·오탐률·사유
  일치 산출, `--live` 로만 실행, 리포트에 확률층·과금 명시, 목표치는 "미확정(실측 후)"로
  표기(기존 TARGETS 3종 무변경, 미측정=판정 없음 규칙 유지). 결정론 판정 대역을
  `testing.py` 에 추가해 `--stub-llm` 배관 검증에 L2 포함하고, **T5 가 꺼둔 대역 e2e 의
  L2-on 커버리지를 복원**한다(스위치 켬 + 판정 대역 주입). **키 선검사**: `--live` + L2
  켜짐이면 측정 시작 전에 `ANTHROPIC_API_KEY` 도 검사한다(spec 6-3 — 측정 도중 키 부재
  크래시는 과금 후 산출물 없음). 판정 키 부재면 측정 2·3 **모두 skip**(사유 기록) —
  L2 를 꺼서 측정 2 만 돌리는 강등 실행 금지(spec 6-3). 겸사: `evaluation.py:766` 의
  "다음 사이클(L2·임계값)" 주석 시제 정정. **측정 2 집계·리포트 토큰 표에 판정 계열 분리 신설**
  (생성·임베딩·판정 — `GoldenOutcome`/`PipelineAgreement` 확장). 실행 조건에 L2 켜짐
  여부·판정 모델 기록. 라이브 리포트 이름(spec 6-3): L2 켜짐 실측의 기본 stem 을 코드가
  `evaluation-live-l2-<n>` 으로 **자동 넘버링**하고, 이름 가드에 **"L2-on 실측 ⇔ l2
  접두" 차원을 명시 스템에도 양방향 추가**하며, 빈 이름 제안(`_next_free_live_stem`)도
  계열을 따르게 한다(기존 양방향 불변식·gitignore 추적 패턴을 만족하는 `evaluation-live`
  접두 유지가 요건). 미끼 기각 재현율 정의(시도 중 최소 1건 기각, 층 무관) 무변경 —
  L2 기각 자동 포함을 테스트로 고정.
- **검증**: 측정 3 대역 실행, 리포트 스키마(md/json — 판정 토큰 계열 포함), 미실행 시
  "미실행+사유" 규칙, 이름 가드 회귀(비실측이 `evaluation-live-l2-*` 거부 + L2-on 실측이
  비-l2 라이브 이름 거부 + L2-off 실측이 l2 이름 거부), T8 픽스처 로더 소비 확인.
- 공유 파일 주의: `evaluation.py`·`scripts/evaluate.py` 는 이 태스크만 쓴다.

### T10 — 문서 갱신 (docs/, AGENTS.md — README 제외)

- **목표**: spec 7절 목록 그대로 — business-rules(L2 판정 규칙 절, `rejected_twice`
  재정의, 2층 구조 현재형, "단일 실패점" 문구를 "스위치로 끌 수 있다"로, **상태 전이
  도표·엔티티 절의 "시도 = L1 판정" 문구**), contracts(층 구분 키·판정 토큰 키·사유
  집합·503 조건(eager)·답변 계약 현재형, **지연 실측치에 L2 꺼짐 조건 병기**),
  security(Anthropic 전송 명시 — 합성 데이터 한정, 실운영 재결정 목록, **판정 프롬프트
  주입 표면 한 줄**), **standards(하드 게이트 3 산출물 목록에 판정 JSON, 재시도 상한 표
  L2 2행, SDK 자동 재시도 문구 provider 중립화, 모듈 경계에 judge.py)**, **루트
  CLAUDE.md·AGENTS.md(절대 규칙 3 — 두 파일 동일 유지)**, operations(둘째 키·볼륨
  재생성·스위치), architecture(대표 흐름 L2 단계 — **단계 번호를 밀지 않는다**, spec 7 —
  ·구성요소 지도·전체 구성 다이어그램·외부 의존 절·경계 절), AGENTS 두 파일(새 모듈
  경계·불변식, "live 마커" 문장 정정 — 마커 등록 자체는 pyproject 에 실존하고, 실존하지
  않는 것은 그 마커를 쓰는 테스트와 기본 제외 장치다; 문장을 현실에 맞게 고치고 마커
  등록 유지 여부는 실행자 판단), **결정 0007 신설 + decisions/index.md**(판정 모델
  선택·배치 판정·fail-closed·"없다" 비대칭 — handoff 의 의도 절을 durable 기록으로),
  결정 0006 문구 한정 수정, `.dryforge` 는 건드리지 않음. 코드 주석 시제 정정 1건:
  `gate.py:193` 의 "다음 사이클 L2" — 하드 게이트 1 완화(동작 무변경·주석 시제 정정만
  허용, 승인됨) 범위다. contracts.py·schema.sql·evaluation.py 의 같은 시제 정정은
  각 소유 태스크(T1·T3·T9) 몫.
  **README·status 갱신은 통째로 라이브 실측 후 단계다**(수치·기능 서술 모두 —
  전시 문서의 서술과 실측값이 엮여 있어 나눠 갱신하면 어긋난다). 실측 전 선반영 금지.
- **검증**: 문서 상호 참조·절 이름 실존 확인 + **단계 번호를 참조하는 코드 주석이
  여전히 유효한지 grep 확인**(문서-only 검증 기준).

## 실행 그래프

```yaml
tasks:
  - id: T1
    depends: []
    risk: RISKY        # 사유 집합·순서가 계약 — 결정론 순서 불변식
  - id: T2
    depends: []
    risk: RISKY        # 재시도 규칙·자동 재시도 차단 불변식
  - id: T3
    depends: []
    risk: MECHANICAL   # DDL 이지만 CHECK 로 모순 상태를 막는 제약 포함
  - id: T4
    depends: [T1, T2]
    risk: RISKY        # 의미 정책 4분면·형식 불일치 반경이 명시 규칙
  - id: T5
    depends: [T1, T4]
    risk: RISKY        # 루프 상한·실패 정책·스위치 — 상태 조정 규칙
  - id: T6
    depends: [T3, T5]
    risk: MECHANICAL
  - id: T7
    depends: [T5, T6]
    risk: MECHANICAL
  - id: T8
    depends: [T1]
    risk: MECHANICAL
  - id: T9
    depends: [T4, T5, T8]
    risk: RISKY        # 미실행≠0·이름 가드·재현율 정의 불변 등 리포트 정직성 규칙
  - id: T10
    depends: [T5, T6, T7, T9]
    risk: NONE
regen_barriers:
  - { after: [T3], run: "docker compose down -v && docker compose up -d --wait && uv run python -m scripts.seed_orders" }
```

- 파도(참고): {T1,T2,T3} → {T4,T8} → {T5} → {T6} → {T7,T9} → {T10}.
- T3 뒤 재생성 장벽: 스키마 변경은 볼륨 재생성으로만 반영된다(마이그레이션 도구 없음).
  정책 인덱싱(`scripts.index_policies`)은 임베딩 과금이 있으므로 장벽에 넣지 않는다 —
  db 테스트는 자체 픽스처로 돌고, 인덱싱은 라이브 실측·실동작 확인 전에 1회 수행한다.
- 공유 파일: `contracts.py`(T1)·`config.py`(T2)·`schema.sql`(T3)·`evaluation.py`(T9)·
  docs(T10)는 각 단독 소유다. `pipeline.py`/`draft.py`/`api.py` 는 T5 가, `api.py` 의
  응답 조립·폼은 T7 이 쓴다 — T5 는 503 조건까지만, 응답 스키마는 T7 만 손댄다.
  `tests/test_evaluation.py` 의 대역 e2e 2건은 T5(L2 꺼짐 주입)와 T9(L2-on 복원)이
  **순서대로** 손댄다 — 파도가 달라 충돌 없음.
- 라이브 실측(측정 2+3, L2-on 3회)은 그래프 밖이다 — 과금이므로 완료 게이트 통과 후
  사용자 요청 시 실행하고, 그 후 README·status 실측값을 반영한다(handoff 하드 게이트).
