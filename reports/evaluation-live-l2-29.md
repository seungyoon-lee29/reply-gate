# Reply-Gate 평가 리포트

측정은 **결정론 층과 확률 층을 분리**한다 — LLM 비결정성이 게이트 자체의 정확도
수치를 흔들지 못하게 하기 위해서다.

## 실행 조건

- 실행 시각(UTC): `2026-08-21T04:02:46+00:00`
- 생성 LLM: OpenAI `gpt-5.6-terra` (effort=기본값)
- 임베딩: OpenAI `text-embedding-3-small` (1536차원)
- L2 판정: 켜짐 / 판정 모델: Anthropic `claude-sonnet-5` (effort=기본값)
- 검색 전략: vector+rewrite · 임베딩 1536차원
- 유사도 임계값: 0.3 / top k: 5
- L1 픽스처: 35건 (`data/l1_fixtures.jsonl`)
- 골든셋: 30건 (`data/golden_set.jsonl`)
- 판정 픽스처: 11건 (`data/judge_fixtures.jsonl`)
- OPENAI_API_KEY 설정 여부: 설정됨
- ANTHROPIC_API_KEY 설정 여부: 설정됨
- 과금 실행: 예 · 측정 범위: full

### 조건 지문 (대조 가능성)

선언 없이 달라진 조건끼리는 대조하지 않는다. **선언된 실험 변인**은 대조를 진행하고
차이 목록을 병기한다.

| 항목 | 값 | 선언 |
| --- | --- | :---: |
| `label_version` | golden-79e0da454f3a |  |
| `retrieval_labels_version` | labels-7fb4ceef7aa6 |  |
| `acceptance_cut` | 0.3 |  |
| `abstention_gate_statistic` | rank1_minus_rank_k_spread |  |
| `abstention_tau` | 0.06 |  |
| `abstention_undefined_policy` | abstain-771b954a3abd |  |
| `query_rewrite` | on |  |
| `retrieval_order` | db=[embedding <=> %s, evidence_id] · merge=[a>b>c] |  |
| `embedding_model` | text-embedding-3-small |  |
| `embedding_dimensions` | 1536 |  |
| `top_k` | 5 |  |
| `generation_model` | gpt-5.6-terra |  |
| `judge_model` | claude-sonnet-5 |  |
| `judge_effort` | 기본값 |  |
| `judge_thinking` | 미전송(계열 기본) |  |
| `judge_prompt_version` | judge-e26f6c8e0b84 |  |
| `judge_fixture_version` | fixture-2b8f33392b36 |  |
| `judge_prompt_caching` | off |  |
| `draft_rule_version` | draftrules-a0e153753e62 |  |
| `l1_fixture_version` | l1fixture-6eb235f9dcd0 |  |
| `sql_guard_version` | sqlguard-1de0777824a4 |  |
| `measurement_scope` | full |  |
| `run_completion` | 중단 없음 |  |
| `generation_effort` | 기본값 |  |
| `l2_enabled` | on |  |

## 목표치 대비

2026-08-05 확정(`docs/tracking/decisions/0006-지표-목표치를-실측-뒤에-확정한다.md`).
미측정 지표와 대역으로 낸 확률 층 수치는 판정하지 않는다.

**합산 일치율은 달성 목표가 아니라 하한 경보선이다**(결정 0006 재확정) — 그 아래로
내려가면 무언가 크게 부서졌다는 신호일 뿐이고, **사이클 성패 판정은 케이스 단위**
(회귀 가드의 비악화 판정 + 케이스별 귀인)가 맡는다. 합산은 병기다.

| 지표 | 성격 | 경계 | 실측 | 판정 |
| --- | --- | --- | ---: | :---: |
| 측정 1 구조적 오류 검출률 | 달성 목표 | ≥ 100% | 100.0% | 달성 |
| 측정 1 정상 초안 오탐률 | 달성 목표 | ≤ 0% | 0.0% | 달성 |
| 측정 2 허용 결과 집합 대비 일치율 | 하한 경보선 | ≥ 75% | 100.0% | 달성 |

## 측정 1 — L1 게이트 단위 정확도 (결정론)

고정 초안+근거 쌍에 `gate.evaluate_draft` 를 직접 적용했다. **LLM 호출 0회, 100% 재현.**
신뢰성 서사의 헤드라인 수치는 이것이다.

- 픽스처 총수: **35건** (위반 26 / 정상 9)
- **구조적 오류 검출률: 100.0%** (26/26)
- **정상 초안 오탐률: 0.0%** (0/9)
- 사유 목록까지 정확히 일치: 100.0% (35/35)

### 사유 4종별 내역

| 사유 | 기대 픽스처 | 검출 | 검출률 | 오발화(기대하지 않은 발화) |
| --- | ---: | ---: | ---: | ---: |
| `schema_violation` | 9 | 9 | 100.0% | 0 |
| `missing_citation` | 4 | 4 | 100.0% | 0 |
| `invalid_citation` | 5 | 5 | 100.0% | 0 |
| `pii_detected` | 13 | 13 | 100.0% | 0 |

모든 픽스처가 기대 판정·기대 사유 목록과 일치했다.

## 측정 2 — 파이프라인 판정 일치율 (end-to-end)

- 골든셋 30건 처리
- **허용 결과 집합 대비 일치율: 100.0%** (30/30) — **초안 전 인계 경로 포함이며 L1 판정만의 지표가 아니다.**
- **미끼 문의(reject_bait)의 기각 재현율: 20.0%** (1/5) — 목표 없는 관측값이다(결정 0006·0008).
- 정상 PII 에코 감시 케이스: 1건 중 금지 사유 발화 0건
- 지연 p50: 8697 ms / p95: 40254 ms (파이프라인 `run` 의 벽시계 시간 — 처리 기록 저장은 포함하지 않는다)
- 검색 단계 폴백: 0건 (전건 재작성 성공 — 검색 구성이 실행 조건 그대로 돌았다)
- 기권 게이트 통계량 미정의: 0건 (전건에서 통계량이 정의됐거나 게이트가 돌지 않았다)

### 문의 1건당 토큰 (생성·임베딩·판정·검색 구분)

provider 와 단가가 다른 계열을 합산하면 건당 비용 지표가 무너진다 — 네 계열은
끝까지 분리해서 센다. L2 미실행이면 판정 계열은, 재작성을 쓰지 않았으면 검색
계열은 0 이다.

| 계열 | 합계 | 건당 |
| --- | ---: | ---: |
| 생성 입력 | 30750 | 1025.0 |
| 생성 출력 | 3973 | 132.4 |
| 생성 소계 | 34723 | 1157.4 |
| 임베딩 | 771 | 25.7 |
| 판정 입력 | 74295 | 2476.5 |
| 판정 출력 | 15828 | 527.6 |
| 판정 소계 | 90123 | 3004.1 |
| 검색 입력 | 3810 | 127.0 |
| 검색 출력 | 409 | 13.6 |
| 검색 소계 | 4219 | 140.6 |
| **합산** | 129836 | **4327.9** |

- 생성 계열 프롬프트 캐시(생성 입력 토큰과 별도 칸): write 3152 / read 0 — 위 '생성 입력'은 **캐시 적중분을 포함한 총 입력**이다(빼지도 더하지도 않는다)
- 검색 계열 프롬프트 캐시(검색 입력 토큰과 별도 칸): write 0 / read 0 — 위 '검색 입력'은 **캐시 적중분을 포함한 총 입력**이다(빼지도 더하지도 않는다)

### 단계별 지연 (구간 아홉)

밖으로 나가는 호출 여섯(의도 분류 · 질의 재작성 · 질의 임베딩 · 조회문 생성 ·
초안 생성 · L2 판정)과 코드만 도는 셋(벡터 검색 · 조회 실행 · 게이트 판정)이다.
한 구간의 시간은 그 구간의 **총 벽시계**이고 재시도·형식 실패·예외로 죽은 호출을
포함한다. **미측정은 0 이 아니다** — 돌지 않은 구간은 분모에서 빠진다(0 을 섞어
평균 내면 그 구간이 실제보다 빨라 보인다). 재생성이 돈 문의는 초안·게이트·판정
구간이 시도별로 쌓이고, 시도별 값은 리포트 JSON 의 `attempt_durations` 에 있다.

| 구간 | 측정 케이스 | 미측정 | 합계 ms | 평균 ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `intent` | 30 | 0 | 46778.2 | 1559.3 | 1268.9 | 2662.8 |
| `query_rewrite` | 20 | 10 | 28667.7 | 1433.4 | 1369.1 | 1810.0 |
| `inquiry_embedding` | 20 | 10 | 4973.5 | 248.7 | 189.8 | 469.3 |
| `policy_search` | 20 | 10 | 537.1 | 26.9 | 25.1 | 43.2 |
| `sql_generation` | 9 | 21 | 14544.5 | 1616.1 | 1611.3 | 2161.3 |
| `sql_execution` | 11 | 19 | 87.3 | 7.9 | 6.5 | 20.8 |
| `draft` | 22 | 8 | 54201.3 | 2463.7 | 1920.7 | 4358.6 |
| `gate` | 22 | 8 | 13.4 | 0.6 | 0.5 | 1.1 |
| `l2_judge` | 22 | 8 | 194754.2 | 8852.5 | 6506.7 | 33173.3 |

### 종결 분포

- 최종 상태: answered 22건, escalated 8건
- 인계 사유: missing_order_ref 2건, no_evidence 4건, order_not_found 2건

### 케이스별 채택 근거

- `G01`: `policy:refund:2-1`, `policy:refund:2-5`, `policy:refund:2-2`, `policy:exchange:3-1`, `policy:refund:2-4`
- `G02`: `policy:shipping:1-4`, `policy:shipping:1-3`, `policy:shipping:1-7`, `policy:shipping:1-6`, `policy:shipping:1-2`
- `G03`: `policy:exchange:3-2`, `policy:exchange:3-1`, `policy:exchange:3-5`, `policy:exchange:3-4`, `policy:exchange:3-3`
- `G04`: `policy:exchange:3-1`, `policy:support:4-6`
- `G05`: `policy:support:4-3`, `policy:support:4-4`, `policy:support:4-5`, `policy:support:4-2`, `policy:exchange:3-2`
- `G06`: `policy:shipping:1-6`, `policy:shipping:1-5`, `policy:exchange:3-2`, `policy:support:4-3`, `policy:support:4-5`
- `G07`: `policy:support:4-7`
- `G08`: `policy:shipping:1-7`, `policy:exchange:3-4`, `policy:exchange:3-3`, `policy:refund:2-7`, `policy:refund:2-5`
- `G09`: `sql:522a454a-2468-4f1b-b5af-ab003db8f624:1`
- `G10`: `sql:fb3cca91-1ed9-4653-b06f-3e8d3c67f45a:1`
- `G11`: `sql:b661aae1-d4e4-4ca4-9eb3-affdfc73a3b2:1`
- `G12`: `policy:refund:2-4`, `policy:support:4-3`, `policy:refund:2-1`, `policy:refund:2-6`, `policy:refund:2-2`, `sql:c90f8081-9913-4b4c-a0db-40ca9284f3d5:1`
- `G13`: `sql:bb434644-e022-4ae8-a575-94b1d4e631db:1`
- `G14`: `sql:234dc51f-c3b3-473d-89b7-bf9e4bda8568:1`
- `G15`: `policy:refund:2-5`, `policy:exchange:3-3`, `policy:shipping:1-7`, `policy:shipping:1-6`, `policy:shipping:1-3`, `sql:3adcaac6-6c01-40dd-a59a-40ce5eaa8c05:1`
- `G16`: `policy:support:4-1`, `policy:support:4-2`, `policy:support:4-5`, `policy:support:4-4`, `policy:support:4-3`
- `G17`: `policy:support:4-1`, `policy:support:4-2`, `policy:support:4-5`, `policy:support:4-4`, `policy:support:4-7`
- `G18`: `policy:refund:2-6`, `policy:support:4-2`, `policy:refund:2-4`, `policy:refund:2-2`, `policy:shipping:1-7`
- `G19`: `policy:support:4-2`, `policy:support:4-1`, `policy:support:4-5`, `policy:support:4-4`, `policy:support:4-7`
- `G20`: `policy:support:4-2`, `policy:support:4-1`, `policy:support:4-5`, `policy:support:4-4`, `policy:support:4-7`
- `G21`: 없음
- `G22`: 없음
- `G23`: 없음
- `G24`: 없음
- `G25`: 없음
- `G26`: 없음
- `G27`: 없음
- `G28`: 없음
- `G29`: `sql:8bd0bd4b-964c-476a-9a58-fc276659b5d4:1`
- `G30`: `sql:c1f6e66b-1de2-466b-9f36-9ba61667a8ea:1`

모든 문의가 허용 결과 집합 안에서 종결했다.

### 검색 실패 / 생성 문제 분해

- 검색 정답 라벨: `data/retrieval_labels.jsonl`
- 생성 문제: **2건** — 정답 조항을 **전부** 채택했지만 기각·인계
- 검색 실패 합계: **3건** (전부 누락 3건 · 일부 누락 0건)
- 근거 없이 답변 확정: **7건** — 정답 조항이 빠진 채 게이트를 통과했다
- 빈 정답 정상 인계: **5건** — 앞의 분류에 포함하지 않음
- 빈 정답 비정상 종결: **0건** — 정상 인계 집계에서 제외

#### 케이스별 판정 근거

- `G01`: **생성 문제** — 정답 근거 `policy:refund:2-1`, `policy:refund:2-2` / 채택 근거 `policy:refund:2-1`, `policy:refund:2-5`, `policy:refund:2-2`, `policy:exchange:3-1`, `policy:refund:2-4` / 최종 answered·라벨 일치
- `G09`: **근거 없이 답변 확정** — 정답 근거 `policy:shipping:1-5` / 채택 근거 `sql:522a454a-2468-4f1b-b5af-ab003db8f624:1` / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G10`: **근거 없이 답변 확정** — 정답 근거 `policy:shipping:1-5` / 채택 근거 `sql:fb3cca91-1ed9-4653-b06f-3e8d3c67f45a:1` / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G11`: **근거 없이 답변 확정** — 정답 근거 `policy:shipping:1-1` / 채택 근거 `sql:b661aae1-d4e4-4ca4-9eb3-affdfc73a3b2:1` / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G13`: **근거 없이 답변 확정** — 정답 근거 `policy:refund:2-4` / 채택 근거 `sql:bb434644-e022-4ae8-a575-94b1d4e631db:1` / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G14`: **근거 없이 답변 확정** — 정답 근거 `policy:exchange:3-5` / 채택 근거 `sql:234dc51f-c3b3-473d-89b7-bf9e4bda8568:1` / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G18`: **생성 문제** — 정답 근거 `policy:refund:2-4`, `policy:refund:2-6` / 채택 근거 `policy:refund:2-6`, `policy:support:4-2`, `policy:refund:2-4`, `policy:refund:2-2`, `policy:shipping:1-7` / 최종 answered·라벨 일치
- `G21`: **빈 정답 정상 인계** — 정답 근거 없음 / 채택 근거 없음 / 검색 0건 종료
- `G22`: **빈 정답 정상 인계** — 정답 근거 없음 / 채택 근거 없음 / 검색 0건 종료
- `G23`: **빈 정답 정상 인계** — 정답 근거 없음 / 채택 근거 없음 / 검색 0건 종료
- `G24`: **빈 정답 정상 인계** — 정답 근거 없음 / 채택 근거 없음 / 검색 0건 종료
- `G25`: **검색 실패(전부 누락)** — 정답 근거 `policy:shipping:1-5` / 채택 근거 없음 / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G26`: **검색 실패(전부 누락)** — 정답 근거 `policy:shipping:1-1` / 채택 근거 없음 / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G27`: **검색 실패(전부 누락)** — 정답 근거 `policy:shipping:1-5` / 채택 근거 없음 / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G28`: **빈 정답 정상 인계** — 정답 근거 없음 / 채택 근거 없음 / 주문 단계 사전 인계 — 구조적 사유가 이긴다(계약상 정상)
- `G29`: **근거 없이 답변 확정** — 정답 근거 `policy:support:4-5` / 채택 근거 `sql:8bd0bd4b-964c-476a-9a58-fc276659b5d4:1` / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G30`: **근거 없이 답변 확정** — 정답 근거 `policy:support:4-5` / 채택 근거 `sql:c1f6e66b-1de2-466b-9f36-9ba61667a8ea:1` / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)

## 측정 3 — L2 판정 단위 정확도 (확률 층)

고정 claim 집합을 **판정기에 직접** 흘려 기대 판정과 대조했다(무엇으로 판정했는지는
아래 판정 모델 항목이 들고 있다). 측정 1 과 같은 모양의 수치지만 이 실행은
**확률 층이고 과금된다** — 재실행하면 값이 달라진다.
목표치: **없음** (무목표 관측 — 미측정을 미달로도 달성으로도 적지 않는
규칙과 같은 이유로, 경계가 없는 지표에 판정을 붙이지 않는다).

- 판정 모델: Anthropic `claude-sonnet-5` (effort=기본값)
- 실측 여부: 실제 판정 모델(과금)
- 픽스처 총수: **11건** (기각 기대 6 / 통과 기대 5)
- **L2 검출률: 100.0%** (6/6)
- **L2 오탐률: 0.0%** (0/5)
- 사유 목록까지 정확히 일치: 90.9% (10/11)
- claim 단위 판정 일치: 100.0% (19/19)
- 모순 근거쌍: 기대 4건 중 3건 검출 (기대 밖 검출 0건)
- 판정 토큰: 입력 30042 / 출력 5665 (픽스처당 3246.1)
- 판정 호출 지연: 평균 6474.4 ms / p50 6299.1 ms / p95 13459.0 ms (측정 11건 / 미측정 0건 — 재시도와 판정 실패 호출을 포함한 총 벽시계다)
- 판정 프롬프트 캐시(판정 입력 토큰과 별도 칸): write 0 / read 0 — 위 '판정 토큰 입력'은 **캐시 적중분을 제외한 비캐시 입력**이다

### 사유 2종별 내역

| 사유 | 기대 픽스처 | 검출 | 검출률 | 오발화(기대하지 않은 발화) |
| --- | ---: | ---: | ---: | ---: |
| `unsupported_claim` | 4 | 4 | 100.0% | 0 |
| `contradictory_evidence` | 3 | 2 | 66.7% | 0 |

### 기대와 어긋난 픽스처

- `J11` (combined): 기대 reject[unsupported_claim, contradictory_evidence] → 실제 reject[unsupported_claim]

## 회귀 가드 — 이중 기준선 두 줄 보고

- 이번 실측 세트: 2/3회 `evaluation-live-l2-28`, `evaluation-live-l2-29`
  - 조건 불일치로 세트에서 제외: `evaluation-live-l2-27` — 조건이 다르다: judge_prompt_caching: 기준선 `on` → 이번 `off` · measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-26` — 조건이 다르다: judge_prompt_caching: 기준선 `on` → 이번 `off` · measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-25` — 조건이 다르다: judge_prompt_caching: 기준선 `on` → 이번 `off` · measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-24` — 조건이 다르다: judge_model: 기준선 `claude-opus-5` → 이번 `claude-sonnet-5` · measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-23` — 조건이 다르다: judge_model: 기준선 `claude-opus-5` → 이번 `claude-sonnet-5` · measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-22` — 조건이 다르다: judge_model: 기준선 `claude-opus-5` → 이번 `claude-sonnet-5` · measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-21` — 조건이 다르다: judge_model: 기준선 `claude-haiku-4-5` → 이번 `claude-sonnet-5` · measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-20` — 조건이 다르다: judge_model: 기준선 `claude-haiku-4-5` → 이번 `claude-sonnet-5` · measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-19` — 조건이 다르다: judge_model: 기준선 `claude-haiku-4-5` → 이번 `claude-sonnet-5` · measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-18` — 조건이 다르다: judge_prompt_version: 기준선 `judge-7fad677f9ac6` → 이번 `judge-e26f6c8e0b84` · measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-17` — 조건이 다르다: judge_prompt_version: 기준선 `judge-7fad677f9ac6` → 이번 `judge-e26f6c8e0b84` · measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-16` — 조건이 다르다: judge_prompt_version: 기준선 `judge-7fad677f9ac6` → 이번 `judge-e26f6c8e0b84` · measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-15` — 조건이 다르다: measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-14` — 조건이 다르다: measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-13` — 조건이 다르다: measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-12` — 조건이 다르다: judge_fixture_version: 기준선 `fixture-836322e97320` → 이번 `fixture-2b8f33392b36` · `evaluation-live-l2-11` — 조건이 다르다: judge_fixture_version: 기준선 `fixture-836322e97320` → 이번 `fixture-2b8f33392b36` · `evaluation-live-l2-10` — 조건이 다르다: judge_fixture_version: 기준선 `fixture-836322e97320` → 이번 `fixture-2b8f33392b36` · `evaluation-live-l2-6` — 조건이 다르다: acceptance_cut: 기준선 `0.5` → 이번 `0.3` · `evaluation-live-l2-5` — 조건이 다르다: acceptance_cut: 기준선 `0.5` → 이번 `0.3` · `evaluation-live-l2-4` — 조건이 다르다: acceptance_cut: 기준선 `0.5` → 이번 `0.3` · `evaluation-live-l2-3` — 조건 지문이 확인되지 않는다(미상 18개: label_version, retrieval_labels_version, abstention_gate_statistic 외 15개) · `evaluation-live-l2-2` — 조건 지문이 확인되지 않는다(미상 18개: label_version, retrieval_labels_version, abstention_gate_statistic 외 15개) · `evaluation-live-l2-1` — 조건 지문이 확인되지 않는다(미상 18개: label_version, retrieval_labels_version, abstention_gate_statistic 외 15개)
- **판정: 대조 불가** — 승격 기준선(구속) 줄이 결정한다. 직전 라이브 줄은 경보이고 판정을 발동시키지 않는다.
- 승격: 2026-08-19 · 사용자 (사이클 4 · 관측 요약 보고 뒤 확인) — 현 기본값 조합(재작성 켜짐 · 컷 0.30 · 캐싱 없음 · T6 이전 판정 프롬프트)의 첫 풀셋 라이브 3회. 관측 넷을 모두 충족했다 — G17 재작성 유지 3/3 · G18·G01·G02 상충쌍 동시 채택 각 3/3 · G21-G24 판정 층 인계 3/3 · G04 정답 조항 채택 + 정상 답변 3/3. 측정 1 은 앞선 여섯 세트와 동일하고 측정 3 은 3회 중 2회가 동일(1회는 오탐 1건 — 9세트 만의 첫 관측). 새 회귀 가드가 붙은 코드로 산출돼 근거 부분 손실 검사가 온전히 작동하는 첫 세트다. (참조 `data/promoted_baseline.json`)

### 승격 기준선 (구속) — **대조 불가**

- 사유: 선언되지 않은 조건 불일치 — 어긋난 항목: abstention_gate_statistic: 기준선 `미배선` → 이번 `rank1_minus_rank_k_spread` · abstention_tau: 기준선 `미배선` → 이번 `0.06` · judge_fixture_version: 기준선 `fixture-836322e97320` → 이번 `fixture-2b8f33392b36`
- 기준선 산출물: `evaluation-live-l2-7`, `evaluation-live-l2-8`, `evaluation-live-l2-9`
- 기준선 출처: `data/promoted_baseline.json`
- **대조 불가 — 선언 없이 어긋난 항목**: abstention_gate_statistic: 기준선 `미배선` → 이번 `rank1_minus_rank_k_spread` · abstention_tau: 기준선 `미배선` → 이번 `0.06` · judge_fixture_version: 기준선 `fixture-836322e97320` → 이번 `fixture-2b8f33392b36`
- 지문 미상 항목(기준선 또는 이번 실측에 값이 없다 — 같다고도 다르다고도 적지 않는다): `abstention_undefined_policy`, `retrieval_order`, `judge_thinking`, `draft_rule_version`, `l1_fixture_version`, `sql_guard_version`, `run_completion`
- 참고: 측정 3 은 무변경 검사 대상이 아니다 — 판정 층 개선의 성공이 무관한 축의 원복을 발동시키면 안 되기 때문이다. 기준선 100.0% / 이번 100.0% — 두 세트의 값이 같다.
- 참고: 합산 일치율은 하한 경보선이고 사이클 판정은 케이스 단위가 맡는다.

### 직전 라이브 (경보) — **대조 불가**

- 사유: 선언되지 않은 조건 불일치 — 어긋난 항목: judge_prompt_caching: 기준선 `on` → 이번 `off` · measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full`
- 기준선 산출물: `evaluation-live-l2-27`
- 기준선 출처: `reports/ 자동 탐색`
- **대조 불가 — 선언 없이 어긋난 항목**: judge_prompt_caching: 기준선 `on` → 이번 `off` · measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full`
- 지문 미상 항목(기준선 또는 이번 실측에 값이 없다 — 같다고도 다르다고도 적지 않는다): `abstention_undefined_policy`, `retrieval_order`, `judge_thinking`, `draft_rule_version`, `l1_fixture_version`, `sql_guard_version`, `run_completion`
- 참고: 측정 3 은 무변경 검사 대상이 아니다 — 판정 층 개선의 성공이 무관한 축의 원복을 발동시키면 안 되기 때문이다. 기준선 100.0% / 이번 100.0% — 두 세트의 값이 같다.
- 참고: 합산 일치율은 하한 경보선이고 사이클 판정은 케이스 단위가 맡는다.

## 한계 (과장하지 않는다)

- **L1 은 패턴형 PII 만 본다.** 전화번호·이메일·주민등록번호처럼 정규식으로 잡히는 값만
  검사한다. 이름·주소 등 비패턴형 개인정보는 정규식으로 잡을 수 없어 **L1 의 검사 대상이
  아니다**(L2 도 근거-주장 정합만 보므로 대상이 아니다). 검출률 수치를 "개인정보 전반"으로
  읽으면 안 된다.
- **L1 은 내용의 진위를 보지 않는다.** citation 존재·무결성·스키마·PII 만 검사한다.
  근거를 인용했지만 내용이 근거와 어긋나는 답변은 L1 이 아니라 **L2 의미 검증**이 잡는다 —
  **L2 를 끈 실행에는 그 층이 통째로 없다**(실행 조건의 "L2 판정" 항목을 함께 읽어야 한다).
- **측정 2·3 은 확률 층이다.** 초안 생성과 판정이 비결정론이므로 재실행하면 값이 달라지고
  실제 모델 실행은 과금된다. 측정 1 만 100% 재현된다.
- **측정 2 의 일치율에는 초안 전 인계 경로가 포함된다** — 근거 0건·주문번호 없음·주문
  없음으로 끝난 건도 분모에 들어가므로 **L1 판정만의 지표가 아니다**.
- **측정 3 에는 목표치가 없다.** 무목표 관측이므로 이 수치에는 달성·미달 판정이 붙지 않는다.

## 이월 (다음 사이클)

- L1 필터링에 의한 L2 호출 감소율
- RAG 검색 품질 단계별 개선표
- 비패턴형 개인정보(이름·주소) 검출
