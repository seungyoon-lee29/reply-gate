# Reply-Gate 평가 리포트

측정은 **결정론 층과 확률 층을 분리**한다 — LLM 비결정성이 게이트 자체의 정확도
수치를 흔들지 못하게 하기 위해서다.

## 실행 조건

- 실행 시각(UTC): `2026-08-19T05:25:00+00:00`
- 생성 LLM: OpenAI `gpt-5.6-terra` (effort=기본값)
- 임베딩: OpenAI `text-embedding-3-small` (1536차원)
- L2 판정: 켜짐 / 판정 모델: Anthropic `claude-sonnet-5` (effort=기본값)
- 검색 전략: vector+rewrite · 임베딩 1536차원
- 유사도 임계값: 0.3 / top k: 5
- L1 픽스처: 27건 (`data/l1_fixtures.jsonl`)
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
| `abstention_gate_statistic` | rank1_minus_rank_k_spread | **선언된 실험 변인** |
| `abstention_tau` | 0.06 | **선언된 실험 변인** |
| `query_rewrite` | on |  |
| `embedding_model` | text-embedding-3-small |  |
| `embedding_dimensions` | 1536 |  |
| `top_k` | 5 |  |
| `generation_model` | gpt-5.6-terra |  |
| `judge_model` | claude-sonnet-5 |  |
| `judge_effort` | 기본값 |  |
| `judge_prompt_version` | judge-e26f6c8e0b84 |  |
| `judge_fixture_version` | fixture-836322e97320 |  |
| `judge_prompt_caching` | off |  |
| `measurement_scope` | full |  |
| `generation_effort` | 기본값 |  |
| `l2_enabled` | on |  |

선언된 실험 변인: `abstention_gate_statistic`, `abstention_tau`

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

- 픽스처 총수: **27건** (위반 19 / 정상 8)
- **구조적 오류 검출률: 100.0%** (19/19)
- **정상 초안 오탐률: 0.0%** (0/8)
- 사유 목록까지 정확히 일치: 100.0% (27/27)

### 사유 4종별 내역

| 사유 | 기대 픽스처 | 검출 | 검출률 | 오발화(기대하지 않은 발화) |
| --- | ---: | ---: | ---: | ---: |
| `schema_violation` | 9 | 9 | 100.0% | 0 |
| `missing_citation` | 4 | 4 | 100.0% | 0 |
| `invalid_citation` | 5 | 5 | 100.0% | 0 |
| `pii_detected` | 6 | 6 | 100.0% | 0 |

모든 픽스처가 기대 판정·기대 사유 목록과 일치했다.

## 측정 2 — 파이프라인 판정 일치율 (end-to-end)

- 골든셋 30건 처리
- **허용 결과 집합 대비 일치율: 100.0%** (30/30) — **초안 전 인계 경로 포함이며 L1 판정만의 지표가 아니다.**
- **미끼 문의(reject_bait)의 기각 재현율: 0.0%** (0/5) — 목표 없는 관측값이다(결정 0006·0008).
- 정상 PII 에코 감시 케이스: 1건 중 금지 사유 발화 0건
- 지연 p50: 9159 ms / p95: 25409 ms (파이프라인 `run` 의 벽시계 시간 — 처리 기록 저장은 포함하지 않는다)
- 검색 단계 폴백: 0건 (전건 재작성 성공 — 검색 구성이 실행 조건 그대로 돌았다)

### 문의 1건당 토큰 (생성·임베딩·판정·검색 구분)

provider 와 단가가 다른 계열을 합산하면 건당 비용 지표가 무너진다 — 네 계열은
끝까지 분리해서 센다. L2 미실행이면 판정 계열은, 재작성을 쓰지 않았으면 검색
계열은 0 이다.

| 계열 | 합계 | 건당 |
| --- | ---: | ---: |
| 생성 입력 | 30045 | 1001.5 |
| 생성 출력 | 3815 | 127.2 |
| 생성 소계 | 33860 | 1128.7 |
| 임베딩 | 757 | 25.2 |
| 판정 입력 | 73575 | 2452.5 |
| 판정 출력 | 13393 | 446.4 |
| 판정 소계 | 86968 | 2898.9 |
| 검색 입력 | 3626 | 120.9 |
| 검색 출력 | 394 | 13.1 |
| 검색 소계 | 4020 | 134.0 |
| **합산** | 125605 | **4186.8** |

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
- `G08`: `policy:shipping:1-7`, `policy:exchange:3-3`, `policy:exchange:3-4`, `policy:refund:2-7`, `policy:shipping:1-6`
- `G09`: `sql:20ba0280-3d26-472d-8b39-aa00f07ac8da:1`
- `G10`: `sql:e3d41a1c-b254-411f-97f4-b64f6a6675c2:1`
- `G11`: `sql:8352b48e-5558-459d-926a-955dcbc0d2e3:1`
- `G12`: `policy:support:4-3`, `policy:refund:2-4`, `policy:refund:2-1`, `policy:refund:2-6`, `policy:refund:2-3`, `sql:dd9c6418-dacf-4768-988c-f03af28111c2:1`
- `G13`: `sql:2161d03d-3698-4b6c-a562-119a7a9aeed9:1`
- `G14`: `sql:dacbb0fb-e361-46f7-9de9-804ab6274a9d:1`
- `G15`: `policy:refund:2-5`, `policy:exchange:3-3`, `policy:shipping:1-7`, `policy:shipping:1-6`, `policy:shipping:1-3`, `sql:36c7b710-cfa5-42e5-be5b-bdd483cf21e0:1`
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
- `G29`: `sql:d4e0493f-4021-420c-bd55-72289f1fb846:1`
- `G30`: `sql:3ac0961a-54e6-4327-86a6-ff4b956d902f:1`

모든 문의가 허용 결과 집합 안에서 종결했다.

### 검색 실패 / 생성 문제 분해

- 검색 정답 라벨: `data/retrieval_labels.jsonl`
- 생성 문제: **1건** — 정답 조항을 **전부** 채택했지만 기각·인계
- 검색 실패 합계: **4건** (전부 누락 4건 · 일부 누락 0건)
- 근거 없이 답변 확정: **6건** — 정답 조항이 빠진 채 게이트를 통과했다
- 빈 정답 정상 인계: **5건** — 앞의 분류에 포함하지 않음
- 빈 정답 비정상 종결: **0건** — 정상 인계 집계에서 제외

#### 케이스별 판정 근거

- `G01`: **생성 문제** — 정답 근거 `policy:refund:2-1`, `policy:refund:2-2` / 채택 근거 `policy:refund:2-1`, `policy:refund:2-5`, `policy:refund:2-2`, `policy:exchange:3-1`, `policy:refund:2-4` / 최종 answered·라벨 일치
- `G09`: **근거 없이 답변 확정** — 정답 근거 `policy:shipping:1-5` / 채택 근거 `sql:20ba0280-3d26-472d-8b39-aa00f07ac8da:1` / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G10`: **근거 없이 답변 확정** — 정답 근거 `policy:shipping:1-5` / 채택 근거 `sql:e3d41a1c-b254-411f-97f4-b64f6a6675c2:1` / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G11`: **근거 없이 답변 확정** — 정답 근거 `policy:shipping:1-1` / 채택 근거 `sql:8352b48e-5558-459d-926a-955dcbc0d2e3:1` / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G13`: **근거 없이 답변 확정** — 정답 근거 `policy:refund:2-4` / 채택 근거 `sql:2161d03d-3698-4b6c-a562-119a7a9aeed9:1` / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G14`: **근거 없이 답변 확정** — 정답 근거 `policy:exchange:3-5` / 채택 근거 `sql:dacbb0fb-e361-46f7-9de9-804ab6274a9d:1` / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G21`: **빈 정답 정상 인계** — 정답 근거 없음 / 채택 근거 없음 / 검색 0건 종료
- `G22`: **빈 정답 정상 인계** — 정답 근거 없음 / 채택 근거 없음 / 검색 0건 종료
- `G23`: **빈 정답 정상 인계** — 정답 근거 없음 / 채택 근거 없음 / 검색 0건 종료
- `G24`: **빈 정답 정상 인계** — 정답 근거 없음 / 채택 근거 없음 / 검색 0건 종료
- `G25`: **검색 실패(전부 누락)** — 정답 근거 `policy:shipping:1-5` / 채택 근거 없음 / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G26`: **검색 실패(전부 누락)** — 정답 근거 `policy:shipping:1-1` / 채택 근거 없음 / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G27`: **검색 실패(전부 누락)** — 정답 근거 `policy:shipping:1-5` / 채택 근거 없음 / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G28`: **빈 정답 정상 인계** — 정답 근거 없음 / 채택 근거 없음 / 주문 단계 사전 인계 — 구조적 사유가 이긴다(계약상 정상) / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G29`: **근거 없이 답변 확정** — 정답 근거 `policy:support:4-5` / 채택 근거 `sql:d4e0493f-4021-420c-bd55-72289f1fb846:1` / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)
- `G30`: **검색 실패(전부 누락)** — 정답 근거 `policy:support:4-5` / 채택 근거 `sql:3ac0961a-54e6-4327-86a6-ff4b956d902f:1` / **정책 검색 미실행**(의도 분류가 `order` 로 라우팅해 정책 검색이 실행되지 않았다)

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
- 판정 토큰: 입력 29928 / 출력 6552 (픽스처당 3316.4)

### 사유 2종별 내역

| 사유 | 기대 픽스처 | 검출 | 검출률 | 오발화(기대하지 않은 발화) |
| --- | ---: | ---: | ---: | ---: |
| `unsupported_claim` | 4 | 4 | 100.0% | 0 |
| `contradictory_evidence` | 3 | 2 | 66.7% | 0 |

### 기대와 어긋난 픽스처

- `J11` (combined): 기대 reject[unsupported_claim, contradictory_evidence] → 실제 reject[unsupported_claim]

## 회귀 가드 — 이중 기준선 두 줄 보고

- 이번 실측 세트: 2/3회 `evaluation-live-l2-10`, `evaluation-live-l2-11`
  - 조건 불일치로 세트에서 제외: `evaluation-live-l2-6` — 조건이 다르다: acceptance_cut: 기준선 `0.3` → 이번 `0.5` · `evaluation-live-l2-5` — 조건이 다르다: acceptance_cut: 기준선 `0.3` → 이번 `0.5` · `evaluation-live-l2-4` — 조건이 다르다: acceptance_cut: 기준선 `0.3` → 이번 `0.5` · `evaluation-live-l2-3` — 조건 지문이 확인되지 않는다(미상 11개: label_version, retrieval_labels_version, abstention_gate_statistic 외 8개) · `evaluation-live-l2-2` — 조건 지문이 확인되지 않는다(미상 11개: label_version, retrieval_labels_version, abstention_gate_statistic 외 8개) · `evaluation-live-l2-1` — 조건 지문이 확인되지 않는다(미상 11개: label_version, retrieval_labels_version, abstention_gate_statistic 외 8개)
- **판정: 보류** — 승격 기준선(구속) 줄이 결정한다. 직전 라이브 줄은 경보이고 판정을 발동시키지 않는다.
- 승격: 2026-08-19 · 사용자 (사이클 4 · 관측 요약 보고 뒤 확인) — 현 기본값 조합(재작성 켜짐 · 컷 0.30 · 캐싱 없음 · T6 이전 판정 프롬프트)의 첫 풀셋 라이브 3회. 관측 넷을 모두 충족했다 — G17 재작성 유지 3/3 · G18·G01·G02 상충쌍 동시 채택 각 3/3 · G21-G24 판정 층 인계 3/3 · G04 정답 조항 채택 + 정상 답변 3/3. 측정 1 은 앞선 여섯 세트와 동일하고 측정 3 은 3회 중 2회가 동일(1회는 오탐 1건 — 9세트 만의 첫 관측). 새 회귀 가드가 붙은 코드로 산출돼 근거 부분 손실 검사가 온전히 작동하는 첫 세트다. (참조 `data/promoted_baseline.json`)

### 승격 기준선 (구속) — **보류**

- 사유: 판정 보류 — 조건이 확인된 실측 2/3 — 세트가 아직 안 찼다 (조건 불일치로 제외: `evaluation-live-l2-6` — 조건이 다르다: acceptance_cut: 기준선 `0.3` → 이번 `0.5` · `evaluation-live-l2-5` — 조건이 다르다: acceptance_cut: 기준선 `0.3` → 이번 `0.5` · `evaluation-live-l2-4` — 조건이 다르다: acceptance_cut: 기준선 `0.3` → 이번 `0.5` · `evaluation-live-l2-3` — 조건 지문이 확인되지 않는다(미상 11개: label_version, retrieval_labels_version, abstention_gate_statistic 외 8개) · `evaluation-live-l2-2` — 조건 지문이 확인되지 않는다(미상 11개: label_version, retrieval_labels_version, abstention_gate_statistic 외 8개) · `evaluation-live-l2-1` — 조건 지문이 확인되지 않는다(미상 11개: label_version, retrieval_labels_version, abstention_gate_statistic 외 8개)). 아래 항목은 잠정 관측이다.
- 기준선 산출물: `evaluation-live-l2-7`, `evaluation-live-l2-8`, `evaluation-live-l2-9`
- 기준선 출처: `data/promoted_baseline.json`
- 선언된 실험 변인(대조 진행): abstention_gate_statistic: 기준선 `미배선` → 이번 `rank1_minus_rank_k_spread` · abstention_tau: 기준선 `미배선` → 이번 `0.06` (짝인 `embedding_model` 도 함께 달라졌다)
- **일치 감소(경보)**: `G01` — 기준선 3/3 → 이번 2/2 · `G02` — 기준선 3/3 → 이번 2/2 · `G03` — 기준선 3/3 → 이번 2/2 · `G04` — 기준선 3/3 → 이번 2/2 · `G05` — 기준선 3/3 → 이번 2/2 · `G06` — 기준선 3/3 → 이번 2/2 · `G07` — 기준선 3/3 → 이번 2/2 · `G08` — 기준선 3/3 → 이번 2/2 · `G09` — 기준선 3/3 → 이번 2/2 · `G10` — 기준선 3/3 → 이번 2/2 · `G11` — 기준선 3/3 → 이번 2/2 · `G12` — 기준선 3/3 → 이번 2/2 · `G13` — 기준선 3/3 → 이번 2/2 · `G14` — 기준선 3/3 → 이번 2/2 · `G15` — 기준선 3/3 → 이번 2/2 · `G16` — 기준선 3/3 → 이번 2/2 · `G17` — 기준선 3/3 → 이번 2/2 · `G18` — 기준선 3/3 → 이번 2/2 · `G19` — 기준선 3/3 → 이번 2/2 · `G20` — 기준선 3/3 → 이번 2/2 · `G21` — 기준선 3/3 → 이번 2/2 · `G22` — 기준선 3/3 → 이번 2/2 · `G23` — 기준선 3/3 → 이번 2/2 · `G24` — 기준선 3/3 → 이번 2/2 · `G25` — 기준선 3/3 → 이번 2/2 · `G26` — 기준선 3/3 → 이번 2/2 · `G27` — 기준선 3/3 → 이번 2/2 · `G28` — 기준선 3/3 → 이번 2/2 · `G29` — 기준선 3/3 → 이번 2/2 · `G30` — 기준선 3/3 → 이번 2/2
- 미판정: 정답 근거 ID 를 어느 실측의 귀인 절에서도 관측하지 못한 케이스 — 양쪽 모두 정답 근거를 **전부 채택**한 것으로 읽히지만(귀인 절은 그런 케이스를 싣지 않는다) ID 를 이름으로 확인하지는 못했다: `G04`, `G05`, `G06`, `G07`, `G08`, `G15`, `G16`, `G17`, `G19`, `G20`
- 참고: 선언된 실험 변인이 있어 대조를 진행한다 — 차이: abstention_gate_statistic: 기준선 `미배선` → 이번 `rank1_minus_rank_k_spread` · abstention_tau: 기준선 `미배선` → 이번 `0.06` (짝인 `embedding_model` 도 함께 달라졌다)
- 참고: 측정 3 은 무변경 검사 대상이 아니다 — 판정 층 개선의 성공이 무관한 축의 원복을 발동시키면 안 되기 때문이다. 기준선 100.0% / 이번 100.0% — 두 세트의 값이 같다.
- 참고: 합산 일치율은 하한 경보선이고 사이클 판정은 케이스 단위가 맡는다.

### 직전 라이브 (경보) — **보류**

- 사유: 판정 보류 — 조건이 확인된 실측 2/3 — 세트가 아직 안 찼다 (조건 불일치로 제외: `evaluation-live-l2-6` — 조건이 다르다: acceptance_cut: 기준선 `0.3` → 이번 `0.5` · `evaluation-live-l2-5` — 조건이 다르다: acceptance_cut: 기준선 `0.3` → 이번 `0.5` · `evaluation-live-l2-4` — 조건이 다르다: acceptance_cut: 기준선 `0.3` → 이번 `0.5` · `evaluation-live-l2-3` — 조건 지문이 확인되지 않는다(미상 11개: label_version, retrieval_labels_version, abstention_gate_statistic 외 8개) · `evaluation-live-l2-2` — 조건 지문이 확인되지 않는다(미상 11개: label_version, retrieval_labels_version, abstention_gate_statistic 외 8개) · `evaluation-live-l2-1` — 조건 지문이 확인되지 않는다(미상 11개: label_version, retrieval_labels_version, abstention_gate_statistic 외 8개)). 아래 항목은 잠정 관측이다.
- 기준선 산출물: `evaluation-live-l2-7`, `evaluation-live-l2-8`, `evaluation-live-l2-9`
- 기준선 출처: `reports/ 자동 탐색`
- 선언된 실험 변인(대조 진행): abstention_gate_statistic: 기준선 `미배선` → 이번 `rank1_minus_rank_k_spread` · abstention_tau: 기준선 `미배선` → 이번 `0.06` (짝인 `embedding_model` 도 함께 달라졌다)
- **일치 감소(경보)**: `G01` — 기준선 3/3 → 이번 2/2 · `G02` — 기준선 3/3 → 이번 2/2 · `G03` — 기준선 3/3 → 이번 2/2 · `G04` — 기준선 3/3 → 이번 2/2 · `G05` — 기준선 3/3 → 이번 2/2 · `G06` — 기준선 3/3 → 이번 2/2 · `G07` — 기준선 3/3 → 이번 2/2 · `G08` — 기준선 3/3 → 이번 2/2 · `G09` — 기준선 3/3 → 이번 2/2 · `G10` — 기준선 3/3 → 이번 2/2 · `G11` — 기준선 3/3 → 이번 2/2 · `G12` — 기준선 3/3 → 이번 2/2 · `G13` — 기준선 3/3 → 이번 2/2 · `G14` — 기준선 3/3 → 이번 2/2 · `G15` — 기준선 3/3 → 이번 2/2 · `G16` — 기준선 3/3 → 이번 2/2 · `G17` — 기준선 3/3 → 이번 2/2 · `G18` — 기준선 3/3 → 이번 2/2 · `G19` — 기준선 3/3 → 이번 2/2 · `G20` — 기준선 3/3 → 이번 2/2 · `G21` — 기준선 3/3 → 이번 2/2 · `G22` — 기준선 3/3 → 이번 2/2 · `G23` — 기준선 3/3 → 이번 2/2 · `G24` — 기준선 3/3 → 이번 2/2 · `G25` — 기준선 3/3 → 이번 2/2 · `G26` — 기준선 3/3 → 이번 2/2 · `G27` — 기준선 3/3 → 이번 2/2 · `G28` — 기준선 3/3 → 이번 2/2 · `G29` — 기준선 3/3 → 이번 2/2 · `G30` — 기준선 3/3 → 이번 2/2
- 미판정: 정답 근거 ID 를 어느 실측의 귀인 절에서도 관측하지 못한 케이스 — 양쪽 모두 정답 근거를 **전부 채택**한 것으로 읽히지만(귀인 절은 그런 케이스를 싣지 않는다) ID 를 이름으로 확인하지는 못했다: `G04`, `G05`, `G06`, `G07`, `G08`, `G15`, `G16`, `G17`, `G19`, `G20`
- 참고: 선언된 실험 변인이 있어 대조를 진행한다 — 차이: abstention_gate_statistic: 기준선 `미배선` → 이번 `rank1_minus_rank_k_spread` · abstention_tau: 기준선 `미배선` → 이번 `0.06` (짝인 `embedding_model` 도 함께 달라졌다)
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
