# Reply-Gate 평가 리포트

측정은 **결정론 층과 확률 층을 분리**한다 — LLM 비결정성이 게이트 자체의 정확도
수치를 흔들지 못하게 하기 위해서다.

## 실행 조건

- 실행 시각(UTC): `2026-08-19T08:09:46+00:00`
- 생성 LLM: 미실행
- 임베딩: 미실행
- L2 판정: 켜짐 / 판정 모델: Anthropic `claude-sonnet-5` (effort=기본값)
- 검색 전략: vector+rewrite · 임베딩 1536차원
- 유사도 임계값: 0.3 / top k: 5
- L1 픽스처: 27건 (`data/l1_fixtures.jsonl`)
- 골든셋: 30건 (`data/golden_set.jsonl`)
- 판정 픽스처: 11건 (`data/judge_fixtures.jsonl`)
- OPENAI_API_KEY 설정 여부: 설정됨
- ANTHROPIC_API_KEY 설정 여부: 설정됨
- 과금 실행: 예 · 측정 범위: measurement_1_3_only

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
| `query_rewrite` | on |  |
| `embedding_model` | text-embedding-3-small |  |
| `embedding_dimensions` | 1536 |  |
| `top_k` | 5 |  |
| `generation_model` | gpt-5.6-terra |  |
| `judge_model` | claude-sonnet-5 |  |
| `judge_effort` | 기본값 |  |
| `judge_prompt_version` | judge-e26f6c8e0b84 |  |
| `judge_fixture_version` | fixture-2b8f33392b36 | **선언된 실험 변인** |
| `judge_prompt_caching` | off |  |
| `measurement_scope` | measurement_1_3_only |  |
| `generation_effort` | 기본값 |  |
| `l2_enabled` | on |  |

선언된 실험 변인: `judge_fixture_version`

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
| 측정 2 허용 결과 집합 대비 일치율 | 하한 경보선 | ≥ 75% | 미측정 | 미측정 |

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

**미실행 (사유: 측정 2 를 실행 선택에서 제외했다 (`--measurements`) — 고르지 않은 측정이다)**

수치를 0 이나 빈 값으로 채우지 않는다 — 미실행은 미실행으로 남긴다.

### 검색 실패 / 생성 문제 분해

**미산출 (사유: 측정 2 미실행: 측정 2 를 실행 선택에서 제외했다 (`--measurements`) — 고르지 않은 측정이다)**

미산출을 0건·빈 집계·성공으로 대체하지 않는다.

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
- 판정 토큰: 입력 30042 / 출력 5854 (픽스처당 3263.3)

### 사유 2종별 내역

| 사유 | 기대 픽스처 | 검출 | 검출률 | 오발화(기대하지 않은 발화) |
| --- | ---: | ---: | ---: | ---: |
| `unsupported_claim` | 4 | 4 | 100.0% | 0 |
| `contradictory_evidence` | 3 | 2 | 66.7% | 0 |

### 기대와 어긋난 픽스처

- `J11` (combined): 기대 reject[unsupported_claim, contradictory_evidence] → 실제 reject[unsupported_claim]

## 회귀 가드 — 이중 기준선 두 줄 보고

- 이번 실측 세트: 1/3회 `evaluation-live-l2-13`
  - 조건 불일치로 세트에서 제외: `evaluation-live-l2-12` — 조건이 다르다: judge_fixture_version: 기준선 `fixture-2b8f33392b36` → 이번 `fixture-836322e97320` · measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-11` — 조건이 다르다: judge_fixture_version: 기준선 `fixture-2b8f33392b36` → 이번 `fixture-836322e97320` · measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-10` — 조건이 다르다: judge_fixture_version: 기준선 `fixture-2b8f33392b36` → 이번 `fixture-836322e97320` · measurement_scope: 기준선 `measurement_1_3_only` → 이번 `full` · `evaluation-live-l2-6` — 조건이 다르다: acceptance_cut: 기준선 `0.3` → 이번 `0.5` · `evaluation-live-l2-5` — 조건이 다르다: acceptance_cut: 기준선 `0.3` → 이번 `0.5` · `evaluation-live-l2-4` — 조건이 다르다: acceptance_cut: 기준선 `0.3` → 이번 `0.5` · `evaluation-live-l2-3` — 조건 지문이 확인되지 않는다(미상 11개: label_version, retrieval_labels_version, abstention_gate_statistic 외 8개) · `evaluation-live-l2-2` — 조건 지문이 확인되지 않는다(미상 11개: label_version, retrieval_labels_version, abstention_gate_statistic 외 8개) · `evaluation-live-l2-1` — 조건 지문이 확인되지 않는다(미상 11개: label_version, retrieval_labels_version, abstention_gate_statistic 외 8개)
- **판정: 대조 불가** — 승격 기준선(구속) 줄이 결정한다. 직전 라이브 줄은 경보이고 판정을 발동시키지 않는다.
- 승격: 2026-08-19 · 사용자 (사이클 4 · 관측 요약 보고 뒤 확인) — 현 기본값 조합(재작성 켜짐 · 컷 0.30 · 캐싱 없음 · T6 이전 판정 프롬프트)의 첫 풀셋 라이브 3회. 관측 넷을 모두 충족했다 — G17 재작성 유지 3/3 · G18·G01·G02 상충쌍 동시 채택 각 3/3 · G21-G24 판정 층 인계 3/3 · G04 정답 조항 채택 + 정상 답변 3/3. 측정 1 은 앞선 여섯 세트와 동일하고 측정 3 은 3회 중 2회가 동일(1회는 오탐 1건 — 9세트 만의 첫 관측). 새 회귀 가드가 붙은 코드로 산출돼 근거 부분 손실 검사가 온전히 작동하는 첫 세트다. (참조 `data/promoted_baseline.json`)

### 승격 기준선 (구속) — **대조 불가**

- 사유: 선언되지 않은 조건 불일치 — 어긋난 항목: abstention_gate_statistic: 기준선 `미배선` → 이번 `rank1_minus_rank_k_spread` · abstention_tau: 기준선 `미배선` → 이번 `0.06` (짝인 `embedding_model` 도 함께 달라졌다) · measurement_scope: 기준선 `full` → 이번 `measurement_1_3_only`
- 기준선 산출물: `evaluation-live-l2-7`, `evaluation-live-l2-8`, `evaluation-live-l2-9`
- 기준선 출처: `data/promoted_baseline.json`
- **대조 불가 — 선언 없이 어긋난 항목**: abstention_gate_statistic: 기준선 `미배선` → 이번 `rank1_minus_rank_k_spread` · abstention_tau: 기준선 `미배선` → 이번 `0.06` (짝인 `embedding_model` 도 함께 달라졌다) · measurement_scope: 기준선 `full` → 이번 `measurement_1_3_only`
- 선언된 실험 변인(대조 진행): judge_fixture_version: 기준선 `fixture-836322e97320` → 이번 `fixture-2b8f33392b36`
- 참고: 선언된 실험 변인이 있어 대조를 진행한다 — 차이: judge_fixture_version: 기준선 `fixture-836322e97320` → 이번 `fixture-2b8f33392b36`
- 참고: 측정 3 은 무변경 검사 대상이 아니다 — 판정 층 개선의 성공이 무관한 축의 원복을 발동시키면 안 되기 때문이다. 기준선 100.0% / 이번 100.0% — 두 세트의 값이 같다.
- 참고: 합산 일치율은 하한 경보선이고 사이클 판정은 케이스 단위가 맡는다.

### 직전 라이브 (경보) — **대조 불가**

- 사유: 선언되지 않은 조건 불일치 — 어긋난 항목: measurement_scope: 기준선 `full` → 이번 `measurement_1_3_only`
- 기준선 산출물: `evaluation-live-l2-10`, `evaluation-live-l2-11`, `evaluation-live-l2-12`
- 기준선 출처: `reports/ 자동 탐색`
- **대조 불가 — 선언 없이 어긋난 항목**: measurement_scope: 기준선 `full` → 이번 `measurement_1_3_only`
- 선언된 실험 변인(대조 진행): judge_fixture_version: 기준선 `fixture-836322e97320` → 이번 `fixture-2b8f33392b36`
- 참고: 선언된 실험 변인이 있어 대조를 진행한다 — 차이: judge_fixture_version: 기준선 `fixture-836322e97320` → 이번 `fixture-2b8f33392b36`
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
