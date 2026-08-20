# 핸드오프 — **사이클 간 · 전면 감사 세션 진행 중** (사이클 4 종료 / 사이클 5 미시작)

**진행 중인 사이클이 없다.** 3-doc 은 `.dryforge/004/` 에 있다. 지금 열려 있는 것은 사이클이
아니라 **사이클 1~4 전면 감사 세션**이고, 그 산출이 아래 "지금 워킹트리" 절이다.

**유지 규칙: 덧붙이는 곳이 아니라 현재 상태로 갈아 끼우는 곳이다.** 남기는 기준은
"다음 사람이 알아야 하는데 **코드도 git 도 결정 기록도** 말해주지 않는 것" 하나다.

## 지금 워킹트리 — **커밋되지 않은 변경 29개가 있다** (2026-08-20)

`HEAD` 는 `f89fcba` 이고 `origin/main` 과 동기화돼 있다. 그 **위에** 미커밋 변경이 얹혀 있다.
**사이클 5 를 시작하기 전에 이 묶음을 커밋할지 사용자에게 확인하라** — 감사 산출이 통째로
워킹트리에만 있다.

```
 M .gitignore                          reports 화이트리스트를 retrieval-{strategies,chunking}-live-* 로 축소
 M README.md                           테스트 건수 1,044→1,071 · deselected 881→908
 M docs/architecture.md                의존 지도 4칸 정정
 M docs/contracts.md                   상충쌍 예시를 실존 쌍(1-3↔1-4)으로
 M docs/engineering-notes.md           신규 3건 — 접기 기준 · 아카이브 인용 4차 재발 · 배선 빠진 필드
 M docs/operations.md                  "컬럼 추가는 예외" 병기
 M docs/standards.md                   하드 게이트 6~9 신설 — 정본 목록이 5번에서 끊겨 있었다
 M docs/tracking/findings.md           27번(L1 우회)·28번(구조 검사 한계) 신설, 27-1 은 해결 표시
 M docs/tracking/pricing.md            "200배"→166.5배 · "네 칸"→아홉 칸
 M docs/tracking/status.md             실측 정정 5건
 M docs/tracking/decisions/0011·0018·0019   하드 게이트 번호 · J04 정정
 M scripts/compare_retrieval.py        _reject_ignored — --min-containment 무시의 대칭 검사
 M src/reply_gate/config.py            하드 게이트 10 → 9 (정본 목록에 없던 번호였다)
 M src/reply_gate/contracts.py         __all__ += is_policy_evidence_id
 M src/reply_gate/evidence.py          **C-1 수정** — _pii_shaped 가 접은 뒤 본다
 M src/reply_gate/gate.py              __all__ += fold_for_detection (export 만, 판정 로직 무변경)
 M src/reply_gate/regression_guard.py  _attribution_unknown_note + measurement3_executed 배선
 M src/reply_gate/retrieval_eval.py    **evaluate_retrieval 삭제(64줄)** + 청킹 리포트에 ranked_unit_ids
 M tests/AGENTS.md                     가드가 잡는 "표면"을 적으라는 규율
 M tests/test_chunking_grid.py         청킹 리포트가 순위를 싣는지
 M tests/test_durable_citations.py     검사 대상 non-empty + **하드 게이트 번호 인용 가드 2건**
 M tests/test_evaluation.py            --stub-llm 실행 전후 policy_chunks 출처 대조
 M tests/test_evidence.py              **C-1 회귀 5건** — 접기 계열 4 + 직접 컬럼 보존 1
 M tests/test_gate.py                  잎 노드 검사 신설
 M tests/test_regression_guard.py      non-empty 단언 + 귀인 사유 2건 + 측정 3 미실행/미상 2건
 M tests/test_retrieval_eval.py        단일 설정 검사를 직접 조립으로 + **전략 리포트 덮어쓰기 검사**
 M .dryforge/handoff.md                이 파일
```

**삭제는 `evaluate_retrieval` 하나(64줄)뿐이다** — 나머지는 배선과 기록이다.

검증(이 상태에서 실행): `ruff check` 0 · `ruff format --check` 0(157파일) · `mypy` 0(63파일) ·
`pytest` **1071 passed** · `pytest -m db` **163 passed / skip 0** · 문서 링크·앵커 0건.
DB 무오염 확인(`policy_chunks` = `text-embedding-3-small` / 1536 / 26행).

기존 1056 → 1060 은 이 세션이 더한 가드 4건, 1060 → 1065 는 C-1 회귀 5건,
1065 → 1067 은 하드 게이트 번호 가드 2건,
1067 → 1070 은 필드 배선 회귀 3건, 1070 → 1071 은 전략 리포트 덮어쓰기 검사다.

## 다음 첫 행동 하나

**사이클 5 를 기획한다.** 사이클 간 감사는 끝났고 승인 대기도 없다. 아래 "사이클 5 가 들고
갈 것" 이 3-doc 의 입력이고, 그 밑의 "처분된 …" 절들은 무엇이 왜 지워지고 무엇이 왜 남았는지의
정본이다.

### 사이클 5 가 들고 갈 것 — **재등재 한 묶음**

**1순위는 승격 기준선 재등재다.** 현 기본값(재작성 · 컷 0.30 · 기권 게이트 τ=0.06 · 교정된
판정 픽스처)으로 **풀셋 라이브를 돌린 세트가 하나도 없다** — 재등재 전까지 회귀 가드의 구속
줄은 영구히 "대조 불가"다. 사이클 4 에는 하지 않기로 사용자가 정했다. 재등재는 **풀셋 라이브
3회 구매 + 사람의 결정**이고, 함께 정리할 다섯 가지가 [findings 23](../docs/tracking/findings.md) 에 있다.

**같은 실행에 실어야 하는 수정** — 전부 채택 축이나 L1 판정의 **동작을 바꾼다.** 어차피 새
라이브를 사는 그 자리가 아니면 **고칠 때마다 기준선이 한 번 더 어긋난다.**

| 출처 | 항목 | 무엇이 바뀌나 |
|---|---|---|
| [findings 26](../docs/tracking/findings.md) | 1 · 2 · 4 · 7 | 채택 축 |
| [findings 27](../docs/tracking/findings.md) | **27-2** — `Pd` 대시 12종 + `Mn`(`U+0301`·`U+FE0F`) 미접힘 | L1 초안 판정 |
| | **27-3** — CTE·파생테이블·`COALESCE` 가 정상 에코를 오기각 | L1 오탐률(헤드라인 지표) |
| | **27-4** — `cast` 대상 타입 미검사로 `pg_catalog` 유출 | 안전장치 2층 |

- **27-2 는 `Pd` 와 `Mn` 을 함께** 넣어야 한다. 하나만 넣으면 나머지 계열이 그대로 남는다.
  회귀 테스트에는 **자리 사이 배치**를 박아라 — 끝에 붙이면 원래도 기각된다.
- **27-3 은 결정 사안이다.** fail-closed 를 유지할지 provenance 를 넓힐지가 갈리고,
  그 대가인 오탐률이 어느 문서에도 수치로 없다.
- **27-1 은 이미 끝났다** — 조건 보존이라 재등재를 기다리지 않았다.

**열어 둔 것**: [findings 19](../docs/tracking/findings.md)(`missing_citation` → 재생성 → 통과
처리 기록) — 재현에 확률 층이 걸린 과금 반복이라 사이클 5 필수는 아니다.

**아직 검사 없는 곳**: `docs/standards.md` 의 "사이클 문서를 절 라벨로 인용하지 않는다" 가드는
백틱 안·코드펜스 안·`§` 없는 하이픈 표기(`스펙 4-3`)를 못 보고, `README.md`·`CLAUDE.md`·
`AGENTS.md` 는 양쪽 가드 어디에도 안 들어간다.

### 처분된 삭제 안건 — **경계를 다시 그은 뒤 지운 것은 하나뿐이다**

**⚠ `retrieval_eval.py` 를 "죽은 코드 덩어리"라고 부르지 마라.** 최상위 정의 159개 중
**148개가 프로덕션 진입점에서 도달**하고, 이 모듈이 커밋된 라이브 리포트를 만들었으며
어제(`0a6ab3c`)도 수정됐다.

**지운 것 — `evaluate_retrieval` 하나 (64줄).** 지운 이유는 "안 불린다"가 아니라 **중복
구현**이다: 이 함수의 컷 스윕 루프와 최적점 선택이 전략 사다리의 `_sweep_point` ·
`_best_sweep_cutoff` 와 같은 규칙을 두 번 쓰고 있었고, 그중 한 쪽은 어떤 실행 경로도
부르지 않았다. `__all__` 1항목도 함께 걷었다.

**남긴 것 — 리포트 층 7개** (`RetrievalEvaluation` · `write_report` · `_json_report` ·
`_markdown_report` · `_report_slug` · `_report_paths` · `_next_report_paths`). 테스트에서만
도달하지만 **죽은 게 아니라 경계가 안 그어진 것**이다. `tests/test_retrieval_eval.py` 의
전용 테스트는 지우지 않고 **조립을 명시하도록 고쳤다**(`retrieve_cases` → `score_retrieval`
→ `RetrievalEvaluation`) — 그 테스트가 **유일하게** 지키는 것이 덮어쓰기 방어이기 때문이다.

**같이 지우면 안 되는 것**: `score_retrieval`·`CaseScore`·`RetrievalScore`·`AggregateMetrics`.
`score_retrieval` 은 `tests/test_chunking_grid.py:293` 이 **패리티 오라클**로 쓰고(항등 매핑이면
격자 채점이 한 자리도 다르지 않아야 한다), 나머지 셋은 그 함수의 반환 타입이다.

**덮어쓰기 검사도 신설했다** — `_next_strategy_report_paths` 는 커밋된 리포트 12개를 만든
writer 인데 검사가 없었다(`tests/test_retrieval_eval.py` 의 "전략 리포트는 기존 산출물을
덮어쓰지 않는다"). 3회 연속 실행이 `-2`·`-3` 를 만들고 앞선 파일이 바이트 단위로 불변인지
본다. **검사가 붙잡는 층은 경로 선택기다** — writer 의 `open("x")` 재귀는 경쟁 상태 방어라
순차 실행으로 탈 수 없다. 다만 선택기를 망가뜨리면 덮어쓰기가 아니라 `RecursionError` 로
터진다(음성 대조로 실측). **조용히 덮이는 경로는 없다.**

**⚠ 이 파일에서 무언가를 지울 때 `ast.lineno` 를 시작 줄로 쓰지 마라.** `lineno` 는 데코레이터가
아니라 `class`/`def` 줄을 가리킨다. 고아가 된 `@dataclass(frozen=True)` 가 바로 아래 정의에
붙어 `TypeError: Cannot overwrite attribute __setattr__` 로 **수집이 통째로 죽는다.**
ruff·mypy 는 통과하므로 pytest 를 돌려야 잡힌다. 실제로 드라이런에서 밟았다.

### 처분된 안건 — **읽는 곳이 0인 필드 2개는 지우지 않고 배선했다**

둘 다 "필요 없나"를 다시 확인한 결과 **생산자가 이미 정확한 값을 쥐고 있고 소비 지점만
없는** 경우였다. 전문은 `docs/engineering-notes.md` "읽는 곳이 0인 필드는 죽은 게 아니라
배선이 빠진 것이었다".

- `RunSummary.measurement3_executed` → `_measurement3_note` 가 **미실행과 미상을 가른다.**
  `evaluation-live-{1,2,3}.json` 은 L2 이전 세대라 측정 3 절 자체가 없는데 리포트가 그것을
  "미상"이라 적고 있었다.
- `ChunkingCaseScore.ranked_unit_ids` → 청킹 리포트 JSON 이 **순위를 함께 싣는다.** 자매
  산출물인 전략 리포트는 같은 자리에 `ranked_hits` 를 싣는데 청킹만 빠져 있었다.

### C-1 (L1 우회 경로) — **끝났다 (2026-08-20)**

`evidence._pii_shaped` 가 `fold_for_detection` 을 거쳐 보게 고쳤고 `gate.__all__` 이 접기
함수를 연다. **`gate.py` 의 판정 로직·픽스처·리포트는 그대로다.** 조건 보존이 전수로 확인돼
(`_pii_shaped` 뒤집힘 **0 / 32,104**) 재등재를 기다리지 않았다.

회귀는 `tests/test_evidence.py` 5건이고 **음성 대조를 실제로 돌렸다** — 수정을 되돌리면 4건이
깨지고, 승인된 직접 컬럼 보존 검사 1건은 양쪽에서 통과한다(그게 조건 보존 쪽 가드다).

경위는 `docs/engineering-notes.md` 의 "접기 기준이 층마다 달라 근거 필터가 게이트보다
헐거웠다" 로 옮겼다. findings 27-1 은 해결 표시만 남겼다.

**C-2·C-3·C-4 는 손대지 않았다** — 전부 findings 27 에 있고, C-2 는 초안 판정을 바꾸므로
재등재 묶음 쪽이다.

## 이 세션이 한 일 (요약 — 상세는 findings 27·28)

사이클 1~4 를 6축 병렬 감사했다: 죽은 코드 · 치명 안전장치 · 리포트 산출물 · 문서 정합 ·
테스트 가드 실효성 · 런타임 배선.

- **불변식 22개 중 21개 성립**, 위반은 불변식 6 하나(= C-1).
- **리포트는 삭제 대상 0건** — 커밋된 38쌍 전부가 근거로 살아 있다. 4개는 `adoption_axis.py` 가
  **런타임에 로드**하고 3개는 테스트가 직접 읽는다. 짝 불일치 0 · 결번 0 · 빈/중단 리포트 0.
- **`.retrieval-cache/` 는 보존** — 523개 중 **351개가 과금된 OpenAI 임베딩**(3-large 7.40MB /
  3-small 5.05MB / 무료 대역 0.61MB). 기능 의존은 없지만 결정 0014·0015 의 "무과금 재현"을
  이 캐시가 지탱한다.
- **`기획-입력.md` 보존** — `.dryforge/003/spec.md:25` 가 링크한다. 다만 `:64` 의 Sonnet 단가가
  `pricing.md` 정본(인상 철회)과 충돌하니 "이력이다" 한 줄 얹는 것을 권한다.
- **`Settings.bulk_generation_model` 보존** — 코드 참조 0이지만 `pricing.md:74`·결정 0010 이
  "실행 경로에 없다"고 명시적으로 다룬다.
- **문서 사실 오류 11건을 고쳤다**(2026-08-20). 감사 보고를 그대로 믿지 않고 **항목마다 리포트
  JSON 을 python 으로 재집계해 독립 재검증**한 뒤 확인된 것만 반영했다:
  `pricing.md:144`·`README.md:1369` **"200배"→166.5배**(반올림 값을 나눈 산물이었다) ·
  `pricing.md:34` "네 칸"→**아홉 칸** · J04 "판 ② 이후 흔들리지 않았다"→**14/15, `-25` 에서
  뒤집혔다**(status·README·결정 0018 세 곳) · `status.md:209` "전 구간 oracle≥blind"→
  **컷 0.40 이상에서만**(52지점 중 12곳 역전, 전부 컷 0.35 이하 — **제품 기본 컷 0.30 이 그
  구간 안이다**) · `status.md:238` 재작성 토큰 ≈120→**실측 133.8~148.0** · `status.md:371`
  `rejected_twice` 4건/회→**G21–G24 한정** · `status.md:13` 미채택 사유 둘 뒤바뀜 ·
  `README.md:309`·`:937`·`status.md:396` G28 "3회 중 2회 기권"→**검색이 돈 2회 중 1회**
  (`-11` 은 기권이 아니라 **미실행**) · `contracts.md:46`~`:59` 의 `policy:shipping:3-1` →
  **실존 상충쌍 `1-3`↔`1-4`**(짝인 `1-2` 도 상충 상대가 없어 함께 갈았다) ·
  `operations.md:84` "DDL 이 전부 `CREATE TABLE IF NOT EXISTS`" → **컬럼 추가는 예외**라고
  병기(이 안내를 믿으면 컬럼 하나 추가에 볼륨 삭제 + **유료 재색인**을 한다) ·
  `architecture.md` 의존 지도 4칸(`config`→`retrieval_strategies` · `api`+`policy_index` ·
  `evaluation`+`query_rewrite`,`retrieval_labels` · 잎 노드 문장에서 `config.py` 제거).
- **반증 1건 — 고치지 않았다.** 감사가 `README.md:783` 의 *"볼륨 재생성 값을 못 한다"* 를
  파손된 문장으로 보고했으나, `값을 하다`(값어치를 하다)는 정상 관용구이고 `README.md:1641`·
  `status.md:225` 가 같은 표현을 독립적으로 쓴다. 최초 도입 커밋부터 문구가 동일해 잘린 흔적도
  없다. **감사 보고를 검증 없이 반영했으면 멀쩡한 문장을 망가뜨릴 뻔했다.**

## 함정 (실제로 밟은 것만)

**이 세션에서 새로 밟은 것 넷:**

- **같은 워크트리에 다른 Claude 세션이 동시에 붙어 있다.** 내가 `tests/test_regression_guard.py`
  에 넣은 편집이 **다른 에이전트의 원복 작업에 통째로 지워졌다**(Edit 은 성공을 보고했다).
  파일을 고친 뒤에는 **`grep` 으로 실재를 확인하라** — Edit 성공 응답은 그 뒤의 덮어쓰기를
  말해주지 않는다. 병렬 세션이 같은 DB 를 쓰면 `apply_schema` 에서 **교착(DeadlockDetected)** 도
  난다.
- **`cp` 로 원복해도 pytest 가 stale `.pyc` 를 쓴다.** 변이를 넣고 되돌린 뒤 테스트가 계속
  빨간불이었는데, 모듈을 직접 import 하면 정상이었다 — pytest 의 assertion-rewrite 캐시가
  변이 판을 들고 있었다. **`find . -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +`
  와 `rm -rf .pytest_cache` 를 원복 직후에 함께 돌려라.**
- **`git check-ignore -v` 로 무시 여부를 판정하지 마라.** negation 패턴(`!reports/...`)에
  매칭되면 종료 코드 0 에 그 줄을 찍는데, 그건 "무시됨"이 아니라 "무시 해제됨"이다.
  **실제 파일을 하나 만들고 `git status --porcelain --ignored` 로 `??` 인지 `!!` 인지 봐라.**
- **PII 우회 문자는 배치가 가른다.** `U+0301`·`U+FE0F` 는 숫자열 **안쪽**에 끼워야 뚫리고
  (`010-1́234-5678` → pass) 끝에 붙이면 기각된다(`010-1234-5678́` → reject). `[0-9]{3,4}` 가
  끊기느냐의 문제다. **회귀 테스트에는 자리 사이 배치를 박아라.** `Cf`(`U+200B`)는 자리
  사이여도 막힌다 — 그래서 접기 보강은 `Pd`(대시) 하나로는 부족하고 **`Mn` 도 같이** 봐야 한다.

**이전부터 있던 것:**

- **τ 는 임베딩 모델에 묶인다.** τ=0.06 은 `-3-small` 1536 에서만 분리하고 `-3-large`
  계열은 여유가 음수다(−0.0114 / −0.0160). 임베딩을 건드리면 게이트가 **조용히
  무력해진다.** 런타임 방어는 없고 리포트 지문의 짝 하나가 전부다(findings 26-7).
- **실측이 끝난 뒤 제품 동작을 바꾸면 그 라이브의 조건이 무효가 된다.** 커밋된 리포트는
  재생성에 과금이 들고 확률 층이라 같은 값도 안 나온다. **다만 "동작을 바꾼다"를 자동으로
  가정하지 마라** — 27-1 처럼 전수로 재서 판정 변화 0건이면 조건은 보존된다.
- **커밋된 라이브 리포트를 고치려고 다시 사지 마라.** 틀린 것이 사람이 읽는 줄이고 기계가
  읽는 값이 맞으면 정정은 문서 쪽이다. 사이클 4 에 두 번 그렇게 처리했다.
- **짝 지문은 짝이 실제로 달라졌는지 값을 보고 적어라.** 짝 관계를 **설명**하는 문장과
  짝이 **달라졌다고 보고**하는 문장은 다르다.
- **가드가 "미판정"이라고 적을 때 사유를 지어내지 마라.** `regression_guard` 가 산출물이 들고
  있는 `attribution_reason` 을 버리고 "옛 산출물이라 필드가 없다"는 고정 문장을 찍고 있었다 —
  `evaluation-live-l2-{16,17,18,23,24}.md` 에 그 거짓 문장이 영구히 남았다(이번에 코드는 고쳤고
  산출물은 규칙 7 대로 보존한다).
- **사이클 문서를 절 라벨(`spec §4-3`)로 인용하지 마라.** 규칙은 `docs/standards.md` 에 있고
  `tests/test_durable_citations.py` 가 막는다. **가드에 사각이 있다** — 백틱 안·코드펜스 안·
  `§` 없는 하이픈 표기(`스펙 4-3`)는 안 잡히고, `README.md`·`CLAUDE.md`·`AGENTS.md` 는 양쪽
  가드 어디에도 안 들어간다. "하드 게이트 10" 이 정확히 그 사각으로 새어 들어왔고 — **번호만
  적는 형태는 세 번째 가드로 막았다**(정본 목록의 번호 집합과 대조). 남은 사각은 그대로다.
- **선언값을 잴 때 인자 없는 `Settings()` 를 쓰지 마라** — `conftest.declared_settings()` 를 쓴다.
- **자격 증명이 새는 경로는 둘이었다.** ① 설정 객체 repr, ② 에이전트가 읽은 `.env` 가 세션
  기록에 평문으로 남는 것. 값이 필요하면 `bool(settings.<필드>)` 로 존재만 확인한다.
  **세 번째가 이번에 확인됐다: `Settings.model_dump()` 는 `repr=False` 를 무시하고 자격 증명
  4종을 평문으로 낸다.** 지금 실행 경로에서 부르는 곳은 없고 `tests/test_config_secrets.py` 는
  `field.repr` 만 본다. `SecretStr` 로 옮기면 둘 다 막힌다.
- **가드를 문자열 스캔으로 짓지 마라.** AST 로 보면 주석은 애초에 없고 docstring 은 제외할 수 있다.
- **구조 검사의 검사 대상은 재귀로 유도하고, 목록이 비면 실패하게 하라.** 관례 전체는
  **`tests/AGENTS.md`** 에 있다.
- **병렬 실행은 리포트 번호를 미리 갈라 줘야 한다** — `--report-stem` 으로 구간을 나눈다.
- **확률 층 장면에 "항상"을 붙이지 마라.** "3/3"은 관측이지 결정론이 아니다.
- **`--stub-llm` 의 롤백은 실제로 걸린다** — 이번에 실행으로 확인했고(색인 26행 바이트 동일),
  `tests/test_evaluation.py` 가 이제 실행 전후 출처를 대조한다. 새는 조건은 **리팩터링 한 줄**
  이다(`finally: app_conn.rollback()` → `with connect(...)`).
- **브랜치 강제 삭제(`-D`)는 `~/.claude/hooks/block-dangerous-git.sh` 가 막는다** — 명령문 안에
  그 문자열이 있기만 해도 차단되므로 문서에 예시로 적지도 마라. `git restore` 도 막힌다.
- **`.claude/hooks/protect-live-reports.sh` 가 `reports/` 아래 추적 파일의 수정·삭제를 막는다.**
  읽기(cat·jq)는 통과하고 `cp` 는 방향과 무관하게 막힌다.
- **설계 세션은 없다.** 문서로 안 풀리면 추측하지 말고 사용자에게 묻는다.

## 옆 세션

`reply-gate-21`(`uds:/tmp/cc-socks/26652.sock`)이 같은 저장소에 붙어 사용자 질의에 답하고 있다.
**C-1 을 독립 재현했고 표가 정확히 일치했다.** `U+0301` 배치 건도 그쪽이 잡아 줬다.
**C-1 수정 소유권은 이 세션**이라고 서로 합의했다. 그쪽은 코드를 건드리지 않는다.

## 하드 게이트

`CLAUDE.md` "절대 깨지 않는 것" 7개와 `docs/standards.md` 가 정본이다.
**과금은 명시적으로만 하고 회당 상한은 $10 이다.**
이 세션은 **과금 실행을 한 번도 하지 않았다** — 모든 검증이 무과금 경로였다.
