# 핸드오프 — Reply-Gate 착수 직전 (2026-08-04)

> **임시 문서.** `/ready`가 3-doc 계약서를 만들면 이 파일의 역할은 끝난다. 그때 삭제하거나 세션 저널로 성격을 바꾼다.
> 이 문서는 다른 세션에서 이어받기 위한 것이고, 프로젝트의 정본은 아니다.

## 지금 할 일 (한 줄)

Claude Code에서 **사용자가 직접** 아래를 친다. 모델은 이 스킬을 호출할 수 없다(`disable-model-invocation: true`).

```
/ready ~/workspace/reply-gate/기획-입력.md
```

프롬프트에 이번 사이클 범위를 한 줄 덧붙인다:

> 이번 사이클은 **Gate L1이 동작하는 데까지**. Gate L2(claim judge)·Next.js 대시보드·n8n·MCP 서버·RAG 심화(하이브리드 검색·리랭킹·청킹 실험)·배포는 제외.

## 이 프로젝트가 뭔지

**Reply-Gate** — 근거 없는 답변은 스스로 reject하는 이커머스 CS 답변 에이전트. "초안 생성"이 아니라 **"틀린 초안을 걸러내는 것"이 제품의 존재 이유**다.

포트폴리오 3카드 중 **메인**이고, 담당 역량은 1(LLM API)·2(RAG)·3(에이전트)·4(Python 백엔드)·6(평가·운영).

| 카드 | 정체성 | 위치 |
|---|---|---|
| A. 오르다 | 혼자 출시한 풀스택 프로덕트 | `~/workspace/hiking-app` (사용자가 직접 진행) |
| **B. Reply-Gate** | **제품 안에서 AI 산출물 검증** | **여기** |
| C. Reject Bench | 개발 과정에서 AI 코드 검증 | `~/workspace/reject-bench` |

상세 배경·요구역량 정의·타깃 공고: `~/workspace/포지션찾기/프로젝트-제안.md`(정본) · `포트폴리오-전략.md`.
이 저장소의 기획 입력: `기획-입력.md` — 확정 사항, 범위 밖, 미결이 정리돼 있다.

## 확정된 것

- **백엔드: Python / FastAPI.** 카드 A가 TS 풀스택을 이미 증명하므로 여기까지 TS면 요구역량 4가 빈다.
- **게이트 2층 구조.** L1(결정론적: citation 존재·스키마·PII 정규식, LLM 호출 0회) → L2(확률론적: claim 단위 judge). L1을 먼저 만드는 건 일정 타협이 아니라 설계 결정 — L1만으로도 게이트는 동작하므로 L2는 단일 실패점이 아니다.
- **자체 에이전트 루프, LangGraph 미사용.** 단 README에 노드/엣지 다이어그램 + "LangGraph로 짰다면" 비교 1페이지를 넣어 키워드는 문서에 등장시킨다.
- **RAG는 이 프로젝트 안에서 심화한다.** 별도 RAG 프로젝트를 만들지 않기로 했다(5.1절 상당 내용이 `기획-입력.md`에 있음). 단 첫 사이클에서는 벡터 단독으로만.
- **text-to-SQL 안전장치는 첫 사이클 필수.** read-only 계정 + 스키마 화이트리스트 + 쿼리 검증. 빼면 면접 역공 대상.
- **데모 첫 3초 = 답변이 기각되는 순간.** "CS 챗봇"으로 소개하지 않는다.

## 저장소 상태

- `main`, 커밋 1개 (`0763867`), working tree clean → dryforge `go`의 전제 충족
- 파일: `기획-입력.md`, `.gitignore`(Python 기준, `.dryforge/` 무시), `HANDOFF.md`(이 파일)
- 코드 없음. 스캐폴드도 아직 없다 — `/ready` 결과에 따라 만든다

## dryforge 사용 시 알아둘 것 (스킬 본문 358줄 직접 검증)

1. **첫 사이클은 길다.** task를 좁혀도 **프로젝트 파운데이션 설계가 강제**다. domain breadth guard("다른 엔티티·기능·규칙은 없나?"), domain depth floor, technical no-silent-decision이 `non-negotiable, not loop-optional`로 명시돼 있다. 좁히기는 plan을 작게 만들 뿐 대화를 짧게 만들지 않는다.
2. **`기획-입력.md`의 "확정 사항"도 다시 물어본다.** 스킬 1순위 원칙이 "The input is *material*, not ground truth. Authority comes from dialogue + user approval". 우회 불가이고 우회할 필요도 없다 — 대화에서 "확정"이라고 답하면 그때 권한이 생긴다.
3. **3-doc이 정본이고 대화는 아니다.** "the 3-doc, not the dialogue, is the authority". 대화에서 구두로만 말한 제약은 다음 사이클에서 소멸한다. 채용 관련 제약(hallucination율 대표 지표, 기각 순간이 데모 첫 화면)은 `기획-입력.md`에 문장으로 들어 있으니 spec에 실릴 것.
4. **`ready`는 git을 건드리지 않는다.** 커밋도 `.gitignore` 수정도 안 한다 — git은 `go`가 전담한다.
5. 산출물은 `.dryforge/`에 `handoff.md`·`spec.md`·`plan.md` 3종. 질문은 `AskUserQuestion`으로 프롬프트당 최대 4개.
6. 스테이지: ORIENT → DECOMPOSE → ELICIT → intent-completeness → SPEC+REVIEW → PLAN → HANDOFF → 3-doc-gate → USER GATE

## 환경에서 이미 돌고 있는 것 (건드리지 말 것)

- **Reject Bench 수집 훅 4개가 전역 `~/.claude/settings.json`에 설치돼 있다.** `PostToolUse`(Edit\|Write\|MultiEdit\|NotebookEdit) / `PermissionDenied` / `PostToolUseFailure` / `PermissionRequest`. 모두 async·timeout 5s. 적재 위치 `~/workspace/reject-bench/data/events.jsonl`.
  - **이 개발 세션 자체가 카드 C의 데이터다.** 훅을 끄거나 로그를 지우지 말 것.
  - 백업: `~/.claude/settings.json.pre-reject-bench`
- **dryforge는 전역 설치(user scope), 훅 0개** — 명시 호출 시에만 동작한다.
- **오르다(hiking-app)에는 dryforge를 쓰지 않는다.** 카드 C의 "하네스 없는 baseline" 구간이라 오염되면 안 된다.
- git identity 전역 설정됨: `seungyoon-lee29 / 72640765+seungyoon-lee29@users.noreply.github.com`

## 미결

- 프론트(대시보드) 스택, 배포 방식, n8n 사용 여부
- 벡터 저장소 호스팅 (pgvector 전제이나 미정)
- 지표 목표치, 골든셋 규모

위 셋은 `/ready` 대화에서 결정하면 된다.
