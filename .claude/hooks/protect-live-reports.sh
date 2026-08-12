#!/usr/bin/env bash
# 커밋된 라이브 실측 리포트의 사후 편집을 거부한다 (PreToolUse).
#
# 이 저장소의 불변식: **커밋된 라이브 리포트는 사후 편집하지 않는다.**
# 근거 — CLAUDE.md "평가용으로 심은 장치를 제거하지 않는다",
#        docs/tracking/status.md "두 벌 다 사후 편집하지 않는다",
#        docs/tracking/findings.md 9번(키 이름이 어긋나도 옛 산출물은 그대로 둔다).
# 확률 층의 산출물이라 재실행해도 같은 값이 안 나오고, 재실행은 과금된다.
#
# 이미 있는 방어와의 관계 — 이건 3층이 아니라 **다른 경로**를 막는다:
#   * src/reply_gate/evaluation.py:250 — 하네스가 기존 라이브 리포트를 덮어쓰는 것을 막는다.
#   * 이 훅 — 에이전트·사람이 Edit/Write/Bash 로 손대는 것을 막는다.
#
# 보호 대상의 정의는 하드코딩하지 않는다: **reports/ 아래의 git 추적 파일**이다.
# .gitignore 가 이미 `reports/*` 를 무시하고 `evaluation-live*.{md,json}` 만 되살리므로,
# 추적 여부가 곧 "커밋된 실측 산출물인가"와 같은 뜻이다. 새 리포트는 커밋 전까지 자유롭고
# 커밋되는 순간 잠긴다 — 규칙이 .gitignore 를 따라 자동으로 갱신된다.
#
# 입력: stdin 으로 PreToolUse 훅 페이로드 JSON.
# 출력: 차단할 때만 permissionDecision=deny JSON. 그 밖에는 아무것도 쓰지 않고 exit 0.
# 실패 방향: 판단할 수 없으면 **통과시킨다**(fail-open). 훅이 개발을 멈추게 하면 안 되고,
#            진짜 불변식은 커밋 리뷰와 위 하네스 가드가 함께 지킨다.

set -uo pipefail

payload=$(cat)

root=$(git rev-parse --show-toplevel 2>/dev/null) || root="${CLAUDE_PROJECT_DIR:-$PWD}"

deny() {
	jq -n --arg reason "$1" '{
		hookSpecificOutput: {
			hookEventName: "PreToolUse",
			permissionDecision: "deny",
			permissionDecisionReason: $reason
		}
	}'
	exit 0
}

# reports/ 아래의 git 추적 파일인가.
is_locked() {
	case "$1" in
	reports/*) ;;
	*) return 1 ;;
	esac
	git -C "$root" ls-files --error-unmatch -- "$1" >/dev/null 2>&1
}

# 절대·상대 경로를 저장소 상대 경로로 만든다.
to_relative() {
	local path=$1
	case "$path" in
	"$root"/*) printf '%s' "${path#"$root"/}" ;;
	/*) printf '%s' "$path" ;;
	*) printf '%s' "${path#./}" ;;
	esac
}

reason_for() {
	printf '%s' "커밋된 라이브 실측 리포트는 사후 편집하지 않는다 — $1

이 파일은 git 이 추적하는 과금 산출물이고, 확률 층이라 재실행해도 같은 값이 나오지 않는다
(CLAUDE.md · docs/tracking/status.md '두 벌 다 사후 편집하지 않는다' · findings.md 9번).

대신 할 것:
  * 새 수치가 필요하면 새 실행을 만든다 — uv run python -m scripts.evaluate --live
    (하네스가 -1 -2 -3 으로 번호를 올려 붙이므로 기존 리포트를 건드리지 않는다)
  * 라벨·기준이 바뀌어 옛 수치를 다시 읽어야 하면, 리포트를 고치지 말고
    docs/tracking/status.md 에 재채점을 **병기**한다 (결정 0008 이 그 선례다)
  * 정말 고쳐야 한다면 사용자에게 먼저 확인받는다"
}

tool=$(printf '%s' "$payload" | jq -r '.tool_name // ""')

case "$tool" in
Edit | Write | MultiEdit | NotebookEdit)
	target=$(printf '%s' "$payload" | jq -r '.tool_input.file_path // .tool_input.notebook_path // ""')
	[ -n "$target" ] || exit 0
	relative=$(to_relative "$target")
	if is_locked "$relative"; then
		deny "$(reason_for "$relative")"
	fi
	;;
Bash)
	command=$(printf '%s' "$payload" | jq -r '.tool_input.command // ""')
	[ -n "$command" ] || exit 0

	# 명령이 잠긴 리포트를 언급하는가. 읽기만 하는 명령은 여기서 걸려도 아래에서 통과한다.
	locked_hit=""
	while IFS= read -r candidate; do
		[ -n "$candidate" ] || continue
		if is_locked "$(to_relative "$candidate")"; then
			locked_hit=$candidate
			break
		fi
	done <<<"$(printf '%s' "$command" | grep -oE '[^[:space:]"'"'"';|&()]*reports/[^[:space:]"'"'"';|&()]+' || true)"

	[ -n "$locked_hit" ] || exit 0

	# 언급만으로는 막지 않는다 — cat·jq·grep 으로 읽는 것은 정상 작업이다.
	# 파일을 바꾸거나 지우는 형태일 때만 거부한다.
	#
	# 알려진 과차단: `cp <잠긴 리포트> /tmp/x` 처럼 **읽기 방향**의 복사도 막는다.
	# 방향 판별(마지막 인자가 목적지)은 `;`·`&&`·파이프로 이어붙인 명령에서 신뢰할 수 없어
	# 넣지 않았다. 잠긴 리포트를 저장소 밖으로 복사하는 것은 "밖에서 고쳐 되넣기"의 앞단계라
	# 막는 편을 택했고, 읽기는 cat·jq·Read 로 이미 열려 있다. 조용히 통과시키는 것보다
	# 눈에 보이게 막는 쪽이 이 저장소의 실패 방향에 맞다.
	if printf '%s' "$command" | grep -qE \
		-e '(^|[;&|(]|[[:space:]])(rm|mv|cp|tee|shred|truncate|dd|install)[[:space:]]' \
		-e '(^|[;&|(]|[[:space:]])(sed|perl|ruby)[^;&|]*[[:space:]]-i' \
		-e '(^|[;&|(]|[[:space:]])git[[:space:]]+(rm|mv|restore|checkout)[[:space:]]' \
		-e '>[[:space:]]*[^[:space:];&|]*reports/'; then
		deny "$(reason_for "$locked_hit")"
	fi
	;;
esac

exit 0
