"""소스 주석이 **아카이브될 설계 문서**를 인용하지 못하게 막는다.

사이클 문서(`.dryforge/spec.md`·`plan.md`·`handoff.md`)는 사이클이 끝나면
`.dryforge/00N/` 으로 옮겨진다. 그 순간 소스에 남은 인용은 **존재하지 않는 경로**를
가리킨다 — 사이클 3 종료 뒤 실제로 14곳이 그렇게 깨졌고, 같은 사고가 이번이 세 번째다
(docs/engineering-notes.md "소스 주석이 아카이브될 설계 문서를 인용함").

그래서 인용은 durable 문서(`docs/*.md`·`AGENTS.md`)로만 한다. 이 테스트가 그 규율을
사람의 기억에서 코드로 옮긴다 — 앞의 두 번은 규율로 막으려다 재발했다.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

#: 검사 대상. `.dryforge/` 자체와 `docs/` 는 제외한다 — 사이클 문서가 자기를 인용하는 것도,
#: 추적 문서가 사이클 문서를 인용하는 것도 아카이브와 함께 옮겨 다니는 서사의 일부다.
_SCANNED = (
    Path("src"),
    Path("scripts"),
    Path("tests"),
    Path("db"),
    Path("docker-compose.yml"),
)

_SUFFIXES = {".py", ".sql", ".yml", ".yaml", ".md"}

#: `\bspec\b` 는 `_inspect_schema`·`import inspect` 를 잡지 않는다(단어 경계).
_FORBIDDEN = re.compile(r"\.dryforge|\bspec\b|스펙", re.IGNORECASE)


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for target in _SCANNED:
        path = _ROOT / target
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix in _SUFFIXES
                and "__pycache__" not in candidate.parts
            )
    return files


def test_소스가_사이클_설계_문서를_인용하지_않는다() -> None:
    offenders: list[str] = []
    for path in _scanned_files():
        if path.name == Path(__file__).name:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _FORBIDDEN.search(line):
                offenders.append(f"{path.relative_to(_ROOT)}:{number}: {line.strip()}")

    assert not offenders, (
        "사이클 문서 인용은 아카이브 시점에 깨진다. durable 문서(docs/*.md·AGENTS.md)의 "
        "절 제목을 인용하라:\n" + "\n".join(offenders)
    )


def test_가드가_실제로_잡는_문장이_있다() -> None:
    """음성 대조 — 검사기가 항상 통과하는 빈 검사가 아님을 확인한다."""
    assert _FORBIDDEN.search("(`.dryforge/spec.md` §8-1)")
    assert _FORBIDDEN.search("평가 리포트가 그것을 집계한다(spec §8-1).")
    assert _FORBIDDEN.search("이번 사이클 스펙 4-1")
    # 단어 경계 덕분에 통과해야 하는 것들
    assert not _FORBIDDEN.search("inspection = _inspect_schema(raw_draft)")
    assert not _FORBIDDEN.search("import inspect")


# ── docs/ 쪽 가드 ───────────────────────────────────────────────────────────
#
# `docs/` 에는 위 금지어를 그대로 걸 수 없다 — 하네스 배치를 설명하는 문서와 이 사고를
# 기록한 문서가 `.dryforge`·`spec` 을 정당하게 쓴다. 대신 **깨지는 인용의 형태**를 막는다:
# 절 기호(`§`) 인용은 **같은 파일의 번호 붙은 절**을 가리킬 때만 통과한다. 사이클 문서의
# `§4-3` 은 그 파일에 4번 절이 없으므로 걸리고, `docs/problem.md` 가 자기 7번 절을
# 가리키는 `§7` 은 통과한다. allowlist 가 필요 없는 것이 이 형태의 값이다.

_DOCS = Path("docs")

#: 절 기호 인용. `§4-3` 의 선두 번호(`4`)로 같은 파일의 절 번호와 대조한다.
_SECTION_REF = re.compile(r"§\s*(\d+)")

#: 번호 붙은 절 제목 — `## 7. 실측이 …` · `### 4-3. 지표`.
_NUMBERED_HEADING = re.compile(r"^#+\s+(\d+)[.\-]")

#: 코드 스팬·코드 블록. 금지 형태를 **예시로 인용하는 문장**은 백틱 안에 있어야 한다.
_CODE_SPAN = re.compile(r"`[^`]*`")
_FENCE = re.compile(r"^\s*```")


def _strip_code(text: str) -> list[tuple[int, str]]:
    """코드 블록을 통째로 빼고, 남은 줄에서 코드 스팬을 지운다."""
    kept: list[tuple[int, str]] = []
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        kept.append((number, _CODE_SPAN.sub("", line)))
    return kept


def test_docs_의_절_인용이_같은_파일_안에서_풀린다() -> None:
    offenders: list[str] = []
    for path in sorted((_ROOT / _DOCS).rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        own = {int(m.group(1)) for m in map(_NUMBERED_HEADING.match, text.splitlines()) if m}
        for number, line in _strip_code(text):
            for ref in _SECTION_REF.finditer(line):
                if int(ref.group(1)) not in own:
                    offenders.append(f"{path.relative_to(_ROOT)}:{number}: {line.strip()}")

    assert not offenders, (
        "사이클 문서를 절 라벨로 인용하면 아카이브 시점에 복원 불가능해진다 "
        '(docs/standards.md "사이클 문서를 절 라벨로 인용하지 않는다"). '
        "규칙을 내용으로 옮겨 적어라:\n" + "\n".join(offenders)
    )


def test_docs_가드가_실제로_잡는다() -> None:
    """음성 대조 — 절 번호가 없는 파일에서 절 인용이 잡히는지 확인한다."""
    lines = _strip_code("# 결정 0014\n\n채택 규칙(spec §4-3)이 순서를 구속한다.\n")
    assert any(_SECTION_REF.search(line) for _, line in lines)
    # 백틱 안의 예시와 자기 절 인용은 통과해야 한다
    assert not any(_SECTION_REF.search(line) for _, line in _strip_code("`spec §4-3` 같은 인용"))
    assert _NUMBERED_HEADING.match("## 7. 실측이 문제정의를 어떻게 바꿨나")
    assert _NUMBERED_HEADING.match("### 4-3. 채택 규칙")
