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
