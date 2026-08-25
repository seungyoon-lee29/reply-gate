"""문서 사이의 상대 링크와 `#fragment` 앵커를 전수로 검사한다 — **무과금·오프라인**.

**왜 있는가.** 사이클 5 의 병합 조건이 문서 변경마다 *"링크와 앵커를 프래그먼트까지 검사해
0건 깨짐"* 을 요구했는데 **그 검사가 저장소에 없었다**([미해결 34](../docs/tracking/findings.md)).
판정 수단이 없는 게이트라, 실물이 0건인 것은 사람이 손으로 확인한 값이었고 다음 변경이
깨뜨려도 스위트는 초록이었다.

**두 가지를 본다.**

1. **대상 파일이 존재하는가** — 상대 경로가 가리키는 파일.
2. **`#fragment` 가 대상 파일의 실제 헤딩 슬러그와 일치하는가** — 파일만 보고 앵커를 안 보면
   *"0건 깨짐"* 이 거짓 음성이 된다. 이 저장소는 그 거짓 음성을 실제로 한 번 겪었다
   (`docs/engineering-notes.md` 의 링크 검사 절).

**자기 링크(`[x](#anchor)`)의 경로를 정규화한다.** 경로가 비면 대상은 **자기 파일**이다 —
`Path("")` 을 그대로 두고 대조하면 앵커가 통째로 거짓 양성이 된다(실제로 한 번 그렇게 읽었다).

**코드 펜스 안은 보지 않는다.** 펜스 안의 `# 주석` 이 헤딩으로 세어지면 없는 앵커가 생기고,
펜스 안의 예시 경로가 깨진 링크로 잡힌다.

```bash
uv run python -m scripts.check_links          # 깨진 것만 출력, 0건이면 종료 코드 0
uv run python -m scripts.check_links --list   # 검사한 링크 전수를 함께 출력
```

**외부 링크(`http`·`https`·`mailto`)는 대상 밖이다** — 네트워크를 타야 하고, 이 검사의
목적은 저장소 안의 상호 참조가 이동·개명에 어긋나지 않는가다.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

#: 저장소 루트 — 이 파일 기준으로 유도한다. 손으로 적으면 옮길 때 조용히 어긋난다.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: 검사 대상 문서 집합([미해결 34](../docs/tracking/findings.md) 이 정한 범위) — `docs/**` ·
#: `README.md` · `CLAUDE.md` · `AGENTS.md`. `AGENTS.md` 는 루트뿐 아니라 모듈별 파일도 같은
#: 이름이라 함께 든다.
DOC_GLOBS: tuple[str, ...] = ("docs/**/*.md", "README.md", "CLAUDE.md", "**/AGENTS.md")

#: 검사에서 빼는 것들 — 이름이 아니라 **성질**로 가른다.
#:
#: **숨김 디렉터리는 통째로 대상이 아니다.** 가상환경·도구 캐시·사이클 문서 아카이브가 전부
#: 거기 있고, 아카이브는 자기 이동 **이전** 경로를 들고 있어 지금 기준으로는 깨진 것이 정상이다.
#: 이름을 하나씩 열거하면 새 숨김 디렉터리가 생길 때마다 조용히 대상에 들어온다.
_EXCLUDED_NAMES: frozenset[str] = frozenset({"node_modules"})

#: 인라인 링크와 이미지. `[표시](대상)` · `![대체](대상)` 둘 다 파일을 가리킨다.
_LINK = re.compile(r"!?\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")

#: ``` 또는 ~~~ 로 열고 닫는 코드 펜스.
_FENCE = re.compile(r"^\s{0,3}(?P<mark>```+|~~~+)")

#: ATX 헤딩. setext(`===` 밑줄) 은 이 저장소가 쓰지 않는다.
_HEADING = re.compile(r"^(?P<level>#{1,6})\s+(?P<text>.*?)\s*#*\s*$")

#: 인라인 코드 스팬. 여는 백틱 수만큼으로 닫힌다.
_CODE_SPAN = re.compile(r"(`+)(?:(?!\1).)*?\1")

#: 네트워크를 타는 대상 — 이 검사의 범위 밖이다.
_EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//)", re.IGNORECASE)


@dataclass(frozen=True)
class Link:
    """문서 하나가 내놓은 상대 링크 하나."""

    #: 링크를 담은 파일 (저장소 루트 기준 상대 경로).
    source: str
    #: 1부터 세는 줄 번호.
    line: int
    #: 원문에 적힌 대상 그대로.
    raw: str
    #: `#` 앞의 경로 부분. 비어 있으면 자기 파일이다.
    path: str
    #: `#` 뒤의 앵커. 없으면 빈 문자열.
    fragment: str


@dataclass(frozen=True)
class Breakage:
    """깨진 링크 하나와 그 사유."""

    link: Link
    reason: str

    def __str__(self) -> str:
        return f"{self.link.source}:{self.link.line}: {self.link.raw} — {self.reason}"


def strip_code_fences(text: str) -> list[str]:
    """펜스 **안쪽 줄을 빈 줄로 바꾼** 줄 목록. 줄 번호를 보존하려고 지우지 않고 비운다."""
    lines = text.splitlines()
    kept: list[str] = []
    closing: str | None = None
    for line in lines:
        match = _FENCE.match(line)
        if closing is None:
            if match is not None:
                closing = match.group("mark")[:3]
                kept.append("")
                continue
            kept.append(line)
        else:
            kept.append("")
            if match is not None and match.group("mark").startswith(closing):
                closing = None
    return kept


def slugify(heading: str) -> str:
    """GitHub 이 헤딩에 붙이는 앵커 슬러그.

    서식(``**굵게**`` · `` `코드` `` · `[표시](대상)`)을 먼저 벗기고, 소문자로 내린 뒤
    **낱말 문자·하이픈·공백이 아닌 것을 지우고** 공백을 하이픈으로 바꾼다. 한글은 낱말
    문자라 그대로 남고, 문장부호(`.`·`,`·`(`·`—`)는 사라진다 — 그래서 공백에 둘러싸인
    em dash 자리에는 하이픈이 **둘** 남는다.
    """
    text = _LINK.sub(lambda m: m.group("text"), heading)
    text = re.sub(r"[*_`~]", "", text)
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return text.strip().replace(" ", "-")


def heading_slugs(text: str) -> tuple[str, ...]:
    """문서가 내놓는 앵커 전수 — **중복은 GitHub 처럼 `-1`·`-2` 를 붙인다.**"""
    seen: Counter[str] = Counter()
    slugs: list[str] = []
    for line in strip_code_fences(text):
        match = _HEADING.match(line)
        if match is None:
            continue
        base = slugify(match.group("text"))
        if not base:
            continue
        count = seen[base]
        seen[base] += 1
        slugs.append(base if count == 0 else f"{base}-{count}")
    return tuple(slugs)


def blank_code_spans(line: str) -> str:
    """인라인 코드 스팬을 같은 길이의 공백으로 바꾼다 — **열 위치를 보존한다.**

    마크다운에서 `` `[x](y)` `` 는 링크가 아니라 글자다. 검사하면 문서가 링크 문법 자체를
    **예시로 설명하는 자리**가 깨진 링크로 잡힌다(이 저장소에 실제로 둘 있다). 헤딩 쪽은
    반대다 — GitHub 이 코드 스팬의 안쪽 글자를 슬러그에 담으므로 거기서는 지우지 않는다.
    """
    return _CODE_SPAN.sub(lambda m: " " * len(m.group(0)), line)


def extract_links(source: str, text: str) -> tuple[Link, ...]:
    """문서 하나가 내놓는 **저장소 안** 링크 전수. 외부 링크는 여기서 떨어진다."""
    links: list[Link] = []
    for number, raw_line in enumerate(strip_code_fences(text), start=1):
        line = blank_code_spans(raw_line)
        for match in _LINK.finditer(line):
            target = match.group("target")
            if _EXTERNAL.match(target):
                continue
            path, _, fragment = target.partition("#")
            links.append(Link(source=source, line=number, raw=target, path=path, fragment=fragment))
    return tuple(links)


def documents(root: Path = REPO_ROOT) -> tuple[Path, ...]:
    """검사 대상 문서 전수 — **글롭에서 유도한다.** 손으로 적으면 새 문서가 검사 밖으로 나간다."""
    found: set[Path] = set()
    for pattern in DOC_GLOBS:
        for path in root.glob(pattern):
            parts = path.relative_to(root).parts
            hidden = any(part.startswith(".") for part in parts[:-1])
            if path.is_file() and not hidden and not (_EXCLUDED_NAMES & set(parts)):
                found.add(path)
    return tuple(sorted(found))


def check(root: Path = REPO_ROOT) -> tuple[tuple[Link, ...], tuple[Breakage, ...]]:
    """`(검사한 링크 전수, 깨진 것)`.

    앵커 대조는 **대상이 마크다운일 때만** 한다 — 다른 파일 형식의 앵커는 이 검사가 판정할
    수 없고, 판정할 수 없는 것을 통과로도 실패로도 적지 않는다.
    """
    slug_cache: dict[Path, frozenset[str]] = {}

    def slugs_of(path: Path) -> frozenset[str]:
        if path not in slug_cache:
            slug_cache[path] = frozenset(heading_slugs(path.read_text(encoding="utf-8")))
        return slug_cache[path]

    checked: list[Link] = []
    broken: list[Breakage] = []
    for document in documents(root):
        source = document.relative_to(root).as_posix()
        for link in extract_links(source, document.read_text(encoding="utf-8")):
            checked.append(link)
            # 경로가 비면 자기 파일이다 — 정규화하지 않으면 앵커가 통째로 거짓 양성이 된다.
            target = (document.parent / link.path).resolve() if link.path else document
            if not target.exists():
                broken.append(Breakage(link, "대상 파일이 없다"))
                continue
            if not link.fragment:
                continue
            if target.is_dir() or target.suffix.lower() != ".md":
                broken.append(Breakage(link, "마크다운이 아닌 대상에 앵커가 붙었다"))
                continue
            if link.fragment not in slugs_of(target):
                broken.append(Breakage(link, f"'{target.relative_to(root)}' 에 그 앵커가 없다"))
    return tuple(checked), tuple(broken)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="검사한 링크를 전수로 출력한다")
    args = parser.parse_args(argv)

    checked, broken = check()
    if args.list:
        for link in checked:
            print(f"{link.source}:{link.line}: {link.raw}")

    print(f"문서 {len(documents())}개 · 링크 {len(checked)}개 · 깨짐 {len(broken)}건")
    for breakage in broken:
        print(f"  {breakage}", file=sys.stderr)
    return 1 if broken else 0


if __name__ == "__main__":  # pragma: no cover - 진입점
    raise SystemExit(main())
