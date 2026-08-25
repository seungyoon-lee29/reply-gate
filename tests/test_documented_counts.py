"""문서가 인용하는 **검증 건수**가 실제 실행값과 같다 — 저장소가 네 번 놓친 자리다.

**왜 이 파일이 있는가.** 이 저장소는 검증 건수 인용이 실측과 갈린 것을 **네 번** 겪었다
(`62e87ad` → `4cc81ea` "같은 함정 두 번째" → `9b524ea` → 사이클 2 적대 리뷰가 찾은
`ruff format` 183↔184). 사이클 5 감사의 사전 부검이 원인을 한 줄로 적었다 —
*"`grep -rn "1,431" tests/ src/` → **0건**. 테스트 하나만 늘어도 인용 네 곳이 동시에 틀리고
스위트는 초록이다."* 수치의 정본이 **"마지막에 손으로 다시 센 사람"** 이었다.

**그 규율을 검사로 바꾼다.** 문서에 적힌 건수를 긁어 실제 실행값과 한 곳씩 대조한다.

**모집단을 규칙으로 정한다** — 감사가 지적한 두 번째 결함이 *"갱신 대상의 모집단이 없다"*
였다. 그래서 자리를 손으로 열거하지 않고 **문서 전수를 훑고**, 다음 하나만 뺀다:

> **날짜가 붙은 `실행값`(`**…실행값**(YYYY-MM-DD)`) 문단은 과거 기록이므로 대조하지 않는다.**
> 그 줄은 그 날 돌린 출력의 사본이고, 나중에 참이 되도록 고치는 것이 오히려 거짓말이다.
> **줄이 아니라 문단이 단위다** — 기록의 수치는 다음 줄로 넘어가는 일이 흔하다.

날짜가 없는 인용은 **지금 상태에 대한 주장**이므로 전부 대조 대상이다.

**닫지 못한 것은 닫았다고 적지 않는다.** 둘이 남는다. ① 이 검사는 **건수만** 본다 — 문서가
인용하는 **헤드라인 모집단**(미끼 재현 분모 · `G06` 계보 등)은 커밋된 라이브 리포트에서
재집계해야 하고 이 파일의 범위가 아니다. ② **세션 핸드오프는 대상이 아니다** — 숨김
디렉터리에 있고 아카이브와 함께 옮겨 다니므로, 소스가 그 경로를 인용하는 것 자체를
`tests/test_durable_citations.py` 가 막는다. 그쪽 건수는 사람이 맞춘다.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest
from scripts.check_links import REPO_ROOT, documents

#: 과거 기록 줄 — 날짜가 붙은 `실행값`. 대조하지 않는다(위 모집단 규칙).
_DATED_RECORD = re.compile(r"실행값\*{0,2}\s*\(\d{4}-\d{2}-\d{2}\)")

#: 여러 줄에 걸친 인용을 한 줄로 잇지 않는다 — 줄 번호가 오류 메시지의 절반이다.
_NUMBER = r"([\d,]+)"


@dataclass(frozen=True)
class _인용:
    """문서 한 줄이 주장하는 건수 하나."""

    자리: str
    이름: str
    값: int


#: `(이름, 정규식)` — 이름은 아래 `_실측()` 의 키와 같다. **정규식은 명령 문면에 앵커를
#: 건다** — 숫자만 긁으면 본문의 다른 수치가 딸려 온다.
_패턴: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pytest 통과", re.compile(rf"uv run pytest\s+#\s*{_NUMBER}\s*passed")),
    (
        "pytest -m db 통과",
        re.compile(rf"uv run pytest -m db\b[^\n]*?{_NUMBER}\s*passed"),
    ),
    (
        "pytest -m db 제외",
        re.compile(rf"uv run pytest -m db\b[^\n]*?passed,\s*{_NUMBER}\s*deselected"),
    ),
    (
        "ruff format 파일",
        re.compile(rf"uv run ruff format --check \.\s*#\s*{_NUMBER}\s*files"),
    ),
    ("mypy 파일", re.compile(rf"uv run mypy\s+#[^\n]*?{_NUMBER}\s*files")),
    ("pytest -m db 통과", re.compile(rf"통합 테스트\({_NUMBER}건\)")),
    (
        "pytest 통과",
        re.compile(rf"\*\*{_NUMBER}건\*\*\s*\(그중 DB 통합"),
    ),
    (
        "pytest -m db 통과",
        re.compile(rf"\(그중 DB 통합\s*\*\*{_NUMBER}건\*\*\)"),
    ),
    (
        "ruff format 파일",
        re.compile(rf"`ruff format --check`\s*\*{{0,2}}{_NUMBER}\s*파일"),
    ),
    ("mypy 파일", re.compile(rf"`mypy`\s*\*{{0,2}}{_NUMBER}\s*파일")),
    ("pytest 통과", re.compile(rf"`pytest`\s*\*\*{_NUMBER}\s*통과")),
    ("pytest -m db 통과", re.compile(rf"`pytest -m db`\s*\*\*{_NUMBER}\s*통과")),
)


def _대상_문서() -> tuple[Path, ...]:
    """링크 검사와 같은 문서 집합 — 숨김 디렉터리의 세션 문서는 대상이 아니다."""
    return documents()


def _자리(문서: Path) -> str:
    """저장소 안이면 루트 기준 상대 경로, 밖이면 파일 이름 — 음성 대조가 `tmp_path` 를 쓴다."""
    try:
        return 문서.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return 문서.name


def 인용_전수(문서들: tuple[Path, ...] | None = None) -> tuple[_인용, ...]:
    """문서가 **지금 상태로 주장하는** 건수 전수. 날짜 붙은 기록은 문단째 뺀다.

    **줄이 아니라 문단이 단위다.** 기록 줄의 수치는 다음 줄로 넘어가는 일이 흔해서
    (`… 실행값**(2026-08-21): ruff check 0 · ruff format --check` ⏎ `182 파일 · …`),
    줄 단위로 걸러내면 넘어간 절반이 "지금 상태 주장"으로 잘못 읽힌다 — 실제로 그렇게 읽혀
    과거 기록 세 문단이 어긋남으로 잡혔다.
    """
    찾은: list[_인용] = []
    for 문서 in 문서들 if 문서들 is not None else _대상_문서():
        상대 = _자리(문서)
        기록_문단 = False
        for 번호, 줄 in enumerate(문서.read_text(encoding="utf-8").splitlines(), start=1):
            if not 줄.strip():
                기록_문단 = False
            elif _DATED_RECORD.search(줄):
                기록_문단 = True
            if 기록_문단:
                continue
            for 이름, 패턴 in _패턴:
                for 매치 in 패턴.finditer(줄):
                    찾은.append(
                        _인용(
                            자리=f"{상대}:{번호}", 이름=이름, 값=int(매치.group(1).replace(",", ""))
                        )
                    )
    return tuple(찾은)


# ── 실측 — 도구를 실제로 돌린 출력만 쓴다 ───────────────────────────────────


def _도구(*명령: str) -> str:
    완료 = subprocess.run(명령, capture_output=True, text=True, cwd=REPO_ROOT, check=False)
    return 완료.stdout + 완료.stderr


@cache
def _ruff_포맷_파일수() -> int | None:
    """`ruff format --check .` 이 센 파일 수. 도구가 없으면 `None`."""
    실행파일 = shutil.which("ruff")
    if 실행파일 is None:
        return None
    출력 = _도구(실행파일, "format", "--check", ".")
    수 = [int(값) for 값 in re.findall(r"(\d+)\s+files?\s+(?:already formatted|would be)", 출력)]
    return sum(수) if 수 else None


@cache
def _mypy_파일수() -> int | None:
    """`mypy` 가 검사한 소스 파일 수. 성공·실패 두 문면을 모두 읽는다."""
    출력 = _도구(sys.executable, "-m", "mypy")
    매치 = re.search(r"(?:no issues found in|checked)\s+(\d+)\s+source files?", 출력)
    return int(매치.group(1)) if 매치 else None


def _실측(session: pytest.Session) -> dict[str, int | None]:
    """이름 → 실행값. `None` 은 **이 실행에서 잴 수 없었다**는 뜻이다(0 이 아니다)."""
    선택된 = list(session.items)
    db = [item for item in 선택된 if item.get_closest_marker("db") is not None]
    return {
        "pytest 통과": len(선택된),
        "pytest -m db 통과": len(db),
        "pytest -m db 제외": len(선택된) - len(db),
        "ruff format 파일": _ruff_포맷_파일수(),
        "mypy 파일": _mypy_파일수(),
    }


def _부분_실행_사유(config: pytest.Config) -> str | None:
    """전체 스위트가 아니면 그 사유. 부분 실행에서 건수를 대조하면 거짓 빨강이 된다."""
    if config.option.keyword:
        return f"-k {config.option.keyword!r} 로 고른 실행"
    if config.option.markexpr:
        return f"-m {config.option.markexpr!r} 로 고른 실행"
    경로 = [값 for 값 in config.invocation_params.args if not 값.startswith("-")]
    return f"경로를 지정한 실행: {' '.join(경로)}" if 경로 else None


# ── 구속 ────────────────────────────────────────────────────────────────────


def test_인용_자리가_비어_있지_않다() -> None:
    """자리를 못 찾으면 아래 대조가 조용히 초록이 된다 — 검사기의 첫 실패 모드다."""
    인용들 = 인용_전수()

    assert {인용.이름 for 인용 in 인용들} == {
        "pytest 통과",
        "pytest -m db 통과",
        "pytest -m db 제외",
        "ruff format 파일",
        "mypy 파일",
    }
    # 같은 건수를 여러 자리가 인용한다는 사실 자체가 이 검사의 존재 이유다 — 네 곳이 동시에
    # 틀어졌던 사고가 그 모양이었다. 상수 대신 **자리 수 > 이름 수**로 잰다.
    assert len(인용들) > len({인용.이름 for 인용 in 인용들})
    assert {인용.자리.split(":")[0] for 인용 in 인용들} == {"README.md"}, (
        "지금 상태를 주장하는 건수 인용은 전시 문서 한 곳에 모여 있다 — 늘어나면 여기를 갱신한다"
    )


def test_문서가_인용한_건수가_실행값과_같다(
    request: pytest.FixtureRequest, pytestconfig: pytest.Config
) -> None:
    """**이 파일의 구속력 있는 검사다.**

    실패하면 고칠 것은 코드가 아니라 문서다 — 실행값이 정본이고 인용이 사본이다.
    """
    사유 = _부분_실행_사유(pytestconfig)
    if 사유 is not None:
        pytest.skip(f"건수 대조는 전체 스위트에서만 성립한다 — {사유}. `uv run pytest` 로 확인한다")

    실측 = _실측(request.session)
    못_잰_것 = sorted(이름 for 이름, 값 in 실측.items() if 값 is None)
    어긋난_것 = [
        f"{인용.자리}: {인용.이름} 인용 {인용.값} ≠ 실행값 {실측[인용.이름]}"
        for 인용 in 인용_전수()
        if 실측[인용.이름] is not None and 인용.값 != 실측[인용.이름]
    ]

    assert 어긋난_것 == [], (
        "문서의 검증 건수 인용이 실행값과 갈렸다 — **명령을 실제로 돌린 출력만** 옮겨 적는다:\n"
        + "\n".join(어긋난_것)
    )
    assert 못_잰_것 == [], f"실행값을 재지 못한 항목이 있다(0 으로 채우지 않는다): {못_잰_것}"


# ── 음성 대조 — 훑기가 실제로 어긋남을 잡는다 ───────────────────────────────


def test_훑기가_어긋난_인용을_잡는다(tmp_path: Path) -> None:
    """실제로 났던 사고 모양 그대로 — 숫자 하나만 틀린 문서."""
    문서 = tmp_path / "README.md"
    문서.write_text("```bash\nuv run pytest              # 1,452 passed\n```\n", encoding="utf-8")

    인용들 = 인용_전수((문서,))

    assert [(인용.이름, 인용.값) for 인용 in 인용들] == [("pytest 통과", 1452)]


def test_훑기가_날짜_붙은_과거_기록을_문단째_대조하지_않는다(tmp_path: Path) -> None:
    """과거 실측 줄을 나중에 참이 되도록 고치는 것이 오히려 거짓말이다.

    **줄이 아니라 문단이 단위다** — 아래 둘째 줄이 그 함정의 실물이다. 줄 단위로 걸러내면
    넘어간 절반(`182 파일 · … 1,431 통과`)이 "지금 상태 주장"으로 잡힌다.
    """
    문서 = tmp_path / "operations.md"
    문서.write_text(
        "**사이클 5 종료 시점의 실행값**(2026-08-21): `ruff check` 0 · `ruff format --check`\n"
        "182 파일 · `mypy` 75 파일 0 · `pytest` **1,431 통과** ·"
        " `pytest -m db` **176 통과 / skip 0**.\n"
        "\n"
        "`pytest` **9,999 통과** 는 문단이 끝난 뒤라 다시 대조 대상이다.\n",
        encoding="utf-8",
    )

    assert [(인용.이름, 인용.값) for 인용 in 인용_전수((문서,))] == [("pytest 통과", 9999)]


def test_훑기가_명령_문면_없는_숫자를_긁지_않는다(tmp_path: Path) -> None:
    """앵커 없이 숫자만 긁으면 본문의 다른 수치가 건수로 둔갑한다."""
    문서 = tmp_path / "status.md"
    문서.write_text("미끼 재현 12/95 · `G06` 18/19 · 지문 25칸.\n", encoding="utf-8")

    assert 인용_전수((문서,)) == ()
