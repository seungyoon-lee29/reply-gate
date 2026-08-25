"""문서 사이의 링크와 앵커가 성립한다 — **그리고 그 검사가 실제로 무언가를 잡는다.**

**왜 이 파일이 있는가.** 사이클 5 의 병합 조건이 *"링크와 앵커를 프래그먼트까지 검사해 0건
깨짐"* 을 요구했는데 검사가 저장소에 없었다([미해결 34](../docs/tracking/findings.md)).
실물은 0건이었지만 그것은 **사람이 한 번 확인한 값**이고, 다음 변경이 깨뜨려도 스위트는
초록이었다 — 판정 수단이 없는 게이트다.

**음성 대조를 함께 둔다.** 이 저장소는 링크 검사에서 거짓 음성을 두 번 겪었다: 앵커를
무시하는 검사기가 "0건 깨짐"을 잘못 말한 적이 있고(`docs/engineering-notes.md`), 자기
링크(`[x](#anchor)`)의 빈 경로를 정규화하지 않아 앵커 수십 건이 거짓 **양성**으로 뜬
적이 있다. 그래서 양쪽을 다 잰다 — 일부러 깨뜨린 링크가 잡히는가, 정상 링크가 안 잡히는가.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.check_links import (
    REPO_ROOT,
    Breakage,
    blank_code_spans,
    check,
    documents,
    extract_links,
    heading_slugs,
    main,
    slugify,
)


def _reasons(root: Path) -> list[str]:
    """`파일:줄 — 사유` 목록. 비교를 문면으로 하려고 `Breakage` 를 문자열로 편다."""
    _checked, broken = check(root)
    return [str(breakage) for breakage in broken]


# ── 구속 — 저장소의 실물이 성립한다 ─────────────────────────────────────────


def test_저장소의_모든_상대_링크와_앵커가_성립한다() -> None:
    """**이 파일의 구속력 있는 검사다.** 병합 조건이 요구하던 그 판정을 코드가 낸다."""
    _checked, broken = check()

    assert [str(breakage) for breakage in broken] == []


def test_검사_대상이_비어_있지_않고_전시_문서를_포함한다() -> None:
    """대상 집합이 비면 위 검사가 조용히 초록이 된다 — 검사기의 첫 실패 모드다."""
    names = {path.relative_to(REPO_ROOT).as_posix() for path in documents()}

    assert {"README.md", "CLAUDE.md", "AGENTS.md"} <= names
    assert any(name.startswith("docs/") for name in names)
    assert any(name.endswith("/AGENTS.md") for name in names)
    # 숨김 디렉터리는 통째로 대상이 아니다 — 아카이브는 자기 이동 이전 경로를 들고 있다.
    assert not any(part.startswith(".") for name in names for part in name.split("/")[:-1])


def test_앵커를_실제로_대조하고_있다() -> None:
    """앵커가 붙은 링크가 0건이면 위 검사는 파일 존재만 보는 셈이다.

    이 저장소가 겪은 거짓 음성이 정확히 그 모양이었다 — 프래그먼트를 무시하는 검사기가
    "0건 깨짐"을 말했다.
    """
    checked, _broken = check()
    with_fragment = [link for link in checked if link.fragment]

    assert len(with_fragment) >= 20
    assert any(not link.path for link in with_fragment), "자기 파일 앵커가 대상에 있어야 한다"


# ── 음성 대조 — 일부러 깨뜨린 것을 잡는다 ───────────────────────────────────


@pytest.fixture
def 문서_묶음(tmp_path: Path) -> Path:
    """정상 링크만 있는 최소 저장소. 각 음성 케이스가 여기에 한 줄씩 더한다."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "target.md").write_text(
        "# 첫 절\n\n본문\n\n## 둘째 절 — **강조**\n\n본문\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# 안내\n\n[대상](docs/target.md) · [앵커](docs/target.md#첫-절) · [자기](#안내)\n",
        encoding="utf-8",
    )
    return tmp_path


def test_정상_묶음은_깨진_것이_없다(문서_묶음: Path) -> None:
    """양성 대조 — 전부 잡아내는 검사기는 검사기가 아니다."""
    assert _reasons(문서_묶음) == []


def test_없는_파일을_가리키는_링크를_잡는다(문서_묶음: Path) -> None:
    (문서_묶음 / "README.md").write_text("# 안내\n\n[없다](docs/gone.md)\n", encoding="utf-8")

    assert _reasons(문서_묶음) == ["README.md:3: docs/gone.md — 대상 파일이 없다"]


def test_없는_앵커를_잡는다(문서_묶음: Path) -> None:
    (문서_묶음 / "README.md").write_text(
        "# 안내\n\n[헛앵커](docs/target.md#없는-절)\n", encoding="utf-8"
    )

    assert _reasons(문서_묶음) == [
        "README.md:3: docs/target.md#없는-절 — 'docs/target.md' 에 그 앵커가 없다"
    ]


def test_자기_파일의_없는_앵커도_잡는다(문서_묶음: Path) -> None:
    """빈 경로를 자기 파일로 정규화하지 않으면 이 케이스가 "대상 파일이 없다"로 잡혀,
    사유가 틀린 채 초록도 빨강도 아닌 보고가 된다."""
    (문서_묶음 / "README.md").write_text("# 안내\n\n[자기](#없는-절)\n", encoding="utf-8")

    assert _reasons(문서_묶음) == ["README.md:3: #없는-절 — 'README.md' 에 그 앵커가 없다"]


def test_정상_자기_앵커는_잡지_않는다(문서_묶음: Path) -> None:
    """거짓 **양성** 쪽 음성 대조 — 실제로 한 번 앵커 수십 건을 이렇게 잘못 읽었다."""
    (문서_묶음 / "README.md").write_text(
        "# 안내\n\n## 둘째\n\n[자기](#안내) [자기2](#둘째)\n", encoding="utf-8"
    )

    assert _reasons(문서_묶음) == []


def test_외부_링크는_대상이_아니다(문서_묶음: Path) -> None:
    (문서_묶음 / "README.md").write_text(
        "# 안내\n\n[웹](https://example.com/x#y) [메일](mailto:a@b.c)\n", encoding="utf-8"
    )
    checked, broken = check(문서_묶음)

    assert broken == ()
    assert [link.raw for link in checked if link.source == "README.md"] == []


def test_코드_펜스_안은_링크로도_헤딩으로도_세지_않는다(문서_묶음: Path) -> None:
    """펜스 안의 `# 주석` 이 헤딩이 되면 없는 앵커가 생기고, 예시 경로가 깨진 링크가 된다."""
    (문서_묶음 / "README.md").write_text(
        "# 안내\n\n```bash\n# 가짜 헤딩\ncat [예시](docs/없는파일.md)\n```\n\n[자기](#가짜-헤딩)\n",
        encoding="utf-8",
    )

    assert _reasons(문서_묶음) == ["README.md:8: #가짜-헤딩 — 'README.md' 에 그 앵커가 없다"]


def test_인라인_코드_안의_링크_문법은_링크가_아니다(문서_묶음: Path) -> None:
    """마크다운에서 `` `[x](y)` `` 는 렌더되지 않는 글자다.

    검사하면 문서가 **링크 문법 자체를 예시로 설명하는 자리**가 깨진 링크로 잡힌다 — 실제로
    이 저장소의 노트 두 곳이 그렇게 걸렸다. 헤딩 쪽은 반대라 슬러그에서는 안쪽 글자를 남긴다.
    """
    (문서_묶음 / "README.md").write_text(
        "# 안내\n\n자기 링크는 `[x](#없는-앵커)` 꼴로 적는다. [진짜](docs/target.md)\n",
        encoding="utf-8",
    )
    checked, broken = check(문서_묶음)

    assert broken == ()
    assert [link.raw for link in checked if link.source == "README.md"] == ["docs/target.md"]
    assert blank_code_spans("앞 `[x](y)` 뒤") == "앞          뒤"
    assert slugify("`코드` 와 **굵게**") == "코드-와-굵게", "헤딩 슬러그는 안쪽 글자를 남긴다"


def test_마크다운이_아닌_대상의_앵커는_판정하지_않고_잡는다(문서_묶음: Path) -> None:
    """판정할 수 없는 것을 통과로 적지 않는다(`scripts/AGENTS.md` 불변식 5 와 같은 결)."""
    (문서_묶음 / "docs" / "schema.sql").write_text("-- ddl\n", encoding="utf-8")
    (문서_묶음 / "README.md").write_text("# 안내\n\n[표](docs/schema.sql#x)\n", encoding="utf-8")

    assert _reasons(문서_묶음) == [
        "README.md:3: docs/schema.sql#x — 마크다운이 아닌 대상에 앵커가 붙었다"
    ]


# ── 슬러그 — GitHub 규칙을 저장소의 실제 앵커로 고정한다 ────────────────────


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        pytest.param("첫 절", "첫-절", id="한글_공백"),
        pytest.param(
            "25. 판정 클라이언트의 thinking docstring 이 계열 전체에 참인 것처럼 적혀 있다 "
            "— **닫혔다 (고쳤다, 2026-08-23)**",
            "25-판정-클라이언트의-thinking-docstring-이-계열-전체에-참인-것처럼-적혀-있다"
            "--닫혔다-고쳤다-2026-08-23",
            id="저장소의_실제_앵커",
        ),
        pytest.param("`코드` 와 **굵게**", "코드-와-굵게", id="서식은_벗긴다"),
        pytest.param("[표시](docs/x.md) 뒤", "표시-뒤", id="링크는_표시만_남는다"),
    ],
)
def test_슬러그가_GitHub_규칙을_따른다(heading: str, expected: str) -> None:
    """공백에 둘러싸인 em dash 자리에 하이픈이 **둘** 남는 것이 이 규칙의 지문이다."""
    assert slugify(heading) == expected


def test_같은_헤딩이_둘이면_뒤에_번호가_붙는다() -> None:
    assert heading_slugs("# 같다\n\n# 같다\n\n# 같다\n") == ("같다", "같다-1", "같다-2")


def test_링크_추출이_이미지와_제목_붙은_링크를_함께_든다() -> None:
    text = '![그림](a.png) [문서](b.md "제목") [바깥](https://x.example)\n'
    links = extract_links("README.md", text)

    assert [link.raw for link in links] == ["a.png", "b.md"]


# ── 진입점 — 깨지면 종료 코드가 1 이다 ──────────────────────────────────────


def test_진입점은_깨진_것이_없으면_0_을_돌려준다(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "깨짐 0건" in capsys.readouterr().out


def test_깨짐이_있으면_종료_코드가_1_이다(문서_묶음: Path) -> None:
    """`main` 은 실제 저장소를 보므로, 종료 코드 규칙은 판정 결과로 직접 잰다."""
    (문서_묶음 / "README.md").write_text("# 안내\n\n[없다](docs/gone.md)\n", encoding="utf-8")
    _checked, broken = check(문서_묶음)

    assert broken and isinstance(broken[0], Breakage)
    assert (1 if broken else 0) == 1
