"""설정에 담긴 자격 증명이 **표시·덤프 어느 경로로도** 평문으로 새지 않는다.

**왜 이 파일이 있는가.** 사이클 4 에서 `assert ... Settings(...) ...` 형태의 단언 하나가
실패하면서 pytest 가 설정 객체를 통째로 repr 했고, OpenAI API 키가 출력에 평문으로
실렸다. 저장소에 커밋된 적은 없지만 CI 로그·트레이스백·공유된 실패 출력은 전부 같은
경로다 — `docs/security.md` 의 비밀 관리 규칙이 막으려는 것이 바로 그것이다.

**그때 막은 것은 표시 경로 하나였다.** 자격 증명 필드에 `Field(repr=False)` 를 걸어
`repr`/`str` 을 닫았지만, `model_dump()`·`model_dump_json()`·`dict(settings)` 처럼 설정
객체를 **통째로 덤프하는 경로는 그 설정을 보지 않는다** — 표시에서 뺀 필드가 덤프에는
평문 그대로 실렸다. 사이클 5 에서 네 필드를 **비밀 전용 타입**(`pydantic.SecretStr`)으로
옮겨 두 경로를 함께 닫았고, 이 파일이 그것을 **덤프 경로 전수**로 지킨다.

**접속 문자열도 닫았다(미해결 24).** `settings.database_url` 은 값을 조립한 **새
문자열**이라 오래 필드 규칙 밖에 있었다 — 담는 것과 평문 `str` 로 내놓는 것을 가르지
않았기 때문이다. 사이클 5 감사가 카나리아로 실제 유출을 재현한 뒤(결정 0023 이 걸어 둔
조건), 조립 결과를 비밀 전용 타입으로 감싸 표시·덤프를 함께 닫았다.

**닫지 못한 경로는 닫았다고 적지 않는다.**
`print(settings.openai_api_key.get_secret_value())` 처럼 값을 직접 꺼내 찍는 것은 어떤
설정 클래스도 막지 못한다 — 그건 리뷰가 막는다. 꺼내는 자리의 **수**가 늘지 않는 것은
아래 자리 세기가 지킨다.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import pprint
import re
import traceback
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field, SecretStr

import reply_gate
from reply_gate.config import Settings
from tests.conftest import declared_settings

#: 애플리케이션 패키지 루트. 파일 목록은 **재귀로 유도**한다 — 손으로 적으면 하위 패키지가
#: 생기는 순간 그 안의 런타임 코드가 조용히 검사 밖으로 나간다(`tests/AGENTS.md` 불변식 4).
_패키지_루트 = Path(reply_gate.__file__).resolve().parent

#: 자격 증명으로 읽어야 하는 필드 이름.
#:
#: **넓게 잡고 예외를 등재한다 — 규칙을 좁히지 않는다.** 처음에는 `judge_max_output_tokens`
#: 오탐을 피하려고 `token` 을 규칙에서 **뺐다.** 그러면 지금 있는 넷은 전부 잡히지만 앞으로
#: 추가될 `*_token`·`*_credential` 형 비밀이 "전수" 검사 밖으로 **조용히** 나간다 — 규칙 축소는
#: 미래의 필드를 함께 지운다. 예외 등재는 그렇지 않다: 새 필드는 자동으로 걸리고, 비밀이 아닌
#: 것만 사유와 함께 여기 이름으로 남는다(사이클 5 감사 §3-6 의 재작성).
_비밀_이름 = re.compile(r"(api_key|password|secret|token|credential)")

#: 이름은 걸리지만 비밀이 아닌 필드 — **사유를 함께 적는다.** 아래 검사가 이 목록이 실제
#: 필드인지, 그리고 규칙에 실제로 걸리는지를 양쪽으로 확인한다(죽은 예외 금지).
_비밀_아님: dict[str, str] = {
    "judge_max_output_tokens": "판정 응답 토큰 상한 — 수치 설정이고 자격 증명이 아니다",
}


def _비밀_필드_이름() -> list[str]:
    """이름이 비밀로 읽히는 설정 필드 **전수**. 손으로 관리하지 않는다.

    다음에 추가될 자격 증명 필드도 같은 검사에 자동으로 걸리게 하려는 것이다 — 목록을
    손으로 적으면 새 필드가 조용히 가드 밖으로 나간다.
    """
    return sorted(
        name for name in Settings.model_fields if _비밀_이름.search(name) and name not in _비밀_아님
    )


def _카나리아_값() -> dict[str, str]:
    """비밀 필드마다 서로 다른 카나리아. 어느 필드가 샜는지 이름으로 갈라 보려는 것이다."""
    return {name: f"카나리아-{name}-값" for name in _비밀_필드_이름()}


def _카나리아_설정() -> Settings:
    """카나리아만 채운 설정. **`.env` 도 프로세스 환경도 읽지 않는다.**

    인자 없는 `Settings()` 를 쓰면 개발자의 실제 키가 이 검사의 덤프 문자열에 실린다 —
    누출 검사를 하면서 누출 경로를 하나 더 만드는 셈이다.
    """
    return declared_settings(**_카나리아_값())


def _덤프_경로(대상: Any) -> dict[str, str]:
    """설정 객체가 문자열이 되는 경로 전수. **키는 경로 이름, 값은 그 산출물이다.**

    표시(`repr`/`str`)만 훑으면 수정 전에도 초록이다 — 사고가 난 다음 자리가 **덤프**이고,
    덤프는 `Field(repr=False)` 를 보지 않는다. 직렬화 여러 종·중첩·복사본까지 함께 훑는다.
    """
    덤프 = 대상.model_dump()
    json_덤프 = 대상.model_dump(mode="json")
    try:
        raise RuntimeError(f"조립 실패: {대상!r} / {대상}")
    except RuntimeError as exc:
        트레이스백 = "".join(traceback.format_exception(exc))
    return {
        "repr()": repr(대상),
        "str()": str(대상),
        "f-string": f"{대상}",
        "format()": format(대상),
        "pprint.pformat(설정)": pprint.pformat(대상),
        "예외 트레이스백": 트레이스백,
        "model_dump()": repr(덤프),
        "str(model_dump())": str(덤프),
        "pprint.pformat(model_dump())": pprint.pformat(덤프),
        "model_dump(mode='json')": repr(json_덤프),
        "json.dumps(model_dump(mode='json'))": json.dumps(json_덤프, ensure_ascii=False),
        "model_dump_json()": 대상.model_dump_json(),
        "model_dump_json(indent=2)": 대상.model_dump_json(indent=2),
        "model_dump(exclude_none=True)": repr(대상.model_dump(exclude_none=True)),
        "dict(설정)": repr(dict(대상)),
        "list(설정)": repr(list(대상)),
        "vars(설정)": repr(vars(대상)),
        "설정.__dict__": repr(대상.__dict__),
        "중첩 dict": repr({"settings": 대상, "dump": 덤프}),
        "중첩 list": repr([대상, 덤프]),
        "model_copy(deep=True)": repr(대상.model_copy(deep=True)),
        "copy.deepcopy": repr(copy.deepcopy(대상)),
    }


def _샌_자리(대상: Any, 카나리아: dict[str, str]) -> list[str]:
    """`경로:필드` 목록. **평문 자체는 담지 않는다** — 실패 출력이 곧 누출이 되지 않도록."""
    return sorted(
        f"{경로}:{이름}"
        for 경로, 산출물 in _덤프_경로(대상).items()
        for 이름, 값 in 카나리아.items()
        if 값 in 산출물
    )


# ── 음성: 어느 덤프 경로에도 평문이 없다 ────────────────────────────────────


def test_설정을_통째_덤프해도_자격_증명이_평문으로_실리지_않는다() -> None:
    """**이 파일의 구속력 있는 검사다.**

    "필드에 표시 억제 속성이 붙어 있다"는 수정 전에도 초록이었다. 실제로 뚫려 있던 것은
    `model_dump()`·`model_dump_json()`·`dict(settings)` 같은 **덤프 경로**이고, 그 경로는
    `Field(repr=False)` 를 보지 않는다.
    """
    카나리아 = _카나리아_값()
    샌_것 = _샌_자리(_카나리아_설정(), 카나리아)
    assert not 샌_것, (
        "자격 증명이 설정 덤프로 평문 유출됐다 (경로:필드) — 비밀 전용 타입으로 옮겨야 "
        f"표시와 덤프가 함께 닫힌다: {', '.join(샌_것)}"
    )


def test_덤프_경로_훑기가_평문을_실제로_잡는다() -> None:
    """**음성 대조** — 훑기 자체가 빈 검사가 아님을 같은 파일이 증명한다.

    표시만 가린 필드(`str` + `repr=False`)를 같은 훑기에 넣으면 덤프 경로에서 잡혀야 한다.
    잡히지 않으면 위 검사는 통과해도 아무것도 지키지 않는다(`tests/AGENTS.md` 불변식 3).
    """

    class _표시만_가린_설정(BaseModel):
        openai_api_key: str = Field(default="", repr=False)

    카나리아 = {"openai_api_key": "카나리아-표시만-가린-값"}
    잡힌_것 = _샌_자리(_표시만_가린_설정(**카나리아), 카나리아)

    assert "model_dump()" in {자리.split(":", 1)[0] for 자리 in 잡힌_것}, (
        "덤프 경로 훑기가 표시만 가린 평문을 잡지 못한다 — 검사기가 비어 있다"
    )
    assert "repr()" not in {자리.split(":", 1)[0] for 자리 in 잡힌_것}, (
        "`repr=False` 는 표시만 닫는다는 전제가 깨졌다"
    )


# ── 음성: 표시 경로(사고가 처음 난 자리)도 계속 닫혀 있다 ──────────────────


def test_설정을_repr_해도_자격_증명이_평문으로_실리지_않는다() -> None:
    카나리아 = _카나리아_값()
    settings = _카나리아_설정()
    찍힌_것 = repr(settings) + str(settings) + f"{settings}"
    샌_것 = sorted(name for name, value in 카나리아.items() if value in 찍힌_것)
    assert not 샌_것, f"자격 증명이 repr 로 샜다: {', '.join(샌_것)}"


# ── 구조: 이름이 비밀로 읽히는 필드 전수 ────────────────────────────────────


def test_비밀로_읽히는_필드는_전부_비밀_전용_타입이다() -> None:
    """새로 추가되는 자격 증명 필드가 조용히 평문 타입으로 들어오는 것을 막는다.

    위 덤프 검사는 **지금 있는 필드**만 본다. 이 검사는 이름이 비밀로 읽히는 필드 전부를
    훑어 다음에 추가될 것까지 잡는다 — 비밀 전용 타입이면 표시와 덤프가 함께 닫힌다.
    """
    평문_필드 = sorted(
        name
        for name in _비밀_필드_이름()
        if Settings.model_fields[name].annotation is not SecretStr
    )
    assert not 평문_필드, (
        "자격 증명 필드는 `SecretStr` 이어야 한다 — 평문 `str` 은 표시를 가려도 "
        f"`model_dump()` 계열 덤프에 그대로 실린다: {', '.join(평문_필드)}"
    )


def test_비밀로_읽히는_필드는_전부_repr_에서_빠져_있다() -> None:
    """비밀 전용 타입 위에 표시 억제를 **겹쳐** 둔다.

    `SecretStr` 만으로도 표시는 마스킹되지만, 필드를 repr 에서 아예 빼면 마스킹 문자열조차
    설정 덤프에 자리를 만들지 않는다. 두 겹을 유지하는 것이 사고가 난 경로에 대한 이 저장소의
    답이다.
    """
    새는_필드 = sorted(name for name in _비밀_필드_이름() if Settings.model_fields[name].repr)
    assert not 새는_필드, (
        "자격 증명 필드는 `Field(repr=False)` 여야 한다 — repr 에 실리면 설정 객체가 실린 "
        f"자리마다 값이 평문으로 따라간다: {', '.join(새는_필드)}"
    )


def test_비밀_필드_목록이_비어_있지_않다() -> None:
    """**음성 대조** — 위 전수 검사들이 빈 집합을 훑고 초록이 되는 것을 막는다.

    이름 규칙이 어긋나 아무 필드도 걸리지 않으면 전수 검사는 통과하지만 아무것도 지키지
    않는다. 실제로 지키는 자격 증명 넷이 목록에 들어 있는지 확인한다.
    """
    assert set(_비밀_필드_이름()) == {
        "anthropic_api_key",
        "openai_api_key",
        "postgres_app_password",
        "postgres_ro_password",
    }


# ── 양성: 실행 경로가 값을 정상적으로 얻는다 ────────────────────────────────


def test_값이_필요한_곳은_명시적으로_꺼내_쓴다() -> None:
    """가린 것은 **표시와 덤프**이지 값이 아니다.

    꺼내는 자리가 코드에 드러나야 어디서 평문이 되는지 셀 수 있다 — 그래서 읽기는
    `get_secret_value()` 라는 이름 붙은 호출이다.
    """
    카나리아 = _카나리아_값()
    settings = _카나리아_설정()
    틀린_것 = sorted(
        name
        for name, value in 카나리아.items()
        if getattr(settings, name).get_secret_value() != value
    )
    assert not 틀린_것, f"자격 증명 값이 꺼내지지 않는다: {', '.join(틀린_것)}"


def test_접속_문자열도_비밀_전용_타입이라_표시_덤프로_새지_않는다() -> None:
    """**미해결 24 를 닫은 자리다** — 예전에는 이 경로가 의도적으로 열려 있었다.

    조립된 DSN 은 비밀번호를 담을 수밖에 없다(담지 않으면 접속이 안 된다). 그러나 **담는
    것과 평문 `str` 로 내놓는 것은 다른 문제**다 — 예전 구현은 `str` 을 돌려줘서 이 속성을
    로그·오류·설정 덤프에 그대로 실으면 비밀번호가 따라갔다. 사이클 5 감사가 카나리아로
    그 사실을 재현했고(결정 0023 이 "재현이 먼저다"로 미뤄 둔 조건), 여기서 감싸 닫는다.

    값이 필요한 자리는 여전히 `.get_secret_value()` 로 드러난다 — 아래 자리 세기가 그 수를
    센다.
    """
    settings = declared_settings(
        postgres_app_password="canary-app-pw",
        postgres_ro_password="canary-ro-pw",
    )
    # **설정 객체를 단언식 밖에 둔다** — 실패 출력이 객체를 통째로 repr 한다(불변식 9).
    앱_dsn = settings.database_url
    ro_dsn = settings.readonly_database_url
    표시 = f"{앱_dsn!r} {앱_dsn} {ro_dsn!r} {ro_dsn}"

    assert isinstance(앱_dsn, SecretStr)
    assert isinstance(ro_dsn, SecretStr)
    assert "canary-app-pw" not in 표시
    assert "canary-ro-pw" not in 표시
    assert 앱_dsn.get_secret_value() == (
        "postgresql://reply_gate_app:canary-app-pw@localhost:5433/reply_gate"
    )
    assert ro_dsn.get_secret_value() == (
        "postgresql://reply_gate_ro:canary-ro-pw@localhost:5433/reply_gate"
    )


def test_접속_문자열이_설정_덤프_전수에서도_평문으로_실리지_않는다() -> None:
    """**속성이라 필드 덤프에는 안 실린다** — 그래서 조립해 넣은 자리를 따로 훑는다.

    닫혔다는 주장의 실제 내용은 "이 값을 어디에 실어도 비밀번호가 안 따라온다"이므로,
    필드 훑기와 같은 덤프 경로 전수에 DSN 을 **직접 얹어** 확인한다.
    """
    settings = declared_settings(postgres_app_password="canary-dump-pw")

    class _DSN_을_담은_모델(BaseModel):
        dsn: SecretStr

    카나리아 = {"dsn": "canary-dump-pw"}
    샌_것 = _샌_자리(_DSN_을_담은_모델(dsn=settings.database_url), 카나리아)

    assert not 샌_것, f"접속 문자열이 덤프 경로로 샜다: {', '.join(샌_것)}"


def _평문으로_꺼내는_자리() -> dict[str, int]:
    """`src/reply_gate` 안에서 `.get_secret_value()` 를 부르는 파일과 횟수.

    **문자열 스캔이 아니라 AST 다** — 주석·docstring 에 이름이 스치기만 해도 세면
    검사가 스스로 흐려진다(`tests/AGENTS.md` 불변식 7).
    """
    자리: dict[str, int] = {}
    for 경로 in sorted(_패키지_루트.rglob("*.py")):
        tree = ast.parse(경로.read_text(encoding="utf-8"))
        횟수 = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_secret_value"
        )
        if 횟수:
            자리[경로.name] = 횟수
    return 자리


def test_평문으로_꺼내는_자리는_세_모듈_여섯_줄뿐이다() -> None:
    """**꺼내는 자리를 셀 수 있게 두는 것이 이 설계의 요점이다.**

    비밀 전용 타입이 막는 것은 실수로 흘러나가는 표시·덤프이지, 값을 꺼내는 것 자체가
    아니다(꺼내지 못하면 접속도 호출도 못 한다). 그래서 지켜야 할 불변식은 "꺼내지 않는다"가
    아니라 **"꺼내는 자리가 코드에 드러나고 그 수가 늘지 않는다"** 이다.

    - `config.py` 2회 — 접속 문자열 조립(앱 계정·read-only 계정)에 쓸 비밀번호를 꺼낸다.
    - `db.py` 1회 — 조립된 DSN 을 psycopg 에 넘기기 직전. **미해결 24 를 닫으면서 생긴
      자리다** — 접속 문자열을 `SecretStr` 로 감싼 대가로, 평문이 되는 지점이 "속성을 읽는
      모든 곳"에서 **이 한 줄**로 좁아졌다.
    - `llm.py` 3회 — SDK 생성자 인자(OpenAI 생성·Anthropic 판정·OpenAI 임베딩).

    새 자리가 생기면 이 검사가 깨진다. 깨졌을 때 할 일은 숫자를 올리는 것이 아니라 **그
    자리가 정말 필요한지 먼저 따지는 것**이다.
    """
    assert _평문으로_꺼내는_자리() == {"config.py": 2, "db.py": 1, "llm.py": 3}


def test_비밀번호가_없으면_접속_문자열에_자격_증명_구획이_없다() -> None:
    """**양성 대조** — 빈 비밀번호에서도 DSN 이 정상 조립된다(조건 보존)."""
    settings = declared_settings()
    앱_dsn = settings.database_url.get_secret_value()
    ro_dsn = settings.readonly_database_url.get_secret_value()

    assert 앱_dsn == "postgresql://reply_gate_app@localhost:5433/reply_gate"
    assert ro_dsn == "postgresql://reply_gate_ro@localhost:5433/reply_gate"


# ── 선언값 헬퍼 — 테스트가 개발자의 `.env` 를 재지 않게 한다 ─────────────────


def test_선언값_헬퍼는_환경도_env_파일도_읽지_않는다(monkeypatch: pytest.MonkeyPatch) -> None:
    """인자 없는 `Settings()` 는 `.env` 와 환경 변수를 함께 읽는다.

    "기본값이 결정대로인가"를 재는 테스트가 그것을 쓰면 결정 기록 대신 **로컬 환경**을
    재게 된다 — 개발자가 `ABSTENTION_TAU` 를 잠깐 바꿔 두면 코드와 결정이 갈렸는데도
    초록이 뜬다(사이클 4 리뷰 advisory). 이 검사는 헬퍼가 실제로 그 둘을 차단하는지 본다.
    """
    monkeypatch.setenv("ABSTENTION_TAU", "0.99")

    assert Settings().abstention_tau == 0.99, "인자 없는 생성은 환경을 읽는다(전제 확인)"
    assert declared_settings().abstention_tau == 0.06
    assert os.environ["ABSTENTION_TAU"] == "0.99", "헬퍼는 지운 환경을 되돌려 놓는다"


def test_선언값_헬퍼도_명시한_덮어쓰기는_받는다() -> None:
    """차단 대상은 **환경**이지 호출자의 명시적 인자가 아니다."""
    assert declared_settings(abstention_tau=0.02).abstention_tau == 0.02


def test_비밀_아님_예외가_죽어_있지_않다() -> None:
    """예외 목록은 **살아 있어야** 한다 — 죽은 예외는 규칙을 조용히 넓힌다.

    두 방향을 함께 본다. ① 이름이 실제 설정 필드인가(필드가 개명·삭제되면 예외가 유령이
    된다). ② 그 이름이 규칙에 **실제로 걸리는가**(안 걸리면 예외가 필요 없는데 남아 있어,
    다음 사람이 "이건 이미 빠져 있구나"로 잘못 읽는다).
    """
    유령 = sorted(name for name in _비밀_아님 if name not in Settings.model_fields)
    안_걸리는_것 = sorted(name for name in _비밀_아님 if not _비밀_이름.search(name))

    assert 유령 == [], f"예외로 적힌 이름이 설정 필드가 아니다 — 개명·삭제 뒤 남은 유령이다: {유령}"
    assert 안_걸리는_것 == [], (
        f"규칙에 걸리지도 않는 이름이 예외에 있다 — 예외가 아무것도 하지 않는다: {안_걸리는_것}"
    )
    assert all(사유.strip() for 사유 in _비밀_아님.values()), "예외에는 사유를 함께 적는다"


def test_규칙이_토큰_계열_비밀을_다시_잡는다() -> None:
    """**음성 대조** — 규칙을 넓힌 이유가 실제로 성립하는지 잰다.

    예전 규칙(`api_key|password|secret`)에서는 `*_token`·`*_credential` 형 이름이 통째로
    빠졌다. 그 이름들이 지금 규칙에 걸리고, 예외로 등재된 것만 빠지는지 확인한다.
    """
    걸리는_것 = [
        이름
        for 이름 in ("slack_bot_token", "gcp_credential", "refresh_token")
        if _비밀_이름.search(이름)
    ]

    assert 걸리는_것 == ["slack_bot_token", "gcp_credential", "refresh_token"]
    assert not re.compile(r"(api_key|password|secret)").search("slack_bot_token"), (
        "예전 규칙이 토큰 계열을 놓쳤다는 전제가 이 검사의 근거다"
    )
    assert "judge_max_output_tokens" not in _비밀_필드_이름()
