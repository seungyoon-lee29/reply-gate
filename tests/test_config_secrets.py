"""설정에 담긴 자격 증명이 `repr` 로 새지 않는다.

**왜 이 파일이 있는가.** 사이클 4 에서 `assert ... Settings(...) ...` 형태의 단언 하나가
실패하면서 pytest 가 설정 객체를 통째로 repr 했고, OpenAI API 키가 출력에 평문으로
실렸다. 저장소에 커밋된 적은 없지만 CI 로그·트레이스백·공유된 실패 출력은 전부 같은
경로다 — `docs/security.md` 의 비밀 관리 규칙이 막으려는 것이 바로 그것이다.

값을 읽는 쪽은 그대로다(`settings.openai_api_key`). 막는 것은 **repr 경로 하나**이고,
그것이 사고가 실제로 난 경로다. `print(settings.openai_api_key)` 처럼 값을 직접 찍는
것은 어떤 설정 클래스도 막지 못한다 — 그건 리뷰가 막는다.
"""

from __future__ import annotations

import re

from reply_gate.config import Settings

#: 자격 증명으로 읽어야 하는 필드 이름. `judge_max_output_tokens` 같은 이름이 걸리지
#: 않도록 "token" 단독은 넣지 않는다 — 실제 비밀은 키·비밀번호·시크릿 세 갈래다.
_비밀_이름 = re.compile(r"(api_key|password|secret)")

_카나리아 = {
    "openai_api_key": "카나리아-OPENAI-KEY",
    "anthropic_api_key": "카나리아-ANTHROPIC-KEY",
    "postgres_app_password": "카나리아-APP-PASSWORD",
    "postgres_ro_password": "카나리아-RO-PASSWORD",
}


def test_설정을_repr_해도_자격_증명이_평문으로_실리지_않는다() -> None:
    settings = Settings(**_카나리아)
    찍힌_것 = repr(settings) + str(settings) + f"{settings}"
    샌_것 = sorted(name for name, value in _카나리아.items() if value in 찍힌_것)
    assert not 샌_것, f"자격 증명이 repr 로 샜다: {', '.join(샌_것)}"

    # 값 자체는 그대로 읽혀야 한다 — 가린 것은 표시이지 값이 아니다.
    for name, value in _카나리아.items():
        assert getattr(settings, name) == value


def test_비밀로_읽히는_필드는_전부_repr_에서_빠져_있다() -> None:
    """새로 추가되는 자격 증명 필드가 조용히 repr 에 실리는 것을 막는다.

    위 검사는 지금 있는 넷만 본다. 이 검사는 **이름이 비밀로 읽히는 필드 전부**를 훑어
    다음에 추가될 것까지 잡는다.
    """
    새는_필드 = sorted(
        name
        for name, field in Settings.model_fields.items()
        if _비밀_이름.search(name) and field.repr
    )
    assert not 새는_필드, (
        "자격 증명 필드는 `Field(repr=False)` 여야 한다 — repr 에 실리면 설정 객체가 실린 "
        f"자리마다 값이 평문으로 따라간다: {', '.join(새는_필드)}"
    )
