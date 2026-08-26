"""τ 가 검증되지 않은 임베딩 조건에서는 **제품 조립이 죽는다.**

**왜 이 파일이 있는가.** τ=0.06 은 `text-embedding-3-small` 1536차원에서만 기권군과
채택군을 가른다. `-3-large` 계열에서는 두 군의 분리 여유가 **음수**라 어떤 τ 도 두 군을
가르지 못한다(`docs/engineering-notes.md` "τ 는 임베딩 모델에 묶인다"). 그런데
`EMBEDDING_MODEL` 을 올리고 재색인까지 정상으로 마치면 임베딩 출처 검사는 통과하고,
게이트는 계속 돌지만 **아무것도 거르지 못하거나 정답 조항을 통째로 자른다** — 어느 쪽이든
오류가 아니라 "검색 품질"로 보인다.

그전까지 있던 장치는 실행 조건 지문의 짝(τ↔`embedding_model`) 하나뿐이었고, 그것은
**사후에 "대조 불가"로 드러나는 것**이지 실행을 막지 않는다. 이 파일이 지키는 것은
**조립 시점 거부**다.

**요청당 오류가 아니라 조립 오류인 이유**: 이것은 런타임 데이터 상태가 아니라 **설정
조합**의 문제이고 조립 시점에 알 수 있다. 이 저장소는 같은 모양의 문제 — 스위치는 켜졌는데
배선이 없는 상태 — 를 이미 조립 시점 오류로 다룬다(`pipeline.PipelineWiringError` ·
`evidence.RetrievalWiringError`). "기본값·`None` 이 조용히 끄는 fail-open 배선을 금지한다"가
그 규칙이고, 여기가 정확히 같은 자리다.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from reply_gate import adoption_axis, evidence, retrieval_eval
from reply_gate.config import (
    VALIDATED_ABSTENTION_EMBEDDINGS,
    AbstentionGateWiringError,
    Settings,
    get_settings,
)
from reply_gate.retrieval_strategies import (
    AbstentionGate,
    AbstentionStatistic,
    apply_abstention_gate,
)
from tests.conftest import declared_settings

#: τ 가 검증되지 않은 임베딩 조건들. 손계산 표(`docs/engineering-notes.md`)에서 분리 여유가
#: 음수로 나온 두 조건 + 차원만 어긋난 경우.
_검증되지_않은_조건 = (
    ("text-embedding-3-large", 1536),
    ("text-embedding-3-large", 3072),
    ("text-embedding-3-small", 3072),
    ("BAAI/bge-m3", 1024),
)


# ── 배선을 읽는 헬퍼 — 문자열 스캔이 아니라 AST 다 ──────────────────────────
#
# 소스를 문자열로 훑어 가드를 켜고 끄면, 주석·docstring 에 이름이 스치기만 해도 가드가
# 스스로 꺼진다(`tests/AGENTS.md` 불변식 7). AST 로 보면 주석은 애초에 없고 docstring 은
# 호출이 아니다 — 실제로 **부르는** 것만 남는다.


def _호출_이름들(tree: ast.AST) -> set[str]:
    """AST 서브트리 안에서 실제로 호출되는 이름 전부(`f()` 의 `f`, `x.f()` 의 `f`)."""
    이름: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                이름.add(func.attr)
            elif isinstance(func, ast.Name):
                이름.add(func.id)
    return 이름


def _모듈_호출_이름들(모듈: ModuleType) -> set[str]:
    경로 = Path(inspect.getsourcefile(모듈) or "")
    return _호출_이름들(ast.parse(경로.read_text(encoding="utf-8")))


def _함수_호출_이름들(함수: Callable[..., object]) -> set[str]:
    """메서드 소스는 클래스 안이라 들여쓰기가 남아 있다 — 떼어내야 파싱된다."""
    return _호출_이름들(ast.parse(textwrap.dedent(inspect.getsource(함수))))


# ── 양성: 현재 기본값 조립은 통과한다 ────────────────────────────────────────


def test_현재_기본값_조립은_통과한다() -> None:
    """**양성 대조** — 전부 거부하는 가드는 가드가 아니다.

    실측이 정당화하는 기본값(τ=0.06 · `text-embedding-3-small` d1536)에서는 게이트가
    그대로 조립돼야 한다. 선언값은 `declared_settings()` 로 잰다 — 인자 없는 `Settings()` 는
    개발자의 `.env` 를 함께 읽어 "기본값이 결정대로인가"가 로컬 환경 측정이 된다.
    """
    gate = declared_settings().abstention_gate()

    assert gate == AbstentionGate(statistic=AbstentionStatistic.SPREAD, tau=0.06)


def test_검증된_조건을_명시해도_통과한다() -> None:
    """환경 변수로 같은 값을 명시한 실행도 같은 조건이다."""
    settings = declared_settings(
        embedding_model="text-embedding-3-small", embedding_dimensions=1536
    )
    # **설정 객체를 단언식 밖에 둔다** — 실패 출력이 객체를 통째로 repr 한다
    # (`tests/AGENTS.md` 불변식 9).
    gate = settings.abstention_gate()

    assert gate is not None


def test_검증_목록이_현행_기본값을_담고_있다() -> None:
    """**음성 대조** — 목록이 비면 아래 거부 검사들은 통과해도 아무것도 지키지 않는다.

    반대로 목록이 넓어지면(누군가 τ 재산출 없이 조건을 늘리면) 이 검사가 그 사실을 드러낸다.
    목록을 넓히는 것은 사람이 하는 명시적 작업이고 승격과 같은 자격이다.
    """
    등재된_조건 = set(VALIDATED_ABSTENTION_EMBEDDINGS)

    assert 등재된_조건 == {("text-embedding-3-small", 1536)}


# ── 음성: 검증되지 않은 조건의 조립이 죽는다 ────────────────────────────────


@pytest.mark.parametrize(("model", "dimensions"), _검증되지_않은_조건)
def test_검증되지_않은_임베딩_조건은_조립에서_죽는다(model: str, dimensions: int) -> None:
    settings = declared_settings(embedding_model=model, embedding_dimensions=dimensions)

    with pytest.raises(AbstentionGateWiringError) as 잡힌:
        settings.abstention_gate()

    문면 = str(잡힌.value)
    assert model in 문면, "실행 조건을 이름으로 찍어야 무엇을 고칠지 알 수 있다"
    assert str(dimensions) in 문면


def test_수집기_조립이_그_자리에서_죽는다() -> None:
    """게이트를 **얻는 자리가 하나**라서, 제품 실행 경로의 조립이 같은 오류로 끝난다.

    `EvidenceCollector.__init__` 이 `Settings.abstention_gate()` 로만 게이트를 얻으므로
    가드를 그 한 자리에 걸면 수집기 조립이 자동으로 덮인다 — 조립자마다 검사를 복제하면
    새 조립자가 생길 때 조용히 빠진다.
    """
    settings = declared_settings(
        embedding_model="text-embedding-3-large", embedding_dimensions=3072
    )

    with pytest.raises(AbstentionGateWiringError):
        settings.abstention_gate()

    부르는_것 = _함수_호출_이름들(evidence.EvidenceCollector.__init__)
    assert "abstention_gate" in 부르는_것, (
        "수집기가 게이트를 다른 경로로 얻으면 이 가드가 조립을 덮지 못한다"
    )


# ── 공격: 우회 경로가 없다 ──────────────────────────────────────────────────


@pytest.mark.parametrize("tau", [0.0, 0.02, 0.06, 0.5, 1.0])
def test_τ_만_바꿔서는_통과하지_못한다(tau: float) -> None:
    """`-3-large` 계열에서는 **어떤 τ 도** 기권군과 채택군을 가르지 못한다.

    그래서 가드가 보는 것은 τ 값이 아니라 **임베딩 조건**이다. τ 를 만지는 것으로 조건
    불일치를 무마할 수 있으면 가드는 그저 눈금 하나짜리 장식이 된다.
    """
    settings = declared_settings(
        embedding_model="text-embedding-3-large", embedding_dimensions=3072, abstention_tau=tau
    )

    with pytest.raises(AbstentionGateWiringError):
        settings.abstention_gate()


@pytest.mark.parametrize(
    "이름",
    [
        "VALIDATED_ABSTENTION_EMBEDDINGS",
        "ABSTENTION_VALIDATED_EMBEDDINGS",
        "VALIDATED_ABSTENTION_EMBEDDINGS_JSON",
        "ABSTENTION_GATE_VALIDATED_EMBEDDINGS",
        "ABSTENTION_TAU_VALIDATED_EMBEDDINGS",
    ],
)
def test_환경_변수로_검증_목록을_무력화할_수_없다(
    monkeypatch: pytest.MonkeyPatch, 이름: str
) -> None:
    """검증 목록은 **설정 필드가 아니라 모듈 상수**여야 한다.

    설정 필드로 두면 `.env` 한 줄이 가드를 통째로 끄고, 그 실행은 "게이트를 켜고 돌았다"로
    집계된다. τ 를 재산출했을 때의 명시적 경로는 **사람이 이 상수를 고치는 것**이고,
    코드에 우회 플래그를 두지 않는다 — 승격과 같은 자격이다.
    """
    monkeypatch.setenv(이름, "text-embedding-3-large:3072")
    settings = Settings(embedding_model="text-embedding-3-large", embedding_dimensions=3072)

    with pytest.raises(AbstentionGateWiringError):
        settings.abstention_gate()


def test_검증_목록은_설정_필드가_아니다() -> None:
    """구조 검사 — 목록 이름이 설정 필드로 승격되면 환경 변수 표면이 다시 열린다."""
    필드로_읽히는_것 = sorted(
        name for name in Settings.model_fields if "validated" in name or "embeddings" in name
    )
    assert not 필드로_읽히는_것, (
        f"검증 목록은 환경에서 덮어쓸 수 없어야 한다: {', '.join(필드로_읽히는_것)}"
    )


def test_게이트를_끄면_조립이_죽지_않는다() -> None:
    """**경계를 명시적으로 못박는 검사다** — 게이트 꺼짐은 우회가 아니다.

    가드가 막는 것은 **조용히 무력해진 게이트**다. 게이트를 명시적으로 끈 실행에는 무력해질
    게이트가 없고, 그 사실이 실행 조건 지문에 `"꺼짐"` 으로 그대로 남아 조용하지도 않다.
    반대로 꺼짐까지 죽이면 임베딩을 올리는 사이클이 게이트를 끄고 실측하는 정상 경로마저
    막히고, 축별 원복(게이트 한 줄 끄기)이라는 이 저장소의 원복 수단이 사라진다.
    판단 근거는 결정 기록 0023 에 있다.
    """
    settings = declared_settings(
        abstention_gate_enabled=False,
        embedding_model="text-embedding-3-large",
        embedding_dimensions=3072,
    )
    gate = settings.abstention_gate()  # 단언식 밖에서 먼저 뽑는다(불변식 9)

    assert gate is None


def test_게이트를_끄지_않은_채로는_같은_조건이_죽는다() -> None:
    """위 검사의 짝 — 꺼짐 경로가 켜짐 경로까지 열어 주지 않는지 본다."""
    settings = declared_settings(
        abstention_gate_enabled=True,
        embedding_model="text-embedding-3-large",
        embedding_dimensions=3072,
    )

    with pytest.raises(AbstentionGateWiringError):
        settings.abstention_gate()


# ── 오프라인 비교·격자 도구는 대상이 아니다 ────────────────────────────────


@pytest.mark.parametrize("모듈", [retrieval_eval, adoption_axis])
def test_오프라인_도구는_설정_게이트를_얻지_않는다(모듈: ModuleType) -> None:
    """오프라인 비교·격자 도구는 τ 를 **명시 인자로** 받아 여러 임베딩 조건을 일부러 훑는다.

    그것이 그 도구들의 일이므로 방어를 거기 걸면 실측 자체가 막힌다 — τ 재산출이
    불가능해지면 가드가 자기가 요구하는 명시적 경로를 스스로 막는 셈이다. 그래서 이 도구들은
    `AbstentionGate` 를 직접 조립하고 `Settings.abstention_gate()` 를 부르지 않는다.
    """
    assert "abstention_gate" not in _모듈_호출_이름들(모듈)


def test_설정_게이트를_얻는_모듈은_같은_헬퍼에_잡힌다() -> None:
    """**음성 대조** — 위 검사는 *부재*만 단언하므로 헬퍼가 빈 집합을 돌려줘도 초록이다.

    같은 헬퍼를 실제로 게이트를 얻는 모듈(`evidence.py`)에 걸어 `abstention_gate` 가
    **나오는지** 본다. 나오지 않으면 위 검사는 아무것도 지키지 않는다는 뜻이다
    (`tests/AGENTS.md` 불변식 3).
    """
    부르는_것 = _모듈_호출_이름들(evidence)

    assert "abstention_gate" in 부르는_것
    # 헬퍼가 "무엇이든 다 들어 있는 집합"을 돌려주는 것도 아님을 함께 못박는다.
    assert "이런_이름의_호출은_없다" not in 부르는_것


def test_오프라인_격자가_검증되지_않은_조건에서도_돈다() -> None:
    """행동 검사 — 구조만 보면 다른 경로로 가드가 새어 들어와도 모른다."""
    gate = AbstentionGate(statistic=AbstentionStatistic.SPREAD, tau=0.02)
    verdict = apply_abstention_gate(gate, (0.9, 0.1, 0.05))

    assert verdict.abstains is False


# ── 폭발 반경: 방어는 그것이 지키는 축까지만 죽인다 ─────────────────────────
#
# 조립 거부는 옳지만 **반경이 지키는 축보다 넓으면** 관계없는 경로가 함께 죽는다.
# 실제로 그랬다: 임베딩 모델을 한 줄 바꾸면 임베딩을 한 번도 쓰지 않는 무과금 측정 1 이
# 리포트 0개로 죽고, 기권 게이트를 쓰지 않는 조회 전용 라우트가 500 이 됐다.
# `api.py` 의 판정 키 선검사가 같은 교훈을 이미 적어 두었다 — *"`get_service`(Depends)는
# GET 라우트와 공유하므로 거기에 넣으면 조회 전용 경로까지 죽는다"*.


def _미검증_조건을_환경에_건다(monkeypatch: pytest.MonkeyPatch) -> None:
    """실행 경로가 실제로 읽는 자리(설정 캐시)를 갈아 끼운다."""
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")
    get_settings.cache_clear()
    monkeypatch.setattr(
        "reply_gate.api.get_settings",
        lambda: declared_settings(
            embedding_model="text-embedding-3-large", embedding_dimensions=3072
        ),
    )


def test_조회_전용_조립은_τ_미검증_조건에서도_선다(monkeypatch: pytest.MonkeyPatch) -> None:
    """조회는 파이프라인을 쓰지 않으므로 파이프라인 조립 오류에 걸리면 안 된다.

    `get_service` 가 파이프라인을 **미리** 조립하던 동안 `GET /inquiries/{id}` 가
    이 조건에서 죽었다. 조립을 접수 경로로 미룬 것이 이 검사가 지키는 계약이다.
    """
    from reply_gate import api

    _미검증_조건을_환경에_건다(monkeypatch)

    opener = api.get_service()  # 여기서 죽으면 조회 라우트가 함께 죽는다

    assert callable(opener)


def test_접수는_같은_조건에서_여전히_조립_거부로_죽는다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """음성 대조 — 반경을 좁힌 것이 방어를 끈 것이 되면 안 된다.

    이 검사가 없으면 "조회를 살렸다"가 곧 "접수도 살렸다"가 돼도 스위트가 초록이다.
    """
    from reply_gate import api
    from reply_gate.pipeline import build_pipeline

    _미검증_조건을_환경에_건다(monkeypatch)
    설정 = declared_settings(embedding_model="text-embedding-3-large", embedding_dimensions=3072)

    with pytest.raises(AbstentionGateWiringError):
        build_pipeline(
            generation_client=api.build_generation_client(설정),
            embedding_client=api.build_embedding_client(설정),
            settings=설정,
        )


def test_조회는_파이프라인을_부르지_않는다() -> None:
    """구조 검사 — 조립을 미뤄도 조회가 그것을 부르면 반경이 그대로다."""
    from reply_gate import api

    assert "_open_pipeline" not in inspect.getsource(api.InquiryService.fetch)
    assert "_open_pipeline" in inspect.getsource(api.InquiryService.process), (
        "접수가 파이프라인을 얻는 자리를 잃으면 이 검사가 지키려는 대상이 사라진다"
    )


def test_무과금_지문_산출은_τ_미검증_조건에서도_돈다() -> None:
    """측정 1 은 임베딩을 한 번도 쓰지 않는다 — 지문 조립이 그것을 죽이면 안 된다.

    죽었을 때의 대가가 컸다: 채점이 **끝난 뒤** 지문 조립에서 터져 `exit 1` · 리포트
    0개였다. 계산을 해 놓고 버리는 모양이다.
    """
    from tests.test_condition_fingerprint import _script_fingerprint

    지문 = _script_fingerprint(
        settings=declared_settings(
            embedding_model="text-embedding-3-large", embedding_dimensions=3072
        )
    )

    assert 지문["abstention_gate_statistic"] == "미조립(τ 미검증 임베딩 조건)"
    assert 지문["abstention_tau"] == "미조립(τ 미검증 임베딩 조건)"


def test_미조립은_꺼짐도_미상도_아니다() -> None:
    """음성 대조 — 세 상태가 뭉개지면 대조가 거짓말을 한다.

    `꺼짐` 으로 적으면 **게이트를 끄고 돈 실행**과 구분되지 않고, `미상` 으로 적으면
    회귀 가드가 그것을 **관용**해(대조는 미상을 어긋남으로 세지 않는다) 조립이 거부된
    조건이 조용히 대조를 통과한다. 셋은 서로 다른 조건이다.
    """
    from tests.test_condition_fingerprint import _script_fingerprint

    미조립 = _script_fingerprint(
        settings=declared_settings(
            embedding_model="text-embedding-3-large", embedding_dimensions=3072
        )
    )["abstention_tau"]
    꺼짐 = _script_fingerprint(
        settings=declared_settings(
            embedding_model="text-embedding-3-large",
            embedding_dimensions=3072,
            abstention_gate_enabled=False,
        )
    )["abstention_tau"]
    정상 = _script_fingerprint()["abstention_tau"]

    assert 꺼짐 == "꺼짐"
    assert 정상 == "0.06"
    assert len({미조립, 꺼짐, 정상, "미상"}) == 4
