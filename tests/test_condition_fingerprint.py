"""실행 조건 지문 확장 — 새 항목 일곱 · 결정론 · 미상 처리 · 기권 사유 배선.

**지문 항목을 더하면 그 이전에 커밋된 라이브 전부가 그 항목 "미상"이 되어 대조 계보가 한 번
끊긴다.** 그 대가를 치르는 만큼 이 파일이 지키는 것은 셋이다.

* **더한 항목이 실제로 움직인다.** 내용 지문이 상수로 굳으면 계보만 끊고 얻는 것이 없다 —
  그래서 항목마다 "그 항목이 재는 규칙을 바꾸면 값이 바뀐다"를 뮤테이션으로 못박는다.
* **내용 지문이 결정론이다.** 같은 코드·같은 데이터면 프로세스가 달라도 같은 값이어야 한다.
  파이썬 해시 랜덤화는 프로세스마다 다르므로 **별도 프로세스에서도** 확인한다.
* **미상을 관용하는 대조 술어와 확인된 동일성만 인정하는 세트 편입 술어의 차이가 남는다.**
  관용을 편입에까지 끌고 가면 지문이 통째로 없는 산출물이 "전부 미상 = 어긋남 없음"으로
  읽혀 3회 정족수를 채운다.

기권 게이트의 **미정의 사유**가 근거 묶음 → 파이프라인 결과 → 평가 리포트로 이어지는
배선도 여기서 검사한다. 사유는 처분(발동/열림)과 별개로 밖으로 나가야 한다 — 처분만
읽으면 "2건 미만이라 열어 뒀다"와 "정의됐고 통과했다"가 산출물에서 같아진다.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, cast

import psycopg
import pytest
from psycopg.rows import DictRow
from scripts import evaluate

from reply_gate import evaluation, gate, llm, retrieval_strategies, sql_guard
from reply_gate.config import Settings
from reply_gate.contracts import Claim, EscalationReason, InquiryStatus, IntentSource
from reply_gate.evaluation import (
    DEFAULT_GOLDEN_SET_PATH,
    DEFAULT_JUDGE_FIXTURES_PATH,
    DEFAULT_L1_FIXTURES_PATH,
    EvaluationReport,
    ExpectedOutcomeSet,
    GoldenCase,
    RunConditions,
    SkippedMeasurement,
    build_report,
    evaluate_case,
    measure_gate_accuracy,
    measure_pipeline_agreement,
    render_markdown,
    report_to_json,
    utc_now_iso,
)
from reply_gate.evidence import EvidenceCollection
from reply_gate.pipeline import InquiryPipeline, ProcessedInquiry, new_inquiry_id
from reply_gate.regression_guard import (
    FINGERPRINT_FIELDS,
    PAIRED_FINGERPRINT_FIELDS,
    ConditionFingerprint,
    load_run_summary,
)
from reply_gate.retrieval_strategies import AbstentionUndefined
from tests.conftest import declared_settings

_ROOT = Path(__file__).resolve().parents[1]
_NO_CONN = cast(psycopg.Connection[DictRow], None)

#: 이번 확장이 더하는 일곱 항목. **이 목록이 다음 재등재의 입력이다** — 이름이 갈리면
#: 사람이 참조 파일에 옮겨 적은 값이 가드가 찾는 자리에 없다.
NEW_FINGERPRINT_FIELDS: Final[tuple[str, ...]] = (
    "run_completion",
    "judge_thinking",
    "draft_rule_version",
    "l1_fixture_version",
    "sql_guard_version",
    "retrieval_order",
    "abstention_undefined_policy",
)

#: 지문 확장 **이전에** 커밋된 라이브 산출물. 새 항목이 하나도 없어야 한다.
_LEGACY_LIVE_STEM: Final = "evaluation-live-l2-7"


# ── 대역과 조립 ─────────────────────────────────────────────────────────────


def _args(*, l1_fixtures: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        golden_set=DEFAULT_GOLDEN_SET_PATH,
        judge_fixtures=DEFAULT_JUDGE_FIXTURES_PATH,
        l1_fixtures=DEFAULT_L1_FIXTURES_PATH if l1_fixtures is None else l1_fixtures,
        stub_llm=False,
    )


def _script_fingerprint(
    *, settings: Settings | None = None, l1_fixtures: Path | None = None
) -> dict[str, str]:
    """진입점이 만드는 명시 지문 — 실행 인자에 묶인 항목이 여기 있다."""
    resolved = declared_settings() if settings is None else settings
    return evaluate._condition_fingerprint(
        args=_args(l1_fixtures=l1_fixtures), settings=resolved, run_settings=resolved
    )


def _conditions(
    *,
    fingerprint: dict[str, str] | None = None,
    aborted: tuple[str, ...] = (),
    declared: tuple[str, ...] = (),
) -> RunConditions:
    return RunConditions(
        started_at=utc_now_iso(),
        generation="OpenAI `gpt-5.6-terra` (effort=기본값)",
        embedding="OpenAI `text-embedding-3-small` (1536차원)",
        embedding_dimensions=1536,
        judge="Anthropic `claude-sonnet-5` (effort=기본값)",
        retrieval_strategy="vector+rewrite",
        similarity_threshold=0.3,
        top_k=5,
        l1_fixture_count=1,
        golden_case_count=1,
        judge_fixture_count=1,
        l1_fixtures_path=str(DEFAULT_L1_FIXTURES_PATH),
        golden_set_path=str(DEFAULT_GOLDEN_SET_PATH),
        judge_fixtures_path=str(DEFAULT_JUDGE_FIXTURES_PATH),
        api_key_present=False,
        judge_api_key_present=False,
        l2_enabled=True,
        measurement2_is_real=True,
        measurement3_is_real=True,
        billed=True,
        condition_fingerprint=_script_fingerprint() if fingerprint is None else fingerprint,
        declared_experiment_fields=declared,
        aborted_measurements=aborted,
    )


def _current(**overrides: Any) -> ConditionFingerprint:
    return _conditions(**overrides).fingerprint()


def _legacy() -> ConditionFingerprint:
    return load_run_summary(_ROOT / "reports" / f"{_LEGACY_LIVE_STEM}.json").fingerprint


# ── 계약 A — 새 항목 일곱이 지문에 실린다 ───────────────────────────────────


def test_새_지문_항목_일곱이_실행_조건에_실린다() -> None:
    """하나라도 빠지면 이번 사이클이 바꾼 축이 산출물에 흔적을 남기지 않는다."""
    values = _current().values

    missing = [name for name in NEW_FINGERPRINT_FIELDS if values.get(name) is None]
    assert missing == [], missing


def test_새_지문_항목_일곱이_회귀_가드의_필수_목록에_있다() -> None:
    """필수 목록에 없으면 **항목이 통째로 사라진** 산출물끼리 "같다"로 읽힌다."""
    assert set(NEW_FINGERPRINT_FIELDS) <= set(FINGERPRINT_FIELDS)


def test_판정_사고_과정_축은_effort_칸_옆의_새_칸이다() -> None:
    """effort 는 "얼마나"이고 thinking 은 "켜는가"다 — 한 칸에 접으면 축이 하나 사라진다."""
    values = _current().values

    assert values["judge_effort"] == "기본값"
    assert values["judge_thinking"] is not None
    assert values["judge_thinking"] != values["judge_effort"]


def test_리포트_JSON_의_조건_지문에도_새_항목이_실린다() -> None:
    """가드가 읽는 것은 산출물 JSON 이다 — 메모리에만 있으면 대조에 못 쓴다."""
    payload = report_to_json(_stub_report())
    values = payload["conditions"]["condition_fingerprint"]

    missing = [name for name in NEW_FINGERPRINT_FIELDS if values.get(name) is None]
    assert missing == [], missing


# ── 계약 B — 내용 지문의 결정론 ─────────────────────────────────────────────


def test_코드_지문은_같은_프로세스에서_두_번_불러도_같다() -> None:
    from reply_gate.evaluation import code_condition_fingerprint

    assert code_condition_fingerprint() == code_condition_fingerprint()


_SUBPROCESS_SNIPPET: Final = (
    "import json;"
    "from reply_gate.evaluation import code_condition_fingerprint;"
    "print(json.dumps(code_condition_fingerprint(), ensure_ascii=False, sort_keys=True))"
)


def _fingerprint_in_subprocess(seed: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    # 같은 인터프리터를 고정 인자로 부른다 — 셸을 태우지 않는다.
    completed = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SNIPPET],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(_ROOT),
        env=env,
        timeout=120,
    )
    return cast(dict[str, str], json.loads(completed.stdout))


def test_코드_지문은_해시_시드가_달라도_같다() -> None:
    """집합을 정렬하지 않고 지문에 담으면 **프로세스마다 값이 달라진다.**

    파이썬 문자열 해시는 프로세스마다 무작위 시드를 받는다 — 같은 코드가 다른 값을 내면
    다음 실행이 자기 자신과 "대조 불가"가 되어 지문이 스스로를 무력화한다.
    """
    from reply_gate.evaluation import code_condition_fingerprint

    first = _fingerprint_in_subprocess("0")
    second = _fingerprint_in_subprocess("31337")

    assert first == second
    assert first == code_condition_fingerprint()


# ── 계약 C — 내용 지문이 실제로 문다 (뮤테이션) ─────────────────────────────


def _draft_rule_version() -> str:
    from reply_gate.evaluation import code_condition_fingerprint

    return code_condition_fingerprint()["draft_rule_version"]


def _sql_guard_version() -> str:
    from reply_gate.evaluation import code_condition_fingerprint

    return code_condition_fingerprint()["sql_guard_version"]


def test_접기_문자_집합이_바뀌면_초안_판정_지문이_움직인다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _draft_rule_version()
    monkeypatch.setattr(evaluation, "NUMERIC_SEPARATOR_VARIANTS", frozenset({"-"}))

    assert _draft_rule_version() != before


def test_접기_동작이_바뀌면_초안_판정_지문이_움직인다(monkeypatch: pytest.MonkeyPatch) -> None:
    """접기는 **이름이 아니라 동작**으로 담는다 — 이름만 담으면 본문 변경이 안 보인다."""
    before = _draft_rule_version()
    monkeypatch.setattr(evaluation, "fold_numeric_for_detection", lambda text: text)

    assert _draft_rule_version() != before


def test_패턴_집합이_줄면_초안_판정_지문이_움직인다(monkeypatch: pytest.MonkeyPatch) -> None:
    before = _draft_rule_version()
    monkeypatch.setattr(evaluation, "DEFAULT_PII_PATTERNS", gate.DEFAULT_PII_PATTERNS[:-1])

    assert _draft_rule_version() != before


def test_기각_사유_순서가_바뀌면_초안_판정_지문이_움직인다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """사유 순서는 결정론 계약의 일부다 — 순서가 바뀌면 같은 초안이 다른 사유열을 낸다."""
    before = _draft_rule_version()
    monkeypatch.setattr(evaluation, "REASON_ORDER", tuple(reversed(gate.REASON_ORDER)))

    assert _draft_rule_version() != before


def test_지문이_게이트_모듈의_정본_이름을_읽는다() -> None:
    """뮤테이션이 소비자 쪽 이름을 갈아 끼우므로, **정본에서 가져왔음**을 따로 못박는다.

    패턴형 개인정보 정의의 단독 소유자는 게이트 모듈이다 — 여기서 자기 목록을 만들면
    지문이 실행 경로와 다른 규칙을 재게 된다(층별 정의 분기는 실제로 뚫린 경로다).
    """
    # 재수출 표면이 아니라 **모듈 전역**을 본다 — 평가 모듈은 정의의 소유자가 아니므로
    # 이 이름들을 밖으로 다시 내보내지 않는다.
    module_globals = vars(evaluation)

    assert module_globals["DEFAULT_PII_PATTERNS"] is gate.DEFAULT_PII_PATTERNS
    assert module_globals["NUMERIC_SEPARATOR_VARIANTS"] is gate.NUMERIC_SEPARATOR_VARIANTS
    assert module_globals["fold_for_detection"] is gate.fold_for_detection
    assert module_globals["fold_numeric_for_detection"] is gate.fold_numeric_for_detection
    assert module_globals["REASON_ORDER"] is gate.REASON_ORDER


def test_초안_판정_지문은_주석_한_줄에_흔들리지_않는다() -> None:
    """양성 대조 — 소스 파일을 통째로 해시하면 주석 한 줄이 계보를 끊는다.

    같은 규칙을 두 번 읽으면 같은 값이어야 하고, 그 값은 **선언과 동작**에서만 나온다.
    """
    assert _draft_rule_version() == _draft_rule_version()
    assert _draft_rule_version().startswith("draftrules-")


def test_캐스트_허용_타입이_줄면_조회_가드_지문이_움직인다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _sql_guard_version()
    monkeypatch.setattr(sql_guard, "ALLOWED_CAST_TYPES", ("text",))

    assert _sql_guard_version() != before


def test_유래_승인_규칙이_좁아지면_조회_가드_지문이_움직인다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """빈칸 채우기 계열의 승인은 **선언이 아니라 코드**에 있다 — 탐침이 그것을 잡는다."""
    before = _sql_guard_version()
    monkeypatch.setattr(sql_guard, "_COALESCE_FUNCTION_TYPES", ())

    assert _sql_guard_version() != before


def test_조회_가드_탐침이_실제로_유래를_가른다() -> None:
    """양성 대조 — 전부 거부하거나 전부 승인하는 탐침이면 뮤테이션이 우연히 문다."""
    from reply_gate.evaluation import sql_guard_probe_outcomes

    outcomes = sql_guard_probe_outcomes()

    assert any(item.startswith("ok|") and item != "ok|" for item in outcomes), outcomes
    assert any(item == "ok|" for item in outcomes), outcomes
    assert any(item.startswith("reject|") for item in outcomes), outcomes


def test_기권_미정의_처분을_뒤집으면_지문이_움직인다(monkeypatch: pytest.MonkeyPatch) -> None:
    """**미정의 처리를 바꿔도 기존 지문 값은 전혀 움직이지 않는다** — 이 칸이 그 자리다."""
    from reply_gate.evaluation import code_condition_fingerprint

    before = code_condition_fingerprint()["abstention_undefined_policy"]
    monkeypatch.setattr(AbstentionUndefined, "abstains", property(lambda self: True))

    assert code_condition_fingerprint()["abstention_undefined_policy"] != before


def test_기권_미정의_경계가_바뀌면_지문이_움직인다(monkeypatch: pytest.MonkeyPatch) -> None:
    """처분만 담으면 "1위 코사인 0 이하"의 **경계**가 바뀌어도 값이 안 움직인다."""
    from reply_gate.evaluation import code_condition_fingerprint

    before = code_condition_fingerprint()["abstention_undefined_policy"]
    monkeypatch.setattr(retrieval_strategies, "undefined_statistic_reason", lambda scores: None)

    assert code_condition_fingerprint()["abstention_undefined_policy"] != before


def test_검색_병합의_tie_break_를_없애면_순서_지문이_움직인다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reply_gate.evaluation import code_condition_fingerprint

    before = code_condition_fingerprint()["retrieval_order"]

    def _without_tie_break(
        *,
        original: Sequence[retrieval_strategies.VectorHit],
        rewritten: Sequence[retrieval_strategies.VectorHit],
    ) -> tuple[retrieval_strategies.VectorHit, ...]:
        del rewritten
        return tuple(original)

    monkeypatch.setattr(retrieval_strategies, "merge_rewritten_rankings", _without_tie_break)

    assert code_condition_fingerprint()["retrieval_order"] != before


def test_검색_순서_지문이_DB_정렬_키를_그대로_싣는다() -> None:
    """tie-break 유무를 사람이 산출물에서 바로 읽을 수 있어야 한다."""
    from reply_gate.evaluation import code_condition_fingerprint

    value = code_condition_fingerprint()["retrieval_order"]

    assert "evidence_id" in value, value


def test_정렬_절_추출이_정렬_없음을_구분한다() -> None:
    """음성 대조 — 추출기가 아무것도 못 찾을 때 조용히 옛 값을 남기면 안 된다."""
    from reply_gate.evaluation import order_by_clause

    assert order_by_clause("ORDER BY embedding <=> %s,\n  evidence_id\n") == (
        "embedding <=> %s, evidence_id"
    )
    assert order_by_clause("ORDER BY embedding <=> %s\n") == "embedding <=> %s"
    assert order_by_clause("SELECT 1 FROM policy_chunks\n") == "정렬 없음"


def test_L1_채점표_한_바이트가_바뀌면_지문이_움직인다(tmp_path: Path) -> None:
    """채점표가 조용히 바뀌면 측정 1 의 분모가 바뀐 채 같은 조건으로 읽힌다."""
    before = _script_fingerprint()["l1_fixture_version"]
    copy = tmp_path / "l1_fixtures.jsonl"
    copy.write_bytes(DEFAULT_L1_FIXTURES_PATH.read_bytes() + b"\n")

    assert _script_fingerprint(l1_fixtures=copy)["l1_fixture_version"] != before


def test_L1_채점표를_읽지_못하면_0_이_아니라_미상이다(tmp_path: Path) -> None:
    assert _script_fingerprint(l1_fixtures=tmp_path / "없다.jsonl")["l1_fixture_version"] == "미상"


# ── 계약 D — 중단된 측정이 기계가 읽는 조건에 남는다 ────────────────────────


def test_중단_없이_끝난_실행은_지문에_중단_없음으로_남는다() -> None:
    assert _current().values["run_completion"] == "중단 없음"


def test_중단된_측정이_지문에_이름으로_남는다() -> None:
    """중단 표시가 사람이 읽는 문자열에만 붙으면 **기계가 읽는 조건은 완주와 같다.**"""
    assert _current(aborted=("측정 2",)).values["run_completion"] == "중단: 측정 2"
    assert _current(aborted=("측정 2", "측정 3")).values["run_completion"] == "중단: 측정 2·측정 3"


def test_중단된_실행은_완주한_실행과_대조되지_않는다() -> None:
    comparison = _current(aborted=("측정 2",)).compare(_current())

    assert comparison.comparable is False
    assert [item.field for item in comparison.undeclared_differences] == ["run_completion"]


# ── 계약 E — 옛 산출물의 새 항목은 "미상"이다 ───────────────────────────────


def test_지문_확장_이전_산출물은_새_항목이_미상이다() -> None:
    legacy = _legacy()

    for name in NEW_FINGERPRINT_FIELDS:
        assert legacy.values.get(name) is None, name
        assert legacy.describe(name) == "미상", name


def _extended_legacy() -> ConditionFingerprint:
    """옛 산출물과 **기존 칸이 전부 같고 새 일곱 칸만 있는** 지문.

    저장소의 현재 기본값은 커밋된 기준선과 기권 게이트·판정 픽스처 축에서 이미 갈려 있다.
    그 차이를 섞으면 "확장이 미상으로 읽히는가"를 재는 대신 **다른 축의 드리프트**를 재게
    된다 — 그래서 여기서는 확장 하나만 남긴다.
    """
    legacy = _legacy()
    values: dict[str, str] = {
        name: value for name, value in legacy.values.items() if value is not None
    }
    current = _current().values
    for name in NEW_FINGERPRINT_FIELDS:
        value = current[name]
        assert value is not None, name
        values[name] = value
    return ConditionFingerprint.from_values(values)


def test_옛_산출물과의_대조는_미상을_관용한다() -> None:
    """옛 산출물에 없는 항목 때문에 기준선 대조를 죽이지 않는다."""
    comparison = _extended_legacy().compare(_legacy())

    assert comparison.comparable is True
    assert set(comparison.unknown_fields) == set(NEW_FINGERPRINT_FIELDS)


def test_세트_편입은_미상을_인정하지_않는다() -> None:
    """관용을 편입에까지 끌고 가면 다른 조건의 케이스 결과가 이번 일치 횟수로 계산된다."""
    same, reason = _extended_legacy().same_condition(_legacy())

    assert same is False
    assert "미상" in (reason or "")


def test_지문_확장_이전_산출물끼리도_한_세트로_묶이지_않는다() -> None:
    """**양쪽 다 모르는 것은 "같다"가 아니다.**

    지문이 통째로 없는 산출물이 "전부 미상 = 어긋남 없음"으로 읽혀 3회 정족수를 채우면,
    다른 조건에서 나온 케이스 결과가 이번 실측의 일치 횟수로 계산된다. 확장한 항목을
    **필수 목록에 올리는** 이유가 이것이다 — 그래야 모른다는 사실이 대조에 남는다.
    """
    other = load_run_summary(_ROOT / "reports" / "evaluation-live-l2-8.json").fingerprint

    same, reason = _legacy().same_condition(other)

    assert same is False
    assert "미상" in (reason or "")


def test_세트_편입_술어가_같은_조건은_그대로_받아들인다() -> None:
    """양성 대조 — 전부 거부하는 술어는 술어가 아니다."""
    same, reason = _current().same_condition(_current())

    assert same is True, reason


def test_새_항목이_다르면_선언_없이는_대조_불가다() -> None:
    """음성 대조 — 새 항목이 대조에서 조용히 빠지면 더한 의미가 없다."""
    drifted = dict(_script_fingerprint())
    drifted["sql_guard_version"] = "sqlguard-000000000000"
    comparison = _current(fingerprint=drifted).compare(_current())

    assert comparison.comparable is False
    assert [item.field for item in comparison.undeclared_differences] == ["sql_guard_version"]


# ── 계약 F — 짝으로 읽는 항목의 규칙은 그대로다 ─────────────────────────────


def test_짝_항목_규칙이_그대로다() -> None:
    """τ 는 임베딩 모델을 넘어 이전되지 않는다 — 확장이 이 규칙을 건드리면 안 된다."""
    assert dict(PAIRED_FINGERPRINT_FIELDS) == {"abstention_tau": "embedding_model"}


def test_새_항목은_짝을_만들지_않는다() -> None:
    assert set(PAIRED_FINGERPRINT_FIELDS) & set(NEW_FINGERPRINT_FIELDS) == set()


# ── 계약 G — 판정 호출은 사고 과정 설정을 보내지 않는다 (구조 검사) ─────────


def _sends_thinking(source: str) -> list[str]:
    """판정 요청 조립에 `thinking` 키가 들어가는가 — 문자열 스캔이 아니라 AST 로 본다."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        for key in node.keys:
            if isinstance(key, ast.Constant) and key.value == "thinking":
                found.append(f"line {key.lineno}")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Subscript):
            index = node.slice
            if isinstance(index, ast.Constant) and index.value == "thinking":
                found.append(f"line {node.lineno}")
    return found


def test_판정_호출은_사고_과정_설정을_보내지_않는다() -> None:
    """미전송은 '끔'이 아니라 **계열 기본**이다 — 보내기 시작하면 지문이 거짓말을 한다.

    지문의 `judge_thinking` 은 "설정을 보내지 않는다"를 값으로 적는다. 그 사실이 코드에서
    바뀌는 순간 이 검사가 먼저 죽어, 지문을 고치지 않고 지나갈 수 없게 한다.
    """
    source = textwrap.dedent(inspect.getsource(llm.AnthropicGenerationClient.complete_json))

    assert _sends_thinking(source) == []


def test_사고_과정_검사가_실제로_전송을_잡는다() -> None:
    """음성 대조 — 아무것도 못 잡는 검사기가 아니라는 확인."""
    assert _sends_thinking('request = {"model": m, "thinking": {"type": "enabled"}}\n')
    assert _sends_thinking('request["thinking"] = {"type": "enabled"}\n')
    assert not _sends_thinking('request = {"model": m, "max_tokens": 8000}\n')


# ── 계약 H — 기권 미정의 사유가 리포트까지 간다 ─────────────────────────────


class _ScriptedCollector:
    def __init__(self, collection: EvidenceCollection) -> None:
        self._collection = collection

    def collect(self, **kwargs: Any) -> EvidenceCollection:
        del kwargs
        return self._collection


def _collection(reason: AbstentionUndefined | None) -> EvidenceCollection:
    return EvidenceCollection(
        intent=IntentSource.POLICY,
        evidence=(),
        escalation_reason=EscalationReason.NO_EVIDENCE,
        failed_stage=None,
        sql_snapshots=(),
        sql_failures=(),
        input_tokens=1,
        output_tokens=1,
        embedding_tokens=1,
        abstention_undefined_reason=reason,
    )


def _run_with(reason: AbstentionUndefined | None) -> ProcessedInquiry:
    pipeline = InquiryPipeline(
        collector=cast(Any, _ScriptedCollector(_collection(reason))),
        drafter=cast(Any, None),
        judge=None,
        l2_enabled=False,
    )
    return pipeline.run(
        inquiry_id=new_inquiry_id(),
        content="해외 배송도 되나요?",
        order_no=None,
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )


def test_파이프라인이_기권_미정의_사유를_결과로_옮긴다() -> None:
    """근거 묶음까지 나온 사유가 여기서 끊기면 리포트가 그 분기를 영영 못 본다."""
    processed = _run_with(AbstentionUndefined.NONPOSITIVE_RANK1)

    assert processed.abstention_undefined_reason is AbstentionUndefined.NONPOSITIVE_RANK1


def test_두_사유가_하나로_접히지_않는다() -> None:
    """처분은 반대인데 사유는 둘 다 그대로 나와야 한다 — 한쪽만 실으면 분기가 사라진다."""
    processed = _run_with(AbstentionUndefined.INSUFFICIENT_SCORES)

    assert processed.abstention_undefined_reason is AbstentionUndefined.INSUFFICIENT_SCORES


def test_기권_사유가_없는_실행은_0_이_아니라_None_이다() -> None:
    """양성 대조 — `None` 은 게이트 꺼짐·검색 미실행·정의됨 셋을 함께 덮는다."""
    assert _run_with(None).abstention_undefined_reason is None


def _processed(reason: AbstentionUndefined | None) -> ProcessedInquiry:
    return ProcessedInquiry(
        inquiry_id="00000000-0000-4000-8000-000000000000",
        order_no=None,
        content="문의",
        intent=IntentSource.POLICY,
        status=InquiryStatus.ESCALATED,
        answer=None,
        claims=(Claim(text="답변", citation_ids=("policy:refund:2-1",)),),
        escalation_reason=EscalationReason.NO_EVIDENCE,
        failed_stage=None,
        evidence=(),
        sql_snapshots=(),
        sql_failures=(),
        attempts=(),
        latency_ms=10,
        input_tokens=0,
        output_tokens=0,
        embedding_tokens=0,
        abstention_undefined_reason=reason,
    )


class _ScriptedPipeline:
    def __init__(self, results: list[ProcessedInquiry]) -> None:
        self._results = list(results)

    def run(self, **kwargs: Any) -> ProcessedInquiry:
        del kwargs
        return self._results.pop(0)


def _golden_case(case_id: str) -> GoldenCase:
    return GoldenCase(
        id=case_id,
        category="normal",
        order_no=None,
        content="문의",
        expected=ExpectedOutcomeSet(
            statuses=frozenset({InquiryStatus.ESCALATED}),
            escalation_reasons=frozenset({EscalationReason.NO_EVIDENCE}),
            expect_reject=False,
            forbidden_reject_reasons=frozenset(),
        ),
        note="",
    )


def test_케이스_결과가_기권_미정의_사유를_싣는다() -> None:
    outcome = evaluate_case(
        case=_golden_case("G21"),
        pipeline=cast(Any, _ScriptedPipeline([_processed(AbstentionUndefined.NONPOSITIVE_RANK1)])),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert outcome.abstention_undefined_reason == "rank1_cosine_not_positive"


def test_세트_집계가_미정의_사유를_사유별로_센다() -> None:
    """두 사유는 처분이 반대다 — 한 칸으로 접으면 어느 분기를 탔는지 사라진다."""
    agreement = measure_pipeline_agreement(
        cases=[_golden_case("G21"), _golden_case("G22"), _golden_case("G23")],
        pipeline=cast(
            Any,
            _ScriptedPipeline(
                [
                    _processed(AbstentionUndefined.NONPOSITIVE_RANK1),
                    _processed(AbstentionUndefined.INSUFFICIENT_SCORES),
                    _processed(None),
                ]
            ),
        ),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )

    assert dict(agreement.abstention_undefined_counts) == {
        "rank1_cosine_not_positive": 1,
        "measured_scores_below_two": 1,
    }


def test_리포트_JSON_이_케이스별_기권_사유와_집계를_함께_싣는다() -> None:
    payload = report_to_json(_stub_report())
    measurement2 = payload["measurement_2_pipeline_agreement"]

    assert measurement2["abstention_undefined_counts"] == {"rank1_cosine_not_positive": 1}
    assert [item["abstention_undefined_reason"] for item in measurement2["outcomes"]] == [
        "rank1_cosine_not_positive"
    ]


def test_리포트_문면이_기권_미정의_분기를_적는다() -> None:
    body = render_markdown(_stub_report())

    assert "기권 게이트 통계량 미정의" in body


def _stub_report() -> EvaluationReport:
    agreement = measure_pipeline_agreement(
        cases=[_golden_case("G21")],
        pipeline=cast(Any, _ScriptedPipeline([_processed(AbstentionUndefined.NONPOSITIVE_RANK1)])),
        app_conn=_NO_CONN,
        readonly_conn=_NO_CONN,
    )
    return build_report(
        conditions=_conditions(),
        gate_accuracy=measure_gate_accuracy(()),
        pipeline=agreement,
        judge_accuracy=SkippedMeasurement(reason="대역 — 판정 미실행"),
    )
