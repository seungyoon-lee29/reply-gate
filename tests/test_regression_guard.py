"""회귀 가드 — 이중 기준선·조건 지문·비악화 판정.

이 스위트가 지키는 것은 하나다: **합산 지표가 못 잡는 회귀를 케이스 단위가 잡는다.**
커밋된 산출물이 그 회귀를 그대로 보여준다 — `evaluation-live-l2-1` 은 G18 을
`reject_reasons=["contradictory_evidence"]` 로 끝냈고 `-l2-4`·`-l2-6` 은
`reject_reasons=[]` 에 채택 근거가 `["policy:refund:2-6"]` 단독인데 **6회 전부
`matched=True`** 다. `matched` 하나로는 그 소멸을 볼 수 없다
(`docs/tracking/findings.md` 18번).

그래서 가드의 술어를 `matched` 하나로 되돌리는 변경은 이 파일에서 죽어야 한다.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from reply_gate.regression_guard import (
    DEFAULT_PROMOTED_BASELINE_PATH,
    RUN_SET_SIZE,
    BaselineNotRegistered,
    ConditionFingerprint,
    GuardUnavailable,
    PromotedBaseline,
    RegressionGuard,
    RunSet,
    RunSummary,
    build_regression_guard,
    compare_run_sets,
    fingerprint_from_conditions,
    guard_to_json,
    load_promoted_baseline,
    load_run_summary,
    render_guard_section,
    run_summary_from_payload,
)

_ROOT = Path(__file__).resolve().parents[1]

#: 기준선·새 실측이 공유하는 지문. 이 값이 갈리면 대조 가능성부터 달라진다.
_BASE_FINGERPRINT: Mapping[str, str] = {
    "label_version": "0008-이후",
    "retrieval_labels_version": "labels-aaaa",
    "acceptance_cut": "0.3",
    "abstention_gate_statistic": "미배선",
    "abstention_tau": "미배선",
    "query_rewrite": "on",
    "embedding_model": "text-embedding-3-small",
    "embedding_dimensions": "1536",
    "top_k": "5",
    "generation_model": "gpt-5.6-terra",
    "judge_model": "claude-sonnet-5",
    "judge_effort": "기본값",
    "judge_prompt_version": "p-aaaa",
    "judge_fixture_version": "f-bbbb",
    "judge_prompt_caching": "off",
    "measurement_scope": "full",
}

#: 케이스 관측 1건 — (일치 여부, 정답 근거, 빠진 정답 근거).
#: 정답 근거가 `None` 이면 귀인 절에 실리지 않은 케이스다(= 정답 근거 전부 채택 + 정상 답변).
CaseSpec = tuple[bool, Sequence[str] | None, Sequence[str] | None]


def _payload(
    *,
    started_at: str,
    cases: Mapping[str, CaseSpec],
    fingerprint: Mapping[str, str] | None = None,
    declared: Sequence[str] = (),
    billed: bool = True,
    attribution_computed: bool = True,
    detection_rate: float | None = 1.0,
    false_positive_rate: float | None = 0.0,
    measurement3_detection_rate: float | None = 1.0,
    include_evidence_fields: bool = True,
) -> dict[str, Any]:
    """리포트 JSON 1건을 대역으로 만든다 — 가드가 실제로 읽는 키만 담는다."""
    values = dict(_BASE_FINGERPRINT if fingerprint is None else fingerprint)
    outcomes = [
        {"case_id": case_id, "matched": matched, "adopted_evidence_ids": []}
        for case_id, (matched, _relevant, _missing) in cases.items()
    ]
    attribution_cases = []
    for case_id, (_matched, relevant, missing) in cases.items():
        if relevant is None:
            continue
        entry: dict[str, Any] = {
            "case_id": case_id,
            "classification": "answered_without_relevant_evidence",
            "relevant_evidence_ids": list(relevant),
        }
        if include_evidence_fields:
            entry["missing_relevant_evidence_ids"] = list(missing or ())
        attribution_cases.append(entry)
    attribution: dict[str, Any] = (
        {"computed": True, "cases": attribution_cases}
        if attribution_computed
        else {"computed": False, "reason": "측정 2 미실행: 대역"}
    )
    return {
        "conditions": {
            "started_at": started_at,
            "billed": billed,
            "l2_enabled": True,
            "measurement2_is_real": billed,
            "condition_fingerprint": values,
            "declared_experiment_fields": list(declared),
        },
        "measurement_1_l1_gate_accuracy": {
            "detection_rate": detection_rate,
            "false_positive_rate": false_positive_rate,
        },
        "measurement_2_pipeline_agreement": {
            "executed": True,
            "match_rate": 1.0,
            "outcomes": outcomes,
        },
        "failure_attribution": attribution,
        "measurement_3_l2_judge_accuracy": (
            {"executed": False, "skip_reason": "미요청"}
            if measurement3_detection_rate is None
            else {"executed": True, "detection_rate": measurement3_detection_rate}
        ),
    }


def _write(reports_dir: Path, stem: str, payload: Mapping[str, Any]) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def _promotion(
    path: Path,
    *,
    stems: Sequence[str],
    fingerprint: Mapping[str, str] | None = None,
    supersedes: Sequence[str] = (),
    promoted_at: str = "2026-08-20",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "promotion": {
                    "promoted_at": promoted_at,
                    "promoted_by": "사용자",
                    "reason": "기준선 라이브 3회를 승격한다",
                    "supersedes": list(supersedes),
                },
                "report_stems": list(stems),
                "fingerprint": dict(_BASE_FINGERPRINT if fingerprint is None else fingerprint),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _healthy_cases() -> dict[str, CaseSpec]:
    """정상 3회 — G18 은 상충쌍을 함께 채택했고 나머지는 귀인 절에 실리지 않는다."""
    return {
        "G01": (True, None, None),
        "G04": (True, None, None),
        "G18": (True, ["policy:refund:2-4", "policy:refund:2-6"], []),
    }


def _seed_baseline(
    reports_dir: Path,
    *,
    cases: Mapping[str, CaseSpec] | None = None,
    stems: Sequence[str] = ("evaluation-live-l2-1", "evaluation-live-l2-2", "evaluation-live-l2-3"),
    fingerprint: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> tuple[str, ...]:
    for index, stem in enumerate(stems, start=1):
        _write(
            reports_dir,
            stem,
            _payload(
                started_at=f"2026-08-0{index}T00:00:00+00:00",
                cases=dict(_healthy_cases() if cases is None else cases),
                fingerprint=fingerprint,
                **kwargs,
            ),
        )
    return tuple(stems)


def _run_guard(
    tmp_path: Path,
    *,
    candidate_cases: Sequence[Mapping[str, CaseSpec]],
    baseline_cases: Mapping[str, CaseSpec] | None = None,
    candidate_fingerprint: Mapping[str, str] | None = None,
    declared: Sequence[str] = (),
    promoted: bool = True,
    baseline_kwargs: Mapping[str, Any] | None = None,
    candidate_kwargs: Mapping[str, Any] | None = None,
) -> RegressionGuard:
    """기준선 3회를 심고 새 실측 N 회를 흘린 뒤 마지막 실행의 가드를 돌려준다."""
    reports_dir = tmp_path / "reports"
    baseline_kwargs = dict(baseline_kwargs or {})
    stems = _seed_baseline(reports_dir, cases=baseline_cases, **baseline_kwargs)
    reference = tmp_path / "data" / "promoted_baseline.json"
    if promoted:
        # 등재는 **자기가 가리키는 산출물과 같은 조건**을 적어야 한다 — 어긋나면 가드가
        # 대조 전에 재등재를 요구한다(그 경로는 따로 테스트한다).
        _promotion(reference, stems=stems, fingerprint=baseline_kwargs.get("fingerprint"))

    payloads = [
        _payload(
            started_at=f"2026-08-1{index}T00:00:00+00:00",
            cases=dict(cases),
            fingerprint=candidate_fingerprint,
            declared=declared,
            **(candidate_kwargs or {}),
        )
        for index, cases in enumerate(candidate_cases, start=1)
    ]
    for index, payload in enumerate(payloads[:-1], start=4):
        _write(reports_dir, f"evaluation-live-l2-{index}", payload)

    current_stem = f"evaluation-live-l2-{3 + len(payloads)}"
    current = run_summary_from_payload(payloads[-1], stem=current_stem, source=current_stem)
    guard = build_regression_guard(
        current=current,
        reports_dir=reports_dir,
        promoted_reference_path=reference,
    )
    assert isinstance(guard, RegressionGuard)
    return guard


# ── A. 이중 기준선 · 승격 ────────────────────────────────────────────────────


def test_승격_참조가_없으면_기준선_미등재라고_적는다(tmp_path: Path) -> None:
    """0 이나 "통과"로 채우지 않는다 — 등재되지 않았다는 사실이 그대로 실려야 한다."""
    guard = _run_guard(tmp_path, candidate_cases=[_healthy_cases()] * RUN_SET_SIZE, promoted=False)
    assert guard.binding.verdict == "기준선 미등재"
    assert guard.verdict == "기준선 미등재"
    assert "통과" not in guard.binding.verdict
    rendered = "\n".join(render_guard_section(guard))
    assert "기준선 미등재" in rendered


def test_경보_줄이_승격_세트와_같으면_그_사실을_적는다(tmp_path: Path) -> None:
    """두 줄이 같은 산출물을 대조하면 판정이 같은 것은 **합의가 아니다.**

    승격 직후에는 `reports/` 에 승격 세트 말고 대조할 라이브가 없어 경보 줄이 구속 줄과
    같은 세트를 집는다. 적지 않으면 사용자가 없는 신호(두 검사가 합의했다)를 읽는다.
    """
    guard = _run_guard(tmp_path, candidate_cases=[_healthy_cases()] * RUN_SET_SIZE)
    assert set(guard.alert.baseline_stems) == set(guard.binding.baseline_stems)
    assert any("승격 기준선과 같은 세트" in note for note in guard.alert.notes)
    assert "승격 기준선과 같은 세트" in "\n".join(render_guard_section(guard))


def test_경보_줄이_다른_세트면_같은_세트_표기를_붙이지_않는다(tmp_path: Path) -> None:
    """표기가 항상 붙으면 신호가 아니다 — 실제로 갈릴 때 붙지 않아야 한다."""
    guard = _run_guard(tmp_path, candidate_cases=[_healthy_cases()] * (RUN_SET_SIZE * 2))
    assert set(guard.alert.baseline_stems) != set(guard.binding.baseline_stems)
    assert not any("승격 기준선과 같은 세트" in note for note in guard.alert.notes)


def test_승격_참조가_비어_있으면_빠진_항목을_이름으로_적는다(tmp_path: Path) -> None:
    reference = tmp_path / "promoted_baseline.json"
    reference.write_text(
        json.dumps({"schema_version": 1, "promotion": None, "report_stems": []}),
        encoding="utf-8",
    )
    loaded = load_promoted_baseline(reference)
    assert isinstance(loaded, BaselineNotRegistered)
    assert "promotion" in loaded.reason or "승격" in loaded.reason


def test_저장소의_승격_참조는_사람의_등재_기록을_들고_있다() -> None:
    """등재는 사람이 한다 — 누가·언제·무엇을 근거로가 없으면 구속 판정의 출처가 사라진다.

    (등재 **경로**가 사람뿐이라는 것은 구조 테스트가 따로 못박는다.)
    """
    assert DEFAULT_PROMOTED_BASELINE_PATH.exists()
    loaded = load_promoted_baseline()
    assert isinstance(loaded, PromotedBaseline)
    assert loaded.promoted_at and loaded.promoted_by and loaded.reason
    assert len(loaded.report_stems) == RUN_SET_SIZE


def test_등재된_기준선이_자기가_가리키는_산출물과_맞는다() -> None:
    """실제 등재로 확인한다 — 참조의 조건 지문과 세 산출물이 한 조건이어야 한다."""
    promotion = load_promoted_baseline()
    assert isinstance(promotion, PromotedBaseline)
    runs = [
        run_summary_from_payload(
            json.loads((_ROOT / "reports" / f"{stem}.json").read_text(encoding="utf-8")),
            stem=stem,
            source=stem,
        )
        for stem in promotion.report_stems
    ]
    head = runs[0].fingerprint
    for other in runs[1:]:
        same, reason = other.fingerprint.same_condition(head)
        assert same, reason
    same, reason = ConditionFingerprint(values=promotion.fingerprint.values).same_condition(head)
    assert same, reason


def test_두_줄이_상반되면_승격_기준선이_판정을_가진다(tmp_path: Path) -> None:
    """직전 라이브는 경보일 뿐이다 — 판정을 발동시키지 않는다."""
    reports_dir = tmp_path / "reports"
    stems = _seed_baseline(reports_dir)
    reference = tmp_path / "promoted_baseline.json"
    _promotion(reference, stems=stems)

    # 직전 라이브 3회는 이미 G18 상충 조항을 잃은 세트다 — 그 세트를 기준으로 보면
    # 새 실측은 "악화 없음"이지만, 승격 기준선을 기준으로 보면 미달이다.
    degraded = dict(_healthy_cases())
    degraded["G18"] = (True, ["policy:refund:2-4", "policy:refund:2-6"], ["policy:refund:2-4"])
    for index, stem in enumerate(
        ("evaluation-live-l2-4", "evaluation-live-l2-5", "evaluation-live-l2-6"), start=4
    ):
        _write(
            reports_dir,
            stem,
            _payload(started_at=f"2026-08-1{index}T00:00:00+00:00", cases=degraded),
        )
    for index, stem in enumerate(("evaluation-live-l2-7", "evaluation-live-l2-8"), start=7):
        _write(
            reports_dir,
            stem,
            _payload(started_at=f"2026-08-2{index}T00:00:00+00:00", cases=degraded),
        )
    current = run_summary_from_payload(
        _payload(started_at="2026-08-29T00:00:00+00:00", cases=degraded),
        stem="evaluation-live-l2-9",
        source="evaluation-live-l2-9",
    )
    guard = build_regression_guard(
        current=current, reports_dir=reports_dir, promoted_reference_path=reference
    )
    assert isinstance(guard, RegressionGuard)
    assert guard.binding.role == "구속"
    assert guard.alert.role == "경보"
    assert guard.binding.verdict == "미달"
    assert guard.alert.verdict == "통과"
    assert guard.verdict == "미달", "승격 기준선이 이겨야 한다"
    assert guard.disagreement is True


def test_하네스에_승격을_자동으로_기록하는_경로가_없다() -> None:
    """승격은 사람이 참조 파일을 바꾸는 것뿐이다 — 코드가 쓰면 구속이 자기추인이 된다."""
    offenders = _promotion_writes(_python_sources())
    assert offenders == [], offenders


def test_승격_자동_경로_검사가_실제로_쓰기를_잡는다(tmp_path: Path) -> None:
    """음성 대조 — 검사기가 아무것도 못 잡는 검사기가 아니라는 확인."""
    guilty = tmp_path / "guilty.py"
    guilty.write_text(
        "from pathlib import Path\n"
        "def promote() -> None:\n"
        "    Path('data/promoted_baseline.json').write_text('{}')\n",
        encoding="utf-8",
    )
    assert _promotion_writes([guilty])


def test_승격_자동_경로_검사가_변수를_거친_쓰기도_잡는다(tmp_path: Path) -> None:
    """음성 대조 — 경로를 지역 변수에 한 번 담으면 호출 문면에 이름이 안 남는다."""
    guilty = tmp_path / "guilty_alias.py"
    guilty.write_text(
        "from reply_gate.regression_guard import DEFAULT_PROMOTED_BASELINE_PATH\n"
        "def promote() -> None:\n"
        "    reference = DEFAULT_PROMOTED_BASELINE_PATH\n"
        "    reference.write_text('{}')\n",
        encoding="utf-8",
    )
    assert _promotion_writes([guilty])

    innocent = tmp_path / "innocent.py"
    innocent.write_text(
        "from reply_gate.regression_guard import DEFAULT_PROMOTED_BASELINE_PATH\n"
        "def read() -> str:\n"
        "    reference = DEFAULT_PROMOTED_BASELINE_PATH\n"
        "    return reference.read_text()\n",
        encoding="utf-8",
    )
    assert _promotion_writes([innocent]) == []


def test_재등재된_승격_참조가_새_기준선을_구속한다(tmp_path: Path) -> None:
    """판정 층이 바뀌면 기준선을 다시 등재한다 — 승격과 같은 자격, 사람만 한다."""
    reports_dir = tmp_path / "reports"
    old = _seed_baseline(reports_dir)
    judged = dict(_BASE_FINGERPRINT)
    judged["judge_prompt_version"] = "p-cccc"
    new_stems = ("evaluation-live-l2-4", "evaluation-live-l2-5", "evaluation-live-l2-6")
    _seed_baseline(reports_dir, stems=new_stems, fingerprint=judged)

    reference = tmp_path / "promoted_baseline.json"
    _promotion(reference, stems=new_stems, fingerprint=judged, supersedes=old)
    loaded = load_promoted_baseline(reference)
    assert isinstance(loaded, PromotedBaseline)
    assert loaded.supersedes == tuple(old)

    current = run_summary_from_payload(
        _payload(
            started_at="2026-08-20T00:00:00+00:00", cases=_healthy_cases(), fingerprint=judged
        ),
        stem="evaluation-live-l2-7",
        source="evaluation-live-l2-7",
    )
    guard = build_regression_guard(
        current=current, reports_dir=reports_dir, promoted_reference_path=reference
    )
    assert isinstance(guard, RegressionGuard)
    # 재등재하지 않았다면 판정 프롬프트 차이가 "대조 불가"로 찍혔을 자리다.
    assert guard.binding.baseline_stems == new_stems
    assert guard.binding.verdict != "대조 불가"
    rendered = "\n".join(render_guard_section(guard))
    assert "재등재" in rendered


# ── B. 비악화 판정 (두 겹) ───────────────────────────────────────────────────


def test_기준선_3회_일치가_새_실측_1회로_떨어지면_미달이고_케이스를_찍는다(
    tmp_path: Path,
) -> None:
    broken = dict(_healthy_cases())
    broken["G04"] = (False, None, None)
    guard = _run_guard(
        tmp_path,
        candidate_cases=[broken, broken, _healthy_cases()],
    )
    assert guard.binding.verdict == "미달"
    assert [item.case_id for item in guard.binding.match_shortfalls] == ["G04"]
    assert "G04" in "\n".join(render_guard_section(guard))


def test_기준선_2회_일치가_0회로_무너지면_미달이다(tmp_path: Path) -> None:
    """감시 모집단은 "기준선 3/3" 으로 한정하지 않는다."""
    baseline_ok = _healthy_cases()
    baseline_flaky = dict(_healthy_cases())
    baseline_flaky["G04"] = (False, None, None)
    reports_dir = tmp_path / "reports"
    stems = ("evaluation-live-l2-1", "evaluation-live-l2-2", "evaluation-live-l2-3")
    for index, (stem, cases) in enumerate(
        zip(stems, (baseline_ok, baseline_ok, baseline_flaky), strict=True), start=1
    ):
        _write(
            reports_dir, stem, _payload(started_at=f"2026-08-0{index}T00:00:00+00:00", cases=cases)
        )
    reference = tmp_path / "promoted_baseline.json"
    _promotion(reference, stems=stems)

    collapsed = dict(_healthy_cases())
    collapsed["G04"] = (False, None, None)
    payloads = [
        _payload(started_at=f"2026-08-1{index}T00:00:00+00:00", cases=collapsed)
        for index in range(1, 4)
    ]
    for index, payload in enumerate(payloads[:-1], start=4):
        _write(reports_dir, f"evaluation-live-l2-{index}", payload)
    current = run_summary_from_payload(
        payloads[-1], stem="evaluation-live-l2-6", source="evaluation-live-l2-6"
    )
    guard = build_regression_guard(
        current=current, reports_dir=reports_dir, promoted_reference_path=reference
    )
    assert isinstance(guard, RegressionGuard)
    assert guard.binding.verdict == "미달"
    assert [item.case_id for item in guard.binding.match_collapses] == ["G04"]


def test_기준선_일치_횟수가_줄면_그_자체를_경보로_찍는다(tmp_path: Path) -> None:
    """3/3 → 2/3 는 규칙상 통과지만 감소 자체가 경보다."""
    slipped = dict(_healthy_cases())
    slipped["G01"] = (False, None, None)
    guard = _run_guard(
        tmp_path,
        candidate_cases=[slipped, _healthy_cases(), _healthy_cases()],
    )
    assert guard.binding.verdict == "통과"
    assert [item.case_id for item in guard.binding.match_decreases] == ["G01"]


def test_G18_회귀_재현_matched_는_모두_참인데_상충_근거가_빠지면_미달이다(
    tmp_path: Path,
) -> None:
    """이 사이클을 촉발한 회귀 그대로다 — 합산도 `matched` 도 못 잡은 자리."""
    lost = dict(_healthy_cases())
    lost["G18"] = (True, ["policy:refund:2-4", "policy:refund:2-6"], ["policy:refund:2-4"])
    guard = _run_guard(tmp_path, candidate_cases=[lost] * RUN_SET_SIZE)

    assert guard.binding.match_shortfalls == ()
    assert guard.binding.verdict == "미달", "`matched` 만 보면 통과로 읽힌다"
    losses = guard.binding.evidence_losses
    assert [item.case_id for item in losses] == ["G18"]
    assert losses[0].dropped_evidence_ids == ("policy:refund:2-4",)
    rendered = "\n".join(render_guard_section(guard))
    assert "policy:refund:2-4" in rendered


def test_근거_손실_판정은_케이스별_다수결이다(tmp_path: Path) -> None:
    """3회 중 1회만 빠진 것은 손실이 아니다 — 확률 층의 흔들림까지 미달로 세지 않는다."""
    lost = dict(_healthy_cases())
    lost["G18"] = (True, ["policy:refund:2-4", "policy:refund:2-6"], ["policy:refund:2-4"])
    guard = _run_guard(tmp_path, candidate_cases=[lost, _healthy_cases(), _healthy_cases()])
    assert guard.binding.evidence_losses == ()
    assert guard.binding.verdict == "통과"


def test_기준선에_근거_필드가_없으면_없다고_적고_0으로_채우지_않는다(tmp_path: Path) -> None:
    """커밋된 옛 산출물에는 새 필드가 없다 — 없다고 적을 뿐 손실로도 통과로도 세지 않는다."""
    guard = _run_guard(
        tmp_path,
        candidate_cases=[_healthy_cases()] * RUN_SET_SIZE,
        baseline_kwargs={"attribution_computed": False},
    )
    assert guard.binding.evidence_losses == ()
    assert any("기준선" in note for note in guard.binding.unknown_notes)
    # **통과가 아니다.** 판정의 정의인 검사가 돌지 않았는데 헤드라인이 통과라고 적히면
    # 그것이 곧 "미실행을 통과로 채우기"다.
    assert guard.binding.verdict == "보류"
    assert "근거 부분 손실 검사가 돌지 않았다" in guard.binding.verdict_reason
    assert "근거 부분 손실 없음" not in guard.binding.verdict_reason
    rendered = "\n".join(render_guard_section(guard))
    assert "미산출" in rendered or "없" in rendered


def test_측정_1_무변경_검사는_가드에_남는다(tmp_path: Path) -> None:
    guard = _run_guard(
        tmp_path,
        candidate_cases=[_healthy_cases()] * RUN_SET_SIZE,
        candidate_kwargs={"detection_rate": 0.9},
    )
    assert guard.binding.verdict == "미달"
    assert any("측정 1" in item for item in guard.binding.measurement1_changes)


def test_측정_3_은_무변경_검사_대상이_아니다(tmp_path: Path) -> None:
    """판정 층 개선의 성공이 무관한 축의 원복을 발동시키면 안 된다."""
    guard = _run_guard(
        tmp_path,
        candidate_cases=[_healthy_cases()] * RUN_SET_SIZE,
        candidate_kwargs={"measurement3_detection_rate": 0.5},
    )
    assert guard.binding.verdict == "통과"
    rendered = "\n".join(render_guard_section(guard))
    assert "측정 3 은 무변경 검사 대상이 아니다" in rendered


def test_측정_3_표기는_이_실행에서_실제로_성립하는_것을_적는다(tmp_path: Path) -> None:
    """규칙은 고정이지만 그 뒤 문장은 실행마다 다르다 — 선언하지 않은 실행의 리포트가
    "선언된 실험 변인이라" 라고 적으면 그것은 거짓이다."""
    # ① 아무것도 선언하지 않았는데 값이 달라졌다.
    silent = _run_guard(
        tmp_path / "silent",
        candidate_cases=[_healthy_cases()] * RUN_SET_SIZE,
        candidate_kwargs={"measurement3_detection_rate": 0.5},
    )
    note = next(item for item in silent.binding.notes if "측정 3" in item)
    assert "선언하지 않았다" in note
    assert "선언된 실험 변인" not in note

    # ② 판정 층 변경을 선언했다.
    judged = dict(_BASE_FINGERPRINT)
    judged["judge_prompt_version"] = "p-cccc"
    declared_guard = _run_guard(
        tmp_path / "declared",
        candidate_cases=[_healthy_cases()] * RUN_SET_SIZE,
        candidate_fingerprint=judged,
        declared=["judge_prompt_version"],
        candidate_kwargs={"measurement3_detection_rate": 0.5},
    )
    note = next(item for item in declared_guard.binding.notes if "측정 3" in item)
    assert "판정 층 변경을 선언했다" in note
    assert "judge_prompt_version" in note

    # ③ 값이 그대로면 변화 자체가 없다.
    same = _run_guard(
        tmp_path / "same",
        candidate_cases=[_healthy_cases()] * RUN_SET_SIZE,
    )
    note = next(item for item in same.binding.notes if "측정 3" in item)
    assert "값이 같다" in note


def test_멀쩡한_실행은_어떤_경보도_내지_않는다(tmp_path: Path) -> None:
    """음성 대조 — 전부 거부하는 가드는 가드가 아니다."""
    guard = _run_guard(tmp_path, candidate_cases=[_healthy_cases()] * RUN_SET_SIZE)
    assert guard.verdict == "통과"
    assert guard.binding.match_shortfalls == ()
    assert guard.binding.match_collapses == ()
    assert guard.binding.match_decreases == ()
    assert guard.binding.evidence_losses == ()
    assert guard.binding.measurement1_changes == ()
    assert guard.disagreement is False


def test_실측이_3회에_못_미치면_통과가_아니라_보류다(tmp_path: Path) -> None:
    guard = _run_guard(tmp_path, candidate_cases=[_healthy_cases()])
    assert guard.binding.verdict == "보류"
    assert "1/3" in guard.binding.verdict_reason


def test_대역_실행은_가드를_돌리지_않고_사유를_적는다(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    current = run_summary_from_payload(
        _payload(started_at="2026-08-11T00:00:00+00:00", cases=_healthy_cases(), billed=False),
        stem="evaluation",
        source="evaluation",
    )
    guard = build_regression_guard(current=current, reports_dir=reports_dir)
    assert isinstance(guard, GuardUnavailable)
    assert guard.reason


# ── C. 조건 지문 ─────────────────────────────────────────────────────────────


def test_선언되지_않은_불일치는_대조_불가와_어긋난_항목이다(tmp_path: Path) -> None:
    drifted = dict(_BASE_FINGERPRINT)
    drifted["top_k"] = "8"
    guard = _run_guard(
        tmp_path,
        candidate_cases=[_healthy_cases()] * RUN_SET_SIZE,
        candidate_fingerprint=drifted,
    )
    assert guard.binding.verdict == "대조 불가"
    assert [item.field for item in guard.binding.fingerprint.undeclared_differences] == ["top_k"]
    rendered = "\n".join(render_guard_section(guard))
    assert "대조 불가" in rendered
    assert "top_k" in rendered


def test_세트_제외_사유는_이번_실행과_제외된_실행을_바꿔_말하지_않는다(tmp_path: Path) -> None:
    """제외 사유의 방향이 뒤집히면 회귀를 진단하는 사람이 어느 쪽이 바뀌었는지 오독한다.

    `기준선` 자리에는 **제외된(옛) 실행**의 값이, `이번` 자리에는 **이번 실행**의 값이 와야
    한다. 양쪽 값이 둘 다 실려 있어서 문장만 훑으면 맞아 보이고, 방향만 조용히 틀린다.
    """
    reports_dir = tmp_path / "reports"
    이번_지문 = dict(_BASE_FINGERPRINT)
    이번_지문["judge_model"] = "판정모델-이번"
    옛_지문 = dict(_BASE_FINGERPRINT)
    옛_지문["judge_model"] = "판정모델-옛것"

    stems = _seed_baseline(reports_dir, fingerprint=이번_지문)
    reference = tmp_path / "data" / "promoted_baseline.json"
    _promotion(reference, stems=stems, fingerprint=이번_지문)

    # 승격 세트도 이번 실행도 아닌 옛 실행 — 조건이 달라 세트에서 빠져야 한다.
    _write(
        reports_dir,
        "evaluation-live-l2-4",
        _payload(
            started_at="2026-08-04T00:00:00+00:00",
            cases=_healthy_cases(),
            fingerprint=옛_지문,
        ),
    )
    현재 = run_summary_from_payload(
        _payload(
            started_at="2026-08-05T00:00:00+00:00",
            cases=_healthy_cases(),
            fingerprint=이번_지문,
        ),
        stem="evaluation-live-l2-5",
        source="evaluation-live-l2-5",
    )
    guard = build_regression_guard(
        current=현재, reports_dir=reports_dir, promoted_reference_path=reference
    )
    assert isinstance(guard, RegressionGuard)

    제외 = " · ".join(guard.candidate_exclusions)
    assert "evaluation-live-l2-4" in 제외, 제외
    assert "기준선 `판정모델-옛것` → 이번 `판정모델-이번`" in 제외, 제외


def test_선언된_실험_변인은_대조를_진행하고_차이_목록을_병기한다(tmp_path: Path) -> None:
    """이 구분이 없으면 지문 규칙이 이번 사이클의 핵심 비교를 스스로 죽인다."""
    changed = dict(_BASE_FINGERPRINT)
    changed["acceptance_cut"] = "0.42"
    guard = _run_guard(
        tmp_path,
        candidate_cases=[_healthy_cases()] * RUN_SET_SIZE,
        candidate_fingerprint=changed,
        declared=["acceptance_cut"],
    )
    assert guard.binding.verdict == "통과"
    assert [item.field for item in guard.binding.fingerprint.declared_differences] == [
        "acceptance_cut"
    ]
    assert guard.binding.fingerprint.undeclared_differences == ()
    assert "acceptance_cut" in "\n".join(render_guard_section(guard))


def test_임베딩_모델이_다르면_τ_가_같아도_대조_불가다(tmp_path: Path) -> None:
    """τ 는 임베딩 모델과 한 쌍으로 읽는다 — 손계산에서 모델을 넘어 이전되지 않았다."""
    base = dict(_BASE_FINGERPRINT)
    base["abstention_tau"] = "0.42"
    other = dict(base)
    other["embedding_model"] = "text-embedding-3-large"
    guard = _run_guard(
        tmp_path,
        candidate_cases=[_healthy_cases()] * RUN_SET_SIZE,
        baseline_kwargs={"fingerprint": base},
        candidate_fingerprint=other,
    )
    diverged = {item.field for item in guard.binding.fingerprint.undeclared_differences}
    assert {"embedding_model", "abstention_tau"} <= diverged
    assert guard.binding.verdict == "대조 불가"


def test_임베딩_모델_변경을_선언하면_짝인_τ_도_함께_선언된_것으로_읽는다() -> None:
    base = ConditionFingerprint.from_values({**_BASE_FINGERPRINT, "abstention_tau": "0.42"})
    moved = ConditionFingerprint.from_values(
        {**_BASE_FINGERPRINT, "abstention_tau": "0.42", "embedding_model": "bge-m3"},
        declared=("embedding_model",),
    )
    comparison = moved.compare(base)
    assert comparison.comparable is True
    assert {item.field for item in comparison.declared_differences} == {
        "embedding_model",
        "abstention_tau",
    }


def test_지문에_없는_항목은_통과로_채우지_않고_미상으로_적는다() -> None:
    """옛 산출물에는 새 지문 항목이 없다 — 같다고도 다르다고도 적지 않는다."""
    legacy = ConditionFingerprint.from_values({"acceptance_cut": "0.3"})
    current = ConditionFingerprint.from_values(_BASE_FINGERPRINT)
    comparison = current.compare(legacy)
    assert comparison.comparable is True
    assert "judge_prompt_version" in comparison.unknown_fields
    assert comparison.undeclared_differences == ()


def test_지문_스키마는_값_추가만으로_확장된다() -> None:
    """뒤 작업이 새 축 값을 넣을 때 이 모듈을 다시 열지 않아도 되어야 한다."""
    base = ConditionFingerprint.from_values({**_BASE_FINGERPRINT, "새로운_축": "off"})
    changed = ConditionFingerprint.from_values({**_BASE_FINGERPRINT, "새로운_축": "on"})
    comparison = changed.compare(base)
    assert [item.field for item in comparison.undeclared_differences] == ["새로운_축"]


def test_지문은_필수_항목을_전부_들고_있다() -> None:
    fingerprint = ConditionFingerprint.from_values({})
    for field in (
        "label_version",
        "acceptance_cut",
        "abstention_gate_statistic",
        "abstention_tau",
        "query_rewrite",
        "embedding_model",
        "embedding_dimensions",
        "top_k",
        "generation_model",
        "judge_model",
        "judge_effort",
        "judge_prompt_version",
        "judge_fixture_version",
        "judge_prompt_caching",
        "measurement_scope",
    ):
        assert field in fingerprint.values
        assert fingerprint.values[field] is None


# ── 산출물 형식 ─────────────────────────────────────────────────────────────


def test_가드_JSON_은_판정과_근거를_함께_싣는다(tmp_path: Path) -> None:
    lost = dict(_healthy_cases())
    lost["G18"] = (True, ["policy:refund:2-4", "policy:refund:2-6"], ["policy:refund:2-4"])
    guard = _run_guard(tmp_path, candidate_cases=[lost] * RUN_SET_SIZE)
    payload = guard_to_json(guard)
    assert payload["verdict"] == "미달"
    assert payload["binding"]["role"] == "구속"
    assert payload["alert"]["role"] == "경보"
    assert payload["binding"]["evidence_losses"][0]["case_id"] == "G18"
    assert payload["binding"]["evidence_losses"][0]["dropped_evidence_ids"] == ["policy:refund:2-4"]
    # 직렬화가 실제로 JSON 이어야 한다.
    json.dumps(payload, ensure_ascii=False)


def test_커밋된_라이브_산출물을_그대로_읽을_수_있다() -> None:
    """옛 산출물은 사후 편집하지 않는다 — 가드가 그것들을 읽고 견뎌야 한다."""
    summary = run_summary_from_payload(
        json.loads((_ROOT / "reports" / "evaluation-live-l2-1.json").read_text(encoding="utf-8")),
        stem="evaluation-live-l2-1",
        source="reports/evaluation-live-l2-1.json",
    )
    assert summary.cases["G18"].matched is True
    # 그 시점 산출물에는 귀인 절이 없다 — 채택 근거를 "없음"으로 세지 않는다.
    assert summary.cases["G18"].evidence_unknown is True
    assert summary.billed is True


def test_커밋된_두_세트는_지문이_달라_자동으로_묶이지_않는다() -> None:
    """컷 0.30 세트와 0.50 세트가 한 세트로 묶이면 직전 라이브 줄이 거짓이 된다."""

    def _load(stem: str) -> Any:
        return run_summary_from_payload(
            json.loads((_ROOT / "reports" / f"{stem}.json").read_text(encoding="utf-8")),
            stem=stem,
            source=stem,
        )

    cycle2 = _load("evaluation-live-l2-1")
    cycle3 = _load("evaluation-live-l2-4")
    assert cycle3.fingerprint.compare(cycle2.fingerprint).comparable is False


# ── 구조 검사 도구 ───────────────────────────────────────────────────────────


_WRITE_CALLS = frozenset({"write_text", "write_bytes", "dump", "unlink", "replace", "rename"})


def _python_sources() -> list[Path]:
    files: list[Path] = []
    for target in ("src", "scripts"):
        files.extend(
            path for path in (_ROOT / target).rglob("*.py") if "__pycache__" not in path.parts
        )
    return files


def _promotion_aliases(*, tree: ast.AST, source: str) -> set[str]:
    """승격 참조 경로를 담은 **지역 이름**.

    호출 문면만 훑는 검사는 `reference = DEFAULT_PROMOTED_BASELINE_PATH` 뒤의
    `reference.write_text(...)` 를 놓친다 — 변수를 한 번 거치면 이름이 호출에 안 남기
    때문이다(사이클 4 리뷰 advisory). 두 갈래로 오염을 전파한다: 대입값이 승격 참조를
    가리키거나, 이름 자체가 `promoted` 를 달고 있거나.
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and "promoted" in node.arg.lower():
            aliases.add(node.arg)
            continue
        if isinstance(node, ast.Assign | ast.AnnAssign):
            if node.value is None:
                continue
            segment = (ast.get_source_segment(source, node.value) or "").lower()
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if "promoted_baseline" in segment or "promoted" in target.id.lower():
                    aliases.add(target.id)
    return aliases


def _mentions_promotion(*, node: ast.AST, source: str, aliases: set[str]) -> bool:
    """이 호출이 승격 참조를 건드리는가 — 문면으로든, 오염된 이름으로든."""
    segment = (ast.get_source_segment(source, node) or "").lower()
    if "promoted_baseline" in segment:
        return True
    return any(isinstance(inner, ast.Name) and inner.id in aliases for inner in ast.walk(node))


def _promotion_writes(paths: Sequence[Path]) -> list[str]:
    """승격 참조 파일에 **쓰는** 호출을 찾는다. 읽기는 잡지 않는다."""
    offenders: list[str] = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        if "promoted_baseline" not in source and "PROMOTED_BASELINE" not in source:
            continue
        tree = ast.parse(source)
        aliases = _promotion_aliases(tree=tree, source=source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name not in _WRITE_CALLS and name != "open":
                continue
            if _mentions_promotion(node=node, source=source, aliases=aliases):
                segment = ast.get_source_segment(source, node) or ""
                offenders.append(f"{path.name}:{node.lineno}: {segment.splitlines()[0]}")
    return offenders


@pytest.mark.parametrize("stem", ["evaluation-live-l2-1", "evaluation-live-l2-6"])
def test_모든_커밋된_라이브_리포트가_읽힌다(stem: str) -> None:
    payload = json.loads((_ROOT / "reports" / f"{stem}.json").read_text(encoding="utf-8"))
    summary = run_summary_from_payload(payload, stem=stem, source=stem)
    assert summary.cases
    assert summary.started_at


def test_기준선이_3회_모두_멀쩡했던_케이스도_근거_손실을_감시한다(tmp_path: Path) -> None:
    """**이 층이 가장 지켜야 할 모집단이다.**

    귀인 절은 "정답 조항을 전부 채택하고 정상 답변한" 케이스를 싣지 않는다. 그래서 기준선이
    3회 모두 멀쩡했던 케이스는 산출물에 정답 근거 ID 가 한 번도 안 나온다 — 없는 쪽 라벨을
    있는 쪽에서 빌려 오지 않으면 그 케이스가 통째로 감시 밖으로 나가고, 리포트는 그 자리에
    "근거 부분 손실 없음"이라고 적는다. 커밋된 산출물 기준으로 30건 중 10건이 그 모양이고
    그중에 라이브가 이름으로 지켜봐야 할 케이스가 들어 있다.
    """
    baseline: dict[str, CaseSpec] = {
        "G18": (True, ["policy:refund:2-4", "policy:refund:2-6"], []),
        # 기준선에서 완벽했던 케이스 — 귀인 절에 실리지 않는다.
        "G20": (True, None, None),
    }
    degraded: dict[str, CaseSpec] = {
        "G18": (True, ["policy:refund:2-4", "policy:refund:2-6"], []),
        # 같은 케이스가 정답 조항 하나를 잃었다. `matched` 는 여전히 True 다.
        "G20": (True, ["policy:support:4-1", "policy:support:4-2"], ["policy:support:4-2"]),
    }
    guard = _run_guard(
        tmp_path,
        baseline_cases=baseline,
        candidate_cases=[degraded] * RUN_SET_SIZE,
    )
    assert guard.binding.match_shortfalls == (), "`matched` 는 셋 다 True 다"
    assert guard.binding.verdict == "미달"
    losses = {item.case_id: item for item in guard.binding.evidence_losses}
    assert "G20" in losses
    assert losses["G20"].dropped_evidence_ids == ("policy:support:4-2",)
    rendered = "\n".join(render_guard_section(guard))
    assert "G20" in rendered and "policy:support:4-2" in rendered
    assert "근거 부분 손실 없음" not in rendered


def test_반대_방향_보정은_거짓_손실을_만들지_않는다(tmp_path: Path) -> None:
    """음성 대조 — 기준선이 일부만 채택했고 이번에 전부 채택했으면 손실이 아니라 개선이다."""
    baseline: dict[str, CaseSpec] = {
        "G20": (True, ["policy:support:4-1", "policy:support:4-2"], ["policy:support:4-2"]),
    }
    improved: dict[str, CaseSpec] = {"G20": (True, None, None)}
    guard = _run_guard(
        tmp_path,
        baseline_cases=baseline,
        candidate_cases=[improved] * RUN_SET_SIZE,
    )
    assert guard.binding.evidence_losses == ()
    assert guard.binding.verdict == "통과"


def test_양쪽_모두_관측하지_못한_ID_는_한_줄로_묶어_적는다(tmp_path: Path) -> None:
    """무지를 "손실 없음"으로 읽히게 두지 않는다 — 다만 케이스마다 한 줄씩 쌓지도 않는다."""
    both_clean: dict[str, CaseSpec] = {
        "G03": (True, None, None),
        "G05": (True, None, None),
    }
    guard = _run_guard(
        tmp_path,
        baseline_cases=both_clean,
        candidate_cases=[both_clean] * RUN_SET_SIZE,
    )
    notes = [note for note in guard.binding.unknown_notes if "관측하지 못한" in note]
    assert len(notes) == 1
    assert "`G03`" in notes[0] and "`G05`" in notes[0]
    assert guard.binding.verdict == "통과"


def test_모델_지문은_설명이_아니라_벌거벗은_id_로_비교한다() -> None:
    """옛 산출물의 설명 문자열과 새 실행의 명시 지문이 형식만 달라 어긋나면, 아무것도
    바뀌지 않았는데 커밋된 기준선마다 "대조 불가"가 뜬다 — 지문이 스스로를 무력화한다."""
    baseline = run_summary_from_payload(
        json.loads((_ROOT / "reports" / "evaluation-live-l2-1.json").read_text(encoding="utf-8")),
        stem="evaluation-live-l2-1",
        source="evaluation-live-l2-1",
    ).fingerprint
    assert baseline.values["generation_model"] == "gpt-5.6-terra"
    assert baseline.values["judge_model"] == "claude-sonnet-5"
    # τ 짝인 임베딩 모델도 옛 산출물에서 읽혀야 한다 — 미상이면 짝 규칙이 통째로 무력해진다.
    assert baseline.values["embedding_model"] == "text-embedding-3-small"
    assert baseline.values["judge_effort"] == "기본값"

    current = ConditionFingerprint.from_values(
        {
            "acceptance_cut": "0.3",
            "top_k": "5",
            "generation_model": "gpt-5.6-terra",
            "judge_model": "claude-sonnet-5",
            "embedding_model": "text-embedding-3-small",
            "judge_effort": "기본값",
        }
    )
    comparison = current.compare(baseline)
    assert comparison.undeclared_differences == (), [
        item.describe() for item in comparison.undeclared_differences
    ]
    assert comparison.comparable is True


def test_모델_설명이_없으면_추측하지_않고_미상으로_남긴다() -> None:
    """ "미실행" 처럼 id 를 담지 않은 설명에서 값을 지어내면 조용한 드리프트가 된다."""
    fingerprint = fingerprint_from_conditions(
        {"generation": "미실행", "judge": "미실행", "embedding": "미실행"}
    )
    assert fingerprint.values["generation_model"] is None
    assert fingerprint.values["judge_model"] is None
    assert fingerprint.values["embedding_model"] is None


def test_커밋된_옛_기준선에_대한_판정은_통과가_아니라_보류다() -> None:
    """실제 산출물로 확인한다 — `evaluation-live-l2-1/2/3` 에는 귀인 절이 아예 없다.

    그 세트를 기준선으로 삼으면 근거 부분 손실 검사가 통째로 돌지 않는다. 그때 나오는
    판정이 "통과"이면, 이 가드가 존재하는 이유인 그 회귀를 못 본 채 성공 신호를 낸다.
    """

    def _load(stem: str) -> RunSummary:
        return run_summary_from_payload(
            json.loads((_ROOT / "reports" / f"{stem}.json").read_text(encoding="utf-8")),
            stem=stem,
            source=stem,
        )

    old_set = RunSet(
        label="옛 기준선",
        runs=tuple(_load(f"evaluation-live-l2-{n}") for n in (1, 2, 3)),
    )
    assert all(not run.attribution_computed for run in old_set.runs)
    line = compare_run_sets(
        baseline=old_set,
        candidate=old_set,
        label="승격 기준선",
        role="구속",
    )
    assert line.verdict == "보류"
    assert "돌지 않았다" in line.verdict_reason


def test_대조된_케이스가_없으면_통과로_적지_않는다(tmp_path: Path) -> None:
    """측정 2 를 건너뛴 실측은 케이스가 없다 — 비어 있는 대조는 성공이 아니다."""
    reports_dir = tmp_path / "reports"
    stems = _seed_baseline(reports_dir)
    reference = tmp_path / "promoted_baseline.json"
    _promotion(reference, stems=stems)

    empty = _payload(started_at="2026-08-11T00:00:00+00:00", cases={})
    current = run_summary_from_payload(
        empty, stem="evaluation-live-l2-4", source="evaluation-live-l2-4"
    )
    guard = build_regression_guard(
        current=current, reports_dir=reports_dir, promoted_reference_path=reference
    )
    assert isinstance(guard, RegressionGuard)
    assert guard.binding.verdict == "보류"
    assert "케이스 단위 판정이 돌지 않았다" in guard.binding.verdict_reason


def test_등재된_지문이_산출물과_어긋나면_재등재를_요구한다(tmp_path: Path) -> None:
    """참조 파일은 "승격 대상 리포트 스템 + 조건 지문" 한 쌍이다 — 둘이 갈리면 그 등재는
    자기가 가리키는 기준선을 설명하지 못한다. 조용히 통과시키지 않는다."""
    reports_dir = tmp_path / "reports"
    stems = _seed_baseline(reports_dir)
    reference = tmp_path / "promoted_baseline.json"
    drifted = dict(_BASE_FINGERPRINT)
    drifted["acceptance_cut"] = "0.5"  # 산출물은 0.3 인데 등재는 0.5 라고 적었다
    _promotion(reference, stems=stems, fingerprint=drifted)

    payloads = [
        _payload(started_at=f"2026-08-1{index}T00:00:00+00:00", cases=_healthy_cases())
        for index in range(1, 4)
    ]
    for index, payload in enumerate(payloads[:-1], start=4):
        _write(reports_dir, f"evaluation-live-l2-{index}", payload)
    current = run_summary_from_payload(
        payloads[-1], stem="evaluation-live-l2-6", source="evaluation-live-l2-6"
    )
    guard = build_regression_guard(
        current=current, reports_dir=reports_dir, promoted_reference_path=reference
    )
    assert isinstance(guard, RegressionGuard)
    assert guard.binding.verdict == "대조 불가"
    assert "참조 파일의 조건 지문이 산출물과 다르다" in guard.binding.verdict_reason
    assert "acceptance_cut" in guard.binding.verdict_reason
    assert "재등재" in guard.binding.verdict_reason


def test_등재된_산출물끼리_조건이_다르면_한_세트가_아니다(tmp_path: Path) -> None:
    """섞인 세트를 기준선으로 승격하면 축별 귀인이 처음부터 불가능하다."""
    reports_dir = tmp_path / "reports"
    _write(
        reports_dir,
        "evaluation-live-l2-1",
        _payload(started_at="2026-08-01T00:00:00+00:00", cases=_healthy_cases()),
    )
    other = dict(_BASE_FINGERPRINT)
    other["top_k"] = "8"
    _write(
        reports_dir,
        "evaluation-live-l2-2",
        _payload(started_at="2026-08-02T00:00:00+00:00", cases=_healthy_cases(), fingerprint=other),
    )
    reference = tmp_path / "promoted_baseline.json"
    _promotion(reference, stems=("evaluation-live-l2-1", "evaluation-live-l2-2"))

    current = run_summary_from_payload(
        _payload(started_at="2026-08-11T00:00:00+00:00", cases=_healthy_cases()),
        stem="evaluation-live-l2-3",
        source="evaluation-live-l2-3",
    )
    guard = build_regression_guard(
        current=current, reports_dir=reports_dir, promoted_reference_path=reference
    )
    assert isinstance(guard, RegressionGuard)
    assert guard.binding.verdict == "대조 불가"
    assert "등재된 산출물끼리 조건이 다르다" in guard.binding.verdict_reason
    assert "top_k" in guard.binding.verdict_reason


def test_참조에_적지_않은_지문_항목은_어긋남이_아니다(tmp_path: Path) -> None:
    """음성 대조 — 사람이 지문을 다 적지 않아도 된다. 적은 것만 맞으면 된다."""
    reports_dir = tmp_path / "reports"
    stems = _seed_baseline(reports_dir)
    reference = tmp_path / "promoted_baseline.json"
    _promotion(reference, stems=stems, fingerprint={"acceptance_cut": "0.3"})

    payloads = [
        _payload(started_at=f"2026-08-1{index}T00:00:00+00:00", cases=_healthy_cases())
        for index in range(1, 4)
    ]
    for index, payload in enumerate(payloads[:-1], start=4):
        _write(reports_dir, f"evaluation-live-l2-{index}", payload)
    current = run_summary_from_payload(
        payloads[-1], stem="evaluation-live-l2-6", source="evaluation-live-l2-6"
    )
    guard = build_regression_guard(
        current=current, reports_dir=reports_dir, promoted_reference_path=reference
    )
    assert isinstance(guard, RegressionGuard)
    assert guard.binding.verdict == "통과"


def test_검색_정답_라벨이_바뀌면_거짓_근거_손실을_만들지_않는다(tmp_path: Path) -> None:
    """파이프라인이 **아무것도** 달라지지 않았는데 라벨에 조항 하나를 더한 것만으로
    "근거가 사라졌다"가 되면, 가드가 잡으라는 회귀 대신 라벨 편집을 고발한다.

    빌려 온 라벨은 기준선이 하지 않은 채택을 기준선에 돌린다 — 그래서 라벨 판이 갈리면
    빌려 오지 않는다. 그리고 그 변화는 지문에 **조건 차이**로 실려야 한다.
    """
    # 기준선: G15 는 완벽해서 귀인 절에 실리지 않는다.
    baseline: dict[str, CaseSpec] = {"G15": (True, None, None)}
    # 새 실측: 파이프라인은 그대로인데 라벨에 2-7 이 추가돼 "빠진 정답"으로 잡힌다.
    after_label_edit: dict[str, CaseSpec] = {
        "G15": (True, ["policy:refund:2-6", "policy:refund:2-7"], ["policy:refund:2-7"]),
    }
    edited = dict(_BASE_FINGERPRINT)
    edited["retrieval_labels_version"] = "labels-bbbb"

    guard = _run_guard(
        tmp_path,
        baseline_cases=baseline,
        candidate_cases=[after_label_edit] * RUN_SET_SIZE,
        candidate_fingerprint=edited,
        declared=["retrieval_labels_version"],
    )
    # ① 라벨 변경은 지문이 **조건 차이**로 본다 — 침묵하지 않는다.
    assert [item.field for item in guard.binding.fingerprint.declared_differences] == [
        "retrieval_labels_version"
    ]
    # ② 그리고 거짓 손실을 만들지 않는다.
    assert guard.binding.evidence_losses == ()
    assert guard.binding.verdict != "미달"
    assert any("라벨 판이 다르다" in note for note in guard.binding.unknown_notes)


def test_라벨_변경을_선언하지_않으면_대조_불가로_잡힌다(tmp_path: Path) -> None:
    """지문에 실렸으니 조용한 라벨 드리프트도 이제 보인다."""
    edited = dict(_BASE_FINGERPRINT)
    edited["retrieval_labels_version"] = "labels-bbbb"
    guard = _run_guard(
        tmp_path,
        candidate_cases=[_healthy_cases()] * RUN_SET_SIZE,
        candidate_fingerprint=edited,
    )
    assert guard.binding.verdict == "대조 불가"
    assert [item.field for item in guard.binding.fingerprint.undeclared_differences] == [
        "retrieval_labels_version"
    ]


def test_라벨_판이_같으면_빌려오기는_그대로_작동한다(tmp_path: Path) -> None:
    """음성 대조 — 라벨 방어가 F2 의 감시를 꺼 버리면 안 된다."""
    baseline: dict[str, CaseSpec] = {"G20": (True, None, None)}
    degraded: dict[str, CaseSpec] = {
        "G20": (True, ["policy:support:4-1", "policy:support:4-2"], ["policy:support:4-2"]),
    }
    guard = _run_guard(
        tmp_path,
        baseline_cases=baseline,
        candidate_cases=[degraded] * RUN_SET_SIZE,
    )
    assert guard.binding.verdict == "미달"
    assert guard.binding.evidence_losses[0].dropped_evidence_ids == ("policy:support:4-2",)


def test_라벨_판이_미상이면_빌려온다() -> None:
    """옛 산출물에는 이 지문 항목이 없다 — 모른다고 막으면 새 실측만 다친다."""
    legacy = ConditionFingerprint.from_values({"acceptance_cut": "0.3"})
    current = ConditionFingerprint.from_values(_BASE_FINGERPRINT)
    assert legacy.values["retrieval_labels_version"] is None
    assert current.values["retrieval_labels_version"] == "labels-aaaa"
    comparison = current.compare(legacy)
    assert "retrieval_labels_version" in comparison.unknown_fields


def test_짝이_안_달라졌으면_함께_달라졌다고_적지_않는다(tmp_path: Path) -> None:
    """없던 사실을 적는 것은 값을 틀리게 적는 것과 같은 자격의 결함이다.

    τ 하나만 달라진 실행에서 문면이 "짝인 임베딩 모델도 함께 달라졌다"를 붙이면, 조건
    대조 실패 사유를 읽는 사람이 임베딩 모델까지 바뀐 줄로 안다. 양쪽 값이 문장에 실려
    있지 않은 자리라 대조로도 못 잡는다.
    """
    base = dict(_BASE_FINGERPRINT)
    base["abstention_tau"] = "미배선"
    other = dict(base)
    other["abstention_tau"] = "0.06"
    assert base["embedding_model"] == other["embedding_model"], "짝은 같아야 하는 설정이다"
    guard = _run_guard(
        tmp_path,
        candidate_cases=[_healthy_cases()] * RUN_SET_SIZE,
        baseline_kwargs={"fingerprint": base},
        candidate_fingerprint=other,
    )
    tau = next(
        item
        for item in guard.binding.fingerprint.undeclared_differences
        if item.field == "abstention_tau"
    )
    assert tau.diverged_by_pair is None
    described = tau.describe()
    assert "함께 달라졌다" not in described, described
    assert described == "abstention_tau: 기준선 `미배선` → 이번 `0.06`"
    # 짝이 어긋나지 않았으므로 짝 자신은 어긋난 항목에 없다.
    assert [item.field for item in guard.binding.fingerprint.undeclared_differences] == [
        "abstention_tau"
    ]


def test_짝_때문에_어긋난_항목은_그렇게_적는다(tmp_path: Path) -> None:
    """``abstention_tau: 기준선 `0.42` → 이번 `0.42``` 는 사람이 읽으면 버그로 보인다.

    τ 는 임베딩 모델과 한 쌍이라 값이 같아도 다른 조건일 수 있다 — 그 이유를 문면이 말해야
    읽는 사람이 리포트를 의심하지 않는다.
    """
    base = dict(_BASE_FINGERPRINT)
    base["abstention_tau"] = "0.42"
    other = dict(base)
    other["embedding_model"] = "text-embedding-3-large"
    guard = _run_guard(
        tmp_path,
        candidate_cases=[_healthy_cases()] * RUN_SET_SIZE,
        baseline_kwargs={"fingerprint": base},
        candidate_fingerprint=other,
    )
    tau = next(
        item
        for item in guard.binding.fingerprint.undeclared_differences
        if item.field == "abstention_tau"
    )
    assert tau.baseline == tau.candidate == "0.42"
    described = tau.describe()
    assert "짝인 `embedding_model`" in described
    assert "같은 조건이 아니다" in described
    assert "0.42` → 이번 `0.42" not in described

    payload = guard_to_json(guard)
    entry = next(
        item
        for item in payload["binding"]["fingerprint"]["undeclared_differences"]
        if item["field"] == "abstention_tau"
    )
    assert entry["diverged_by_pair"] == "embedding_model"

    # 양성 대조 — 짝이 없는 항목의 문면은 그대로다.
    model = next(
        item
        for item in guard.binding.fingerprint.undeclared_differences
        if item.field == "embedding_model"
    )
    assert model.describe() == (
        "embedding_model: 기준선 `text-embedding-3-small` → 이번 `text-embedding-3-large`"
    )


# ── 세트 편입 — 정족수를 채우려고 다른 조건을 끌어오지 않는다 ────────────────


def test_승격_직후_1회차_실측은_옛_산출물로_정족수를_채우지_않는다() -> None:
    """**3회 연속 실측의 1회차·2회차가 거짓 미달을 커밋하는 것을 막는다.**

    승격된 기준선(7·8·9)은 이번 세트에서 제외되므로, 9 를 현재 실행으로 두면 조건이 확인된
    실측은 자기 자신 하나뿐이다. 예전에는 지문이 통째로 없는 사이클 2 산출물이 "전부 미상 =
    어긋남 없음"으로 읽혀 정족수를 채웠고, 멀쩡한 케이스 셋이 `1/3` 으로 찍혀 **미달**이
    나왔다. 라이브 리포트는 사후 편집하지 않으므로 그 거짓 판정은 기록에 영구히 남는다.
    """
    reports = _ROOT / "reports"
    current = load_run_summary(reports / "evaluation-live-l2-9.json")
    guard = build_regression_guard(current=current, reports_dir=reports)
    assert isinstance(guard, RegressionGuard)

    # ① 옛 계열이 세트에 섞이지 않는다.
    assert guard.candidate_stems == ("evaluation-live-l2-9",)
    for stale in ("evaluation-live-l2-1", "evaluation-live-l2-2", "evaluation-live-l2-3"):
        assert stale not in guard.candidate_stems
    assert guard.candidate_run_count == 1

    # ② 그래서 판정은 미달도 통과도 아닌 보류다.
    assert guard.verdict == "보류"
    assert guard.binding.match_shortfalls == ()
    assert guard.binding.evidence_losses == ()

    # ③ 왜 정족수를 못 채웠는지가 산출물에 이름으로 남는다.
    assert "1/3" in guard.binding.verdict_reason
    assert "evaluation-live-l2-2" in guard.binding.verdict_reason
    assert any("evaluation-live-l2-2" in item for item in guard.candidate_exclusions)
    payload = guard_to_json(guard)
    assert payload["candidate_exclusions"]
    assert "조건 불일치로 세트에서 제외" in "\n".join(render_guard_section(guard))


def test_조건이_확인된_3회가_모이면_진짜_회귀는_그대로_미달이다(tmp_path: Path) -> None:
    """음성 대조 — 편입을 조인 것이 가드를 무디게 만들면 안 된다."""
    lost = dict(_healthy_cases())
    lost["G18"] = (True, ["policy:refund:2-4", "policy:refund:2-6"], ["policy:refund:2-4"])
    guard = _run_guard(tmp_path, candidate_cases=[lost] * RUN_SET_SIZE)

    assert guard.candidate_run_count == RUN_SET_SIZE
    assert guard.candidate_stems == (
        "evaluation-live-l2-4",
        "evaluation-live-l2-5",
        "evaluation-live-l2-6",
    )
    assert guard.candidate_exclusions == ()
    assert guard.verdict == "미달"
    assert guard.binding.evidence_losses[0].dropped_evidence_ids == ("policy:refund:2-4",)


def test_지문이_확인되지_않는_실행은_같은_조건으로_묶지_않는다() -> None:
    """`comparable` 과 다른 술어다 — 미상은 대조에서는 관용하고 편입에서는 막는다."""
    legacy = ConditionFingerprint.from_values({"acceptance_cut": "0.3"})
    current = ConditionFingerprint.from_values(_BASE_FINGERPRINT)

    # 기준선 대조에서는 여전히 관용한다(옛 산출물 때문에 대조를 죽이지 않는다).
    assert current.compare(legacy).comparable is True
    # 그러나 한 세트로 묶지는 않는다.
    same, reason = legacy.same_condition(current)
    assert same is False
    assert reason is not None and "확인되지 않는다" in reason

    # 양성 대조 — 완전히 같은 지문끼리는 묶인다.
    twin = ConditionFingerprint.from_values(_BASE_FINGERPRINT)
    assert twin.same_condition(current) == (True, None)


def test_실행이_적어_보낸_미상_문자열도_미상으로_읽는다() -> None:
    """ "미상"끼리 같다고 읽으면 확인되지 않은 조건이 확인된 것처럼 묶인다."""
    left = ConditionFingerprint.from_values({**_BASE_FINGERPRINT, "label_version": "미상"})
    right = ConditionFingerprint.from_values({**_BASE_FINGERPRINT, "label_version": "미상"})
    assert left.values["label_version"] is None
    same, reason = left.same_condition(right)
    assert same is False
    assert reason is not None and "label_version" in reason
