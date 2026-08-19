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
    build_regression_guard,
    guard_to_json,
    load_promoted_baseline,
    render_guard_section,
    run_summary_from_payload,
)

_ROOT = Path(__file__).resolve().parents[1]

#: 기준선·새 실측이 공유하는 지문. 이 값이 갈리면 대조 가능성부터 달라진다.
_BASE_FINGERPRINT: Mapping[str, str] = {
    "label_version": "0008-이후",
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
    stems = _seed_baseline(reports_dir, cases=baseline_cases, **(baseline_kwargs or {}))
    reference = tmp_path / "data" / "promoted_baseline.json"
    if promoted:
        _promotion(reference, stems=stems)

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


def test_승격_참조가_비어_있으면_빠진_항목을_이름으로_적는다(tmp_path: Path) -> None:
    reference = tmp_path / "promoted_baseline.json"
    reference.write_text(
        json.dumps({"schema_version": 1, "promotion": None, "report_stems": []}),
        encoding="utf-8",
    )
    loaded = load_promoted_baseline(reference)
    assert isinstance(loaded, BaselineNotRegistered)
    assert "promotion" in loaded.reason or "승격" in loaded.reason


def test_저장소의_승격_참조는_사람이_채울_자리로_커밋되어_있다() -> None:
    """참조 파일 자체는 저장소에 있고, 등재는 비어 있다 — 자동 승격이 없다는 증거다."""
    assert DEFAULT_PROMOTED_BASELINE_PATH.exists()
    loaded = load_promoted_baseline()
    assert isinstance(loaded, BaselineNotRegistered)


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
    assert guard.binding.verdict != "미달"
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
    assert "측정 3" in rendered
    assert "선언된 실험 변인" in rendered


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


def _promotion_writes(paths: Sequence[Path]) -> list[str]:
    """승격 참조 파일에 **쓰는** 호출을 찾는다. 읽기는 잡지 않는다."""
    offenders: list[str] = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        if "promoted_baseline" not in source and "PROMOTED_BASELINE" not in source:
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name not in _WRITE_CALLS and name != "open":
                continue
            segment = ast.get_source_segment(source, node) or ""
            if "promoted_baseline" in segment.lower() or "promoted_baseline_path" in segment:
                offenders.append(f"{path.name}:{node.lineno}: {segment.splitlines()[0]}")
    return offenders


@pytest.mark.parametrize("stem", ["evaluation-live-l2-1", "evaluation-live-l2-6"])
def test_모든_커밋된_라이브_리포트가_읽힌다(stem: str) -> None:
    payload = json.loads((_ROOT / "reports" / f"{stem}.json").read_text(encoding="utf-8"))
    summary = run_summary_from_payload(payload, stem=stem, source=stem)
    assert summary.cases
    assert summary.started_at
