"""평가 하네스 실행 — 측정 1(L1 픽스처)과 측정 2(골든셋 end-to-end)를 돌려 리포트를 낸다.

    uv run python -m scripts.evaluate                 # 측정 1 만 (측정 2 는 미실행 사유 기록)
    uv run python -m scripts.evaluate --stub-llm      # 측정 2 를 결정론 대역으로 (배관 검증)
    uv run python -m scripts.evaluate --live          # 측정 2 를 실제 OpenAI 로 (과금·비결정론)

**측정 1 은 항상 돈다** — LLM 을 호출하지 않으므로 키도 DB 도 필요 없다.

**측정 2 는 명시적 opt-in 이다.** 실제 실행은 과금되고 결과가 재실행마다 달라지므로
기본값으로 돌리지 않는다. 실행하지 않았으면 리포트에 **미실행 사유가 그대로 남는다** —
조용히 0 이나 빈 값을 채워 "돌았다"처럼 보이게 하지 않는다.

**리포트 이름 규칙은 양방향이다: 라이브 이름 ⇔ 실측** (`evaluation.resolve_report_stem`).
실측(과금) 실행만 `evaluation-live*` 에 쓸 수 있고, 실측은 그 이름에만 쓸 수 있으며,
이미 있는 라이브 리포트는 덮어쓸 수 없다. `--live` 를 줬어도 키·DB 문제로 측정 2 가
미실행이면 비실측이므로 기본 이름은 `evaluation` 으로 떨어진다. 검사는 측정 시작 전이다.

`--stub-llm` 은 정책 청크를 **어휘 임베딩 대역**으로 다시 적재해야 하므로, 적재를
트랜잭션 안에서 하고 끝나면 **롤백한다**. 공유 DB 의 실제 임베딩을 덮어쓰지 않는다.

**알려진 일시 공백**: L2 판정(기본 켜짐)에는 아직 결정론 대역이 없어 `--stub-llm` 이
외부 호출 0회를 지킬 수 없다. 그래서 `--stub-llm` + L2 켜짐은 조용히 사이클 1 동작으로
낮춰 돌리지 않고 **명시적 오류로 멈춘다**(`_require_stub_judge_gap`).

같은 이유로 `--live` + L2 켜짐도 지금은 멈춘다(`_require_live_judge_gap`) — 단
**측정 2 가 실제로 돌 조건일 때만**이다: 리포트에 L2 켜짐 여부도 판정 모델도 남지 않고
이름 계열도 사이클 1 실측과 같아, 덮어쓸 수 없는 라이브 산출물이 "L2 포함 여부 불명"인
채로 영구히 남기 때문이다. 키·DB 가 없어 측정 2 가 미실행이면 리포트는 실측 이름을 받지
못하고(덮어쓸 수 있는 `evaluation` 계열) 그 피해가 성립하지 않으므로 막지 않는다 —
측정 1 결과와 "측정 2 미실행 + 사유"를 담은 리포트가 그대로 남는다. 두 가드 모두 하네스
확장 태스크가 걷어낸다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import psycopg
from psycopg.rows import DictRow

from reply_gate.config import Settings, get_settings
from reply_gate.db import connect, database_unavailable_reason, readonly_connect
from reply_gate.evaluation import (
    DEFAULT_GOLDEN_SET_PATH,
    DEFAULT_L1_FIXTURES_PATH,
    DEFAULT_REPORT_DIR,
    DEFAULT_REPORT_STEM,
    LIVE_REPORT_STEM,
    EvaluationReport,
    GoldenCase,
    GoldenOutcome,
    PipelineAgreement,
    ReportStemError,
    RunConditions,
    SkippedMeasurement,
    StubGenerationClient,
    assess_targets,
    build_report,
    display_path,
    load_golden_set,
    load_l1_fixtures,
    measure_gate_accuracy,
    measure_pipeline_agreement,
    resolve_report_stem,
    utc_now_iso,
    write_report,
)
from reply_gate.llm import (
    EmbeddingClient,
    GenerationClient,
    OpenAIEmbeddingClient,
    OpenAIGenerationClient,
)
from reply_gate.pipeline import build_pipeline
from reply_gate.policy_index import index_policy_documents, load_policy_documents
from reply_gate.testing import LexicalEmbeddingClient

#: 대역 임베딩은 실제 모델과 유사도 분포가 달라 기본 임계값(0.3)에서 거의 다 걸러진다.
#: 배관 검증용 실행에서만 쓰는 낮춘 기본값이고, 리포트에 그대로 기록된다.
STUB_SIMILARITY_THRESHOLD = 0.05


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="L1 게이트 단위 정확도(측정 1)와 파이프라인 판정 일치율(측정 2)을 측정한다"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--live",
        action="store_true",
        help="측정 2 를 실제 OpenAI 클라이언트로 실행한다 (과금·비결정론, OPENAI_API_KEY 필요)",
    )
    mode.add_argument(
        "--stub-llm",
        action="store_true",
        help="측정 2 를 결정론 대역으로 실행한다 (하네스 배관 검증용 — 실제 수치가 아니다)",
    )
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET_PATH)
    parser.add_argument("--l1-fixtures", type=Path, default=DEFAULT_L1_FIXTURES_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--report-stem",
        default=None,
        help=(
            f"리포트 파일 이름 조각 (기본: 실측이면 `{LIVE_REPORT_STEM}`, 아니면 "
            f"`{DEFAULT_REPORT_STEM}`). 라이브 이름 ⇔ 실측 규칙과 기존 실측 덮어쓰기 금지가 "
            f"실행 전에 강제된다"
        ),
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="정책 검색 유사도 임계값 (기본: --stub-llm 이면 0.05, 아니면 설정값)",
    )
    parser.add_argument("--top-k", type=int, default=None, help="정책 검색 상위 k (기본: 설정값)")
    return parser


def _skip_reason(*, args: argparse.Namespace, settings: Settings) -> str | None:
    """측정 2 를 실행하지 못하는 사유. 실행 가능하면 `None`."""
    if not args.live and not args.stub_llm:
        key_state = "설정됨" if settings.openai_api_key else "없음"
        return (
            "측정 2 실행이 요청되지 않았다 — 실제 생성 LLM 실행은 과금·비결정론이므로 "
            f"`--live` 로 명시해야 한다 (현재 OPENAI_API_KEY: {key_state}). "
            "배관만 확인하려면 `--stub-llm`"
        )
    if args.live and not settings.openai_api_key:
        return "OPENAI_API_KEY 가 없다 — 측정 2 는 실제 생성 LLM 이 있어야 진짜 수치가 나온다"
    if args.live and settings.l2_enabled and not settings.anthropic_api_key:
        # 시작해 놓고 첫 판정 호출에서 죽으면 30건 중 일부만 과금하고 산출물은 없다.
        return (
            "ANTHROPIC_API_KEY 가 없다 — L2 판정이 켜져 있으면 판정 없이 답변을 확정하지 "
            "않으므로 측정 2 를 시작하지 않는다 (판정을 빼고 재보려면 L2_ENABLED=false)"
        )
    return database_unavailable_reason(settings=settings)


def _measurement_two_settings(*, args: argparse.Namespace, settings: Settings) -> Settings:
    threshold = args.similarity_threshold
    if threshold is None:
        threshold = (
            STUB_SIMILARITY_THRESHOLD if args.stub_llm else settings.vector_similarity_threshold
        )
    return settings.model_copy(
        update={
            "vector_similarity_threshold": float(threshold),
            "vector_top_k": int(args.top_k) if args.top_k is not None else settings.vector_top_k,
        }
    )


def _clients(
    *, args: argparse.Namespace, settings: Settings
) -> tuple[GenerationClient, EmbeddingClient, str, str]:
    """(생성 클라이언트, 임베딩 클라이언트, 생성 설명, 임베딩 설명)."""
    if args.stub_llm:
        embedder = LexicalEmbeddingClient(dimensions=settings.embedding_dimensions)
        return (
            cast(GenerationClient, StubGenerationClient()),
            embedder,
            "결정론 대역 `evaluation.StubGenerationClient` (실제 모델 아님)",
            f"결정론 대역 `testing.LexicalEmbeddingClient`({embedder.dimensions}차원, 어휘 2-gram)",
        )
    # 실제 실행 — 생성·임베딩 모두 OpenAI. 여기서만 실제 API 키를 쓴다.
    return (
        OpenAIGenerationClient(api_key=settings.openai_api_key, model=settings.generation_model),
        OpenAIEmbeddingClient(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        ),
        f"OpenAI `{settings.generation_model}` (effort={settings.generation_effort or '기본값'})",
        f"OpenAI `{settings.embedding_model}` ({settings.embedding_dimensions}차원)",
    )


def _run_measurement_two(
    *,
    cases: tuple[GoldenCase, ...],
    args: argparse.Namespace,
    settings: Settings,
) -> tuple[PipelineAgreement, str, str]:
    generation_client, embedding_client, generation_label, embedding_label = _clients(
        args=args, settings=settings
    )
    pipeline = build_pipeline(
        generation_client=generation_client,
        embedding_client=embedding_client,
        settings=settings,
    )

    def progress(outcome: GoldenOutcome) -> None:
        mark = "OK " if outcome.matched else "MISS"
        status = "접수거부" if outcome.status is None else outcome.status.value
        reason = f"/{outcome.escalation_reason.value}" if outcome.escalation_reason else ""
        print(f"  [{mark}] {outcome.case_id} {status}{reason} ({outcome.latency_ms} ms)")

    app_conn: psycopg.Connection[DictRow] = connect(settings=settings)
    try:
        if args.stub_llm:
            # 대역 임베딩으로 재적재한다. **끝나면 롤백**해 공유 DB 의 실제 임베딩을 지킨다.
            index_policy_documents(
                conn=app_conn, documents=load_policy_documents(), embedder=embedding_client
            )
        with readonly_connect(settings=settings) as ro_conn:
            agreement = measure_pipeline_agreement(
                cases=cases,
                pipeline=pipeline,
                app_conn=app_conn,
                readonly_conn=ro_conn,
                on_outcome=progress,
            )
    finally:
        # 하네스는 처리 기록을 남기지 않는다 — 리포트가 산출물이고 DB 는 그대로 둔다.
        app_conn.rollback()
        app_conn.close()

    return agreement, generation_label, embedding_label


def _require_stub_judge_gap(*, args: argparse.Namespace, settings: Settings) -> None:
    """`--stub-llm` + L2 켜짐은 **지금 돌릴 수 없다** — 알려진 일시 공백이라 명시적으로 죽는다.

    대역 실행은 외부 호출 0회여야 하는데 결정론 **판정** 대역이 아직 없다(하네스 확장
    태스크 몫). 여기서 조용히 `l2_enabled=False` 로 낮춰 돌리면 리포트에는 대역 수치가
    찍히지만 그 수치가 사이클 1 배관을 잰 것인지 L2 를 포함해 잰 것인지 알 수 없게 된다 —
    "미실행 측정을 0 이나 빈 값으로 채우지 않는다"와 같은 이유로 멈춘다.
    """
    if args.stub_llm and settings.l2_enabled:
        raise SystemExit(
            "`--stub-llm` 은 지금 돌릴 수 없다: L2 판정이 켜져 있는데(기본 켜짐) 결정론 판정 "
            "대역이 아직 없어 대역 실행이 실제 판정 모델을 부르게 된다. 이 공백은 하네스 확장 "
            "태스크가 결정론 판정 대역과 함께 닫는다. 지금 배관만 확인하려면 L2_ENABLED=false "
            "로 **명시**하고 다시 실행한다 (그 실행은 사이클 1 동작을 잰 값이다)."
        )


def _require_live_judge_gap(
    *, args: argparse.Namespace, settings: Settings, skip: str | None
) -> None:
    """`--live` + L2 켜짐은 **실측이 실제로 돌 때만** 막는다 — 산출물이 L2 포함 여부를
    말하지 못하기 때문이다.

    라이브 리포트는 덮어쓸 수 없다(`evaluation.resolve_report_stem`). 그런데 지금은
    실행 조건(`RunConditions`)에 L2 켜짐 여부도 판정 모델도 없고, 이름 계열도 사이클 1
    실측이 쓰던 `evaluation-live-<n>` 그대로다. 이대로 돌리면 **"L2 를 포함해 잰 값인지
    알 수 없는" 실측 리포트가 영구히 남는다** — `--stub-llm` 을 막는 것과 같은 모호함이다
    (대역 수치가 무엇을 잰 것인지 알 수 없게 되는 문제).

    **그래서 조건은 `skip is None` 까지다.** 막으려는 피해는 리포트가 실측 이름을 받을
    때만(`measurement2_is_real`, 즉 `--live` + 측정 2 실행 가능) 성립한다. 키도 DB 도
    없는 `--live` 는 측정 2 가 미실행이라 덮어쓸 수 있는 `evaluation` 계열로 떨어지므로,
    여기서 죽이면 무료인 측정 1 산출물과 "측정 2 미실행 + 사유" 기록까지 함께 잃는다
    (`scripts/AGENTS.md` 의 "미실행 측정은 리포트에 미실행 + 사유로 남긴다").

    **이 가드는 뒤 태스크가 걷어낸다**: L2 실측 리포트 이름 계열(`evaluation-live-l2-<n>`)과
    실행 조건의 판정 항목(L2 켜짐·판정 모델)이 들어오는 순간 이 함수를 지운다. 그때까지
    L2 를 포함한 실측은 산출물을 남길 자리가 없다.
    """
    if args.live and settings.l2_enabled and skip is None:
        raise SystemExit(
            "`--live` 는 지금 돌릴 수 없다: L2 판정이 켜져 있는데(기본 켜짐) 리포트가 그 "
            "사실을 담지 못한다 — 실행 조건에 L2 켜짐 여부도 판정 모델도 남지 않고, 이름도 "
            "사이클 1 실측과 같은 `evaluation-live-<n>` 계열이라 덮어쓸 수 없는 산출물이 "
            "'L2 포함 여부 불명'으로 남는다. L2 실측용 이름 계열과 실행 조건 기록이 들어오는 "
            "하네스 확장 태스크가 이 가드를 걷어낸다. 지금 사이클 1 실측을 다시 재려면 "
            "L2_ENABLED=false 로 **명시**하고 다시 실행한다."
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    # 측정 1(무료)조차 시작하기 전에 죽인다 — 돌릴 수 없는 요청은 산출물을 남기지 않는다.
    _require_stub_judge_gap(args=args, settings=settings)
    # 라이브 가드는 미실행 사유를 알아야 판단한다(실측 이름을 받을 실행만 막는다).
    skip = _skip_reason(args=args, settings=settings)
    _require_live_judge_gap(args=args, settings=settings, skip=skip)

    fixtures = load_l1_fixtures(args.l1_fixtures)
    cases = load_golden_set(args.golden_set)

    run_settings = _measurement_two_settings(args=args, settings=settings)
    measurement2_is_real = bool(args.live) and skip is None

    # 리포트 이름은 **측정을 시작하기 전에** 확정한다 — 여기서 거부될 실행이 과금(라이브
    # 30건)이나 대역 재적재를 먼저 하고 나서 산출물만 버리는 일이 없어야 한다.
    try:
        stem = resolve_report_stem(
            requested=args.report_stem,
            live_requested=bool(args.live),
            measurement2_is_real=measurement2_is_real,
            out_dir=args.out_dir,
        )
    except ReportStemError as error:
        raise SystemExit(str(error)) from error

    print(f"측정 1 — L1 픽스처 {len(fixtures)}건 (LLM 호출 0회)")
    # 실행 시각은 측정을 **시작하기 전에** 찍는다 — 라이브 30건은 수 분이 걸려,
    # 끝난 뒤에 찍으면 리포트의 `started_at` 이 실제 시작 시각과 어긋난다.
    started_at = utc_now_iso()
    accuracy = measure_gate_accuracy(fixtures)

    pipeline: PipelineAgreement | SkippedMeasurement
    if skip is not None:
        print(f"측정 2 — 미실행: {skip}")
        pipeline = SkippedMeasurement(reason=skip)
        generation_label = "미실행"
        embedding_label = "미실행"
    else:
        print(f"측정 2 — 골든셋 {len(cases)}건 end-to-end")
        pipeline, generation_label, embedding_label = _run_measurement_two(
            cases=cases, args=args, settings=run_settings
        )

    conditions = RunConditions(
        started_at=started_at,
        generation=generation_label,
        embedding=embedding_label,
        similarity_threshold=run_settings.vector_similarity_threshold,
        top_k=run_settings.vector_top_k,
        l1_fixture_count=len(fixtures),
        golden_case_count=len(cases),
        # 커밋되는 라이브 리포트에 로컬 절대 경로(사용자명 포함)를 남기지 않는다.
        l1_fixtures_path=display_path(args.l1_fixtures),
        golden_set_path=display_path(args.golden_set),
        api_key_present=bool(settings.openai_api_key),
        measurement2_is_real=measurement2_is_real,
    )
    report: EvaluationReport = build_report(
        conditions=conditions, gate_accuracy=accuracy, pipeline=pipeline
    )
    markdown_path, json_path = write_report(report, out_dir=args.out_dir, stem=stem)

    _print_summary(report)
    _print_targets(report)
    print(f"\n리포트: {markdown_path}\n리포트(JSON): {json_path}")
    return 0


def _print_summary(report: EvaluationReport) -> None:
    accuracy = report.gate_accuracy
    detection = accuracy.detection_rate
    false_positive = accuracy.false_positive_rate
    print(
        "\n측정 1 결과 — 검출률 "
        f"{'n/a' if detection is None else f'{detection * 100:.1f}%'} "
        f"({accuracy.violation_detected}/{accuracy.violation_total}), 오탐률 "
        f"{'n/a' if false_positive is None else f'{false_positive * 100:.1f}%'} "
        f"({accuracy.clean_false_positive}/{accuracy.clean_total})"
    )
    if isinstance(report.pipeline, SkippedMeasurement):
        print(f"측정 2 결과 — 미실행 (사유: {report.pipeline.reason})")
        return
    agreement = report.pipeline
    match_rate = agreement.match_rate
    recall = agreement.bait_reject_recall
    print(
        "측정 2 결과 — 일치율 "
        f"{'n/a' if match_rate is None else f'{match_rate * 100:.1f}%'} "
        f"({agreement.matched}/{agreement.total}), 기각 재현율 "
        f"{'n/a' if recall is None else f'{recall * 100:.1f}%'} "
        f"({agreement.bait_reject_reproduced}/{agreement.bait_total})"
    )
    if not report.conditions.measurement2_is_real:
        print("  ※ 위 측정 2 수치는 대역으로 낸 값이다 — 실제 수치가 아니다.")


def _print_targets(report: EvaluationReport) -> None:
    """목표치 판정을 콘솔에도 찍는다 — 회귀(미달)를 리포트 파일을 열어야만 아는 일이 없게."""
    print("목표치 판정:")
    for item in assess_targets(report):
        value = "미측정" if item.value is None else f"{item.value * 100:.1f}%"
        print(f"  [{item.verdict}] {item.target.label} {item.target.describe()} — 실측 {value}")


if __name__ == "__main__":
    raise SystemExit(main())
