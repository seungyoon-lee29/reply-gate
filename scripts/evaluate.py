"""평가 하네스 실행 — 측정 1(L1 픽스처)·측정 2(골든셋 end-to-end)·측정 3(판정 픽스처)을
돌려 리포트를 낸다.

    uv run python -m scripts.evaluate                 # 측정 1 만 (측정 2·3 은 미실행 사유 기록)
    uv run python -m scripts.evaluate --stub-llm      # 측정 2 를 결정론 대역으로 (배관 검증)
    uv run python -m scripts.evaluate --live          # 측정 2·3 을 실제 모델로 (과금·비결정론)

**측정 1 은 항상 돈다** — LLM 을 호출하지 않으므로 키도 DB 도 필요 없다.

**측정 2·3 은 명시적 opt-in 이다.** 실제 실행은 과금되고 결과가 재실행마다 달라지므로
기본값으로 돌리지 않는다. 실행하지 않았으면 리포트에 **미실행 사유가 그대로 남는다** —
조용히 0 이나 빈 값을 채워 "돌았다"처럼 보이게 하지 않는다.

**측정 3 은 `--live` + L2 켜짐에서만 돈다.** 판정 모델을 실제로 부르는 확률 층이라
대역으로는 낼 수 없는 수치이고, L2 꺼짐 기준선 실행에서 돌면 "꺼짐 기준선"이 판정 비용을
쓰는 이상한 실행이 된다. 그리고 **측정 2 와 실행 조건을 공유한다**: 과금 산출물이
덮어쓸 수 있는 이름에 남지 않으려면 둘의 실측 여부가 갈리면 안 된다.

**키 선검사는 측정 시작 전이다.** `--live` + L2 켜짐이면 `OPENAI_API_KEY` 와
`ANTHROPIC_API_KEY` 를 **둘 다** 본다. 판정 키 부재를 측정 도중 발견하면 과금만 하고
산출물 없이 죽는다. 판정 키가 없으면 측정 2·3 을 **모두** 건너뛴다 — L2 를 꺼서 측정 2 만
돌리는 **강등 실행은 금지**다(기준선과 실행 조건이 오염된다).

**리포트 이름 규칙은 두 겹이고 둘 다 양방향이다** (`evaluation.resolve_report_stem`):
라이브 이름 ⇔ 실측, 그리고 **L2 켜짐 실측 ⇔ `evaluation-live-l2` 접두**. 실측(과금)
실행만 `evaluation-live*` 에 쓸 수 있고, 이미 있는 라이브 리포트는 덮어쓸 수 없다.
`--live` 를 줬어도 키·DB 문제로 측정 2 가 미실행이면 비실측이므로 기본 이름은
`evaluation` 으로 떨어진다. 검사는 측정 시작 전이다.

`--stub-llm` 은 정책 청크를 **어휘 임베딩 대역**으로 다시 적재해야 하므로, 적재를
트랜잭션 안에서 하고 끝나면 **롤백한다**. 공유 DB 의 실제 임베딩을 덮어쓰지 않는다.
판정도 결정론 대역(`testing.StubJudge`)으로 갈아 끼우므로 **외부 호출 0회**다.
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
    DEFAULT_JUDGE_FIXTURES_PATH,
    DEFAULT_L1_FIXTURES_PATH,
    DEFAULT_REPORT_DIR,
    DEFAULT_REPORT_STEM,
    LIVE_L2_REPORT_STEM,
    LIVE_REPORT_STEM,
    EvaluationReport,
    GoldenCase,
    GoldenOutcome,
    JudgeAccuracy,
    JudgeFixture,
    L2Outcome,
    PipelineAgreement,
    ReportStemError,
    RunConditions,
    SkippedMeasurement,
    StubGenerationClient,
    assess_targets,
    build_report,
    display_path,
    load_golden_set,
    load_judge_fixtures,
    load_l1_fixtures,
    measure_gate_accuracy,
    measure_judge_accuracy,
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
from reply_gate.pipeline import Judging, build_judge, build_pipeline
from reply_gate.policy_index import index_policy_documents, load_policy_documents
from reply_gate.testing import LexicalEmbeddingClient, StubJudge, build_stub_pipeline

#: 대역 임베딩은 실제 모델과 유사도 분포가 달라 기본 임계값(0.3)에서 거의 다 걸러진다.
#: 배관 검증용 실행에서만 쓰는 낮춘 기본값이고, 리포트에 그대로 기록된다.
STUB_SIMILARITY_THRESHOLD = 0.05


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "L1 게이트 단위 정확도(측정 1)·파이프라인 판정 일치율(측정 2)·"
            "L2 판정 단위 정확도(측정 3)를 측정한다"
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--live",
        action="store_true",
        help=(
            "측정 2·3 을 실제 모델로 실행한다 "
            "(과금·비결정론, OPENAI_API_KEY + L2 켜짐이면 ANTHROPIC_API_KEY 필요)"
        ),
    )
    mode.add_argument(
        "--stub-llm",
        action="store_true",
        help=(
            "측정 2 를 결정론 대역(생성·임베딩·판정)으로 실행한다 "
            "(하네스 배관 검증용 — 실제 수치가 아니다. 측정 3 은 돌지 않는다)"
        ),
    )
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET_PATH)
    parser.add_argument("--l1-fixtures", type=Path, default=DEFAULT_L1_FIXTURES_PATH)
    parser.add_argument("--judge-fixtures", type=Path, default=DEFAULT_JUDGE_FIXTURES_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--report-stem",
        default=None,
        help=(
            f"리포트 파일 이름 조각 (기본: L2 켜짐 실측이면 `{LIVE_L2_REPORT_STEM}-<n>`, "
            f"꺼짐 실측이면 `{LIVE_REPORT_STEM}`, 실측이 아니면 `{DEFAULT_REPORT_STEM}`). "
            f"라이브 이름 ⇔ 실측 · L2 켜짐 ⇔ l2 접두 규칙과 기존 실측 덮어쓰기 금지가 "
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
        # 여기서 L2 를 꺼서 측정 2 만 돌리는 **강등은 하지 않는다** — 그렇게 나온 값은
        # 꺼짐 기준선인데 실행 조건은 켜짐으로 요청된 실행이라 둘 다 오염된다.
        return (
            "ANTHROPIC_API_KEY 가 없다 — L2 판정이 켜져 있으면 판정 없이 답변을 확정하지 "
            "않으므로 측정 2·3 을 시작하지 않는다 (판정을 빼고 재보려면 L2_ENABLED=false 로 "
            "명시한다 — 그 실행은 꺼짐 기준선이다)"
        )
    return database_unavailable_reason(settings=settings)


def _judge_skip_reason(
    *, args: argparse.Namespace, settings: Settings, skip: str | None
) -> str | None:
    """측정 3 을 실행하지 못하는 사유. 실행 가능하면 `None`.

    `--live` + L2 켜짐이 아니면 애초에 돌지 않고, 그 조건을 만족해도 **측정 2 의 사유를
    그대로 물려받는다**: 키 부재도, DB 부재도 마찬가지다. DB 자체는 측정 3 에 필요 없지만,
    두 측정의 실측 여부가 갈리면 과금된 판정 수치가 실측 이름을 받지 못하는 리포트
    (덮어쓸 수 있는 `evaluation` 계열)에 실린다 — 재생성에 돈이 드는 산출물이 재생성이
    공짜인 산출물과 같은 경로를 쓰는, 한 번 겪은 그 사고다.
    """
    if not args.live:
        return (
            "측정 3 은 판정 모델을 실제로 부르는 확률 층이라 `--live` 로만 돈다 — "
            "결정론 대역으로는 판정 정확도를 잴 수 없다"
        )
    if not settings.l2_enabled:
        return (
            "L2 판정이 꺼져 있다 — 이 실행은 꺼짐 기준선이므로 판정 계열 측정을 돌리지 않는다 "
            "(L2_ENABLED=true 로 다시 실행한다)"
        )
    if skip is not None:
        return f"측정 2 와 같은 사유로 미실행: {skip}"
    return None


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


def _judge_label(*, args: argparse.Namespace, settings: Settings, skip: str | None) -> str:
    """실행 조건에 남는 판정 모델 설명 — 대역 수치를 실제 수치로 읽지 않게 하는 기록이다."""
    if not settings.l2_enabled:
        return "L2 꺼짐 (판정 미실행)"
    if skip is not None:
        return "미실행"
    if args.stub_llm:
        return "결정론 대역 `testing.StubJudge` (실제 모델 아님)"
    return f"Anthropic `{settings.judge_model}` (effort={settings.judge_effort or '기본값'})"


def _run_measurement_two(
    *,
    cases: tuple[GoldenCase, ...],
    args: argparse.Namespace,
    settings: Settings,
) -> tuple[PipelineAgreement, str, str]:
    generation_client, embedding_client, generation_label, embedding_label = _clients(
        args=args, settings=settings
    )
    pipeline = (
        # 대역 실행은 판정도 대역이어야 외부 호출 0회다 — `build_pipeline` 은 판정자를
        # 설정에서 조립하므로 주입 구멍이 없다(`testing.build_stub_pipeline` 의 존재 이유).
        build_stub_pipeline(
            generation_client=generation_client,
            embedding_client=embedding_client,
            judge=StubJudge(),
            settings=settings,
        )
        if args.stub_llm
        else build_pipeline(
            generation_client=generation_client,
            embedding_client=embedding_client,
            settings=settings,
        )
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


def _run_measurement_three(
    *, fixtures: tuple[JudgeFixture, ...], settings: Settings
) -> JudgeAccuracy:
    """측정 3 — 판정 픽스처를 **실제 판정기**에 흘린다(과금).

    파이프라인을 거치지 않고 판정기를 직접 부른다: 재는 것이 판정 층 하나이므로 초안
    생성·근거 수집이 끼면 무엇을 잰 것인지 흐려진다.
    """
    judge: Judging = build_judge(settings)

    def progress(outcome: L2Outcome) -> None:
        if outcome.error is not None:
            print(f"  [FAIL] {outcome.fixture_id} 판정 실패: {outcome.error}")
            return
        mark = "OK " if outcome.reasons_matched else "MISS"
        actual = "판정 없음" if outcome.actual_verdict is None else outcome.actual_verdict.value
        print(
            f"  [{mark}] {outcome.fixture_id} 기대 {outcome.expected_verdict.value} / 실제 {actual}"
        )

    return measure_judge_accuracy(fixtures=fixtures, judge=judge, on_outcome=progress)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    # 키·DB 선검사는 **측정 시작 전**이다 — 도중에 발견하면 과금만 하고 산출물이 없다.
    skip = _skip_reason(args=args, settings=settings)
    judge_skip = _judge_skip_reason(args=args, settings=settings, skip=skip)

    fixtures = load_l1_fixtures(args.l1_fixtures)
    cases = load_golden_set(args.golden_set)
    judge_fixtures = load_judge_fixtures(args.judge_fixtures)

    run_settings = _measurement_two_settings(args=args, settings=settings)
    measurement2_is_real = bool(args.live) and skip is None

    # 리포트 이름은 **측정을 시작하기 전에** 확정한다 — 여기서 거부될 실행이 과금(라이브
    # 30건)이나 대역 재적재를 먼저 하고 나서 산출물만 버리는 일이 없어야 한다.
    try:
        stem = resolve_report_stem(
            requested=args.report_stem,
            live_requested=bool(args.live),
            measurement2_is_real=measurement2_is_real,
            l2_enabled=settings.l2_enabled,
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

    judge_accuracy: JudgeAccuracy | SkippedMeasurement
    if judge_skip is not None:
        print(f"측정 3 — 미실행: {judge_skip}")
        judge_accuracy = SkippedMeasurement(reason=judge_skip)
    else:
        print(f"측정 3 — 판정 픽스처 {len(judge_fixtures)}건 (실제 판정 모델·과금)")
        judge_accuracy = _run_measurement_three(fixtures=judge_fixtures, settings=run_settings)

    conditions = RunConditions(
        started_at=started_at,
        generation=generation_label,
        embedding=embedding_label,
        judge=_judge_label(args=args, settings=settings, skip=skip),
        similarity_threshold=run_settings.vector_similarity_threshold,
        top_k=run_settings.vector_top_k,
        l1_fixture_count=len(fixtures),
        golden_case_count=len(cases),
        judge_fixture_count=len(judge_fixtures),
        # 커밋되는 라이브 리포트에 로컬 절대 경로(사용자명 포함)를 남기지 않는다.
        l1_fixtures_path=display_path(args.l1_fixtures),
        golden_set_path=display_path(args.golden_set),
        judge_fixtures_path=display_path(args.judge_fixtures),
        api_key_present=bool(settings.openai_api_key),
        judge_api_key_present=bool(settings.anthropic_api_key),
        l2_enabled=settings.l2_enabled,
        measurement2_is_real=measurement2_is_real,
    )
    report: EvaluationReport = build_report(
        conditions=conditions,
        gate_accuracy=accuracy,
        pipeline=pipeline,
        judge_accuracy=judge_accuracy,
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
    else:
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

    if isinstance(report.judge_accuracy, SkippedMeasurement):
        print(f"측정 3 결과 — 미실행 (사유: {report.judge_accuracy.reason})")
        return
    judged = report.judge_accuracy
    detection_rate = judged.detection_rate
    false_positive_rate = judged.false_positive_rate
    print(
        "측정 3 결과 — L2 검출률 "
        f"{'n/a' if detection_rate is None else f'{detection_rate * 100:.1f}%'} "
        f"({judged.violation_detected}/{judged.violation_total}), 오탐률 "
        f"{'n/a' if false_positive_rate is None else f'{false_positive_rate * 100:.1f}%'} "
        f"({judged.clean_false_positive}/{judged.clean_total})"
        f"{f', 판정 실패 {judged.error_total}건' if judged.error_total else ''}"
    )
    print("  ※ 측정 3 은 확률 층이고 목표치는 미확정이다 — 달성·미달 판정이 붙지 않는다.")


def _print_targets(report: EvaluationReport) -> None:
    """목표치 판정을 콘솔에도 찍는다 — 회귀(미달)를 리포트 파일을 열어야만 아는 일이 없게."""
    print("목표치 판정:")
    for item in assess_targets(report):
        value = "미측정" if item.value is None else f"{item.value * 100:.1f}%"
        print(f"  [{item.verdict}] {item.target.label} {item.target.describe()} — 실측 {value}")


if __name__ == "__main__":
    raise SystemExit(main())
