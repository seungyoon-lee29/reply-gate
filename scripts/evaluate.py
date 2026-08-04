"""평가 하네스 실행 — 측정 1(L1 픽스처)과 측정 2(골든셋 end-to-end)를 돌려 리포트를 낸다.

    uv run python -m scripts.evaluate                 # 측정 1 만 (측정 2 는 미실행 사유 기록)
    uv run python -m scripts.evaluate --stub-llm      # 측정 2 를 결정론 대역으로 (배관 검증)
    uv run python -m scripts.evaluate --live          # 측정 2 를 실제 OpenAI 로 (과금·비결정론)

**측정 1 은 항상 돈다** — LLM 을 호출하지 않으므로 키도 DB 도 필요 없다.

**측정 2 는 명시적 opt-in 이다.** 실제 실행은 과금되고 결과가 재실행마다 달라지므로
기본값으로 돌리지 않는다. 실행하지 않았으면 리포트에 **미실행 사유가 그대로 남는다** —
조용히 0 이나 빈 값을 채워 "돌았다"처럼 보이게 하지 않는다.

`--stub-llm` 은 정책 청크를 **어휘 임베딩 대역**으로 다시 적재해야 하므로, 적재를
트랜잭션 안에서 하고 끝나면 **롤백한다**. 공유 DB 의 실제 임베딩을 덮어쓰지 않는다.
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
    EvaluationReport,
    GoldenCase,
    GoldenOutcome,
    PipelineAgreement,
    RunConditions,
    SkippedMeasurement,
    StubGenerationClient,
    build_report,
    load_golden_set,
    load_l1_fixtures,
    measure_gate_accuracy,
    measure_pipeline_agreement,
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
    parser.add_argument("--report-stem", default="evaluation")
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    fixtures = load_l1_fixtures(args.l1_fixtures)
    cases = load_golden_set(args.golden_set)

    print(f"측정 1 — L1 픽스처 {len(fixtures)}건 (LLM 호출 0회)")
    # 실행 시각은 측정을 **시작하기 전에** 찍는다 — 라이브 30건은 수 분이 걸려,
    # 끝난 뒤에 찍으면 리포트의 `started_at` 이 실제 시작 시각과 어긋난다.
    started_at = utc_now_iso()
    accuracy = measure_gate_accuracy(fixtures)

    run_settings = _measurement_two_settings(args=args, settings=settings)
    skip = _skip_reason(args=args, settings=settings)
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
        l1_fixtures_path=str(args.l1_fixtures),
        golden_set_path=str(args.golden_set),
        api_key_present=bool(settings.openai_api_key),
        measurement2_is_real=bool(args.live) and skip is None,
    )
    report: EvaluationReport = build_report(
        conditions=conditions, gate_accuracy=accuracy, pipeline=pipeline
    )
    markdown_path, json_path = write_report(report, out_dir=args.out_dir, stem=args.report_stem)

    _print_summary(report)
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


if __name__ == "__main__":
    raise SystemExit(main())
