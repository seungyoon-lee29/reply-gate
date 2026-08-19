"""평가 하네스 실행 — 측정 1(L1 픽스처)·측정 2(골든셋 end-to-end)·측정 3(판정 픽스처)을
돌려 리포트를 낸다.

    uv run python -m scripts.evaluate                 # 측정 1 만 (측정 2·3 은 미실행 사유 기록)
    uv run python -m scripts.evaluate --stub-llm      # 측정 2 를 결정론 대역으로 (배관 검증)
    uv run python -m scripts.evaluate --live          # 측정 2·3 을 실제 모델로 (과금·비결정론)

**측정 1 은 항상 돈다** — LLM 을 호출하지 않으므로 키도 DB 도 필요 없다. 판정 픽스처
파일이 없거나 라벨이 깨져 있어도 마찬가지다: 그 실패는 **측정 3 의 미실행 사유**로 강등되고
측정 1·2 는 그대로 산출된다. **측정 2·3 이 예상 밖 예외로 중단돼도 같다** — 어느 쪽이 죽어도
리포트는 반드시 쓰인다. 중단을 트레이스백으로 알리면 이미 과금이 끝난 산출물(골든셋 30건
중 완주분)과 무료인 측정 1 산출물이 함께 사라지기 때문이다. 중단은 **리포트의 미실행
사유**(예외 종류와 메시지를 그대로 담는다)와 종료 코드로만 알린다.

**종료 코드가 1 이 되는 경우는 둘뿐이다 — ① 측정 3 이 돌 조건이었는데 미실행,
② 사용자 중단(Ctrl-C). 그 밖에는 0.**
①의 "돌 조건"은 `--live` 로 선검사(키·DB)를 통과해 **측정 2 가 실측으로 돌 조건이었고**
**L2 도 켜져 있던** 실행이다. 완주했든 도중에 중단됐든 같다: 측정 2 가 중단되면 측정 3 도
잇지 않으므로(추가 과금을 하면 안 된다) 그 실행도 1 이 된다. 판정 픽스처 로드 실패로
측정 3 이 미실행인 것도 마찬가지다 — 골든셋 30건을 사고 판정 수치는 못 낸 실행이 래퍼·CI 에
"성공"으로 읽히면 안 된다.
②는 **과금 여부와 무관하다**: `--stub-llm`·기본 실행에서 Ctrl-C 를 눌러도 1 이다. 리포트는
그대로 쓰되(과금분 보존) 중단 사실을 종료 코드가 들고 간다.
반면 **`--live` 없이 도는 평범한 실행**에서 측정 3 이 "`--live` 아님" 사유로 미실행인 것과,
**L2 꺼짐 기준선 실행**(`L2_ENABLED=false`)에서 측정 3 이 설계상 안 도는 것은 **정상이므로
0** 이다 — 목표치를 전부 달성한 꺼짐 기준선이 실패로 읽히면 안 된다.

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
라이브 이름 ⇔ **과금 실행**, 그리고 **L2 켜짐 실측 ⇔ `evaluation-live-l2` 접두**. 과금
실행만 `evaluation-live*` 에 쓸 수 있고, 이미 있는 라이브 리포트는 덮어쓸 수 없다.
`--live` 를 줬어도 키·DB 문제로 선검사를 통과하지 못하면 과금이 아니므로 기본 이름은
`evaluation` 으로 떨어진다. 검사는 측정 시작 전이다.

**실행마다 회귀 가드가 두 줄로 보고한다** — 승격 기준선(구속) + 직전 라이브(경보).
승격은 `data/promoted_baseline.json` 을 **사람이** 바꾸는 것뿐이고, 이 스크립트를 포함해
저장소 어디에도 그 파일을 쓰는 경로가 없다.

`--stub-llm` 은 정책 청크를 **어휘 임베딩 대역**으로 다시 적재해야 하므로, 적재를
트랜잭션 안에서 하고 끝나면 **롤백한다**. 공유 DB 의 실제 임베딩을 덮어쓰지 않는다.
판정도 결정론 대역(`testing.StubJudge`)으로 갈아 끼우므로 **외부 호출 0회**다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, cast

import psycopg
from psycopg.rows import DictRow

from reply_gate.config import Settings, get_settings
from reply_gate.contracts import RejectReason
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
    attach_regression_guard,
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
from reply_gate.judge import JUDGE_SYSTEM_PROMPT
from reply_gate.llm import (
    EmbeddingClient,
    GenerationClient,
    OpenAIEmbeddingClient,
    OpenAIGenerationClient,
)
from reply_gate.pipeline import Judging, build_judge, build_pipeline
from reply_gate.policy_index import index_policy_documents, load_policy_documents
from reply_gate.regression_guard import (
    GuardUnavailable,
    RegressionGuard,
    content_digest,
    text_digest,
)
from reply_gate.retrieval_labels import DEFAULT_RETRIEVAL_LABELS_PATH
from reply_gate.testing import LexicalEmbeddingClient, StubJudge, build_stub_pipeline

#: 대역 임베딩은 실제 모델과 유사도 분포가 달라 기본 임계값(0.3)에서 거의 다 걸러진다.
#: 배관 검증용 실행에서만 쓰는 낮춘 기본값이고, 리포트에 그대로 기록된다.
STUB_SIMILARITY_THRESHOLD = 0.05

#: 선택 가능한 **과금** 측정. 측정 1 은 무과금·결정론(LLM 호출 0회)이라 선택 대상이 아니고
#: 항상 돈다 — 리포트의 헤드라인 수치가 거기서 나오기 때문이다.
BILLED_MEASUREMENTS: Final[frozenset[str]] = frozenset({"2", "3"})


def selected_measurements(args: argparse.Namespace) -> frozenset[str]:
    """`--measurements` 를 해석한다. 잘못된 값은 **측정 시작 전에** 거부한다."""
    raw = [item.strip() for item in str(args.measurements).split(",") if item.strip()]
    unknown = sorted(set(raw) - BILLED_MEASUREMENTS)
    if unknown:
        raise SystemExit(
            f"거부: --measurements 에 알 수 없는 값이 있다: {', '.join(unknown)} "
            f"(고를 수 있는 값: {', '.join(sorted(BILLED_MEASUREMENTS))})"
        )
    if not raw:
        raise SystemExit(
            "거부: --measurements 가 비었다 — 과금 측정을 하나도 고르지 않은 실행은 "
            "측정 1 만 도는 기본 실행과 같다. 그때는 `--live` 를 빼라."
        )
    return frozenset(raw)


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
            "고른 과금 측정을 실제 모델로 실행한다 (과금·비결정론). 측정 2 는 "
            "OPENAI_API_KEY + DB, 측정 3 은 L2 켜짐 + ANTHROPIC_API_KEY 를 요구한다 — "
            "무엇을 고를지는 `--measurements` 다"
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
    parser.add_argument(
        "--judge-model",
        default=None,
        metavar="모델",
        help=(
            "L2 판정 모델 (기본: 설정값). 판정 모델 계열 비교를 **실행 인자로만** 하기 "
            "위한 것이라 설정 기본값은 바뀌지 않는다 — 고른 값은 실행 조건 지문의 "
            "`judge_model` 에 그대로 실린다"
        ),
    )
    parser.add_argument(
        "--measurements",
        default=",".join(sorted(BILLED_MEASUREMENTS)),
        metavar="번호목록",
        help=(
            "돌릴 **과금 측정**을 고른다 (쉼표 구분: `2`·`3`·`2,3`. 기본: 둘 다). "
            "측정 1 은 무과금·결정론이라 항상 돈다. 예: `--live --measurements 3` 은 "
            "판정 픽스처만 사는 측정 3 단독 실측이고, DB 도 OPENAI_API_KEY 도 필요 없다. "
            "고르지 않은 측정은 리포트에 '미실행 + 사유'로 적힌다(0 채움 금지)."
        ),
    )
    parser.add_argument(
        "--declare-experiment",
        action="append",
        metavar="지문항목",
        help=(
            "이번 실행이 **의도적으로 바꾼** 조건 지문 항목(반복 가능). 선언된 항목의 차이는 "
            "대조를 막지 않고 기준선 줄에 차이 목록으로 병기된다. 선언하지 않은 불일치만 "
            "'대조 불가'다 — 조용한 조건 드리프트를 잡는 것이 지문의 존재 이유다."
        ),
    )
    return parser


def _skip_reason(
    *, args: argparse.Namespace, settings: Settings, selected: frozenset[str]
) -> str | None:
    """측정 2 를 실행하지 못하는 사유. 실행 가능하면 `None`.

    **선택 제외를 가장 먼저 본다** — 측정 2 를 고르지 않은 실행에까지 DB·생성 키를
    요구하면 측정 3 단독 실측이 있지도 않은 전제 때문에 죽는다.
    """
    if "2" not in selected:
        return "측정 2 를 실행 선택에서 제외했다 (`--measurements`) — 고르지 않은 측정이다"
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
    *, args: argparse.Namespace, settings: Settings, skip: str | None, selected: frozenset[str]
) -> str | None:
    """측정 3 을 실행하지 못하는 사유. 실행 가능하면 `None`.

    측정 3 이 실제로 필요로 하는 것은 **판정 키 + L2 켜짐**뿐이다 — DB 도, 생성 키도,
    골든셋도 필요 없다(입력은 판정 픽스처다). 그래서 선검사도 그것만 본다.

    **측정 2 의 사유를 무조건 물려받지 않는다.** 예전에는 그렇게 했고 근거는 "두 측정의
    실측 여부가 갈리면 과금된 판정 수치가 덮어쓸 수 있는 이름으로 샌다"였는데, 그 근거는
    리포트 이름의 자격이 **과금 실행 여부**로 바뀌면서 사라졌다(`scripts/AGENTS.md`
    불변식 7·11). 이제 측정 3 단독 실측도 라이브 이름을 받는다.

    다만 **측정 2 를 함께 고른 실행**에서는 여전히 물려받는다: 사용자가 요청한 것은 두
    측정이 함께 도는 한 세트인데 한쪽만 사는 것은 요청되지 않은 부분 구매이고,
    "강등 실행 금지"(불변식 10)와 같은 규칙이다.
    """
    if "3" not in selected:
        return "측정 3 을 실행 선택에서 제외했다 (`--measurements`) — 고르지 않은 측정이다"
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
    if not settings.anthropic_api_key:
        # 시작해 놓고 첫 판정 호출에서 죽으면 과금만 하고 산출물이 없다 — 선검사가 먼저다.
        return "ANTHROPIC_API_KEY 가 없다 — 판정 모델을 부를 수 없다"
    if "2" in selected and skip is not None:
        return f"측정 2 와 같은 사유로 미실행: {skip}"
    return None


def _run_settings(*, args: argparse.Namespace, settings: Settings) -> Settings:
    """이번 실행의 설정 — 실행 인자가 덮은 값까지 반영한 것.

    **설정 기본값은 바뀌지 않는다.** 여기서 만든 값이 실행 조건 지문에 그대로 실리므로,
    인자로 덮은 조건은 산출물에서 그대로 읽힌다.
    """
    threshold = args.similarity_threshold
    if threshold is None:
        threshold = (
            STUB_SIMILARITY_THRESHOLD if args.stub_llm else settings.vector_similarity_threshold
        )
    return settings.model_copy(
        update={
            "vector_similarity_threshold": float(threshold),
            "vector_top_k": int(args.top_k) if args.top_k is not None else settings.vector_top_k,
            "judge_model": (
                str(args.judge_model) if args.judge_model is not None else settings.judge_model
            ),
        }
    )


#: 중단된 측정 2 의 실행 조건에 붙는 꼬리표. 실행 조건 항목은 "무엇으로 돌렸는가"만이 아니라
#: **결과**도 적는 자리라(미실행이면 "미실행"이다), 중단도 구분되어 남아야 한다.
_ABORTED_MARK: Final = " (측정 2 중단)"


def _client_labels(*, args: argparse.Namespace, settings: Settings) -> tuple[str, str]:
    """(생성 설명, 임베딩 설명) — 클라이언트를 만들지 않고도 정해진다.

    측정 2 가 중단돼도 **무엇으로 돌리던 중이었는지**는 실행 조건에 남아야 한다: 과금된
    실행이 어느 모델에 돈을 썼는지가 산출물에서 사라지면 중단 리포트를 읽을 수 없다.
    """
    if args.stub_llm:
        return (
            "결정론 대역 `evaluation.StubGenerationClient` (실제 모델 아님)",
            f"결정론 대역 `testing.LexicalEmbeddingClient`"
            f"({settings.embedding_dimensions}차원, 어휘 2-gram)",
        )
    return (
        f"OpenAI `{settings.generation_model}` (effort={settings.generation_effort or '기본값'})",
        f"OpenAI `{settings.embedding_model}` ({settings.embedding_dimensions}차원)",
    )


def _clients(
    *, args: argparse.Namespace, settings: Settings
) -> tuple[GenerationClient, EmbeddingClient, str, str]:
    """(생성 클라이언트, 임베딩 클라이언트, 생성 설명, 임베딩 설명)."""
    generation_label, embedding_label = _client_labels(args=args, settings=settings)
    if args.stub_llm:
        return (
            cast(GenerationClient, StubGenerationClient()),
            LexicalEmbeddingClient(dimensions=settings.embedding_dimensions),
            generation_label,
            embedding_label,
        )
    # 실제 실행 — 생성·임베딩 모두 OpenAI. 여기서만 실제 API 키를 쓴다.
    return (
        OpenAIGenerationClient(api_key=settings.openai_api_key, model=settings.generation_model),
        OpenAIEmbeddingClient(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        ),
        generation_label,
        embedding_label,
    )


def _judge_label(
    *, args: argparse.Namespace, settings: Settings, judging_ran: bool, aborted: bool = False
) -> str:
    """실행 조건에 남는 판정 모델 설명 — 대역 수치를 실제 수치로 읽지 않게 하는 기록이다.

    **"판정이 이 실행 어디에서든 돌았는가"를 본다 — 측정 2 의 사유가 아니다.** 판정은 두
    자리에서 돈다: 측정 2 의 파이프라인 안(L1 통과분 판정)과 측정 3(픽스처 단위 정확도).
    측정 2 의 사유만 보면 **측정 3 단독 실측**이 늘 "미실행"으로 적히고, 같은 문서의 측정 3
    절은 "실제 판정 모델(과금)"이라고 적는다 — 리포트가 자기와 모순된다.
    """
    if not settings.l2_enabled:
        return "L2 꺼짐 (판정 미실행)"
    if not judging_ran:
        return "미실행"
    label = (
        "결정론 대역 `testing.StubJudge` (실제 모델 아님)"
        if args.stub_llm
        else f"Anthropic `{settings.judge_model}` (effort={settings.judge_effort or '기본값'})"
    )
    # 측정 2 가 중단된 실행에서는 이 판정자가 **끝까지 돌지 않았다** — 세 항목(생성·임베딩·
    # 판정)이 같은 꼬리표를 달아야 실행 조건만 읽고 완주한 실행으로 오해하지 않는다.
    return f"{label}{_ABORTED_MARK}" if aborted else label


def _retrieval_strategy_label(settings: Settings) -> str:
    """실행 조건에 남는 검색 전략 조합 (scripts/AGENTS.md 불변식 15).

    **설정에서 유도한다** — 손으로 적으면 스위치를 껐는데 리포트가 켜졌다고 말한다.
    하이브리드·리랭크는 실행 경로에 없으므로(미채택) 여기 나타날 이름이 없다.
    """
    stages = ["vector"]
    if settings.query_rewrite_enabled:
        stages.append("rewrite")
    return "+".join(stages)


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


def _reasons(reasons: tuple[RejectReason, ...]) -> str:
    return "없음" if not reasons else "[" + ", ".join(reason.value for reason in reasons) + "]"


def _run_measurement_three(
    *, fixtures: tuple[JudgeFixture, ...], settings: Settings
) -> JudgeAccuracy:
    """측정 3 — 판정 픽스처를 **실제 판정기**에 흘린다(과금).

    파이프라인을 거치지 않고 판정기를 직접 부른다: 재는 것이 판정 층 하나이므로 초안
    생성·근거 수집이 끼면 무엇을 잰 것인지 흐려진다.
    """
    judge: Judging = build_judge(settings)

    def progress(outcome: L2Outcome) -> None:
        actual = outcome.actual_verdict
        if outcome.error is not None or actual is None:
            # 판정이 나오지 않은 건은 여기서 끝난다 — 아래 마크는 판정이 있는 건만 쓴다.
            print(f"  [FAIL] {outcome.fixture_id} 판정 실패: {outcome.error}")
            return
        # 같은 줄이 찍는 것은 verdict 다. 사유만 어긋난 건을 `MISS` 로 찍으면 "기대 reject /
        # 실제 reject" 옆에 실패 표시가 붙어, 리포트가 성공으로 세는 건을 콘솔이 뒤집는다.
        if outcome.reasons_matched:
            mark, detail = "OK  ", ""
        elif outcome.verdict_matched:
            mark = "사유"
            detail = (
                f" — 사유 불일치: 기대 {_reasons(outcome.expected_reasons)}"
                f" / 실제 {_reasons(outcome.actual_reasons)}"
            )
        else:
            mark, detail = "MISS", ""
        print(
            f"  [{mark}] {outcome.fixture_id} 기대 {outcome.expected_verdict.value} "
            f"/ 실제 {actual.value}{detail}"
        )

    return measure_judge_accuracy(fixtures=fixtures, judge=judge, on_outcome=progress)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    selected = selected_measurements(args)
    # 키·DB 선검사는 **측정 시작 전**이다 — 도중에 발견하면 과금만 하고 산출물이 없다.
    skip = _skip_reason(args=args, settings=settings, selected=selected)
    judge_skip = _judge_skip_reason(args=args, settings=settings, skip=skip, selected=selected)
    # 측정 3 이 **돌 조건이었는가** — 판정 선검사(선택 · `--live` · L2 켜짐 · 판정 키)를
    # 통과했는가다. 그 뒤의 중단은 "돌 조건이었는데 못 돌았다"이지 "돌 조건이 아니었다"가
    # 아니므로 여기서 잡아 둔다. **종료 코드 규칙만** 이 값을 본다
    # (`scripts/AGENTS.md` 불변식 12 ①).
    measurement3_was_due = bool(args.live) and judge_skip is None

    fixtures = load_l1_fixtures(args.l1_fixtures)
    cases = load_golden_set(args.golden_set)
    # 판정 픽스처 로드 실패는 **측정 3 만** 접는다 — 키도 DB 도 필요 없는 측정 1 의 산출물을
    # 판정 픽스처 파일이 인질로 잡으면, 무료 실행이 트레이스백으로 죽으면서 아무것도 안 남는다.
    judge_fixtures: tuple[JudgeFixture, ...] = ()
    judge_fixture_error: str | None = None
    try:
        judge_fixtures = load_judge_fixtures(args.judge_fixtures)
    # `KeyboardInterrupt` 는 여기서 잡지 않는다 — 아직 아무것도 과금되지 않았고, 삼키면
    # Ctrl-C 를 누른 실행이 그대로 측정 2(과금)로 넘어간다.
    except Exception as error:
        judge_fixture_error = (
            f"판정 픽스처를 읽지 못했다 (`{display_path(args.judge_fixtures)}`): "
            f"{type(error).__name__}: {error}"
        )
        print(f"경고 — {judge_fixture_error}")
    if judge_fixture_error is not None:
        # 이미 다른 사유로 미실행이어도 로드 실패는 함께 남긴다 — 사유 하나를 다른 사유가
        # 덮으면 리포트만 보고는 픽스처 파일이 깨진 것을 알 수 없다.
        judge_skip = (
            judge_fixture_error if judge_skip is None else f"{judge_skip} / {judge_fixture_error}"
        )

    run_settings = _run_settings(args=args, settings=settings)
    # 측정 2 가 **실측으로 돌 조건이었는가**. 도중에 중단돼도 이 값은 True 로 남는다:
    # 이름은 이미 확정됐고(측정 시작 전 결정), 그 이름이 곧 "이 실행은 과금될 수 있었다"는
    # 저장소에 남는 기록이다.
    measurement2_is_real = bool(args.live) and skip is None
    # 측정 3 이 **실제로 돌 것인가** — 판정 픽스처 로드 실패까지 반영한 값이다. 로드가
    # 실패하면 판정 호출은 **한 번도** 일어나지 않으므로 그 실행은 과금 실행이 아니다.
    measurement3_will_run = bool(args.live) and judge_skip is None
    # **과금 실행 여부**가 리포트 이름의 자격을 판정한다 — "측정 2 실측 여부"는 그 목적의
    # 낡은 대리 변수였다. 둘 중 **하나라도 실제로 살 것**이면 과금 실행이다: 측정 2 를
    # 건너뛰고 판정 픽스처만 사는 측정 3 단독 실측이 라이브 이름을 거부당하면, 그 산출물이
    # gitignore 되는 `evaluation` 스템으로 떨어져 다음 실행에 덮인다
    # (`scripts/AGENTS.md` 불변식 7 — 실제로 한 번 그렇게 잃었다).
    #
    # **"살 조건이었다"가 아니라 "실제로 산다"를 본다.** 판정 픽스처가 깨진 측정 3 단독
    # 실행은 호출을 0회 하고도 라이브 이름을 가져갔고, 그 빈 산출물이 추적 대상이 되어
    # 회귀 가드의 직전 라이브 탐색에서 **머리**가 됐다 — 진짜 경보(`미달`)가 빈 세트와의
    # `대조 불가`로 덮였다. 그래서 알 수 있는 미실행 사유는 이름을 정하기 **전에** 반영한다.
    billed = measurement2_is_real or measurement3_will_run

    # 리포트 이름은 **측정을 시작하기 전에** 확정한다 — 여기서 거부될 실행이 과금(라이브
    # 30건)이나 대역 재적재를 먼저 하고 나서 산출물만 버리는 일이 없어야 한다.
    try:
        stem = resolve_report_stem(
            requested=args.report_stem,
            live_requested=bool(args.live),
            billed=billed,
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
    measurement2_aborted = False
    #: 사용자 중단(Ctrl-C)이 있었는가. 과금 여부와 무관하게 종료 코드로 알린다.
    interrupted = False
    if skip is not None:
        print(f"측정 2 — 미실행: {skip}")
        pipeline = SkippedMeasurement(reason=skip)
        generation_label = "미실행"
        embedding_label = "미실행"
    else:
        print(f"측정 2 — 골든셋 {len(cases)}건 end-to-end")
        try:
            pipeline, generation_label, embedding_label = _run_measurement_two(
                cases=cases, args=args, settings=run_settings
            )
        # `measure_pipeline_agreement` 는 인프라 예외(DB 재기동 등)를 그대로 터뜨리는 것이
        # 설계다 — 그 예외가 여기서 main 밖으로 나가면 **이미 과금된 완주분과 무료인 측정 1
        # 산출물까지** 트레이스백과 함께 사라진다. 측정 3 과 정확히 같은 실패 모양이라 같은
        # 패턴으로 막는다. `BaseException` 까지 넓히는 이유와 재발생시키지 않는 이유는
        # 아래 측정 3 가드의 주석과 같다.
        except BaseException as error:
            aborted = f"측정 2 가 중단됐다: {type(error).__name__}: {error}"
            print(f"측정 2 — 중단: {aborted}")
            pipeline = SkippedMeasurement(reason=aborted)
            measurement2_aborted = True
            interrupted = interrupted or isinstance(error, KeyboardInterrupt)
            attempted_generation, attempted_embedding = _client_labels(
                args=args, settings=run_settings
            )
            generation_label = f"{attempted_generation}{_ABORTED_MARK}"
            embedding_label = f"{attempted_embedding}{_ABORTED_MARK}"
            # 측정 2 가 중단된 실행에서 측정 3 을 **이어서 돌리지 않는다**: 중단 사유가
            # Ctrl-C 면 이어 도는 것이 곧 추가 과금이고, 인프라 고장이면 골든셋 수치 없이
            # 판정 수치만 남은 리포트가 된다(두 측정의 실측 여부가 갈리면 안 된다).
            if judge_skip is None:
                judge_skip = f"측정 2 와 같은 사유로 미실행: {aborted}"

    judge_accuracy: JudgeAccuracy | SkippedMeasurement
    measurement3_is_real = False
    if judge_skip is not None:
        print(f"측정 3 — 미실행: {judge_skip}")
        judge_accuracy = SkippedMeasurement(reason=judge_skip)
    else:
        print(f"측정 3 — 판정 픽스처 {len(judge_fixtures)}건 (실제 판정 모델·과금)")
        try:
            judge_accuracy = _run_measurement_three(fixtures=judge_fixtures, settings=run_settings)
        # `BaseException` 까지 잡는다: `except (Exception, KeyboardInterrupt)` 는
        # `SystemExit`(BaseException 직계)을 놓쳐, 판정 SDK 나 그 의존이 `sys.exit()` 를
        # 부르면 리포트 없이 프로세스가 끝난다. **재발생시키지 않는다** — 여기까지 왔으면
        # 측정 2 는 이미 과금이 끝났고, 그 산출물을 남기는 것이 종료 신호를 그대로
        # 전달하는 것보다 우선이다(중단 사실은 미실행 사유와 종료 코드 1 이 들고 간다).
        except BaseException as error:
            aborted = f"측정 3 이 중단됐다: {type(error).__name__}: {error}"
            print(f"측정 3 — 중단: {aborted}")
            judge_accuracy = SkippedMeasurement(reason=aborted)
            interrupted = interrupted or isinstance(error, KeyboardInterrupt)
        else:
            measurement3_is_real = True

    # ── 종료 코드는 여기 한 곳에서만 정해진다 (규칙이 갈리지 않게) ──────────────
    # 1 이 되는 경우는 둘뿐이다.
    #   ① **측정 3 이 돌 조건이었는데 미실행** — 즉 `--live` 로 판정 선검사를 통과한
    #      실행(선택 포함 · L2 켜짐 · 판정 키 있음)인데 판정 수치가 안 나온 경우. 사유가
    #      무엇이든(픽스처 로드 실패·중단·측정 2 중단으로 인한 연쇄 미실행) 같다 — 돈을
    #      쓰고 판정 수치는 못 낸 실행이 종료 코드로는 성공으로 읽히면 안 되기 때문이다.
    #   ② **사용자 중단(Ctrl-C)** — 과금 여부와 무관하다. 산출물은 남기되(과금분 보존),
    #      중단된 실행이 성공으로 읽히면 안 된다.
    # 반대로 **L2 꺼짐 기준선에서 측정 3 이 안 도는 것은 설계상 정상이라 0** 이다.
    # `--live` 없이 도는 평범한 실행에서 "`--live` 아님" 사유로 미실행인 것도,
    # `--measurements 2` 로 **측정 3 을 고르지 않은** 실행도 마찬가지다.
    measurement3_missing = measurement3_was_due and isinstance(judge_accuracy, SkippedMeasurement)
    exit_code = 1 if interrupted or measurement3_missing else 0

    conditions = RunConditions(
        started_at=started_at,
        generation=generation_label,
        embedding=embedding_label,
        judge=_judge_label(
            args=args,
            settings=settings,
            # 생성·임베딩 설명만 측정 2 의 사유를 따른다. 판정은 측정 2 의 파이프라인 안에서도
            # 돌고 측정 3 에서도 도니, 둘 중 하나라도 돌았으면 무엇으로 돌았는지 적는다.
            judging_ran=skip is None or measurement3_will_run,
            aborted=measurement2_aborted,
        ),
        embedding_dimensions=run_settings.embedding_dimensions,
        retrieval_strategy=_retrieval_strategy_label(run_settings),
        similarity_threshold=run_settings.vector_similarity_threshold,
        top_k=run_settings.vector_top_k,
        l1_fixture_count=len(fixtures),
        golden_case_count=len(cases),
        # 로드하지 못했으면 0 이 아니라 **미실행**이다.
        judge_fixture_count=None if judge_fixture_error is not None else len(judge_fixtures),
        # 커밋되는 라이브 리포트에 로컬 절대 경로(사용자명 포함)를 남기지 않는다.
        l1_fixtures_path=display_path(args.l1_fixtures),
        golden_set_path=display_path(args.golden_set),
        judge_fixtures_path=display_path(args.judge_fixtures),
        api_key_present=bool(settings.openai_api_key),
        judge_api_key_present=bool(settings.anthropic_api_key),
        l2_enabled=settings.l2_enabled,
        measurement2_is_real=measurement2_is_real,
        # 진입점에서 측정 3 이 도는 조건은 `--live` + L2 켜짐 + 키뿐이다 — 완주했으면
        # 실제 판정 모델로 낸 값이다(대역 판정은 `--stub-llm` 이고 그때는 돌지 않는다).
        measurement3_is_real=measurement3_is_real,
        billed=billed,
        measurement_scope=_measurement_scope(selected=selected, skip=skip, judge_skip=judge_skip),
        condition_fingerprint=_condition_fingerprint(
            args=args, settings=settings, run_settings=run_settings
        ),
        # 선언된 실험 변인은 **실행이 명시**한다 — 코드가 추측하지 않는다. 선언되지 않은
        # 지문 불일치는 "대조 불가"로 남고, 그것이 조용한 조건 드리프트를 잡는 자리다.
        declared_experiment_fields=tuple(args.declare_experiment or ()),
    )
    report: EvaluationReport = build_report(
        conditions=conditions,
        gate_accuracy=accuracy,
        pipeline=pipeline,
        judge_accuracy=judge_accuracy,
    )
    report = attach_regression_guard(report, stem=stem, reports_dir=args.out_dir)
    markdown_path, json_path = write_report(report, out_dir=args.out_dir, stem=stem)

    _print_summary(report)
    _print_targets(report)
    _print_regression_guard(report)
    print(f"\n리포트: {markdown_path}\n리포트(JSON): {json_path}")
    return exit_code


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
    if report.conditions.measurement3_is_real:
        print("  ※ 측정 3 은 확률 층이고 목표치는 미확정이다 — 달성·미달 판정이 붙지 않는다.")
    else:
        print("  ※ 위 측정 3 수치는 대역으로 낸 값이다 — 판정 모델의 정확도가 아니다.")


def _measurement_scope(
    *, selected: frozenset[str], skip: str | None, judge_skip: str | None
) -> str:
    """측정 범위 — 풀셋인지 부분인지. 조건 지문의 한 항목이라 대조 가능성을 가른다.

    **고른 것이 아니라 실제로 돌 것**을 적는다 — 골랐지만 선검사에 걸려 못 도는 측정을
    범위에 넣으면 지문이 실행을 잘못 설명한다.
    """
    del selected  # 실제 실행 여부가 이미 두 사유에 반영되어 있다.
    if skip is None and judge_skip is None:
        return "full"
    if skip is not None and judge_skip is None:
        return "measurement_1_3_only"
    if skip is None:
        return "measurement_1_2_only"
    return "measurement_1_only"


def _condition_fingerprint(
    *, args: argparse.Namespace, settings: Settings, run_settings: Settings
) -> dict[str, str]:
    """대조 가능성을 결정하는 지문 값들.

    **새 축을 붙이는 작업은 여기 한 줄을 더하면 된다** — 지문 스키마는 열린 맵이라
    `evaluation.py` 도 `regression_guard.py` 도 다시 열지 않는다. 기권 게이트가 그 예이고,
    지금은 설정값을 그대로 싣는다.

    프롬프트·픽스처·라벨 버전은 **손으로 적지 않고 내용에서 끌어낸다** — 손으로 적는
    버전 문자열은 갱신을 잊는 순간 조용한 드리프트가 된다. 같은 이유로 기권 게이트도
    상수를 다시 적지 않고 `Settings.abstention_gate()` 가 조립한 것을 읽는다 — 설정과
    지문이 갈리면 바뀐 축이 가드에 보이지 않는다.
    """
    gate = run_settings.abstention_gate()
    return {
        # 라벨 버전 = 골든셋 내용 지문. 결정 0008 의 라벨 재정렬 전후가 여기서 갈린다.
        "label_version": content_digest(args.golden_set, prefix="golden-") or "미상",
        # **검색 정답 라벨은 근거 부분 손실 검사의 정답 입력이다.** 여기 없으면 라벨 한 줄
        # 추가가 지문에는 아무 차이도 남기지 않은 채 "근거가 사라졌다"로 읽힌다.
        "retrieval_labels_version": (
            content_digest(DEFAULT_RETRIEVAL_LABELS_PATH, prefix="labels-") or "미상"
        ),
        "acceptance_cut": f"{run_settings.vector_similarity_threshold:g}",
        # 기권 게이트. **끈 실행은 τ=0 이 아니라 "꺼짐"이다** — 0 으로 적으면 "모든 질의를
        # 통과시킨 게이트"와 구분되지 않고, 그 둘은 다른 조건이다.
        # τ 는 `embedding_model` 과 짝으로 읽힌다(`regression_guard.PAIRED_FINGERPRINT_FIELDS`).
        "abstention_gate_statistic": gate.statistic.value if gate is not None else "꺼짐",
        "abstention_tau": f"{gate.tau:g}" if gate is not None else "꺼짐",
        "query_rewrite": "on" if run_settings.query_rewrite_enabled else "off",
        # 대역 실행은 대역이라고 적는다 — 설정값 모델명을 적으면 산출물이 거짓 신고한다.
        "embedding_model": "결정론 대역" if args.stub_llm else run_settings.embedding_model,
        "embedding_dimensions": str(run_settings.embedding_dimensions),
        "top_k": str(run_settings.vector_top_k),
        "generation_model": "결정론 대역" if args.stub_llm else run_settings.generation_model,
        "generation_effort": run_settings.generation_effort or "기본값",
        "judge_model": run_settings.judge_model,
        "judge_effort": run_settings.judge_effort or "기본값",
        "judge_prompt_version": text_digest(JUDGE_SYSTEM_PROMPT, prefix="judge-"),
        "judge_fixture_version": content_digest(args.judge_fixtures, prefix="fixture-") or "미상",
        # 판정 프롬프트 캐싱은 아직 배선이 없다. 배선하면 설정에서 읽도록 바꾼다.
        "judge_prompt_caching": "off",
        "l2_enabled": "on" if settings.l2_enabled else "off",
    }


def _print_regression_guard(report: EvaluationReport) -> None:
    """회귀 판정을 콘솔에도 찍는다 — 리포트 파일을 열어야만 미달을 아는 일이 없게."""
    guard = report.regression_guard
    if isinstance(guard, GuardUnavailable):
        print(f"회귀 가드 — 미산출 (사유: {guard.reason})")
        return
    if not isinstance(guard, RegressionGuard):  # pragma: no cover - 방어
        return
    print(
        f"회귀 가드 — 판정 [{guard.verdict}] (승격 기준선 구속) "
        f"/ 직전 라이브 [{guard.alert.verdict}] (경보)"
    )
    print(f"  {guard.binding.verdict_reason}")
    for finding in (*guard.binding.match_shortfalls, *guard.binding.match_collapses):
        print(f"  - 일치 미달: {finding.describe()}")
    for loss in guard.binding.evidence_losses:
        print(f"  - 근거 부분 손실: {loss.describe()}")


def _print_targets(report: EvaluationReport) -> None:
    """목표치 판정을 콘솔에도 찍는다 — 회귀(미달)를 리포트 파일을 열어야만 아는 일이 없게."""
    print("목표치 판정:")
    for item in assess_targets(report):
        value = "미측정" if item.value is None else f"{item.value * 100:.1f}%"
        print(f"  [{item.verdict}] {item.target.label} {item.target.describe()} — 실측 {value}")


if __name__ == "__main__":
    raise SystemExit(main())
