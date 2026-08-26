"""채택 축의 결정론 — 미정의 사유별 처분과 코사인 동점의 tie-break.

두 구멍이 같은 축(근거 채택)에 있었고 둘 다 **관측 가능한 오동작이 아니어서** 오래 살아
남았다. 그래서 이 파일은 "지금 값이 틀렸다"가 아니라 **"보장이 없다"** 를 검사로 바꾼다.

1. **통계량이 정의되지 않으면 게이트가 열렸다.** 미정의 사유는 둘인데(측정된 점수 2건 미만 ·
   1위 코사인 0 이하) 한 사유로 접혀 둘 다 "발동하지 않음"이 됐다. **처분은 반대여야 한다** —
   1위 코사인이 0 이하면 기권하고, 측정 점수가 2건 미만이면 기권하지 않는다. 뒤엣것은 결함이
   아니라 계약이다: 미정의를 0 으로 채우면 모든 양수 τ 에서 기권이 되어 후보가 하나뿐인
   질의가 근거 없이 인계된다.
2. **사유가 판정 값 안에만 있었다.** 원시연산은 사유를 담아 돌려주는데 채택 함수가
   "기권했나" 한 칸만 읽고 나머지를 버렸다 — 그래서 사유가 근거 묶음에도 리포트에도 없었다.
   **"사유가 판정 값에 실린다"는 이미 참이라 검증 항목이 못 된다.** 여기서 구속력을 갖는
   것은 **근거 묶음이 사유를 들고 나오는가**이다.
3. **검색 결과 순서에 결정론 tie-break 가 없었다.** 파이썬 합집합 병합은 이미 유사도 →
   근거 ID 로 정렬하므로 **파이썬 층 동점 검사는 수정 전에도 통과한다.** 남은 비결정은
   DB 가 상위 `top_k` 를 자를 때 어느 행이 살아남는가 하나라, 구속력 있는 검사는 실제 DB 에
   동점을 심는 `db` 마커 검사다. 동점 행은 **정본 순서(근거 ID 오름차순)와 반대로** 심는다 —
   근거 ID 순으로 심으면 tie-break 가 없는 코드도 우연히 정본 순서를 돌려줘 음성 대조가
   초록으로 태어난다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import psycopg
import pytest
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import DictRow

from reply_gate.adoption_axis import AbstentionStatistic as HandCalcStatistic
from reply_gate.adoption_axis import statistic_value as handcalc_statistic_value
from reply_gate.config import Settings
from reply_gate.evidence import EvidenceCollector, adopt_policy_hits
from reply_gate.llm import GenerationClient, JsonCompletion
from reply_gate.policy_index import (
    DEFAULT_POLICY_DIR,
    PolicySearchHit,
    index_policy_documents,
    load_policy_documents,
    search_policy_chunks,
)
from reply_gate.retrieval_strategies import (
    AbstentionGate,
    AbstentionStatistic,
    AbstentionUndefined,
    abstention_statistic,
    apply_abstention_gate,
    truncate_for_gate,
    undefined_statistic_reason,
)
from reply_gate.testing import LexicalEmbeddingClient

_TOP_K = 5
_CUT = 0.30
#: 결정 0014 채택 구성.
_TAU = 0.06
#: 정책 인덱스의 커밋된 출처. 심는 동점 행도 같은 출처여야 검색이 거부하지 않는다.
_INDEX_MODEL = "text-embedding-3-small"
_INDEX_DIMENSIONS = 1536


def _gate(*, tau: float = _TAU) -> AbstentionGate:
    return AbstentionGate(statistic=AbstentionStatistic.SPREAD, tau=tau)


def _hit(evidence_id: str, similarity: float) -> PolicySearchHit:
    return PolicySearchHit(
        evidence_id=evidence_id,
        document_slug="refund",
        document_title="환불 정책",
        article=evidence_id.rsplit(":", 1)[-1],
        article_title="조항",
        content="본문",
        similarity=similarity,
    )


def _ranked(scores: Sequence[float]) -> tuple[PolicySearchHit, ...]:
    """유사도 내림차순 순위 — DB 검색 결과의 모양이다."""
    return tuple(_hit(f"policy:x:{index}", score) for index, score in enumerate(scores, start=1))


# ---------------------------------------------------------------------------
# 계약 A — 미정의 사유는 둘이고 처분이 반대다. 하나로 접지 않는다.
# ---------------------------------------------------------------------------


def test_미정의_사유는_둘이고_점수열만_보고_갈린다() -> None:
    assert undefined_statistic_reason((0.9,)) is AbstentionUndefined.INSUFFICIENT_SCORES
    assert undefined_statistic_reason(()) is AbstentionUndefined.INSUFFICIENT_SCORES
    assert undefined_statistic_reason((0.0, -0.1)) is AbstentionUndefined.NONPOSITIVE_RANK1
    assert undefined_statistic_reason((-0.2, -0.3)) is AbstentionUndefined.NONPOSITIVE_RANK1
    # 양성 대조 — 정의되는 점수열은 사유를 남기지 않는다.
    assert undefined_statistic_reason((0.5, 0.4)) is None


def test_1위_코사인이_0_이하면_게이트가_기권한다() -> None:
    """모든 후보가 절대 하한 아래라 채택 집합은 어차피 비지만, **판정이 그것을 말해야 한다.**

    "결과가 같으니 상관없다"로 두면 통계량을 신뢰할 수 없었다는 사실이 어디에도 남지 않고,
    컷이 낮아지는 순간 조용히 오동작으로 바뀐다.
    """
    verdict = apply_abstention_gate(_gate(), (0.0, -0.1, -0.2))

    assert verdict.abstains is True
    assert verdict.value is None
    assert verdict.undefined_reason is AbstentionUndefined.NONPOSITIVE_RANK1


def test_측정된_점수가_두_건_미만이면_게이트가_기권하지_않는다() -> None:
    """이쪽은 결함이 아니라 계약이다 — 뒤집으면 후보가 하나뿐인 질의가 근거 없이 인계된다."""
    verdict = apply_abstention_gate(_gate(), (0.9,))

    assert verdict.abstains is False
    assert verdict.value is None
    assert verdict.undefined_reason is AbstentionUndefined.INSUFFICIENT_SCORES


def test_두_사유의_처분이_서로_반대다() -> None:
    """한 사유로 접히면 이 단언이 깨진다 — 접힘이 정확히 원래 결함의 모양이었다."""
    dispositions = {reason: reason.abstains for reason in AbstentionUndefined}

    assert dispositions == {
        AbstentionUndefined.INSUFFICIENT_SCORES: False,
        AbstentionUndefined.NONPOSITIVE_RANK1: True,
    }
    assert len(AbstentionUndefined) == 2


def test_통계량이_정의되는_점수열의_판정은_그대로다() -> None:
    """τ 축 무변경 — 미정의 처리를 갈라도 정의되는 구간의 판정은 손대지 않는다."""
    gate = _gate()
    below = apply_abstention_gate(gate, (0.51, 0.50, 0.49, 0.48, 0.47))
    above = apply_abstention_gate(gate, (0.62, 0.55, 0.53, 0.52, 0.51))
    exact = apply_abstention_gate(
        AbstentionGate(statistic=AbstentionStatistic.SPREAD, tau=0.25),
        (0.5, 0.4, 0.35, 0.3, 0.25),
    )

    assert (below.abstains, below.undefined_reason) == (True, None)
    assert (above.abstains, above.undefined_reason) == (False, None)
    # τ 와 같은 값은 기권시키지 않는다 — 등호 방향도 그대로다.
    assert (exact.value, exact.abstains) == (0.25, False)


# ---------------------------------------------------------------------------
# 계약 B — 근거 묶음이 사유를 들고 나온다. (판정 값에 실리는 것은 이미 참이었다)
# ---------------------------------------------------------------------------


def test_채택_함수의_결과_묶음이_기권_사유를_들고_나온다() -> None:
    """원시연산 → 채택 함수 사이에서 사유가 버려지던 자리다.

    예전 반환값은 후보 튜플 하나였고, 채택 함수가 판정에서 `abstains` 한 칸만 읽었다.
    그래서 사유는 **판정 값 안에서만** 살아 있었고 근거 묶음에도 리포트에도 없었다.
    """
    insufficient = adopt_policy_hits(
        candidates=_ranked((0.9,)), top_k=_TOP_K, similarity_threshold=_CUT, gate=_gate()
    )
    nonpositive = adopt_policy_hits(
        candidates=_ranked((0.0, -0.1)), top_k=_TOP_K, similarity_threshold=_CUT, gate=_gate()
    )

    assert insufficient.undefined_reason is AbstentionUndefined.INSUFFICIENT_SCORES
    assert insufficient.abstained is False
    # 게이트가 열린 채로 남았으므로 항목 축이 뒤에서 자른다.
    assert [hit.evidence_id for hit in insufficient.hits] == ["policy:x:1"]

    assert nonpositive.undefined_reason is AbstentionUndefined.NONPOSITIVE_RANK1
    assert nonpositive.abstained is True
    assert nonpositive.hits == ()


def test_통계량이_정의된_채택은_사유_없이_돌아온다() -> None:
    """양성 대조 — 사유 칸이 늘 차 있으면 그 칸은 아무것도 구분하지 못한다."""
    adoption = adopt_policy_hits(
        candidates=_ranked((0.62, 0.55, 0.53, 0.52, 0.51)),
        top_k=_TOP_K,
        similarity_threshold=_CUT,
        gate=_gate(),
    )

    assert adoption.undefined_reason is None
    assert adoption.abstained is False
    assert len(adoption.hits) == 5


def test_게이트가_돌지_않은_채택은_판정_자체가_없다() -> None:
    """게이트가 돌지 않은 것과 돌았는데 정의된 것은 다른 상태다 — 둘 다 사유는 `None` 이다."""
    gate_off = adopt_policy_hits(
        candidates=_ranked((0.62, 0.55)), top_k=_TOP_K, similarity_threshold=_CUT, gate=None
    )
    no_candidates = adopt_policy_hits(
        candidates=(), top_k=_TOP_K, similarity_threshold=_CUT, gate=_gate()
    )

    assert gate_off.verdict is None
    assert gate_off.undefined_reason is None
    assert len(gate_off.hits) == 2
    assert no_candidates.verdict is None
    assert no_candidates.hits == ()


class _PolicyIntentClient:
    """의도 해석만 대답하는 생성 대역 — 정책 경로 하나만 열면 되는 테스트용이다."""

    def complete_json(self, **kwargs: Any) -> JsonCompletion:
        assert kwargs["stage"] == "intent", f"이 대역이 모르는 단계다: {kwargs['stage']!r}"
        return JsonCompletion(data={"source": "policy"}, input_tokens=10, output_tokens=2)


def _collector(*, top_k: int) -> EvidenceCollector:
    """재작성은 끈다 — 이 테스트가 재는 것은 채택 축이고, 재작성 대본은 잡음이다."""
    return EvidenceCollector(
        generation_client=cast(GenerationClient, _PolicyIntentClient()),
        embedding_client=LexicalEmbeddingClient(dimensions=_INDEX_DIMENSIONS),
        settings=Settings(
            vector_top_k=top_k,
            vector_similarity_threshold=0.0,
            query_rewrite_enabled=False,
            abstention_gate_enabled=True,
        ),
    )


@pytest.fixture
def indexed_policies(app_conn: psycopg.Connection[DictRow]) -> None:
    """저장소의 정책 문서를 결정론 임베딩으로 적재한다 (픽스처 롤백으로 되돌아간다)."""
    index_policy_documents(
        conn=app_conn,
        documents=load_policy_documents(DEFAULT_POLICY_DIR),
        embedder=LexicalEmbeddingClient(dimensions=_INDEX_DIMENSIONS),
    )


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_근거_묶음이_기권_미정의_사유를_들고_나온다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """뒤 태스크가 리포트로 이을 표면이다 — 여기서 끊기면 사유가 산출물에 영영 안 실린다.

    `top_k=1` 이면 게이트가 볼 슬라이스가 1건이라 통계량이 정의되지 않는다(사유 ①).
    게이트는 열린 채로 남고, **그 사실이 근거 묶음에 남는다.**
    """
    result = _collector(top_k=1).collect(
        inquiry_id="00000000-0000-4000-8000-000000000001",
        content="환불 신청 기간이 어떻게 되나요",
        order_no=None,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.abstention_undefined_reason is AbstentionUndefined.INSUFFICIENT_SCORES
    # 게이트는 열린 채로 남았다 — 근거가 통째로 비지 않는다.
    assert result.evidence
    assert result.escalation_reason is None


@pytest.mark.db
@pytest.mark.usefixtures("indexed_policies")
def test_통계량이_정의되면_근거_묶음에_사유가_없다(
    app_conn: psycopg.Connection[DictRow], ro_conn: psycopg.Connection[DictRow]
) -> None:
    """양성 대조 — 제품 기본 `top_k` 에서는 사유 칸이 비어 있어야 한다."""
    result = _collector(top_k=_TOP_K).collect(
        inquiry_id="00000000-0000-4000-8000-000000000002",
        content="환불 신청 기간이 어떻게 되나요",
        order_no=None,
        app_conn=app_conn,
        readonly_conn=ro_conn,
    )

    assert result.abstention_undefined_reason is None
    assert result.evidence


# ---------------------------------------------------------------------------
# 계약 C — 코사인 동점이 절단선에 걸려도 살아남는 행이 같다(DB 마커).
# ---------------------------------------------------------------------------

#: 정본 순서(근거 ID 오름차순)의 **역순**으로 심는다. 갓 삽입된 행은 물리 순서가 삽입
#: 순서라, 근거 ID 순으로 심으면 tie-break 가 없는 코드도 우연히 정본 순서를 돌려준다.
_TIE_ARTICLES = ("9-3", "9-2", "9-1")
_TIE_IDS = tuple(f"policy:tiebreak:{article}" for article in _TIE_ARTICLES)
_CANONICAL = tuple(sorted(_TIE_IDS))


def _tie_vector() -> list[float]:
    """어떤 실제 조항과도 코사인이 같아질 수 없는 단위 벡터."""
    vector = [0.0] * _INDEX_DIMENSIONS
    vector[0] = 1.0
    return vector


def _plant_tied_chunks(conn: psycopg.Connection[DictRow]) -> list[float]:
    """같은 임베딩을 가진 조항 셋을 정본 순서의 역순으로 심는다 (롤백으로 되돌아간다)."""
    register_vector(conn)
    vector = _tie_vector()
    with conn.cursor() as cur:
        for article in _TIE_ARTICLES:
            cur.execute(
                """
                INSERT INTO policy_chunks
                    (evidence_id, document_slug, document_title, article, article_title,
                     content, embedding, embedding_model, embedding_dimensions)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    f"policy:tiebreak:{article}",
                    "tiebreak",
                    "동점 대조 정책",
                    article,
                    "동점 조항",
                    "동점 대조용 본문",
                    Vector(vector),
                    _INDEX_MODEL,
                    _INDEX_DIMENSIONS,
                ),
            )
    return vector


@pytest.mark.db
def test_코사인_동점이_절단선에_걸려도_살아남는_행이_정본_순서다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """26조항 정확 스캔이라 실무상 안정적이지만 **보장이 아니었다.**

    동점이 `top_k` 절단선에 걸리면 어느 행이 살아남는지가 DB 몫이고, 그 순간 채택 집합과
    기권 게이트 통계량이 함께 흔들린다.
    """
    vector = _plant_tied_chunks(app_conn)

    at_cut = search_policy_chunks(
        conn=app_conn,
        query_vector=vector,
        top_k=2,
        embedding_model=_INDEX_MODEL,
        embedding_dimensions=_INDEX_DIMENSIONS,
    )
    whole_group = search_policy_chunks(
        conn=app_conn,
        query_vector=vector,
        top_k=3,
        embedding_model=_INDEX_MODEL,
        embedding_dimensions=_INDEX_DIMENSIONS,
    )

    # 양성 대조 — 심은 셋이 실제로 동점이고 실제 조항들을 앞질렀다.
    assert {hit.similarity for hit in whole_group} == {1.0}
    assert tuple(hit.evidence_id for hit in whole_group) == _CANONICAL
    # 절단선이 동점 한가운데를 지나도 살아남는 둘이 정해져 있다.
    assert tuple(hit.evidence_id for hit in at_cut) == _CANONICAL[:2]


@pytest.mark.db
def test_tie_break_없는_정렬은_정본_순서를_돌려주지_않는다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """음성 대조 — 정렬에서 근거 ID 를 빼면 위 검사가 깨진다는 것을 같은 파일이 증명한다.

    이 검사가 없으면 "동점을 심었지만 DB 가 알아서 정본 순서를 준다"와 구분되지 않는다.
    """
    vector = _plant_tied_chunks(app_conn)

    with app_conn.cursor() as cur:
        cur.execute(
            """
            SELECT evidence_id
            FROM policy_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s
            LIMIT 2
            """,
            (Vector(vector),),
        )
        raw = tuple(str(row["evidence_id"]) for row in cur.fetchall())

    assert set(raw) <= set(_TIE_IDS), "동점 셋이 상위를 차지해야 대조가 성립한다"
    assert raw != _CANONICAL[:2], "tie-break 없는 정렬이 정본 순서를 돌려주면 이 대조가 무의미하다"


# ---------------------------------------------------------------------------
# 계약 D — 손계산 쌍둥이와의 갈림은 **값이 아니라 처분**에서만 받아들인다.
# ---------------------------------------------------------------------------


def test_손계산_쌍둥이는_값은_같고_미정의_사유는_가르지_않는다() -> None:
    """받아들인 갈림을 검사로 못박는다(결정 0022).

    손계산 모듈은 커밋된 산출물을 읽는 **채점자**라 게이트 판정을 내리지 않는다 — 미정의
    입력은 처분이 아니라 즉시 실패다. 그리고 두 구현이 독립이어야 교차 대조가 의미를
    가지므로 여기서 하나로 합치지 않는다. 값이 갈리면 그때는 배선이 어긋난 것이다.
    """
    scores = (0.62, 0.55, 0.53, 0.52, 0.51)

    assert abstention_statistic(AbstentionStatistic.SPREAD, scores) == pytest.approx(
        handcalc_statistic_value(HandCalcStatistic.SPREAD, scores), abs=1e-12
    )
    # 런타임은 사유로 가르고, 손계산은 두 사유 모두 그냥 죽는다.
    for undefined in ((0.9,), (0.0, -0.1)):
        assert undefined_statistic_reason(undefined) is not None
        with pytest.raises(ValueError):
            handcalc_statistic_value(HandCalcStatistic.SPREAD, undefined)


def test_게이트는_여전히_점수열만_받는다() -> None:
    """사유를 갈랐다고 게이트가 케이스 정체성이나 정답을 보게 되지 않는다."""
    import inspect

    assert set(inspect.signature(undefined_statistic_reason).parameters) == {"scores"}
    assert set(inspect.signature(apply_abstention_gate).parameters) == {"gate", "scores"}
    assert set(inspect.signature(truncate_for_gate).parameters) == {"similarities", "top_k"}
