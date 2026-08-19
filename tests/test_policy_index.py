"""정책 문서 파싱·적재·검색 테스트.

적재/검색은 DB 가 필요하지만 **임베딩 API 키는 필요 없다** — 결정론 임베딩 대역으로
배관(청킹 → 적재 → 유사도 정렬 → 임계값 필터)을 끝까지 돌린다. 임베딩 품질 자체는
실제 키가 있어야 확인할 수 있고 그건 골든셋 평가의 몫이다.
"""

from __future__ import annotations

from typing import cast

import psycopg
import pytest
from psycopg.rows import DictRow

from reply_gate.evidence import adopt_policy_hits
from reply_gate.policy_index import (
    DEFAULT_POLICY_DIR,
    PlantedKind,
    PolicyIndexProvenanceError,
    index_policy_documents,
    load_policy_documents,
    parse_policy_document,
    search_policy_chunks,
)
from reply_gate.testing import LexicalEmbeddingClient


def _count(cur: psycopg.Cursor[DictRow], sql: str) -> int:
    """단일 카운트 조회. fetchone() 이 None 이면 그 자체가 실패다."""
    cur.execute(sql)
    row = cur.fetchone()
    assert row is not None
    return int(row["n"])


SAMPLE = """\
---
slug: sample
title: 샘플 정책
---

<!-- planted: none -->
## 9-1 첫 조항

첫 조항 본문이다.

<!-- planted: decoy; note: 전화번호를 뺐다 -->
## 9-2 둘째 조항

둘째 조항 본문이다.
계속 이어지는 줄.
"""


# ── 파싱 (DB 불필요) ────────────────────────────────────────────────────────


def test_조항_단위로_쪼개고_근거_ID_를_붙인다() -> None:
    document = parse_policy_document(SAMPLE)

    assert document.slug == "sample"
    assert [chunk.article for chunk in document.chunks] == ["9-1", "9-2"]
    assert document.chunks[0].evidence_id == "policy:sample:9-1"
    assert document.chunks[1].content == "둘째 조항 본문이다.\n계속 이어지는 줄."


def test_심은_장치를_메타데이터로_읽는다() -> None:
    document = parse_policy_document(SAMPLE)

    assert document.chunks[0].planted is PlantedKind.NONE
    assert document.chunks[1].planted is PlantedKind.DECOY
    assert document.chunks[1].planted_note == "전화번호를 뺐다"


def test_임베딩_텍스트에_문서와_조항_제목이_들어간다() -> None:
    chunk = parse_policy_document(SAMPLE).chunks[0]

    assert "샘플 정책" in chunk.embedding_text
    assert "첫 조항" in chunk.embedding_text


@pytest.mark.parametrize(
    "bad",
    [
        "## 1-1 제목\n본문\n",  # front matter 없음
        "---\nslug: x\n---\n\n## 1-1 제목\n본문\n",  # title 없음
        "---\nslug: x\ntitle: y\n---\n\n본문만 있다\n",  # 조항 없음
        "---\nslug: x\ntitle: y\n---\n\n## 1-1 제목\n\n## 1-2 제목\n본문\n",  # 빈 조항
    ],
)
def test_잘못된_문서는_거부한다(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_policy_document(bad)


# ── 저장소의 실제 정책 문서 (DB 불필요) ─────────────────────────────────────


def test_저장소_정책_문서에_기각_유발_장치가_살아_있다() -> None:
    """미끼·모호·상충 조항은 데모와 지표의 신뢰성 장치다 — 정리 명목으로 사라지면 안 된다."""
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    chunks = [chunk for document in documents for chunk in document.chunks]
    planted = [chunk.planted for chunk in chunks]

    assert len(chunks) >= 20
    assert planted.count(PlantedKind.DECOY) >= 2, "미끼 조항이 없으면 기각 장면이 재현되지 않는다"
    assert planted.count(PlantedKind.AMBIGUOUS) >= 1
    assert planted.count(PlantedKind.CONFLICTING) >= 2, "상충은 최소 2개 조항이 짝을 이룬다"


def test_미끼_조항에는_패턴형_값이_실제로_없다() -> None:
    """미끼가 값을 담고 있으면 모델이 지어낼 이유가 없어져 장치가 무력해진다."""
    import re

    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    decoys = [
        chunk
        for document in documents
        for chunk in document.chunks
        if chunk.planted is PlantedKind.DECOY
    ]
    phone_or_email = re.compile(
        r"(?<![0-9])(1[5-8][0-9]{2}|0[1-9][0-9]?)[-. ]?[0-9]{3,4}[-. ]?[0-9]{4}(?![0-9])|@"
    )

    for chunk in decoys:
        assert not phone_or_email.search(chunk.content), f"{chunk.evidence_id} 에 패턴형 값이 있다"


def test_근거_ID_는_문서_전체에서_유일하다() -> None:
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    ids = [chunk.evidence_id for document in documents for chunk in document.chunks]

    assert len(ids) == len(set(ids))
    assert all(evidence_id.startswith("policy:") for evidence_id in ids)


# ── 적재·검색 (DB 필요, 임베딩 키 불필요) ────────────────────────────────────


@pytest.mark.db
def test_적재_후_조항이_전부_들어가고_재실행해도_중복되지_않는다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    embedder = LexicalEmbeddingClient(dimensions=1536)
    expected = sum(len(document.chunks) for document in documents)

    first = index_policy_documents(conn=app_conn, documents=documents, embedder=embedder)
    second = index_policy_documents(conn=app_conn, documents=documents, embedder=embedder)

    assert first.upserted == expected
    assert second.upserted == expected
    assert second.deleted == 0
    with app_conn.cursor() as cur:
        assert _count(cur, "SELECT count(*) AS n FROM policy_chunks") == expected
        assert _count(cur, "SELECT count(*) AS n FROM policy_chunks WHERE embedding IS NULL") == 0


@pytest.mark.db
def test_문서에서_사라진_조항은_삭제된다(app_conn: psycopg.Connection[DictRow]) -> None:
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    embedder = LexicalEmbeddingClient(dimensions=1536)
    index_policy_documents(conn=app_conn, documents=documents, embedder=embedder)

    trimmed = index_policy_documents(conn=app_conn, documents=documents[:1], embedder=embedder)

    assert trimmed.deleted > 0
    with app_conn.cursor() as cur:
        assert _count(cur, "SELECT count(*) AS n FROM policy_chunks") == len(documents[0].chunks)


@pytest.mark.db
def test_유사도_검색이_관련_조항을_돌려준다(app_conn: psycopg.Connection[DictRow]) -> None:
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    embedder = LexicalEmbeddingClient(dimensions=1536)
    index_policy_documents(conn=app_conn, documents=documents, embedder=embedder)

    query = embedder.embed(stage="inquiry", texts=["환불 신청 기간이 어떻게 되나요"]).vectors[0]
    hits = search_policy_chunks(
        conn=app_conn,
        query_vector=query,
        top_k=5,
        embedding_model=embedder.model,
        embedding_dimensions=embedder.dimensions,
    )

    assert hits, "검색 결과가 비면 안 된다"
    assert hits[0].document_slug == "refund"
    assert hits == sorted(hits, key=lambda hit: hit.similarity, reverse=True)


@pytest.mark.db
def test_검색은_컷_전_상위_top_k_후보를_그대로_돌려준다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """자르는 것은 `LIMIT top_k` 하나다 — 컷은 `evidence.adopt_policy_hits` 가 건다.

    검색이 컷을 먼저 걸면 질의 단위 기권 게이트가 **짧아진 슬라이스**로 산포를 재게 되어
    오프라인 격자와 런타임이 다른 수를 낸다(`docs/tracking/decisions/0014`). 그래서 컷
    미만 후보가 여기서 살아 있는 것이 계약이다.
    """
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    embedder = LexicalEmbeddingClient(dimensions=1536)
    index_policy_documents(conn=app_conn, documents=documents, embedder=embedder)
    query = embedder.embed(stage="inquiry", texts=["환불 신청 기간"]).vectors[0]

    hits = search_policy_chunks(
        conn=app_conn,
        query_vector=query,
        top_k=10,
        embedding_model=embedder.model,
        embedding_dimensions=embedder.dimensions,
    )

    assert len(hits) == 10
    # 양성 대조 — 대역 임베딩의 상위 10건에는 제품 컷(0.30) 미만이 실제로 섞여 있다.
    assert [hit for hit in hits if hit.similarity < 0.30]
    # 그리고 그 후보를 버리는 것은 채택 단계다.
    assert all(
        hit.similarity >= 0.30
        for hit in adopt_policy_hits(
            candidates=hits, top_k=10, similarity_threshold=0.30, gate=None
        )
    )


# ── 임베딩 provenance (불변식: 저장된 벡터와 질의는 같은 공간에서 나와야 한다) ──


@pytest.mark.db
def test_적재는_벡터와_함께_그것을_만든_모델과_차원을_적는다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    embedder = LexicalEmbeddingClient(dimensions=1536)

    index_policy_documents(conn=app_conn, documents=documents, embedder=embedder)

    with app_conn.cursor() as cur:
        cur.execute("SELECT DISTINCT embedding_model, embedding_dimensions FROM policy_chunks")
        assert cur.fetchall() == [
            {"embedding_model": embedder.model, "embedding_dimensions": embedder.dimensions}
        ]


def test_출처를_밝히지_못하는_임베더는_적재하지_않는다() -> None:
    """모델 이름이 빈 벡터는 나중에 같은 공간인지 판정할 근거가 없다.

    커넥션 없이 도는 것 자체가 확인이다 — 거부가 DB 왕복이나 임베딩 호출보다 앞에 있다.
    """
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    embedder = LexicalEmbeddingClient(dimensions=1536, model="   ")

    with pytest.raises(ValueError, match="임베딩 모델 이름이 비어 있다"):
        index_policy_documents(
            conn=cast(psycopg.Connection[DictRow], None), documents=documents, embedder=embedder
        )


@pytest.mark.db
def test_다른_모델의_질의는_유사도를_내지_않고_거부된다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """차원이 같으면 pgvector 는 코사인을 계산해 준다 — 오류가 아니라 **근거 없음**으로
    위장되는 경로이므로, 계산 전에 코드가 막는다."""
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    indexed = LexicalEmbeddingClient(dimensions=1536, model="model-a")
    index_policy_documents(conn=app_conn, documents=documents, embedder=indexed)
    query = indexed.embed(stage="inquiry", texts=["환불 신청 기간"]).vectors[0]

    with pytest.raises(PolicyIndexProvenanceError) as excinfo:
        search_policy_chunks(
            conn=app_conn,
            query_vector=query,
            top_k=5,
            embedding_model="model-b",
            embedding_dimensions=1536,
        )

    assert "model-a" in str(excinfo.value)
    assert "model-b" in str(excinfo.value)
    assert "index_policies" in str(excinfo.value)


@pytest.mark.db
def test_차원만_다른_질의도_거부된다(app_conn: psycopg.Connection[DictRow]) -> None:
    """모델 이름이 같아도 차원이 다르면 다른 공간이다(`3-large` 는 1536·3072 둘 다 낸다)."""
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    embedder = LexicalEmbeddingClient(dimensions=1536, model="model-a")
    index_policy_documents(conn=app_conn, documents=documents, embedder=embedder)
    query = embedder.embed(stage="inquiry", texts=["환불 신청 기간"]).vectors[0]

    with pytest.raises(PolicyIndexProvenanceError):
        search_policy_chunks(
            conn=app_conn,
            query_vector=query,
            top_k=5,
            embedding_model="model-a",
            embedding_dimensions=3072,
        )


@pytest.mark.db
def test_두_공간이_섞인_인덱스는_어느_질의로도_거부된다(
    app_conn: psycopg.Connection[DictRow],
) -> None:
    """적재가 도중에 끊겨 행마다 출처가 다를 수 있다 — 그 상태는 어떤 질의와도 맞지 않는다."""
    documents = load_policy_documents(DEFAULT_POLICY_DIR)
    embedder = LexicalEmbeddingClient(dimensions=1536, model="model-a")
    index_policy_documents(conn=app_conn, documents=documents, embedder=embedder)
    app_conn.execute(
        "UPDATE policy_chunks SET embedding_model = 'model-b' WHERE evidence_id = %s",
        (documents[0].chunks[0].evidence_id,),
    )
    query = embedder.embed(stage="inquiry", texts=["환불 신청 기간"]).vectors[0]

    with pytest.raises(PolicyIndexProvenanceError, match=r"model-a.*model-b"):
        search_policy_chunks(
            conn=app_conn,
            query_vector=query,
            top_k=5,
            embedding_model="model-a",
            embedding_dimensions=1536,
        )


@pytest.mark.db
def test_적재_전_빈_인덱스는_불일치가_아니다(app_conn: psycopg.Connection[DictRow]) -> None:
    """검색할 것이 없는 것과 다른 공간을 비교하는 것은 다르다."""
    app_conn.execute("DELETE FROM policy_chunks")

    hits = search_policy_chunks(
        conn=app_conn,
        query_vector=[0.0] * 1536,
        top_k=5,
        embedding_model="model-a",
        embedding_dimensions=1536,
    )

    assert hits == []
