"""테스트 대역 자체의 성질 검증 — 대역이 틀리면 이 대역을 쓰는 모든 검증이 무의미해진다."""

from __future__ import annotations

from reply_gate.testing import LexicalEmbeddingClient


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def test_같은_입력은_같은_벡터를_준다() -> None:
    client = LexicalEmbeddingClient(dimensions=64)

    first = client.embed(stage="t", texts=["환불은 수령 후 7일 이내"]).vectors[0]
    second = client.embed(stage="t", texts=["환불은 수령 후 7일 이내"]).vectors[0]

    assert first == second


def test_어휘가_겹칠수록_유사도가_높다() -> None:
    client = LexicalEmbeddingClient(dimensions=512)
    query = "환불 신청 기간이 어떻게 되나요"
    related = "환불 신청은 상품 수령 후 7일 이내에 가능합니다"
    unrelated = "임직원 주차장 이용 규정 안내"

    vectors = client.embed(stage="t", texts=[query, related, unrelated]).vectors

    assert _cosine(vectors[0], vectors[1]) > _cosine(vectors[0], vectors[2])


def test_단위벡터를_돌려준다() -> None:
    client = LexicalEmbeddingClient(dimensions=128)

    for text in ["배송", "", "교환 규정"]:
        vector = client.embed(stage="t", texts=[text]).vectors[0]
        assert abs(_cosine(vector, vector) - 1.0) < 1e-9
        assert len(vector) == 128
