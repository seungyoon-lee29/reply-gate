"""테스트 대역 자체의 성질 검증 — 대역이 틀리면 이 대역을 쓰는 모든 검증이 무의미해진다."""

from __future__ import annotations

from typing import Any, cast

from reply_gate.config import Settings
from reply_gate.contracts import Claim, Draft, Evidence, EvidenceSource, RejectReason, Verdict
from reply_gate.llm import GenerationClient
from reply_gate.pipeline import build_pipeline
from reply_gate.testing import LexicalEmbeddingClient, StubJudge, build_stub_pipeline

_EVIDENCE = Evidence(
    id="policy:refund:2-4",
    source=EvidenceSource.POLICY,
    content="[환불 정책 2-4] 검수가 완료되면 3영업일 이내에 환불이 진행됩니다.",
    evidence_text="[환불 정책 2-4] 검수가 완료되면 3영업일 이내에 환불이 진행됩니다.",
)


def _draft(text: str) -> Draft:
    return Draft(claims=(Claim(text=text, citation_ids=(_EVIDENCE.id,)),))


class _UnusableGenerationClient:
    """조립만 확인하는 테스트용 — 호출되면 실패한다."""

    def complete_json(self, **kwargs: Any) -> Any:
        del kwargs
        raise AssertionError("조립 확인 테스트는 생성 클라이언트를 부르지 않는다")


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


# ── 결정론 판정 대역 ────────────────────────────────────────────────────────


def test_판정_대역은_근거에_없는_값을_기각하고_있는_값은_통과시킨다() -> None:
    """양성·음성 짝 — 전부 기각하는 대역은 배관 검증에 쓸 수 없다."""
    judge = StubJudge()

    supported = judge.judge(
        draft=_draft("검수 후 3영업일 이내에 환불됩니다."), evidence=[_EVIDENCE]
    )
    fabricated = judge.judge(
        draft=_draft("검수 후 7영업일 이내에 환불됩니다."), evidence=[_EVIDENCE]
    )

    assert supported.result.verdict is Verdict.PASS
    assert supported.result.reject_reasons == ()
    assert fabricated.result.verdict is Verdict.REJECT
    assert fabricated.result.reject_reasons == (RejectReason.UNSUPPORTED_CLAIM,)
    # claim 판정은 초안의 claim 전부와 1:1 로 대응한다 (판정 계약과 같은 모양).
    assert [judgment.claim_text for judgment in fabricated.result.claim_judgments] == [
        "검수 후 7영업일 이내에 환불됩니다."
    ]
    assert fabricated.input_tokens > 0 and fabricated.output_tokens > 0


def test_판정_대역은_같은_입력에_같은_판정을_준다() -> None:
    first = StubJudge().judge(draft=_draft("검수 후 3영업일."), evidence=[_EVIDENCE])
    second = StubJudge().judge(draft=_draft("검수 후 3영업일."), evidence=[_EVIDENCE])

    assert first == second


def test_판정_대역은_호출을_기록한다() -> None:
    judge = StubJudge()
    judge.judge(draft=_draft("검수 후 3영업일."), evidence=[_EVIDENCE])

    assert len(judge.calls) == 1
    assert judge.calls[0].claims[0].text == "검수 후 3영업일."


def test_대역_조립기는_실제_조립과_같은_협력자를_쓴다() -> None:
    """조합이 어긋나면 대역 실행이 실제 실행과 **다른 것**을 잰다 — 드리프트를 여기서 잡는다."""
    settings = Settings(l2_enabled=True)
    generation = cast(GenerationClient, _UnusableGenerationClient())
    embedding = LexicalEmbeddingClient(dimensions=8)

    real = build_pipeline(
        generation_client=generation, embedding_client=embedding, settings=settings
    )
    stub = build_stub_pipeline(
        generation_client=generation,
        embedding_client=embedding,
        judge=StubJudge(),
        settings=settings,
    )

    assert type(stub._collector) is type(real._collector)
    assert type(stub._drafter) is type(real._drafter)
    assert stub._l2_enabled == real._l2_enabled is True
    # 다른 것은 판정자 하나뿐이다 — 그게 이 조립기의 존재 이유다.
    assert isinstance(stub._judge, StubJudge)
    assert not isinstance(real._judge, StubJudge)
