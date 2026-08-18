"""검색용 질의 재작성 — 폴백 계약·배선 검사·프롬프트 단일 소유.

여기서 지키는 것은 셋이다.

1. **런타임 호출 실패는 폴백이지 인계가 아니다**(docs/business-rules.md "검색 단계 실패").
   대신 조용하지도 않다 — 사유와 실비용 토큰이 그대로 올라온다.
2. **미배선은 폴백이 아니라 조립 시점 오류다.** 기본값·`None` 이 조용히 전략을 끄면 배선을
   빠뜨린 실행이 "재작성을 켜고 돌았다"로 집계된다.
3. **프롬프트는 한 벌이다.** 픽스처를 만든 문장과 런타임이 쓰는 문장이 갈리면 오프라인
   비교표(macro F1 0.734)가 런타임을 더 이상 예측하지 못한다.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

import pytest

from reply_gate.config import Settings
from reply_gate.evidence import RetrievalWiringError, merge_policy_rankings
from reply_gate.llm import GenerationClient, JsonCompletion, LLMCallError, LLMFormatError
from reply_gate.pipeline import MissingCredentialsError
from reply_gate.policy_index import PolicySearchHit
from reply_gate.query_rewrite import (
    QUERY_REWRITE_STAGE,
    REWRITE_JSON_SCHEMA,
    REWRITE_SYSTEM_PROMPT,
    RewriteOutcome,
    build_rewrite_user_prompt,
    rewrite_query,
)
from reply_gate.testing import LexicalEmbeddingClient

_ROOT = Path(__file__).resolve().parents[1]
INQUIRY = "상담원과 통화하고 싶은데 몇 번으로 걸면 되나요?"


class _Client:
    """`GenerationClient` 대역 — 정해진 산출을 돌려주거나 정해진 예외를 던진다."""

    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


def _client(outcome: Any) -> GenerationClient:
    return cast(GenerationClient, _Client(outcome))


def _completion(data: Any, *, input_tokens: int = 11, output_tokens: int = 4) -> JsonCompletion:
    return JsonCompletion(data=data, input_tokens=input_tokens, output_tokens=output_tokens)


# ── 성공 경로 ───────────────────────────────────────────────────────────────


def test_재작성문을_받아_돌려주고_토큰을_싣는다() -> None:
    outcome = rewrite_query(
        client=_client(_completion({"rewritten": " 고객센터 전화 연결 방법 "})),
        inquiry=INQUIRY,
    )

    assert outcome.query == "고객센터 전화 연결 방법"
    assert outcome.fell_back is False
    assert outcome.fallback_reason is None
    assert (outcome.input_tokens, outcome.output_tokens) == (11, 4)


def test_호출은_픽스처를_만든_그_프롬프트와_스키마로_나간다() -> None:
    """조건이 갈리면 비교표가 런타임을 예측하지 못한다 — 그래서 호출 인자를 못박는다."""
    client = _Client(_completion({"rewritten": "고객센터 전화 연결 방법"}))

    rewrite_query(client=cast(GenerationClient, client), inquiry=INQUIRY, effort="low")

    (call,) = client.calls
    assert call["stage"] == QUERY_REWRITE_STAGE
    assert call["system"] == REWRITE_SYSTEM_PROMPT
    assert call["user"] == f"[문의]\n{INQUIRY}"
    assert call["schema"] == REWRITE_JSON_SCHEMA
    assert call["effort"] == "low"


def test_프롬프트에는_문의_원문_말고는_아무것도_없다() -> None:
    """정책·라벨·직전 결과가 섞이면 blind 조건이 깨진다(결정 0010)."""
    assert build_rewrite_user_prompt(inquiry=INQUIRY) == f"[문의]\n{INQUIRY}"


# ── 폴백 경로 — 인계가 아니다 ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("outcome", "expected_reason", "tokens"),
    [
        (
            LLMCallError(stage=QUERY_REWRITE_STAGE, reason="연결이 끊겼다", attempts=2),
            "전송 오류",
            (0, 0),
        ),
        (
            LLMCallError(
                stage=QUERY_REWRITE_STAGE,
                reason="거절 응답",
                attempts=1,
                input_tokens=9,
                output_tokens=2,
            ),
            "전송 오류",
            (9, 2),
        ),
        (
            LLMFormatError(
                stage=QUERY_REWRITE_STAGE,
                detail="JSON 이 아니다",
                input_tokens=6,
                output_tokens=1,
            ),
            "구조화 출력 형식 불일치",
            (6, 1),
        ),
    ],
)
def test_호출_실패는_폴백이고_사유와_실비용_토큰이_남는다(
    outcome: Exception, expected_reason: str, tokens: tuple[int, int]
) -> None:
    result = rewrite_query(client=_client(outcome), inquiry=INQUIRY)

    assert result.query is None
    assert result.fell_back is True
    assert result.fallback_reason is not None
    assert expected_reason in result.fallback_reason
    # 실행됐으나 실패한 호출의 토큰도 실비용이다 — 0 으로 접지 않는다.
    assert (result.input_tokens, result.output_tokens) == tokens


@pytest.mark.parametrize(
    "data", [{"rewritten": ""}, {"rewritten": "   "}, {"rewritten": 7}, {}, []]
)
def test_쓸_수_없는_산출도_폴백이다(data: Any) -> None:
    result = rewrite_query(client=_client(_completion(data)), inquiry=INQUIRY)

    assert result.query is None
    assert result.fallback_reason is not None
    # 형식이 틀려도 호출은 나갔다 — 그 토큰은 실비용이다.
    assert (result.input_tokens, result.output_tokens) == (11, 4)


def test_자격_증명_부재는_폴백으로_위장되지_않는다() -> None:
    """설정 오류를 검색 폴백으로 삼키면, 키 없이 돌린 실행이 503 대신 정상 처리로 집계된다.

    `MissingCredentialsError` 가 `LLMCallError` 를 상속하지 않는 이유가 바로 이것이고,
    여기서 좁게 잡는 것이 그 설계를 실제로 쓰는 자리다.
    """
    with pytest.raises(MissingCredentialsError):
        rewrite_query(client=_client(MissingCredentialsError("키가 없다")), inquiry=INQUIRY)


def test_폴백에는_반드시_사유가_붙는다() -> None:
    """자료형이 조용한 폴백을 만들 수 없게 한다 — 사유 없는 폴백은 구성 불가다."""
    with pytest.raises(ValueError, match="조용한 폴백"):
        RewriteOutcome(query=None, input_tokens=0, output_tokens=0, fallback_reason=None)
    with pytest.raises(ValueError, match="조용한 폴백"):
        RewriteOutcome(query="질의", input_tokens=0, output_tokens=0, fallback_reason="사유")


# ── 배선 (하드 게이트) ──────────────────────────────────────────────────────


def test_스위치_켜짐_더하기_미배선은_조립에서_죽는다() -> None:
    from reply_gate.evidence import EvidenceCollector

    with pytest.raises(RetrievalWiringError, match="재작성 클라이언트가 배선되지 않았다"):
        EvidenceCollector(
            generation_client=_client(_completion({"rewritten": "질의"})),
            embedding_client=LexicalEmbeddingClient(dimensions=1536),
            settings=Settings(query_rewrite_enabled=True),
        )


def test_명시적으로_끄면_클라이언트_없이도_조립된다() -> None:
    """양성 대조 — 전부 거부하는 검사는 검사가 아니다. 끄려면 **명시**해야 한다."""
    from reply_gate.evidence import EvidenceCollector

    collector = EvidenceCollector(
        generation_client=_client(_completion({"rewritten": "질의"})),
        embedding_client=LexicalEmbeddingClient(dimensions=1536),
        settings=Settings(query_rewrite_enabled=False),
    )

    assert collector is not None


def test_제품_기본값은_재작성_켜짐과_컷_0_30_이다() -> None:
    """사이클 3 T10 이 두 축을 갈랐다 — 재작성은 실측이 정당화하고 컷 0.50 은 반증됐다.

    컷 0.50 은 무근거 4건을 기권시키는 대신 G04 의 정답 조항(0.3571)과 G18·G01 의 상충
    조항(0.4676·0.4693)을 함께 잘라 정상 문의를 인계시키고 L2 모순 기각을 없앴다.
    셋 다 0.30 위다.
    """
    settings = Settings()

    assert settings.query_rewrite_enabled is True
    assert settings.vector_similarity_threshold == 0.3


# ── 합집합 합침 규칙 (순수 함수) ────────────────────────────────────────────


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


def _ranked(scores: dict[str, float]) -> list[PolicySearchHit]:
    """유사도 내림차순 순위 — DB 검색 결과의 모양이다."""
    return [
        _hit(evidence_id, similarity)
        for evidence_id, similarity in sorted(scores.items(), key=lambda item: -item[1])
    ]


def test_같은_조항은_더_큰_유사도로_합쳐진다() -> None:
    merged = merge_policy_rankings(
        original=_ranked({"policy:a:1": 0.90, "policy:b:1": 0.20}),
        rewritten=_ranked({"policy:b:1": 0.80, "policy:c:1": 0.40}),
        top_k=5,
    )

    assert [hit.evidence_id for hit in merged] == ["policy:a:1", "policy:b:1", "policy:c:1"]
    assert [hit.similarity for hit in merged] == [0.90, 0.80, 0.40]


def test_합집합이어도_채택_상한은_top_k_그대로다() -> None:
    """재작성이 후보를 2배로 늘려도 채택 상한은 `top_k` 가 그대로 자른다."""
    merged = merge_policy_rankings(
        original=_ranked({"policy:a:1": 0.9, "policy:b:1": 0.8}),
        rewritten=_ranked({"policy:c:1": 0.7, "policy:d:1": 0.6}),
        top_k=3,
    )

    assert [hit.evidence_id for hit in merged] == ["policy:a:1", "policy:b:1", "policy:c:1"]


def test_각_질의의_top_k_를_합친_결과는_전체_순위를_합친_것과_같다() -> None:
    """런타임은 DB 가 이미 자른 top_k 를 합치고, 오프라인 하네스는 전체 순위를 합친다.

    두 결과가 갈리면 비교표가 런타임을 예측하지 못한다. 26개 조항 전체 순위를 만들어
    양쪽을 실제로 대조한다 — 코드 주석의 증명을 값으로 확인하는 자리다.
    """
    top_k = 5
    corpus = [f"policy:doc:{index}" for index in range(26)]
    # 두 질의가 서로 다르게 정렬되도록 어긋난 점수를 준다.
    original_scores = {name: (index * 7 % 26) / 26 for index, name in enumerate(corpus)}
    rewritten_scores = {name: (index * 11 % 26) / 26 for index, name in enumerate(corpus)}

    truncated = merge_policy_rankings(
        original=_ranked(original_scores)[:top_k],
        rewritten=_ranked(rewritten_scores)[:top_k],
        top_k=top_k,
    )
    full = merge_policy_rankings(
        original=_ranked(original_scores),
        rewritten=_ranked(rewritten_scores),
        top_k=top_k,
    )

    assert [hit.evidence_id for hit in truncated] == [hit.evidence_id for hit in full]
    assert len(full) == top_k  # 양성 대조 — 빈 결과끼리 같은 것이 아니다


def test_재작성_순위가_비면_원문_순위_그대로다() -> None:
    """폴백 경로의 값 — 재작성이 없는 것은 사이클 2 동작과 정확히 같아야 한다."""
    original = _ranked({"policy:a:1": 0.9, "policy:b:1": 0.4})

    merged = merge_policy_rankings(original=original, rewritten=(), top_k=5)

    assert [hit.evidence_id for hit in merged] == ["policy:a:1", "policy:b:1"]


# ── 프롬프트 단일 소유 (구조 검사) ──────────────────────────────────────────


def test_픽스처_생성기는_프롬프트를_스스로_정의하지_않고_패키지에서_가져온다() -> None:
    """두 벌이 되면 픽스처와 런타임의 생성 조건이 조용히 갈린다.

    이름을 다시 정의하는지 AST 로 본다 — "import 했으니 같은 것을 쓴다"는 검사는 스크립트가
    자기 상수를 새로 만들어 덮어써도 통과한다.
    """
    source = (_ROOT / "scripts" / "generate_blind_rewrites.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "reply_gate.query_rewrite"
        for alias in node.names
    }
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    } | {
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    shared = {"REWRITE_SYSTEM_PROMPT", "REWRITE_JSON_SCHEMA", "build_rewrite_user_prompt"}
    assert shared <= imported
    assert not shared & assigned
