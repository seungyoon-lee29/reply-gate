"""테스트·오프라인 검증용 대역(fake).

여기 있는 구현은 외부 서비스를 호출하지 않는다. 실제 실행 경로에서는 쓰지 않는다.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from reply_gate.llm import EmbeddingResult

__all__ = ["LexicalEmbeddingClient"]


class LexicalEmbeddingClient:
    """문자 2-gram 해시 기반의 결정론 임베딩 대역.

    실제 임베딩 모델의 의미 유사도를 흉내내지는 않지만, **어휘가 겹치는 문서일수록
    코사인 유사도가 높다**는 성질은 유지한다. 덕분에 API 키 없이도 벡터 검색 배관
    (적재 → 유사도 정렬 → 임계값 필터)을 끝까지 검증할 수 있다.

    한국어는 띄어쓰기가 의미 단위와 어긋나는 경우가 많아 공백 토큰 대신 문자 2-gram 을 쓴다.
    """

    def __init__(self, *, dimensions: int = 1536) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions 는 1 이상이어야 한다")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, *, stage: str, texts: Sequence[str]) -> EmbeddingResult:
        del stage  # 대역은 단계별로 다르게 동작하지 않는다.
        vectors = [self._vector(text) for text in texts]
        total_tokens = sum(len(text) for text in texts)
        return EmbeddingResult(vectors=vectors, total_tokens=total_tokens)

    def _vector(self, text: str) -> list[float]:
        counts = [0.0] * self._dimensions
        normalized = "".join(text.split()).lower()
        grams = [normalized[i : i + 2] for i in range(max(len(normalized) - 1, 0))] or [normalized]
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            counts[int.from_bytes(digest, "big") % self._dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in counts))
        if norm == 0.0:
            # 빈 문자열: 0 벡터 대신 첫 축을 세워 코사인 계산에서 0 나눗셈을 피한다.
            counts[0] = 1.0
            return counts
        return [value / norm for value in counts]
