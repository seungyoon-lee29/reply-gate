"""테스트·오프라인 검증용 대역(fake).

여기 있는 구현은 외부 서비스를 호출하지 않는다. 실제 실행 경로에서는 쓰지 않는다.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Final

from reply_gate.config import Settings
from reply_gate.contracts import ClaimJudgment, Draft, Evidence, JudgeResult, RejectReason, Verdict
from reply_gate.draft import DraftGenerator
from reply_gate.evidence import EvidenceCollector
from reply_gate.judge import JudgeOutcome
from reply_gate.llm import EmbeddingClient, EmbeddingResult, GenerationClient
from reply_gate.pipeline import InquiryPipeline, Judging

__all__ = ["LexicalEmbeddingClient", "StubJudge", "build_stub_pipeline"]


class LexicalEmbeddingClient:
    """문자 2-gram 해시 기반의 결정론 임베딩 대역.

    실제 임베딩 모델의 의미 유사도를 흉내내지는 않지만, **어휘가 겹치는 문서일수록
    코사인 유사도가 높다**는 성질은 유지한다. 덕분에 API 키 없이도 벡터 검색 배관
    (적재 → 유사도 정렬 → 임계값 필터)을 끝까지 검증할 수 있다.

    한국어는 띄어쓰기가 의미 단위와 어긋나는 경우가 많아 공백 토큰 대신 문자 2-gram 을 쓴다.
    """

    #: 벡터 출처 이름. 대역도 출처를 밝혀야 실제 모델의 인덱스와 섞이지 않는다 —
    #: 대역으로 적재한 DB 를 실제 모델로 검색하면 그 자체가 불일치로 걸린다.
    MODEL = "stub:lexical-2gram"

    def __init__(self, *, dimensions: int = 1536, model: str = MODEL) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions 는 1 이상이어야 한다")
        self._dimensions = dimensions
        self._model = model

    @property
    def model(self) -> str:
        return self._model

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


#: 숫자로 시작하는 토막(`3영업일`·`30,000원`·`1588-0000`·`ORD-20260720-0002` 의 숫자 구간).
_NUMBER_RUN: Final = re.compile(r"[0-9][0-9,.\-~]*")
#: 프롬프트 길이에서 유도하는 가짜 토큰 환산 비율(문자 → 토큰).
_CHARS_PER_TOKEN: Final = 4


def _numbers(text: str) -> set[str]:
    """텍스트의 숫자열을 **구분 기호를 지운 값 집합**으로 뽑는다.

    `30,000` 과 `30000`, `3~5` 와 `3`·`5` 가 같은 값으로 대조되게 하려는 것이다 —
    표기 차이로 갈리면 대역이 근거에 있는 값을 "지어낸 값"으로 오판한다.
    """
    values: set[str] = set()
    for match in _NUMBER_RUN.findall(text):
        for part in re.split(r"[.,\-~]", match):
            digits = part.strip()
            if digits:
                values.add(digits.lstrip("0") or "0")
    return values


class StubJudge:
    """`Judging` 대역 — 결정론 판정. **실제 판정 모델이 아니다. 배관 검증 전용.**

    규칙은 하나뿐이다: claim 텍스트의 숫자열 중 **그 claim 이 인용한 근거 원문에 없는
    값**이 하나라도 있으면 그 claim 은 `unsupported_claim` 이다 — "근거에 없는 값을
    지어냈다"의 기계적 근사다. 생성 대역(`evaluation.StubGenerationClient`)이 미끼
    조항에서 값을 지어내는 것과 짝을 이룬다.

    **모순 감지는 하지 않는다**(항상 빈 목록). 어휘 규칙으로 근사하면 금액·기간이 다른
    정책 조항 대부분이 거짓 모순으로 잡혀 골든셋 배관 검증이 통째로 무의미해진다. 그래서
    이 대역으로 낸 측정 3 수치는 **판정 모델의 정확도가 아니다.**

    그 사실을 지키는 안전장치는 둘이고, **실행 조건의 판정 모델 문자열은 그중 하나도
    아니다**(사람이 읽어야만 구분되는 한 줄로는 부족하다):

    * 진입점(`scripts/evaluate.py`)은 `--stub-llm` 에서 측정 3 을 **아예 돌리지 않는다** —
      미실행 사유가 리포트에 남는다.
    * 그래도 대역으로 측정 3 을 만들면(직접 조립·테스트) 리포트가
      `RunConditions.measurement3_is_real=False` 를 들고 JSON 의 `billed`·`deterministic`
      과 마크다운의 확률층·과금 문구가 전부 대역 쪽으로 뒤집힌다.
    """

    def __init__(self) -> None:
        self.calls: list[Draft] = []

    def judge(self, *, draft: Draft, evidence: Sequence[Evidence]) -> JudgeOutcome:
        self.calls.append(draft)
        evidence_by_id = {item.id: item for item in evidence}

        judgments: list[ClaimJudgment] = []
        for claim in draft.claims:
            cited = " ".join(
                evidence_by_id[citation_id].evidence_text
                for citation_id in claim.citation_ids
                if citation_id in evidence_by_id
            )
            fabricated = sorted(_numbers(claim.text) - _numbers(cited))
            judgments.append(
                ClaimJudgment(
                    claim_text=claim.text,
                    verdict=Verdict.REJECT if fabricated else Verdict.PASS,
                    explanation=(
                        f"인용 근거에 없는 값: {', '.join(fabricated)}"
                        if fabricated
                        else "인용 근거가 뒷받침한다"
                    ),
                )
            )

        rejected = any(judgment.verdict is Verdict.REJECT for judgment in judgments)
        result = JudgeResult(
            verdict=Verdict.REJECT if rejected else Verdict.PASS,
            reject_reasons=(RejectReason.UNSUPPORTED_CLAIM,) if rejected else (),
            claim_judgments=tuple(judgments),
            contradictions=(),
        )
        prompt_chars = len(draft.answer_text) + sum(len(item.evidence_text) for item in evidence)
        output_chars = sum(
            len(judgment.claim_text) + len(judgment.explanation) for judgment in judgments
        )
        return JudgeOutcome(
            result=result,
            input_tokens=max(1, prompt_chars // _CHARS_PER_TOKEN),
            output_tokens=max(1, output_chars // _CHARS_PER_TOKEN),
            attempts=1,
        )


def build_stub_pipeline(
    *,
    generation_client: GenerationClient,
    embedding_client: EmbeddingClient,
    judge: Judging,
    settings: Settings,
) -> InquiryPipeline:
    """대역 배관 검증용 파이프라인 — `pipeline.build_pipeline` 과 **같은 협력자**를 쓰되
    판정자만 주입한다.

    `build_pipeline` 은 판정자를 설정에서 조립하므로(실제 Anthropic 클라이언트) 주입
    구멍이 없다. 그 구멍을 뚫는 것은 파이프라인 모듈의 결정이라, 여기서는 같은 조합을
    다시 적는다 — **조합이 어긋나면 대역 실행이 실제 실행과 다른 것을 잰다.**
    `tests/test_testing_doubles.py` 가 두 조립의 협력자 종류를 대조해 드리프트를 잡는다.
    """
    return InquiryPipeline(
        collector=EvidenceCollector(
            generation_client=generation_client,
            embedding_client=embedding_client,
            settings=settings,
        ),
        drafter=DraftGenerator(client=generation_client, effort=settings.generation_effort),
        judge=judge,
        l2_enabled=settings.l2_enabled,
    )
