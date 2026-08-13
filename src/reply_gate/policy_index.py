"""정책 문서 파싱·청킹·인덱싱 — 대표 흐름 3단계의 검색 대상
(docs/architecture.md "구성요소 지도").

**청킹은 조항 단위**다 — 정책 문서는 조항 경계가 곧 의미 단위이기 때문이다(고정 크기 등
전략 비교는 다음 사이클). 청크 1개 = 조항 1개 = 근거 1개이고, 청크 ID 는
`reply_gate.contracts.policy_evidence_id()` 가 정의하는 `policy:<문서 slug>:<조항 번호>` 다.

문서에는 **모호 조항·상충 조항**(내용상 오류 — L1 이 아니라 L2 의 검출 대상이다)과
**미끼 조항**(관련 주제를 다루되 패턴형 값을 일부러 뺀 조항 — 모델이 일반 지식으로 그 값을
채우면 L1 `pii_detected` 에 걸린다)이 심겨 있다. 심은 위치는 문서 안 주석으로 표시되며
이 모듈이 메타데이터로 읽어 들인다. **테스트 데이터 정리 명목으로 제거하지 않는다** —
기각 장면이 재현되지 않으면 제품 서사가 무너진다.

적재는 재실행 가능하다: 같은 `evidence_id` 는 갱신하고, 문서에서 사라진 조항은 지운다.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import DictRow

from reply_gate.contracts import policy_evidence_id
from reply_gate.llm import EmbeddingClient

__all__ = [
    "DEFAULT_POLICY_DIR",
    "IndexResult",
    "PlantedKind",
    "PolicyChunk",
    "PolicyDocument",
    "PolicyIndexProvenanceError",
    "PolicySearchHit",
    "index_policy_documents",
    "load_policy_documents",
    "parse_policy_document",
    "search_policy_chunks",
]

#: 저장소의 정책 문서 원본 위치.
DEFAULT_POLICY_DIR = Path(__file__).resolve().parents[2] / "data" / "policies"

#: 처리 기록·비용 산출에 남는 단계 이름.
POLICY_INDEX_STAGE = "policy_index"

_FRONT_MATTER = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.DOTALL)
_PLANTED = re.compile(r"<!--\s*planted:\s*(?P<kind>[a-z_]+)\s*(?:;\s*note:\s*(?P<note>.*?))?-->")
_HEADING = re.compile(r"^##\s+(?P<article>\S+)\s+(?P<title>.+?)\s*$")


class PolicyIndexProvenanceError(RuntimeError):
    """저장된 벡터가 질의 임베딩과 다른 공간에서 나왔다 — 재색인 없이는 비교가 성립하지 않는다.

    **인계 사유가 아니라 설정 오류다.** `no_evidence` 로 바꾸면 "검색이 못 찾았다"로 집계되어
    낡은 인덱스가 검색 품질 지표로 위장한다(docs/standards.md "오류를 업무 판정으로
    바꾸지 않는다").
    """


class PlantedKind(StrEnum):
    """조항에 일부러 심은 장치의 종류."""

    NONE = "none"
    #: 정량 기준이 없어 답변자가 수치를 지어낼 여지가 있는 조항 (L2 검출 대상).
    AMBIGUOUS = "ambiguous"
    #: 같은 상황에 다른 기준을 제시하는 조항 (L2 검출 대상).
    CONFLICTING = "conflicting"
    #: 패턴형 값을 일부러 뺀 조항 (사이클 1 기각 유발 재료).
    DECOY = "decoy"


@dataclass(frozen=True)
class PolicyChunk:
    """조항 1개 = 근거 1개."""

    evidence_id: str
    document_slug: str
    document_title: str
    article: str
    article_title: str
    content: str
    planted: PlantedKind
    planted_note: str | None

    @property
    def embedding_text(self) -> str:
        """임베딩 대상 텍스트. 조항이 어느 문서·주제인지까지 담아야 검색이 산다."""
        return f"{self.document_title} / {self.article_title}\n{self.content}"


@dataclass(frozen=True)
class PolicyDocument:
    """정책 문서 1개."""

    slug: str
    title: str
    chunks: tuple[PolicyChunk, ...]


@dataclass(frozen=True)
class IndexResult:
    """적재 결과 — 적재/삭제 건수와 임베딩 토큰(건당 비용 산출용)."""

    upserted: int
    deleted: int
    embedding_tokens: int


@dataclass(frozen=True)
class PolicySearchHit:
    """유사도 검색 결과 1건."""

    evidence_id: str
    document_slug: str
    document_title: str
    article: str
    article_title: str
    content: str
    similarity: float


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(text)
    if match is None:
        raise ValueError("정책 문서에 front matter(--- slug/title ---)가 없다")
    fields: dict[str, str] = {}
    for line in match.group("body").splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields, text[match.end() :]


def parse_policy_document(text: str) -> PolicyDocument:
    """마크다운 원본을 조항 단위 청크로 쪼갠다.

    `## <조항번호> <제목>` 이 조항 경계이고, 바로 앞의 `<!-- planted: ... -->` 주석이
    그 조항에 심은 장치를 표시한다.
    """
    fields, body = _parse_front_matter(text)
    slug = fields.get("slug", "")
    title = fields.get("title", "")
    if not slug or not title:
        raise ValueError("정책 문서 front matter 에 slug 와 title 이 모두 있어야 한다")

    chunks: list[PolicyChunk] = []
    pending_kind = PlantedKind.NONE
    pending_note: str | None = None
    current: PolicyChunk | None = None
    lines: list[str] = []

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        content = "\n".join(lines).strip()
        if not content:
            raise ValueError(f"조항 {current.article} 의 본문이 비어 있다")
        chunks.append(
            PolicyChunk(
                evidence_id=current.evidence_id,
                document_slug=current.document_slug,
                document_title=current.document_title,
                article=current.article,
                article_title=current.article_title,
                content=content,
                planted=current.planted,
                planted_note=current.planted_note,
            )
        )
        current = None
        lines.clear()

    for line in body.splitlines():
        planted_match = _PLANTED.search(line)
        if planted_match is not None:
            pending_kind = PlantedKind(planted_match.group("kind"))
            note = planted_match.group("note")
            pending_note = note.strip() if note else None
            continue

        heading = _HEADING.match(line)
        if heading is not None:
            flush()
            article = heading.group("article")
            current = PolicyChunk(
                evidence_id=policy_evidence_id(document_slug=slug, article=article),
                document_slug=slug,
                document_title=title,
                article=article,
                article_title=heading.group("title"),
                content="",
                planted=pending_kind,
                planted_note=pending_note,
            )
            pending_kind = PlantedKind.NONE
            pending_note = None
            continue

        if current is not None:
            lines.append(line)

    flush()
    if not chunks:
        raise ValueError("정책 문서에 조항(`## <번호> <제목>`)이 하나도 없다")
    return PolicyDocument(slug=slug, title=title, chunks=tuple(chunks))


def load_policy_documents(directory: Path = DEFAULT_POLICY_DIR) -> tuple[PolicyDocument, ...]:
    """디렉터리의 모든 정책 문서를 파일명 순서로 읽는다."""
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"정책 문서를 찾을 수 없다: {directory}")
    return tuple(parse_policy_document(path.read_text(encoding="utf-8")) for path in paths)


def _all_chunks(documents: Iterable[PolicyDocument]) -> list[PolicyChunk]:
    chunks = [chunk for document in documents for chunk in document.chunks]
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.evidence_id in seen:
            raise ValueError(f"근거 ID 가 중복된다: {chunk.evidence_id}")
        seen.add(chunk.evidence_id)
    return chunks


def index_policy_documents(
    *,
    conn: psycopg.Connection[DictRow],
    documents: Sequence[PolicyDocument],
    embedder: EmbeddingClient,
) -> IndexResult:
    """조항을 임베딩해 `policy_chunks` 에 적재한다. 재실행하면 중복 없이 갱신된다.

    벡터와 **함께 그 벡터를 만든 모델·차원을 적는다** — 나중에 질의 임베딩이 같은 공간에서
    나왔는지 판정할 근거가 이것뿐이다.
    """
    chunks = _all_chunks(documents)
    if embedder.dimensions <= 0:
        raise ValueError("임베딩 차원은 1 이상이어야 한다")
    if not embedder.model.strip():
        raise ValueError(
            "임베딩 모델 이름이 비어 있다 — 출처를 적을 수 없는 벡터는 적재하지 않는다"
        )

    embedding = embedder.embed(
        stage=POLICY_INDEX_STAGE, texts=[chunk.embedding_text for chunk in chunks]
    )
    if len(embedding.vectors) != len(chunks):
        raise ValueError(
            f"임베딩 개수가 조항 수와 다르다: {len(embedding.vectors)} != {len(chunks)}"
        )

    register_vector(conn)
    with conn.cursor() as cur:
        for chunk, vector in zip(chunks, embedding.vectors, strict=True):
            cur.execute(
                """
                INSERT INTO policy_chunks
                    (evidence_id, document_slug, document_title, article, article_title,
                     content, embedding, embedding_model, embedding_dimensions)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (evidence_id) DO UPDATE SET
                    document_slug        = EXCLUDED.document_slug,
                    document_title       = EXCLUDED.document_title,
                    article              = EXCLUDED.article,
                    article_title        = EXCLUDED.article_title,
                    content              = EXCLUDED.content,
                    embedding            = EXCLUDED.embedding,
                    embedding_model      = EXCLUDED.embedding_model,
                    embedding_dimensions = EXCLUDED.embedding_dimensions
                """,
                (
                    chunk.evidence_id,
                    chunk.document_slug,
                    chunk.document_title,
                    chunk.article,
                    chunk.article_title,
                    chunk.content,
                    vector,
                    embedder.model,
                    embedder.dimensions,
                ),
            )
        # 문서에서 사라진 조항은 남겨두면 존재하지 않는 근거를 검색이 돌려주게 된다.
        cur.execute(
            "DELETE FROM policy_chunks WHERE evidence_id <> ALL(%s)",
            ([chunk.evidence_id for chunk in chunks],),
        )
        deleted = cur.rowcount

    return IndexResult(
        upserted=len(chunks),
        deleted=max(deleted, 0),
        embedding_tokens=embedding.total_tokens,
    )


def _assert_index_provenance(
    cur: psycopg.Cursor[DictRow], *, embedding_model: str, embedding_dimensions: int
) -> None:
    """저장된 벡터가 전부 질의와 같은 공간에서 나왔는지 확인한다 — 아니면 거부한다.

    검색 자체는 거부하지 않는다: 차원이 같으면 pgvector 가 코사인을 계산해 주고, 그 값은
    **오류처럼 보이지 않는다.** 실측에서 다른 모델의 질의는 전 조항이 임계값 아래로 떨어져
    `no_evidence` 로 끝났다 — 낡은 인덱스가 "검색이 못 찾았다"로 위장한 것이다.

    적재 전(행 0개)은 불일치가 아니다. 검색할 것이 없으면 빈 결과와 같다.
    """
    cur.execute(
        """
        SELECT embedding_model, embedding_dimensions, count(*) AS n
        FROM policy_chunks
        GROUP BY embedding_model, embedding_dimensions
        ORDER BY embedding_model, embedding_dimensions
        """
    )
    rows = cur.fetchall()
    if not rows:
        return
    if len(rows) == 1 and (
        rows[0]["embedding_model"] == embedding_model
        and int(rows[0]["embedding_dimensions"]) == embedding_dimensions
    ):
        return
    stored = ", ".join(
        f"{row['embedding_model']}/{int(row['embedding_dimensions'])}차원 {int(row['n'])}건"
        for row in rows
    )
    raise PolicyIndexProvenanceError(
        f"정책 인덱스의 임베딩 출처가 질의와 다르다 — 저장: {stored}, "
        f"질의: {embedding_model}/{embedding_dimensions}차원. "
        "`uv run python -m scripts.index_policies` 로 재색인해야 한다."
    )


def search_policy_chunks(
    *,
    conn: psycopg.Connection[DictRow],
    query_vector: Sequence[float],
    top_k: int,
    similarity_threshold: float,
    embedding_model: str,
    embedding_dimensions: int,
) -> list[PolicySearchHit]:
    """문의 임베딩과 코사인 유사도가 높은 조항 상위 k 개.

    **임계값 미만 결과는 버린다** (docs/architecture.md "대표 흐름" 3단계) — 관련 없는 조항을 근거로
    올리면 초안이 그 위에 문장을 세우고 게이트가 잡아야 할 일이 늘어난다.

    **질의 임베딩의 출처를 함께 받는다.** 저장된 벡터와 다른 공간이면 유사도를 계산하지
    않고 거부한다(fail-closed) — 이 확인을 호출자에게 맡기면 새 호출자가 생길 때마다 다시
    빠뜨릴 수 있고, 빠뜨린 결과는 오류가 아니라 정상 판정처럼 보인다.
    """
    if top_k <= 0:
        return []
    register_vector(conn)
    vector = Vector(list(query_vector))
    with conn.cursor() as cur:
        _assert_index_provenance(
            cur, embedding_model=embedding_model, embedding_dimensions=embedding_dimensions
        )
        cur.execute(
            """
            SELECT evidence_id, document_slug, document_title, article, article_title,
                   content, 1 - (embedding <=> %s) AS similarity
            FROM policy_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (vector, vector, top_k),
        )
        rows = cur.fetchall()

    return [
        PolicySearchHit(
            evidence_id=row["evidence_id"],
            document_slug=row["document_slug"],
            document_title=row["document_title"],
            article=row["article"],
            article_title=row["article_title"],
            content=row["content"],
            similarity=float(row["similarity"]),
        )
        for row in rows
        if float(row["similarity"]) >= similarity_threshold
    ]
