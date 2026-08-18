"""정책 검색 평가용 정답 라벨.

판정 결과 라벨과 검색 관련성 라벨은 서로 다른 축이다. 이 모듈은 골든셋 문의별로
정책 검색기가 찾아야 할 근거 ID 집합만 읽는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from reply_gate.evaluation import DEFAULT_GOLDEN_SET_PATH, load_golden_set
from reply_gate.policy_index import DEFAULT_POLICY_DIR, load_policy_documents

__all__ = [
    "DEFAULT_RETRIEVAL_LABELS_PATH",
    "RetrievalLabel",
    "load_retrieval_labels",
]

_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_RETRIEVAL_LABELS_PATH: Final = _ROOT / "data" / "retrieval_labels.jsonl"
_ROW_KEYS: Final = frozenset({"id", "relevant_evidence_ids", "note"})


@dataclass(frozen=True)
class RetrievalLabel:
    """골든셋 문의 1건의 정책 검색 정답 집합."""

    id: str
    relevant_evidence_ids: frozenset[str]
    note: str


def load_retrieval_labels(
    path: Path = DEFAULT_RETRIEVAL_LABELS_PATH,
    *,
    golden_set_path: Path = DEFAULT_GOLDEN_SET_PATH,
    policy_dir: Path = DEFAULT_POLICY_DIR,
) -> tuple[RetrievalLabel, ...]:
    """검색 정답 라벨을 읽는다.

    교차검증 대상은 **호출자가 실제로 쓸** 골든셋과 정책 코퍼스다. 기본 경로로만 검사하면
    다른 코퍼스로 평가할 때 존재하지 않는 조항이 정답으로 남아 recall 이 조용히 0 으로
    계상된다(fail-closed 가 fail-open 이 되는 경로).
    """
    labels: list[RetrievalLabel] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                raw: object = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} JSON 파싱 실패: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{lineno} 각 줄은 JSON 객체여야 한다")
            row = cast(dict[str, object], raw)
            row_keys = set(row)
            if row_keys != _ROW_KEYS:
                missing = ", ".join(sorted(_ROW_KEYS - row_keys)) or "없음"
                extra = ", ".join(sorted(row_keys - _ROW_KEYS)) or "없음"
                raise ValueError(
                    f"{path}:{lineno} 행 키가 계약과 다르다(누락={missing}, 추가={extra})"
                )
            label_id = row["id"]
            if not isinstance(label_id, str) or not label_id.strip():
                raise ValueError(f"{path}:{lineno} id는 비어 있지 않은 문자열이어야 한다")
            if label_id in seen:
                raise ValueError(f"라벨 ID가 중복된다: {label_id}")
            seen.add(label_id)

            raw_evidence_ids = row["relevant_evidence_ids"]
            if not isinstance(raw_evidence_ids, list):
                raise ValueError(f"{path}:{lineno} relevant_evidence_ids는 배열이어야 한다")
            evidence_ids: list[str] = []
            evidence_seen: set[str] = set()
            for value in raw_evidence_ids:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{path}:{lineno} 근거 ID는 비어 있지 않은 문자열이어야 한다")
                if value in evidence_seen:
                    raise ValueError(f"{path}:{lineno} 근거 ID가 중복된다: {value}")
                evidence_seen.add(value)
                evidence_ids.append(value)

            note = row["note"]
            if not isinstance(note, str):
                raise ValueError(f"{path}:{lineno} note는 문자열이어야 한다")
            labels.append(
                RetrievalLabel(
                    id=label_id,
                    relevant_evidence_ids=frozenset(evidence_ids),
                    note=note,
                )
            )
    golden_ids = {case.id for case in load_golden_set(golden_set_path)}
    unknown_ids = {label.id for label in labels} - golden_ids
    if unknown_ids:
        raise ValueError(f"골든셋에 없는 ID가 검색 라벨에 있다: {', '.join(sorted(unknown_ids))}")
    missing_ids = golden_ids - {label.id for label in labels}
    if missing_ids:
        raise ValueError(f"검색 라벨에서 빠진 골든셋 ID가 있다: {', '.join(sorted(missing_ids))}")
    policy_ids = {
        chunk.evidence_id
        for document in load_policy_documents(policy_dir)
        for chunk in document.chunks
    }
    unknown_evidence_ids = {
        evidence_id
        for label in labels
        for evidence_id in label.relevant_evidence_ids
        if evidence_id not in policy_ids
    }
    if unknown_evidence_ids:
        raise ValueError(
            "정책 문서에 없는 근거 ID가 검색 라벨에 있다: "
            + ", ".join(sorted(unknown_evidence_ids))
        )
    return tuple(labels)
