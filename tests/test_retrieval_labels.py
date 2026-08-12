"""정책 검색 정답 라벨 로더의 fail-closed 계약을 검증한다."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reply_gate.retrieval_labels import (
    DEFAULT_RETRIEVAL_LABELS_PATH,
    load_retrieval_labels,
)


def _rows() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in DEFAULT_RETRIEVAL_LABELS_PATH.read_text(encoding="utf-8").splitlines()
    ]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_저장소_검색_라벨_30건을_정책_원문_기준으로_읽는다() -> None:
    labels = load_retrieval_labels(DEFAULT_RETRIEVAL_LABELS_PATH)
    by_id = {label.id: label.relevant_evidence_ids for label in labels}

    assert len(labels) == 30
    assert by_id == {
        "G01": frozenset({"policy:refund:2-1", "policy:refund:2-2"}),
        "G02": frozenset({"policy:shipping:1-3", "policy:shipping:1-4"}),
        "G03": frozenset({"policy:exchange:3-2"}),
        "G04": frozenset({"policy:support:4-6"}),
        "G05": frozenset({"policy:support:4-3"}),
        "G06": frozenset({"policy:shipping:1-6"}),
        "G07": frozenset({"policy:support:4-7"}),
        "G08": frozenset({"policy:shipping:1-7"}),
        "G09": frozenset({"policy:shipping:1-5"}),
        "G10": frozenset({"policy:shipping:1-5"}),
        "G11": frozenset({"policy:shipping:1-1"}),
        "G12": frozenset({"policy:refund:2-4"}),
        "G13": frozenset({"policy:refund:2-4"}),
        "G14": frozenset({"policy:exchange:3-5"}),
        "G15": frozenset({"policy:refund:2-5"}),
        "G16": frozenset({"policy:support:4-1"}),
        "G17": frozenset({"policy:support:4-1"}),
        "G18": frozenset({"policy:refund:2-6"}),
        "G19": frozenset({"policy:support:4-2"}),
        "G20": frozenset({"policy:support:4-1", "policy:support:4-2"}),
        "G21": frozenset(),
        "G22": frozenset(),
        "G23": frozenset(),
        "G24": frozenset(),
        "G25": frozenset({"policy:shipping:1-5"}),
        "G26": frozenset({"policy:shipping:1-1"}),
        "G27": frozenset({"policy:shipping:1-5"}),
        "G28": frozenset({"policy:refund:2-4"}),
        "G29": frozenset({"policy:support:4-5"}),
        "G30": frozenset({"policy:support:4-5"}),
    }
    assert all(label.note for label in labels)


def test_무근거_문의_네_건의_빈_정답_집합을_보존한다() -> None:
    labels = {label.id: label for label in load_retrieval_labels()}

    empty_ids = {case_id for case_id, label in labels.items() if not label.relevant_evidence_ids}
    assert empty_ids == {"G21", "G22", "G23", "G24"}


def test_골든셋에_없는_ID가_라벨에_있으면_거부한다(tmp_path: Path) -> None:
    path = tmp_path / "unknown-id.jsonl"
    rows = _rows()
    rows.append({"id": "G99", "relevant_evidence_ids": [], "note": "골든셋에 없음"})
    _write_rows(path, rows)

    with pytest.raises(ValueError, match=r"골든셋에 없는 ID.*G99"):
        load_retrieval_labels(path)


def test_골든셋_ID가_라벨에서_빠지면_거부한다(tmp_path: Path) -> None:
    path = tmp_path / "missing-id.jsonl"
    rows = [row for row in _rows() if row["id"] != "G30"]
    _write_rows(path, rows)

    with pytest.raises(ValueError, match=r"검색 라벨에서 빠진 골든셋 ID.*G30"):
        load_retrieval_labels(path)


def test_정책_문서에_없는_근거_ID면_거부한다(tmp_path: Path) -> None:
    path = tmp_path / "unknown-evidence.jsonl"
    rows = _rows()
    rows[0]["relevant_evidence_ids"] = ["policy:refund:9-9"]
    _write_rows(path, rows)

    with pytest.raises(ValueError, match=r"정책 문서에 없는 근거 ID.*policy:refund:9-9"):
        load_retrieval_labels(path)


def test_라벨_행에_계약_밖의_키가_있으면_거부한다(tmp_path: Path) -> None:
    path = tmp_path / "extra-key.jsonl"
    rows = _rows()
    rows[0]["category"] = "normal"
    _write_rows(path, rows)

    with pytest.raises(ValueError, match=r"키.*category"):
        load_retrieval_labels(path)


def test_라벨_ID가_중복되면_거부한다(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-id.jsonl"
    rows = _rows()
    rows[-1]["id"] = "G01"
    _write_rows(path, rows)

    with pytest.raises(ValueError, match=r"라벨 ID가 중복.*G01"):
        load_retrieval_labels(path)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("id", 1, "id는 비어 있지 않은 문자열"),
        ("relevant_evidence_ids", "policy:refund:2-1", "배열이어야"),
        ("relevant_evidence_ids", [1], "근거 ID는 비어 있지 않은 문자열"),
        ("note", None, "note는 문자열"),
    ],
)
def test_라벨_필드_타입이_계약과_다르면_거부한다(
    tmp_path: Path, field: str, bad_value: object, message: str
) -> None:
    path = tmp_path / f"bad-{field}.jsonl"
    rows = _rows()
    rows[0][field] = bad_value
    _write_rows(path, rows)

    with pytest.raises(ValueError, match=message):
        load_retrieval_labels(path)


def test_한_라벨_안의_근거_ID가_중복되면_거부한다(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-evidence.jsonl"
    rows = _rows()
    rows[0]["relevant_evidence_ids"] = ["policy:refund:2-1", "policy:refund:2-1"]
    _write_rows(path, rows)

    with pytest.raises(ValueError, match=r"근거 ID가 중복.*policy:refund:2-1"):
        load_retrieval_labels(path)
