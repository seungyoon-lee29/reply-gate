"""정책 검색 정답 라벨 로더의 fail-closed 계약을 검증한다."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reply_gate.evaluation import DEFAULT_GOLDEN_SET_PATH
from reply_gate.policy_index import DEFAULT_POLICY_DIR
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
        "G18": frozenset({"policy:refund:2-6", "policy:refund:2-4"}),
        "G19": frozenset({"policy:support:4-2"}),
        "G20": frozenset({"policy:support:4-1", "policy:support:4-2"}),
        "G21": frozenset(),
        "G22": frozenset(),
        "G23": frozenset(),
        "G24": frozenset(),
        "G25": frozenset({"policy:shipping:1-5"}),
        "G26": frozenset({"policy:shipping:1-1"}),
        "G27": frozenset({"policy:shipping:1-5"}),
        "G28": frozenset(),
        "G29": frozenset({"policy:support:4-5"}),
        "G30": frozenset({"policy:support:4-5"}),
    }
    assert all(label.note for label in labels)


def test_직접_답하거나_제한할_정책이_없는_다섯_건은_빈_정답이다() -> None:
    labels = {label.id: label for label in load_retrieval_labels()}

    empty_ids = {case_id for case_id, label in labels.items() if not label.relevant_evidence_ids}
    assert empty_ids == {"G21", "G22", "G23", "G24", "G28"}


def test_직접_관련은_정면_조항과_그것과_상충하는_조항까지다() -> None:
    """정답 경계는 두 겹이다(결정 0010) — 정면으로 답하는 조항 + 같은 사안의 상충 조항.

    표현만 다른 G16·G17 이 같은 정답을 갖는 것이 이 규칙의 시금석이다. 전화만 묻는 문의에
    이메일 접수 채널(4-2)까지 넣으면 정답을 하나 찾고도 r@1 이 0.5 로 기록돼 표적 케이스
    판독이 흐려진다. 반대로 G18 은 2-6 과 상충하는 2-4 를 함께 세야 한다 — 검색이 둘을
    올려야 L2 가 모순을 잡는데, 라벨이 2-6 만이면 그 동작이 precision 을 깎는다.
    """
    labels = {label.id: label for label in load_retrieval_labels()}

    assert labels["G16"].relevant_evidence_ids == frozenset({"policy:support:4-1"})
    assert labels["G17"].relevant_evidence_ids == labels["G16"].relevant_evidence_ids
    assert labels["G19"].relevant_evidence_ids == frozenset({"policy:support:4-2"})
    assert labels["G20"].relevant_evidence_ids == frozenset(
        {"policy:support:4-1", "policy:support:4-2"}
    )
    assert labels["G18"].relevant_evidence_ids == frozenset(
        {"policy:refund:2-6", "policy:refund:2-4"}
    )
    assert labels["G25"].relevant_evidence_ids == frozenset({"policy:shipping:1-5"})
    assert labels["G29"].relevant_evidence_ids == frozenset({"policy:support:4-5"})
    assert labels["G30"].relevant_evidence_ids == frozenset({"policy:support:4-5"})
    assert "처리 전제" in labels["G25"].note
    assert "공개 제한" in labels["G30"].note


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


def test_교차검증은_호출자가_실제로_쓸_코퍼스를_기준으로_한다(tmp_path: Path) -> None:
    """기본 경로로만 검사하면 축소된 코퍼스로 돌릴 때 recall 이 조용히 0 으로 계상된다."""
    reduced = tmp_path / "policies"
    reduced.mkdir()
    source = DEFAULT_POLICY_DIR / "shipping.md"
    (reduced / "shipping.md").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    # 기본 코퍼스 기준으로는 통과하는 라벨이,
    assert len(load_retrieval_labels(DEFAULT_RETRIEVAL_LABELS_PATH)) == 30

    # 실제로 검색할 코퍼스에 없는 조항을 정답으로 담고 있으면 로드 시점에 거부된다.
    with pytest.raises(ValueError, match="정책 문서에 없는 근거 ID"):
        load_retrieval_labels(DEFAULT_RETRIEVAL_LABELS_PATH, policy_dir=reduced)


def test_교차검증은_호출자가_준_골든셋을_기준으로_한다(tmp_path: Path) -> None:
    golden = tmp_path / "golden.jsonl"
    rows = [
        json.loads(line)
        for line in DEFAULT_GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    golden.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows[:5]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="골든셋에 없는 ID"):
        load_retrieval_labels(DEFAULT_RETRIEVAL_LABELS_PATH, golden_set_path=golden)
