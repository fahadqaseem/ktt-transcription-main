from __future__ import annotations

from pathlib import Path

from ktt_ocr_eval.core.artifacts import read_predictions
from ktt_ocr_eval.core.schemas import DocumentInput, MethodConfig
from ktt_ocr_eval.orchestration.fingerprint import method_fingerprint
from ktt_ocr_eval.orchestration.runner import run_matrix, scan_completed_keys


def _doc(index: int) -> DocumentInput:
    return DocumentInput(document_id=f"doc{index}", source_path=Path(f"img{index}.png"))


def _method(
    config: dict | None = None,
    *,
    name: str = "ex",
    nickname: str | None = None,
) -> MethodConfig:
    return MethodConfig(
        name=name,
        adapter="example_provider",
        nickname=nickname,
        config=config or {},
    )


def test_fingerprint_depends_on_config_not_name_or_nickname() -> None:
    alpha = _method({"x": 1}, name="a", nickname="Alpha")
    beta = _method({"x": 1}, name="b", nickname="Beta")
    assert method_fingerprint(alpha) == method_fingerprint(beta)
    assert method_fingerprint(alpha) != method_fingerprint(_method({"x": 2}, name="a"))


def test_collect_attaches_provenance_per_record(tmp_path: Path) -> None:
    method = _method({"x": 1}, nickname="My Run")
    predictions_path = run_matrix([_doc(1)], [method], tmp_path)

    predictions = read_predictions(predictions_path)
    assert len(predictions) == 1
    collection = predictions[0].collection
    assert collection is not None
    assert collection.fingerprint == method_fingerprint(method)
    assert collection.adapter == "example_provider"
    assert collection.nickname == "My Run"
    assert collection.config == {"x": 1}


def test_resume_skips_completed_and_reruns_on_config_change(tmp_path: Path) -> None:
    method = _method({"x": 1})
    run_matrix([_doc(1), _doc(2)], [method], tmp_path)
    assert len(scan_completed_keys(tmp_path)) == 2

    rerun_path = run_matrix([_doc(1), _doc(2)], [method], tmp_path)
    assert read_predictions(rerun_path) == []  # all cells already collected -> no-op

    changed_path = run_matrix([_doc(1), _doc(2)], [_method({"x": 2})], tmp_path)
    assert len(read_predictions(changed_path)) == 2  # new fingerprint -> recollected


def test_each_run_writes_its_own_immutable_file(tmp_path: Path) -> None:
    run_matrix([_doc(1)], [_method({"x": 1})], tmp_path)
    run_matrix([_doc(2)], [_method({"x": 2})], tmp_path)
    assert len(sorted(tmp_path.glob("predictions-*.jsonl"))) == 2


def test_no_resume_recollects_everything(tmp_path: Path) -> None:
    method = _method({"x": 1})
    run_matrix([_doc(1)], [method], tmp_path)
    again = run_matrix([_doc(1)], [method], tmp_path, resume=False)
    assert len(read_predictions(again)) == 1
