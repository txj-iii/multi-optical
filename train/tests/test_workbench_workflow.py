from pathlib import Path

from train.workbench_workflow import (
    build_manifest_scene_selection,
    discover_latest_capture_group,
    next_sample_id,
)


def test_next_sample_id_advances_from_existing_workspace(tmp_path: Path) -> None:
    for scene_id in ("SAMPLE_009", "SAMPLE_010", "SAMPLE_015"):
        (tmp_path / scene_id).mkdir()

    assert next_sample_id(tmp_path) == "SAMPLE_016"


def test_discover_latest_capture_group_prefers_newest_directory_with_five_bmps(tmp_path: Path) -> None:
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    older.mkdir()
    newer.mkdir()
    for index in range(5):
        (older / f"old_{index}.bmp").write_bytes(b"old")
        target = newer / f"new_{index}.bmp"
        target.write_bytes(b"new")
    older.touch()
    newer.touch()

    selected = discover_latest_capture_group(tmp_path)

    assert [path.name for path in selected] == [f"new_{index}.bmp" for index in range(4, -1, -1)]


def test_build_manifest_scene_selection_appends_workflow_scene_and_override() -> None:
    scene_ids, overrides, pigment_overrides = build_manifest_scene_selection(
        base_scene_ids=["SAMPLE_036", "SAMPLE_037"],
        workflow_records={
            "SAMPLE_050": {
                "prediction_root": "D:/multi-optical/train/experiments/five_band_predictions/task_specific/ui_workbench_review",
                "pigment_root": "D:/multi-optical/train/experiments/five_band_predictions/task_specific/ui_workbench_pigment",
                "status": "pending_review",
            }
        },
    )

    assert scene_ids == ["SAMPLE_036", "SAMPLE_037", "SAMPLE_050"]
    assert overrides == {
        "SAMPLE_050": Path("D:/multi-optical/train/experiments/five_band_predictions/task_specific/ui_workbench_review")
    }
    assert pigment_overrides == {
        "SAMPLE_050": Path("D:/multi-optical/train/experiments/five_band_predictions/task_specific/ui_workbench_pigment")
    }


def test_build_manifest_scene_selection_keeps_main_version_scene_on_base_root() -> None:
    scene_ids, overrides, pigment_overrides = build_manifest_scene_selection(
        base_scene_ids=["SAMPLE_055"],
        workflow_records={
            "SAMPLE_055": {
                "prediction_root": "D:/multi-optical/train/experiments/five_band_predictions/task_specific/ui_workbench_candidate/SAMPLE_055",
                "pigment_root": "D:/multi-optical/train/experiments/five_band_predictions/task_specific/ui_workbench_pigment",
                "status": "approved",
            }
        },
    )

    assert scene_ids == ["SAMPLE_055"]
    assert overrides == {}
    assert pigment_overrides == {
        "SAMPLE_055": Path("D:/multi-optical/train/experiments/five_band_predictions/task_specific/ui_workbench_pigment")
    }


def test_build_manifest_scene_selection_uses_validation_mode_for_base_scene() -> None:
    scene_ids, overrides, pigment_overrides = build_manifest_scene_selection(
        base_scene_ids=["SAMPLE_055"],
        workflow_records={
            "SAMPLE_055": {
                "prediction_root": "D:/multi-optical/train/experiments/five_band_predictions/task_specific/ui_workbench_candidate/SAMPLE_055",
                "display_prediction_mode": "validation",
                "status": "approved",
            }
        },
    )

    assert scene_ids == ["SAMPLE_055"]
    assert overrides == {}
    assert pigment_overrides == {}


def test_build_manifest_scene_selection_uses_workflow_mode_for_base_scene_override() -> None:
    scene_ids, overrides, pigment_overrides = build_manifest_scene_selection(
        base_scene_ids=["SAMPLE_055"],
        workflow_records={
            "SAMPLE_055": {
                "prediction_root": "D:/multi-optical/train/experiments/five_band_predictions/task_specific/ui_workbench_candidate/SAMPLE_055",
                "display_prediction_mode": "workflow",
                "status": "approved",
            }
        },
    )

    assert scene_ids == ["SAMPLE_055"]
    assert overrides == {
        "SAMPLE_055": Path("D:/multi-optical/train/experiments/five_band_predictions/task_specific/ui_workbench_candidate/SAMPLE_055")
    }
    assert pigment_overrides == {}


def test_generate_review_predictions_export_to_main_validation_root_with_paint_override(tmp_path: Path, monkeypatch) -> None:
    from train import workbench_workflow as workflow
    from PIL import Image
    import numpy as np

    calls: list[dict[str, object]] = []
    compose_calls: list[dict[str, object]] = []

    scene_root = tmp_path / "scenes" / "SAMPLE_050"
    scene_root.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(scene_root / "preview.png")

    def fake_export_five_band_predictions(*args, **kwargs):
        calls.append(kwargs)
        output_root = Path(kwargs["output_root"])
        scene_id = kwargs["scene_ids"][0]
        scene_dir = output_root / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        for head_name in kwargs.get("export_heads", ("paint", "pollution", "aging")):
            (scene_dir / f"{head_name}_pred.png").write_bytes(b"pred")
            (scene_dir / f"{head_name}_overlay.png").write_bytes(b"overlay")
        return [scene_dir]

    monkeypatch.setattr(workflow, "SCENES_ROOT", tmp_path / "scenes")
    monkeypatch.setattr(workflow, "MAIN_PREDICTION_ROOT", tmp_path / "main_predictions")
    monkeypatch.setattr(workflow, "TMP_ROOT", tmp_path / "tmp")
    monkeypatch.setattr(workflow, "export_five_band_predictions", fake_export_five_band_predictions)
    monkeypatch.setattr(workflow, "compose_predictions", lambda **kwargs: compose_calls.append(kwargs))

    result = workflow._generate_review_predictions("SAMPLE_050")

    assert result == tmp_path / "main_predictions"
    assert len(calls) == 2
    validation_call, paint_call = calls
    assert validation_call["checkpoint_path"] == workflow.VALIDATION_CHECKPOINT_PATH
    assert validation_call["output_root"] == tmp_path / "main_predictions"
    assert validation_call["scene_ids"] == ["SAMPLE_050"]
    assert validation_call["composition_mode"] == "conflict_resolved"
    assert validation_call["pollution_shape_filter"] is True
    assert validation_call["pollution_threshold"] == 0.35
    assert paint_call["checkpoint_path"] == workflow.PAINT_OVERRIDE_CHECKPOINT_PATH
    assert paint_call["export_heads"] == ("paint",)
    assert paint_call["output_root"] == tmp_path / "tmp" / "SAMPLE_050" / "paint_override"
    assert paint_call["composition_mode"] == "independent"
    assert compose_calls[0]["paint_root"] == tmp_path / "tmp" / "SAMPLE_050" / "paint_override"
    assert compose_calls[0]["pollution_root"] == tmp_path / "main_predictions"
    assert compose_calls[0]["aging_root"] == tmp_path / "main_predictions"
    assert compose_calls[0]["output_root"] == tmp_path / "main_predictions"


def test_generate_review_predictions_keeps_main_validation_root_for_legacy_scene(tmp_path: Path, monkeypatch) -> None:
    from train import workbench_workflow as workflow

    calls: list[dict[str, object]] = []
    compose_calls: list[dict[str, object]] = []

    def fake_export_five_band_predictions(*args, **kwargs):
        calls.append(kwargs)
        output_root = Path(kwargs["output_root"])
        scene_id = kwargs["scene_ids"][0]
        scene_dir = output_root / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        for head_name in kwargs.get("export_heads", ("paint", "pollution", "aging")):
            (scene_dir / f"{head_name}_pred.png").write_bytes(b"pred")
            (scene_dir / f"{head_name}_overlay.png").write_bytes(b"overlay")
        return [scene_dir]

    monkeypatch.setattr(workflow, "SCENES_ROOT", tmp_path / "scenes")
    monkeypatch.setattr(workflow, "MAIN_PREDICTION_ROOT", tmp_path / "main_predictions")
    monkeypatch.setattr(workflow, "TMP_ROOT", tmp_path / "tmp")
    monkeypatch.setattr(workflow, "export_five_band_predictions", fake_export_five_band_predictions)
    monkeypatch.setattr(workflow, "compose_predictions", lambda **kwargs: compose_calls.append(kwargs))

    result = workflow._generate_review_predictions("SAMPLE_047")

    assert result == tmp_path / "main_predictions"
    assert len(calls) == 1
    assert calls[0]["checkpoint_path"] == workflow.VALIDATION_CHECKPOINT_PATH
    assert compose_calls == []


def test_approve_scene_marks_annotation_accepted_without_training(tmp_path: Path, monkeypatch) -> None:
    from train import workbench_workflow as workflow

    state_path = tmp_path / "workflow_state.json"
    scenes_root = tmp_path / "scenes"
    sample_root = scenes_root / "SAMPLE_052"
    sample_root.mkdir(parents=True)
    state_path.write_text('{"samples": {"SAMPLE_052": {"status": "pending_review", "prediction_root": "review", "annotation_decision": "saved"}}, "updated_at": null}', encoding='utf-8')

    monkeypatch.setattr(workflow, "SCENES_ROOT", scenes_root)
    monkeypatch.setattr(workflow, "refresh_manifest", lambda state: {"samples": []})

    result = workflow.approve_scene("SAMPLE_052", state_path=state_path)

    assert result["record"]["status"] == "approved"
    assert result["record"]["stage"] == "annotation_approved"
    assert result["record"]["annotation_decision"] == "accepted"
    assert result["record"]["prediction_root"] == "review"
    assert result["record"]["display_prediction_mode"] == "validation"
    assert "patch_root" not in result["record"]
    assert "train_dir" not in result["record"]
    assert "last_train_stdout" not in result["record"]
    assert "last_predict_stdout" not in result["record"]


def test_approve_scene_clears_adjust_target_when_accepting_annotation(tmp_path: Path, monkeypatch) -> None:
    from train import workbench_workflow as workflow

    state_path = tmp_path / "workflow_state.json"
    scenes_root = tmp_path / "scenes"
    sample_root = scenes_root / "SAMPLE_047"
    sample_root.mkdir(parents=True)
    state_path.write_text('{"samples": {"SAMPLE_047": {"status": "needs_adjustment", "prediction_root": "review", "annotation_decision": "saved", "adjust_target": "paint"}}, "updated_at": null}', encoding='utf-8')

    monkeypatch.setattr(workflow, "SCENES_ROOT", scenes_root)
    monkeypatch.setattr(workflow, "refresh_manifest", lambda state: {"samples": []})

    result = workflow.approve_scene("SAMPLE_047", state_path=state_path)

    assert result["record"]["status"] == "approved"
    assert result["record"]["adjust_target"] is None
    assert result["record"]["annotation_decision"] == "accepted"


def test_refresh_manifest_includes_default_and_workflow_pigment_roots(tmp_path: Path, monkeypatch) -> None:
    from train import workbench_workflow as workflow

    captured: dict[str, object] = {}

    monkeypatch.setattr(workflow, 'SCENES_ROOT', tmp_path / 'scenes')
    monkeypatch.setattr(workflow, 'MANIFEST_PATH', tmp_path / 'ui' / 'workbench_manifest.json')
    monkeypatch.setattr(workflow, 'MAIN_PREDICTION_ROOT', tmp_path / 'main_predictions')
    monkeypatch.setattr(workflow, 'SAMPLE_RECORD_PATH', tmp_path / 'sample_record.md')
    monkeypatch.setattr(workflow, 'PAINT_OVERRIDE_PREDICTION_ROOT', Path('D:/multi-optical/train/experiments/five_band_predictions/task_specific/retune_9_scene3647_agingmix_4849_v1_selected'))
    monkeypatch.setattr(workflow, '_discover_base_scene_ids', lambda prediction_root=None: ['SAMPLE_048', 'SAMPLE_049'])
    monkeypatch.setattr(workflow, '_load_version_provenance', lambda: None)

    def fake_export_workbench_manifest(**kwargs):
        captured.update(kwargs)
        return {'samples': []}

    monkeypatch.setattr(workflow, 'export_workbench_manifest', fake_export_workbench_manifest)

    workflow.refresh_manifest(
        {
            'samples': {
                'SAMPLE_050': {
                    'prediction_root': 'D:/multi-optical/train/experiments/five_band_predictions/task_specific/ui_workbench_review',
                    'pigment_root': 'D:/multi-optical/train/experiments/five_band_predictions/task_specific/ui_workbench_pigment',
                }
            }
        }
    )

    assert captured['scene_prediction_roots']['SAMPLE_050'] == Path('D:/multi-optical/train/experiments/five_band_predictions/task_specific/ui_workbench_review')
    assert captured['scene_pigment_roots']['SAMPLE_048'] == Path('D:/multi-optical/train/experiments/five_band_predictions/task_specific/retune_9_scene3647_agingmix_4849_v1_selected')
    assert captured['scene_pigment_roots']['SAMPLE_049'] == Path('D:/multi-optical/train/experiments/five_band_predictions/task_specific/retune_9_scene3647_agingmix_4849_v1_selected')
    assert captured['scene_pigment_roots']['SAMPLE_050'] == Path('D:/multi-optical/train/experiments/five_band_predictions/task_specific/ui_workbench_pigment')


def test_load_workflow_state_strips_legacy_candidate_training_fields(tmp_path: Path, monkeypatch) -> None:
    from train import workbench_workflow as workflow

    state_path = tmp_path / "workflow_state.json"
    state_path.write_text(
        """
{
  "samples": {
    "SAMPLE_050": {
      "status": "approved",
      "stage": "candidate_ready",
      "prediction_root": "D:/multi-optical/train/experiments/five_band_predictions/task_specific/ui_workbench_candidate/SAMPLE_050",
      "patch_root": "D:/multi-optical/train/five_band_patches/ui_workbench_candidate/SAMPLE_050/train",
      "train_dir": "D:/multi-optical/train/experiments/five_band_train/task_specific/ui_workbench_candidate/SAMPLE_050",
      "last_train_stdout": "legacy train log",
      "last_predict_stdout": "legacy predict log",
      "annotation_decision": "accepted"
    },
    "SAMPLE_051": {
      "status": "pending_review",
      "stage": "candidate_ready",
      "prediction_root": "D:/multi-optical/train/experiments/five_band_predictions/task_specific/ui_workbench_candidate/SAMPLE_051",
      "patch_root": "D:/multi-optical/train/five_band_patches/ui_workbench_candidate/SAMPLE_051/train",
      "train_dir": "D:/multi-optical/train/experiments/five_band_train/task_specific/ui_workbench_candidate/SAMPLE_051",
      "last_train_stdout": "legacy train log",
      "last_predict_stdout": "legacy predict log",
      "annotation_decision": "pending"
    }
  },
  "updated_at": null
}
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(workflow, "MAIN_PREDICTION_ROOT", tmp_path / "validation_predictions")

    payload = workflow.load_workflow_state(state_path)

    approved = payload["samples"]["SAMPLE_050"]
    pending = payload["samples"]["SAMPLE_051"]
    assert approved["prediction_root"] == str(tmp_path / "validation_predictions")
    assert approved["stage"] == "annotation_approved"
    assert "patch_root" not in approved
    assert "train_dir" not in approved
    assert "last_train_stdout" not in approved
    assert "last_predict_stdout" not in approved

    assert pending["prediction_root"] == str(tmp_path / "validation_predictions")
    assert pending["stage"] == "annotation_saved"
    assert "patch_root" not in pending
    assert "train_dir" not in pending
    assert "last_train_stdout" not in pending
    assert "last_predict_stdout" not in pending


def test_load_workflow_state_promotes_approved_legacy_candidate_stage(tmp_path: Path, monkeypatch) -> None:
    from train import workbench_workflow as workflow

    state_path = tmp_path / "workflow_state.json"
    state_path.write_text(
        """
{
  "samples": {
    "SAMPLE_052": {
      "status": "approved",
      "stage": "candidate_ready",
      "prediction_root": "D:/multi-optical/train/experiments/five_band_predictions/task_specific/ui_workbench_candidate/SAMPLE_052",
      "annotation_decision": "pending"
    }
  },
  "updated_at": null
}
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(workflow, "MAIN_PREDICTION_ROOT", tmp_path / "validation_predictions")

    payload = workflow.load_workflow_state(state_path)
    record = payload["samples"]["SAMPLE_052"]

    assert record["annotation_decision"] == "accepted"
    assert record["stage"] == "annotation_approved"


def test_load_workflow_state_promotes_approved_saved_stage(tmp_path: Path, monkeypatch) -> None:
    from train import workbench_workflow as workflow

    state_path = tmp_path / "workflow_state.json"
    state_path.write_text(
        """
{
  "samples": {
    "SAMPLE_052": {
      "status": "approved",
      "stage": "annotation_saved",
      "prediction_root": "D:/multi-optical/train/experiments/five_band_predictions/task_specific/validation_v10_balanced_softcomp_5056_pollthr035",
      "annotation_decision": "accepted"
    }
  },
  "updated_at": null
}
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(workflow, "MAIN_PREDICTION_ROOT", tmp_path / "validation_predictions")

    payload = workflow.load_workflow_state(state_path)
    record = payload["samples"]["SAMPLE_052"]

    assert record["stage"] == "annotation_approved"
