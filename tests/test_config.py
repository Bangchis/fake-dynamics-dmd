import pytest

from fake_dynamics.config import Config


def test_unknown_config_field_is_not_silently_ignored(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("fake_cfg: 8\n")
    with pytest.raises(ValueError, match="Unknown config"):
        Config.load(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("gradient_accumulation_steps", 0),
        ("gradient_accumulation_steps", 1.5),
        ("gradient_accumulation_steps", True),
        ("distributed_strategy", "fsdp"),
        ("prediction_type", "v_prediction"),
        ("generator_anchors", (999, 0)),
    ],
)
def test_unimplemented_or_incompatible_features_fail(field, value):
    config = Config()
    setattr(config, field, value)
    with pytest.raises(ValueError):
        config.validate()


def test_reference_config_leaves_machine_budget_unset():
    config = Config.load("configs/sdxl.yaml")
    assert "per_device_batch_size" in config.missing_runtime_fields()
    assert "total_generator_updates" in config.missing_runtime_fields()
