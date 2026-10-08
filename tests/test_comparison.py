"""Authored for recipient execution; no tests were run on the author's Mac."""

import copy

import pytest

from fake_dynamics.comparison import audit
from fake_dynamics.config import Config


def manifest():
    cfg = Config(per_device_batch_size=2, gradient_accumulation_steps=32).as_dict()
    return {
        "config": cfg,
        "world_size": 2,
        "global_batch_size": 128,
        "prompt_sha256": "train-prompts",
        "scheduler": {"prediction_type": "epsilon"},
        "training_sha256": "trained-model",
        "provenance": {"generator": {"sha256": "init-G"}, "fake": {"sha256": "init-F"}},
        "environment": {
            "code_revision": "same-commit",
            "code_dirty": False,
            "versions": {"torch": "2.6.0"},
        },
        "state": {
            "generator_updates": 1500,
            "fake_updates": 7500,
            "generator_attempts": 1500,
            "fake_attempts": 7500,
            "fake_in_cycle": 0,
            "elapsed_training_seconds": 100,
        },
    }


def report():
    return {
        "generation": {
            "checkpoint_sha256": "trained-model",
            "complete": True,
            "count": 10000,
            "weights": "ema",
            "prompts_sha256": "eval-prompts",
            "seed_base": 0,
            "seed_policy": "per_sample_seed_plus_prompt_index",
            "batch_size": 1,
            "resolution": 1024,
            "sampler": "four-step",
            "external_cfg": False,
            "watermark": False,
            "quantization": "uint8",
            "scheduler": {"prediction_type": "epsilon"},
            "environment": {"versions": {"torch": "2.6.0"}},
        },
        "reference": {"tree_sha256": "reference-images"},
        "upstream_commit": "evaluator",
        "evaluator_sha256": "code",
        "fid_feature_extractor": "inception",
        "resize": "512",
        "pilot": False,
        "environment": {"versions": {"torch": "2.6.0"}},
        "metrics": {"fid": {"value": 20, "scale": "raw", "source": "evaluator"}},
    }


def test_declared_sweep_detects_unintended_lr_change():
    a, b = manifest(), manifest()
    b["config"]["consistency_weight_max"] = 0.3
    assert not audit(a, b, ["consistency_weight_max"])["blocking_issues"]
    b["config"]["fake_lr"] = 2 * a["config"]["fake_lr"]
    assert "Mismatch: config.fake_lr" in audit(a, b, ["consistency_weight_max"])["blocking_issues"]


@pytest.mark.parametrize("change", ["budget", "data", "init", "skip", "missing", "dirty"])
def test_training_mismatch_blocks_comparison(change):
    a, b = manifest(), manifest()
    if change == "budget":
        b["state"]["generator_updates"] = 5000
    elif change == "data":
        b["prompt_sha256"] = "different-data"
    elif change == "init":
        b["provenance"]["fake"]["sha256"] = "different-F"
    elif change == "skip":
        b["state"]["generator_attempts"] += 1
    elif change == "missing":
        del a["config"]["generator_lr"]
        del b["config"]["generator_lr"]
    else:
        a["environment"]["code_dirty"] = b["environment"]["code_dirty"] = True
    assert audit(a, b)["blocking_issues"]


def test_training_match_keeps_gpu_cost_difference_visible():
    a, b = manifest(), manifest()
    b["state"]["elapsed_training_seconds"] = 200
    result = audit(a, b)
    assert not result["blocking_issues"] and not result["evaluation_audited"]
    assert result["costs"][1]["training_gpu_hours"] == 2 * result["costs"][0]["training_gpu_hours"]


def test_scores_can_differ_but_eval_seed_and_checkpoint_must_match_protocol():
    a, b = manifest(), manifest()
    b["training_sha256"] = "winner-model"
    ra, rb = report(), report()
    rb["generation"]["checkpoint_sha256"] = "winner-model"
    rb["metrics"]["fid"]["value"] = 17
    assert not audit(a, b, left_report=ra, right_report=rb)["blocking_issues"]
    changed = copy.deepcopy(rb)
    changed["generation"]["seed_base"] = 42
    assert audit(a, b, left_report=ra, right_report=changed)["blocking_issues"]
    changed = copy.deepcopy(rb)
    changed["generation"]["checkpoint_sha256"] = "wrong-model"
    assert audit(a, b, left_report=ra, right_report=changed)["blocking_issues"]


def test_missing_metric_on_one_side_and_metric_revision_difference_are_exposed():
    a, b = manifest(), manifest()
    ra, rb = report(), report()
    ra["metrics"]["hps_v3"] = {"value": 0.2, "model_revision": "v1", "scale": "raw"}
    assert audit(a, b, left_report=ra, right_report=rb)["blocking_issues"]
    rb["metrics"]["hps_v3"] = {"value": 0.2, "model_revision": "v2", "scale": "raw"}
    assert audit(a, b, left_report=ra, right_report=rb)["blocking_issues"]
