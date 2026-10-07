"""Recipient-only provenance checks; no real TensorBoard installation is required here."""

import json
import sys
import types

import pytest

from fake_dynamics.telemetry import log_evaluation


class Writer:
    records = []

    def __init__(self, path):
        self.records.append(("open", path))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def add_scalar(self, name, value, step):
        self.records.append(("scalar", name, value, step))

    def add_text(self, name, value, step):
        self.records.append(("text", name, value, step))


@pytest.fixture
def writer(monkeypatch):
    Writer.records = []
    monkeypatch.setitem(
        sys.modules, "torch.utils.tensorboard", types.SimpleNamespace(SummaryWriter=Writer)
    )
    return Writer


def test_completed_report_imports_only_measured_metrics_at_declared_g_step(tmp_path, writer):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "metrics": {
                    "fid": {"value": 20, "scale": "raw", "source": "test"},
                    "hps_v3": {"value": None, "status": "not_available"},
                }
            }
        )
    )
    log_evaluation(report, tmp_path / "run", 1500)
    scalars = [row for row in writer.records if row[0] == "scalar"]
    assert scalars == [("scalar", "evaluation/fid", 20, 1500)]
    assert any(row[0] == "text" and "not_available" in row[2] for row in writer.records)


@pytest.mark.parametrize(
    "metric",
    [
        {"value": float("nan"), "scale": "raw"},
        {"value": 1.0},
        {"value": 1.0, "scale": "raw", "library_version": "test"},
    ],
)
def test_invalid_metric_report_writes_no_partial_dashboard(tmp_path, writer, metric):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"metrics": {"fid": {"value": 20, "scale": "raw"}, "image_reward": metric}})
    )
    with pytest.raises(ValueError):
        log_evaluation(report, tmp_path / "run", 1500)
    assert writer.records == []
