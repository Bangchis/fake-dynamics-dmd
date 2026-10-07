"""Scalar/event logging without storing weights, gradients or every generated image."""

import json
import math
import time
from pathlib import Path


def scalar_fields(record, prefix=""):
    """Flatten numeric diagnostics, preserving absent sparse-bucket observations."""
    result = {}
    for key, value in record.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(scalar_fields(value, name + "/"))
        elif isinstance(value, (int, float)) or value is None:
            result[name] = value
    return result


class TensorBoardLogger:
    def __init__(self, config, rank):
        self.writer = None
        if config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise RuntimeError(
                    "Install this environment's logging extra: pip install -e '.[logging]'"
                ) from error
            if rank != 0:
                return
            self.writer = SummaryWriter(
                str(Path(config.output_dir) / "tensorboard"),
                flush_secs=config.tensorboard_flush_seconds,
            )
            self.writer.add_text(
                "run/config", "```json\n" + json.dumps(config.as_dict(), indent=2) + "\n```", 0
            )

    def record(self, event, record):
        if self.writer is None:
            return
        step = int(record["k_F"] if event == "fake_step" else record["k_G"])
        prefix = {
            "fake_step": "fake_by_kF",
            "generator_step": "generator_by_kG",
            "fixed_probe": "probe_by_kG",
            "performance": "performance_by_kG",
        }.get(event)
        if prefix is None:
            self.writer.add_text(
                "events/" + event, json.dumps(record, default=str), int(record["k_G"])
            )
            return
        for key, value in scalar_fields(record).items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                self.writer.add_scalar(prefix + "/" + key, value, step)

    def text(self, name, content, step=0):
        if self.writer is not None:
            self.writer.add_text(name, content, step)

    def image(self, name, image, step):
        if self.writer is not None:
            self.writer.add_image(name, image, step, dataformats="CHW")

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def event_time():
    return time.time()


def log_evaluation(report_path, run_dir, generator_updates):
    """Import an actual completed evaluation report into the SAME training dashboard."""
    from torch.utils.tensorboard import SummaryWriter

    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    if generator_updates < 0 or not isinstance(report.get("metrics"), dict):
        raise ValueError(
            "Evaluation report must include metrics records with value/scale/provenance"
        )
    values = {}
    for name, metric in report["metrics"].items():
        if not isinstance(metric, dict) or "value" not in metric:
            raise ValueError(f"Metric must be a record with value/scale/provenance: {name}")
        value = metric["value"]
        if value is None:
            continue
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not metric.get("scale")
        ):
            raise ValueError(f"Nonfinite/nonnumeric metric or missing scale: {name}")
        if (
            name in ("image_reward", "hps_v2_1", "hps_v3")
            and not {"library_version", "model_revision", "aggregation"} <= metric.keys()
        ):
            raise ValueError(f"Missing evaluator provenance: {name}")
        values[name] = value
    with SummaryWriter(str(Path(run_dir) / "tensorboard")) as writer:
        for name, value in values.items():
            writer.add_scalar("evaluation/" + name, value, generator_updates)
        writer.add_text("evaluation/provenance", json.dumps(report, indent=2), generator_updates)
