import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel


class Runtime:
    def __init__(self, config):
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if config.distributed_strategy == "single" and self.world_size != 1:
            raise ValueError("torchrun requires distributed_strategy=ddp")
        if config.backend == "sdxl":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "SDXL training needs CUDA; use toy only on a suitable test machine"
                )
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
            if config.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("GPU does not support bf16; choose fp16 or no explicitly")
        else:
            self.device = torch.device("cpu")
            if config.mixed_precision != "no":
                raise ValueError("Toy backend requires mixed_precision='no'")
        if self.world_size > 1:
            torch.distributed.init_process_group("nccl" if self.device.type == "cuda" else "gloo")
        random.seed(config.seed + self.rank)
        np.random.seed(config.seed + self.rank)
        torch.manual_seed(config.seed + self.rank)
        self.config = config

    def wrap(self, model):
        if self.world_size == 1:
            return model
        return DistributedDataParallel(
            model,
            device_ids=[self.device.index] if self.device.type == "cuda" else None,
            broadcast_buffers=False,
        )

    def autocast(self):
        dtype = torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16
        return torch.autocast(
            self.device.type, dtype=dtype, enabled=self.config.mixed_precision != "no"
        )

    def all_true(self, value):
        flag = torch.tensor(int(value), device=self.device)
        if self.world_size > 1:
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        return bool(flag.item())

    def barrier(self):
        if self.world_size > 1:
            torch.distributed.barrier()

    def gather(self, item):
        if self.world_size == 1:
            return [item]
        result = [None] * self.world_size
        torch.distributed.all_gather_object(result, item)
        return result

    def broadcast(self, item):
        if self.world_size == 1:
            return item
        items = [item if self.rank == 0 else None]
        torch.distributed.broadcast_object_list(items, src=0)
        return items[0]

    def close(self):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def environment_manifest():
    import importlib.metadata

    versions = {}
    for name in (
        "torch",
        "numpy",
        "diffusers",
        "transformers",
        "accelerate",
        "Pillow",
        "open-clip-torch",
        "clean-fid",
        "torchvision",
        "safetensors",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, dirty = None, None
    return {
        "versions": versions,
        "code_revision": revision,
        "code_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


class JsonLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **record):
        # Nonfinite diagnostic values are explicit strings, preserving valid JSON.
        def clean(value):
            if isinstance(value, float) and not np.isfinite(value):
                return str(value)
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            return value

        with self.path.open("a") as stream:
            stream.write(json.dumps(clean(record), sort_keys=True, allow_nan=False) + "\n")
