import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_prompts(path):
    """JSONL with explicit ids; plain lines are intentionally not silently reinterpreted."""
    records = []
    seen = set()
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
                raise ValueError(f"{path}:{line_number}: requires a string prompt")
            identifier = str(row.get("id", len(records)))
            if identifier in seen:
                raise ValueError(f"Duplicate prompt id: {identifier}")
            row["id"] = identifier
            seen.add(identifier)
            records.append(row)
    if not records:
        raise ValueError("Prompt file is empty")
    return records


class PromptStream:
    """No worker prefetch: cursor + epoch + independent RNG restore exact prompt order.

    All ranks derive the same global shuffled order; each receives a disjoint slice.
    Last incomplete global batch is dropped, with an explicit minimum-size check.
    """

    def __init__(self, path, batch_size, rank=0, world_size=1, seed=10):
        self.records = read_prompts(path)
        self.file_hash = sha256_file(path)
        self.batch_size, self.rank, self.world_size = batch_size, rank, world_size
        self.global_batch = batch_size * world_size
        if len(self.records) < self.global_batch:
            raise ValueError("Prompt count must be at least the global batch size")
        self.rng = torch.Generator().manual_seed(seed)
        self.epoch, self.cursor = 0, 0
        self.order = torch.randperm(len(self.records), generator=self.rng)

    def next(self):
        if self.cursor + self.global_batch > len(self.records):
            self.epoch += 1
            self.cursor = 0
            self.order = torch.randperm(len(self.records), generator=self.rng)
        start = self.cursor + self.rank * self.batch_size
        selected = self.order[start : start + self.batch_size].tolist()
        self.cursor += self.global_batch
        return [self.records[i]["prompt"] for i in selected]

    def state_dict(self):
        return {
            "file_hash": self.file_hash,
            "batch_size": self.batch_size,
            "world_size": self.world_size,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "order": self.order,
            "rng": self.rng.get_state(),
        }

    def load_state_dict(self, state):
        for key in ("file_hash", "batch_size", "world_size"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Prompt stream resume mismatch: {key}")
        self.epoch, self.cursor, self.order = state["epoch"], state["cursor"], state["order"]
        self.rng.set_state(state["rng"])
