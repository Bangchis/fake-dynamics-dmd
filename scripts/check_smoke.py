"""Recipient-only log/manifest coverage check; does not validate model tensor changes."""

import argparse
import json
from pathlib import Path


def check(run_dir, expected_updates):
    root = Path(run_dir)
    manifest_path = (
        Path(json.loads((root / "latest.json").read_text())["checkpoint"]) / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["state"]["generator_updates"] == expected_updates, "Wrong G counter"
    assert manifest["state"]["fake_updates"] == 5 * expected_updates, "Wrong F:G ratio"
    assert len(manifest["rank_files"]) == 2, "Need both optimizer/RNG rank files"
    for row in manifest["rank_files"]:
        assert (manifest_path.parent / row["file"]).is_file(), "Missing rank state"
    rank_sequences = []
    for rank in range(2):
        rows = [
            json.loads(line)
            for line in (root / f"metrics_rank{rank}.jsonl").read_text().splitlines()
        ]
        generator = [row for row in rows if row["event"] == "generator_step" and row["success"]]
        assert {row["anchor"] for row in generator} == {999, 749, 499, 249}, "Missing G anchor"
        for row in generator:
            assert row["beta"] == 0.05 and row["lambda_cd"] == 0.1
            assert row["cd_active"] == (row["anchor"] != 249), "Wrong CD branch"
        rank_sequences.append([(row["k_G"], row["k_F"], row["anchor"]) for row in generator])
        assert not any(row["event"] == "optimizer_step_skipped" for row in rows), (
            "Investigate skips"
        )
    assert rank_sequences[0] == rank_sequences[1], "Ranks disagree on update counters/anchors"
    print(
        "Both rank logs cover all G anchors/CD branches and expected counters; tensor/numerical checks remain separate."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--expected-updates", type=int, default=8)
    args = parser.parse_args()
    check(args.run_dir, args.expected_updates)
