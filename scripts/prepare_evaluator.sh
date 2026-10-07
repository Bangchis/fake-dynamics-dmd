#!/usr/bin/env bash
set -euo pipefail
# RECIPIENT ONLY; puts upstream code in ignored assets, does not edit its tracked files.
task_upstream="${DMD2_EVAL_DIR:-assets/DMD2}"
if [[ -e "$task_upstream" ]]; then
  echo "Already exists: $task_upstream. Check its revision manually." >&2
  exit 1
fi
git clone https://github.com/tianweiy/DMD2.git "$task_upstream"
git -C "$task_upstream" checkout 8d8fa55633d47cfb81bbc7a892e7248f9518763f
python -m pip install -e '.[eval]'
python -m pip install 'git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'
echo "Evaluator dependencies installed; model weights download on first evaluation."
