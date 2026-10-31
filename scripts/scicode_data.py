"""Stage: prepare SciCode fine-tuning data and the test-split eval bundle.

Downloads the SciCode problems (HF dataset SciCode1/SciCode: dev split with
gold solutions, test split without) and the numeric test data (test_data.h5,
Google Drive - compute nodes have internet access).

Outputs two artifacts:
  scicode-data  - scicode_train.jsonl: SFT rows ({"messages": [...]}) built
                  from the DEV split only (test split is the held-out
                  benchmark; training on it would be contamination)
  scicode-test  - problems_test.jsonl + test_data.h5 for the eval stage
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clearml_finetuning import cml, data, prompts  # noqa: E402

DEV_URL = "https://huggingface.co/datasets/SciCode1/SciCode/resolve/main/problems_dev.jsonl"
TEST_URL = "https://huggingface.co/datasets/SciCode1/SciCode/resolve/main/problems_test.jsonl"
# Google Drive file id for test_data.h5 (1.05 GB) from the SciCode README
H5_ID = "17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW"


def fetch(url, dest, attempts=3):
    last = None
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                with open(dest, "wb") as f:
                    f.write(r.read())
            return
        except Exception as e:
            last = e
            print(f"download failed ({e}); retry {i + 1}/{attempts}: {url}")
    raise last


def fetch_h5(dest, attempts=3):
    last = None
    for i in range(attempts):
        try:
            subprocess.run(
                [sys.executable, "-m", "gdown", H5_ID, "-O", dest],
                check=True, timeout=3600)
            if os.path.getsize(dest) > 100_000_000:
                return
            raise RuntimeError("h5 too small - truncated download")
        except Exception as e:
            last = e
            print(f"h5 download failed ({e}); retry {i + 1}/{attempts}")
    raise last


def build_scicode_rows(problems):
    """One SFT row per dev sub-step (messages format, ready for training)."""
    rows = []
    for p in problems:
        deps = p.get("required_dependencies", "")
        for ss in p["sub_steps"]:
            user = prompts.SCICODE_USER_TEMPLATE.format(
                problem_description=p["problem_description_main"],
                dependencies=deps,
                step_description=ss["step_description_prompt"],
                function_header=ss["function_header"],
            )
            if ss.get("step_background", "").strip():
                user += f"\n\nBackground:\n{ss['step_background']}"
            rows.append({
                "messages": [
                    {"role": "system", "content": prompts.SCICODE_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": ss["ground_truth_code"]},
                ],
                "meta": {"problem_id": p["problem_id"], "step": ss["step_number"]},
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="prepare SciCode data")
    ap.add_argument("--out-dir", default="scicode")
    args = ap.parse_args()

    run = cml.get_run_logger("scicode_data", vars(args))

    os.makedirs(args.out_dir, exist_ok=True)
    dev_path = os.path.join(args.out_dir, "problems_dev.jsonl")
    test_path = os.path.join(args.out_dir, "problems_test.jsonl")
    h5_path = os.path.join(args.out_dir, "test_data.h5")

    print("downloading SciCode problems (HF)")
    fetch(DEV_URL, dev_path)
    fetch(TEST_URL, test_path)
    if not os.path.exists(h5_path):
        print("downloading test_data.h5 (Google Drive, ~1 GB)")
        fetch_h5(h5_path)

    dev = [json.loads(l) for l in open(dev_path) if l.strip()]
    test = [json.loads(l) for l in open(test_path) if l.strip()]
    rows = build_scicode_rows(dev)
    train_path = os.path.join(args.out_dir, "scicode_train.jsonl")
    data.write_jsonl(train_path, rows)
    n_steps = sum(len(p["sub_steps"]) for p in test)
    print(f"dev={len(dev)} problems -> {len(rows)} train rows; "
          f"test={len(test)} problems / {n_steps} sub-steps held out for eval")

    if run:
        run.log({"scicode/dev_problems": len(dev), "scicode/train_rows": len(rows),
                 "scicode/test_problems": len(test), "scicode/test_steps": n_steps},
                step=0)
        from pathlib import Path
        run.task.upload_artifact("scicode-data", Path(train_path), wait_on_upload=True)
        test_dir = os.path.join(args.out_dir, "test_bundle")
        os.makedirs(test_dir, exist_ok=True)
        import shutil
        shutil.copy(test_path, test_dir)
        shutil.copy(h5_path, test_dir)
        run.task.upload_artifact("scicode-test", Path(test_dir), wait_on_upload=True)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
