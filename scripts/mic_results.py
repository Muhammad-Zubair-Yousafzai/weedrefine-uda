"""Summarise a finished MIC run: every evaluation, the final one, and a JSON file.

Finds the newest work dir for --name under MIC's work_dirs, reads its *.log.json,
prints IoU per evaluation, and writes {out}/{name}.json plus a copy of the logs.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from weedrefine.eval.mic_log import evaluations, last_train_value, read_records

NAMES = ["background", "crop", "weed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="config name, e.g. bean_2019_to_2021_s0_10k")
    ap.add_argument("--work_dirs", default="third_party/MIC/seg/work_dirs/local-rose")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    # MIC names work dirs yymmdd_HHMM_{name}_{5 random chars}; match the name exactly
    runs = sorted(p for p in Path(args.work_dirs).glob(f"*_{args.name}_*")
                  if p.name[12:-6] == args.name)
    if not runs:
        raise SystemExit(f"no work dir for {args.name} in {args.work_dirs}")
    run = runs[-1]
    logs = sorted(run.glob("*.log.json"))
    if not logs:
        raise SystemExit(f"no *.log.json in {run}")
    records = read_records(logs[-1])
    evals = evaluations(records, NAMES)
    if not evals:
        raise SystemExit(f"no evaluation in {logs[-1]} (run not finished?)")

    print(f"{run.name}")
    print(f"{'iter':>7} {'mIoU':>6} {'bg':>6} {'crop':>6} {'weed':>6}")
    for e in evals:
        print(f"{e['iter']:>7} {e['mIoU']:6.2f} {e['IoU_background']:6.2f} "
              f"{e['IoU_crop']:6.2f} {e['IoU_weed']:6.2f}")
    masked = last_train_value(records, "masked.decode.loss_seg")
    print(f"last masked.decode.loss_seg: {masked} (should be > 0 by the end)")

    commit = subprocess.run(["git", "log", "-1", "--format=%h"], capture_output=True,
                            text=True).stdout.strip()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(name=args.name, work_dir=str(run), code_commit=commit,
                   final=evals[-1], evaluations=evals, last_masked_loss=masked)
    (out / f"{args.name}.json").write_text(json.dumps(summary, indent=1))
    for f in run.glob("*.log*"):
        shutil.copy2(f, out / f"{args.name}_{f.name}")
    print(f"wrote {out / (args.name + '.json')}")


if __name__ == "__main__":
    main()
