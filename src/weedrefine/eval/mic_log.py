"""Read evaluation results from an MMSegmentation/MIC *.log.json file.

Each line is a JSON record. Training records have "mode": "train" and "iter".
Evaluation records carry "mIoU" and "IoU.<class>" as fractions in [0, 1]; their own
"iter" is the val loader count, so the training iteration is taken from the last
training record before them.
"""
import json


def read_records(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def evaluations(records, class_names):
    """List of {'iter', 'mIoU', 'IoU_<class>'...} in percent, one per evaluation."""
    evals, train_iter = [], 0
    for r in records:
        if "mIoU" in r:
            e = {"iter": train_iter, "mIoU": 100 * r["mIoU"]}
            for c in class_names:
                e[f"IoU_{c}"] = 100 * r[f"IoU.{c}"]
            evals.append(e)
        elif r.get("mode") == "train" and "iter" in r:
            train_iter = r["iter"]
    return evals


def last_train_value(records, key):
    """Value of `key` in the last training record that has it, else None."""
    for r in reversed(records):
        if r.get("mode") == "train" and key in r and "mIoU" not in r:
            return r[key]
    return None
