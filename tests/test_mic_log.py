import json

from weedrefine.eval.mic_log import evaluations, last_train_value, read_records

NAMES = ["background", "crop", "weed"]


def _write(tmp_path, recs):
    p = tmp_path / "x.log.json"
    p.write_text("\n".join([json.dumps({"env_info": "..."})] +
                           [json.dumps(r) for r in recs] + ["not json"]) + "\n")
    return p


def test_evaluations_take_train_iter_and_percent(tmp_path):
    recs = [
        {"mode": "train", "iter": 990, "masked.decode.loss_seg": 0.0},
        {"mode": "train", "iter": 1000, "masked.decode.loss_seg": 0.2},
        {"mode": "val", "iter": 25, "mIoU": 0.5, "IoU.background": 0.9,
         "IoU.crop": 0.4, "IoU.weed": 0.2},
        {"mode": "train", "iter": 2000, "masked.decode.loss_seg": 0.3},
        {"mode": "val", "iter": 25, "mIoU": 0.6, "IoU.background": 0.9,
         "IoU.crop": 0.5, "IoU.weed": 0.4},
    ]
    r = read_records(_write(tmp_path, recs))
    ev = evaluations(r, NAMES)
    assert [e["iter"] for e in ev] == [1000, 2000]
    assert abs(ev[-1]["IoU_weed"] - 40.0) < 1e-9 and abs(ev[0]["mIoU"] - 50.0) < 1e-9
    assert last_train_value(r, "masked.decode.loss_seg") == 0.3
