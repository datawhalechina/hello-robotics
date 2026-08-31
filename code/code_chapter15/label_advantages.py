"""Run Value inference, compute exact n-step advantages, and write new labeled episodes."""

from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import numpy as np
from config import (
    ACP_POSITIVE_RATIO,
    ADVANTAGE_N_STEP,
    labeled_dir,
    raw_dirs,
    value_checkpoint,
)
from dataset import episode_paths, load_episode, scalar
from value_math import (
    EpisodeInfo,
    acp_labels,
    dense_rewards,
    n_step_advantage,
    value_targets,
)


def prompt(task, state, q01, q99):
    norm = np.clip(2 * (state - q01) / np.maximum(q99 - q01, 1e-6) - 1, -1, 1)
    bins = np.digitize(norm, np.linspace(-1, 1, 257, dtype=np.float32)[:-1]) - 1
    bins = np.pad(bins, (0, 32 - len(bins)))
    return f"Task: {str(task).strip().replace('_', ' ').replace(chr(10), ' ')}, State: {' '.join(map(str, bins.tolist()))}\nValue: "


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--value-checkpoint", type=Path)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-step", type=int, default=ADVANTAGE_N_STEP)
    p.add_argument("--positive-ratio", type=float, default=ACP_POSITIVE_RATIO)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    out = labeled_dir(a.round)
    if out.exists():
        if not a.overwrite:
            raise FileExistsError(out)
        shutil.rmtree(out)
    out.mkdir(parents=True)
    import torch
    from transformers import AutoTokenizer
    from value_model import Pistar06Value, expected_from_logits

    ckpt = (a.value_checkpoint or value_checkpoint(a.round)).resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = Pistar06Value.load(ckpt, device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt / "tokenizer")
    q01 = np.asarray(meta["state_q01"])
    q99 = np.asarray(meta["state_q99"])
    paths = episode_paths(raw_dirs(a.round))
    episodes = [load_episode(x) for x in paths]
    maxima = {}
    for ep in episodes:
        tid = int(ep["task_index"][0])
        maxima[tid] = max(maxima.get(tid, 0), len(ep["state"]))
    all_task = []
    all_adv = []
    all_inter = []
    staged = []
    with torch.inference_mode():
        for path, ep in zip(paths, episodes, strict=True):
            n = len(ep["state"])
            eid = int(ep["episode_index"][0])
            tid = int(ep["task_index"][0])
            texts = [prompt(scalar(ep["task"]), x, q01, q99) for x in ep["state"]]
            preds = []
            for start in range(0, n, a.batch_size):
                tok = tokenizer(
                    texts[start : start + a.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=200,
                    return_tensors="pt",
                ).to(device)
                imgs = np.stack(
                    [
                        np.stack(
                            [
                                ep[k][i]
                                for k in ("head_image", "left_image", "right_image")
                            ]
                        )
                        for i in range(start, min(n, start + a.batch_size))
                    ]
                )
                tensor = torch.from_numpy(imgs).permute(0, 1, 4, 2, 3).to(device)
                preds.extend(
                    expected_from_logits(
                        model(tok["input_ids"], tok["attention_mask"], tensor)
                    )
                    .cpu()
                    .numpy()
                )
            info = {eid: EpisodeInfo(eid, tid, n, bool(scalar(ep["success"])))}
            targets = value_targets(np.full(n, eid), np.arange(n), info, maxima)
            rewards = dense_rewards(targets, np.full(n, eid), np.arange(n))
            advantage = n_step_advantage(
                rewards, np.asarray(preds), np.full(n, eid), np.arange(n), a.n_step
            )
            staged.append((path, ep, targets, np.asarray(preds, np.float32), advantage))
            all_task.extend([tid] * n)
            all_adv.extend(advantage)
            all_inter.extend(ep["is_intervention"])
    labels, thresholds = acp_labels(
        np.asarray(all_task),
        np.asarray(all_adv),
        np.asarray(all_inter),
        a.positive_ratio,
        True,
    )
    cursor = 0
    for path, ep, targets, preds, adv in staged:
        n = len(adv)
        payload = {
            **ep,
            "value_target": targets,
            "value_prediction": preds,
            "advantage": adv,
            "acp_indicator": labels[cursor : cursor + n],
        }
        np.savez_compressed(out / path.name, **payload)
        cursor += n
    print(
        {
            "output": str(out),
            "frames": cursor,
            "positive_ratio": float(labels.mean()),
            "thresholds": thresholds,
        }
    )


if __name__ == "__main__":
    main()
