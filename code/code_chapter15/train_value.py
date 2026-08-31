"""Train distributional Pistar06 Value on all data through one round."""

from __future__ import annotations
import argparse
import math
import random
import shutil
from pathlib import Path
import numpy as np
from config import ValueTrainPreset, raw_dirs, value_checkpoint
from dataset import episode_paths, load_episode, scalar, task_max_lengths
from value_math import EpisodeInfo, two_hot, value_targets


class Frames:
    def __init__(self, paths):
        self.rows = []
        maxima = task_max_lengths(paths)
        states = np.concatenate([load_episode(path)["state"] for path in paths], axis=0)
        self.q01 = np.quantile(states, 0.01, axis=0).astype(np.float32)
        self.q99 = np.quantile(states, 0.99, axis=0).astype(np.float32)
        for path in paths:
            ep = load_episode(path)
            if "use_for_value" in ep and not bool(scalar(ep["use_for_value"])):
                continue
            n = len(ep["state"])
            eid = int(ep["episode_index"][0])
            tid = int(ep["task_index"][0])
            info = {eid: EpisodeInfo(eid, tid, n, bool(scalar(ep["success"])))}
            targets = value_targets(np.full(n, eid), np.arange(n), info, maxima)
            for i in range(n):
                self.rows.append((ep, i, float(targets[i])))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ep, j, target = self.rows[i]
        images = np.stack(
            [ep["head_image"][j], ep["left_image"][j], ep["right_image"][j]]
        )
        state = ep["state"][j]
        normalized = np.clip(
            2 * (state - self.q01) / np.maximum(self.q99 - self.q01, 1e-6) - 1, -1, 1
        )
        q = np.digitize(normalized, np.linspace(-1, 1, 257, dtype=np.float32)[:-1]) - 1
        q = np.pad(q, (0, 32 - len(q)))
        prompt = f"Task: {scalar(ep['task'])}, State: {' '.join(map(str, q.tolist()))}\nValue: "
        return images, prompt, target


def cosine_with_warmup(optimizer, preset: ValueTrainPreset, training_steps: int):
    """Evo-RL cosine schedule with a non-zero final learning rate.

    Short debug runs scale both warmup and decay so the complete schedule is
    still visible instead of ending partway through the original 8k schedule.
    """
    import torch

    if training_steps <= 0:
        raise ValueError("training_steps must be positive")
    warmup_steps = preset.warmup_steps
    decay_steps = preset.decay_steps
    if training_steps < decay_steps:
        scale = training_steps / decay_steps
        warmup_steps = int(warmup_steps * scale)
        decay_steps = training_steps

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            start = 1.0 / (warmup_steps + 1)
            return start + (1.0 - start) * step / warmup_steps
        clipped = min(step, decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * clipped / decay_steps))
        alpha = preset.final_lr / preset.lr
        return (1.0 - alpha) * cosine + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--language-model", type=Path)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accumulation", type=int, default=1)
    p.add_argument("--freeze-vision", action="store_true")
    p.add_argument("--freeze-language", action="store_true")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    preset = ValueTrainPreset()
    language_model = (
        a.language_model.expanduser().resolve()
        if a.language_model is not None
        else Path(preset.language_model)
    )
    required = ("config.json", "model.safetensors", "tokenizer_config.json")
    missing = [name for name in required if not (language_model / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"local language model is incomplete: {language_model}; missing: {', '.join(missing)}"
        )
    language_model = str(language_model)
    output = (a.output or value_checkpoint(a.round)).resolve()
    if output.exists():
        if not a.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from value_model import (
        Pistar06Value,
        ValueModelConfig,
        two_hot_cross_entropy,
        expected_from_logits,
    )

    torch.manual_seed(preset.seed)
    np.random.seed(preset.seed)
    random.seed(preset.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = episode_paths(raw_dirs(a.round))
    data = Frames(paths)
    if not len(data):
        raise RuntimeError("empty value dataset")
    tokenizer = AutoTokenizer.from_pretrained(language_model, local_files_only=True)

    def collate(rows):
        images, prompts, targets = zip(*rows, strict=True)
        tok = tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=200,
            return_tensors="pt",
        )
        return (
            torch.from_numpy(np.stack(images)).permute(0, 1, 4, 2, 3),
            tok,
            torch.tensor(targets),
        )

    loader = DataLoader(
        data,
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=preset.workers,
        collate_fn=collate,
        drop_last=False,
    )
    iterator = iter(loader)
    cfg = ValueModelConfig(
        vision_model=preset.vision_model,
        language_model=language_model,
        fusion_dim=preset.fusion_dim,
        dropout=preset.dropout,
        dtype=a.dtype,
        freeze_vision=a.freeze_vision,
        freeze_language=a.freeze_language,
    )
    model = Pistar06Value(cfg).to(device)
    optim = torch.optim.AdamW(
        (x for x in model.parameters() if x.requires_grad),
        lr=preset.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=preset.weight_decay,
    )
    scheduler = cosine_with_warmup(optim, preset, a.steps)
    model.train()
    optim.zero_grad(set_to_none=True)
    for step in range(1, a.steps + 1):
        try:
            images, tok, target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, tok, target = next(iterator)
        logits = model(
            tok["input_ids"].to(device),
            tok["attention_mask"].to(device),
            images.to(device),
        )
        soft = torch.from_numpy(two_hot(target.numpy())).to(device)
        loss = two_hot_cross_entropy(logits, soft) / a.grad_accumulation
        loss.backward()
        if step % a.grad_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), preset.grad_clip)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)
        if step == 1 or step % 200 == 0:
            print(
                f"step={step} loss={loss.item() * a.grad_accumulation:.5f} value={expected_from_logits(logits).mean().item():.4f}",
                flush=True,
            )
    model.save(
        output,
        {
            "round": a.round,
            "steps": a.steps,
            "task_max_lengths": task_max_lengths(paths),
            "state_q01": data.q01.tolist(),
            "state_q99": data.q99.tolist(),
            "seed": preset.seed,
        },
    )
    tokenizer.save_pretrained(output / "tokenizer")
    print(output)


if __name__ == "__main__":
    main()
