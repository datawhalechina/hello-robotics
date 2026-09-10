"""Run Motus Stage 3 with Chapter-16 G2 data semantics and memory-safe LoRA.

The Motus model forward pass, trimodal losses, scheduler and evaluator are reused.
This launcher only adds four workstation/task-specific corrections:

1. measured state and commanded action are loaded from separate files;
2. train/validation episodes are disjoint and limited to the three color tasks;
3. Accelerate gradient accumulation counts *optimizer* steps correctly;
4. LoRA-only checkpoints avoid writing the frozen 8B base model.
"""

from __future__ import annotations

import json
import re
import runpy
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from data_utils import dataset_summary, find_checkpoint_file
from g2_dataset_adapter import install_g2_dataset_adapter
from lora import apply_lora, load_lora_adapter, save_lora_adapter
from motus_compat import configure_attention, install_motus_imports
from settings import (
    MOTUS_ROOT,
    OFFICIAL_CLEAN_EPISODES,
    OFFICIAL_RANDOMIZED_EPISODES,
    OFFICIAL_RELEASE_STEPS,
)


def config_argument() -> Path:
    try:
        index = sys.argv.index("--config")
        return Path(sys.argv[index + 1]).expanduser().resolve()
    except (ValueError, IndexError):
        raise SystemExit("train_stage3.py requires --config <yaml>")


def preflight(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("training_mode") != "finetune":
        raise ValueError("Stage 3 requires top-level training_mode: finetune")
    chapter = Path(__file__).resolve().parent
    dataset = (chapter / config["dataset"]["dataset_dir"]).resolve()
    stage2 = (chapter / config["finetune"]["checkpoint_path"]).resolve()
    find_checkpoint_file(stage2)
    summary = dataset_summary(dataset)
    if summary["episodes"] == 0:
        raise FileNotFoundError(
            f"no complete processed episodes under {dataset}; "
            "run prepare_data.py again so commands/*.pt are created"
        )
    if int(config["common"]["state_dim"]) != 16 or int(config["common"]["action_dim"]) != 16:
        raise ValueError("Chapter-16 G2 Stage 3 must use 16D state/action")
    if config.get("finetune", {}).get("train_scope") != "lora":
        raise ValueError("This Chapter-16 launcher intentionally supports LoRA only")
    expected = OFFICIAL_CLEAN_EPISODES + OFFICIAL_RANDOMIZED_EPISODES
    configured_steps = int(config["training"]["max_steps"])
    accumulation = int(config["training"].get("gradient_accumulation_steps", 1))
    print(
        f"preflight: {summary['episodes']} complete state+command episodes; "
        f"optimizer_steps={configured_steps}; grad_accum={accumulation}; Stage-2={stage2}",
        flush=True,
    )
    if "smoke" not in path.stem and summary["episodes"] < expected:
        print(
            f"[note] This is a single pick-place task family with {summary['episodes']} episodes. "
            f"The released multi-task Motus_robotwin2 scale was {expected}; it is not a required "
            "quota for this one environment.",
            flush=True,
        )
    if path.name == "stage3_g2.yaml" and configured_steps != OFFICIAL_RELEASE_STEPS:
        raise ValueError(f"release reproduction config must use {OFFICIAL_RELEASE_STEPS} steps")


def configure_lora(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    finetune = config.get("finetune", {})
    from models.motus import Motus

    original_init = Motus.__init__

    def lora_init(self, model_config):
        original_init(self, model_config)
        apply_lora(self, finetune.get("lora"))

    Motus.__init__ = lora_init
    return finetune.get("lora", {})


def _move_batch(batch, device, dtype):
    first = batch["first_frame"].to(device, dtype=dtype)
    video = batch["video_frames"].to(device, dtype=dtype)
    language = batch["language_embedding"]
    if language is not None:
        language = language.to(device, dtype=dtype)
    state = batch.get("initial_state")
    if state is not None:
        state = state.to(device, dtype=dtype)
    actions = batch["action_sequence"].to(device, dtype=dtype)
    vlm = batch["vlm_inputs"]
    if vlm is not None:
        vlm = {key: value.to(device) if torch.is_tensor(value) else value for key, value in vlm.items()}
    return first, video, language, state, actions, vlm


def install_accumulating_loop(trainer_class, namespace) -> None:
    """Replace the official per-microbatch step with true Accelerate accumulation."""
    logger = namespace["logger"]
    evaluate_model = namespace["evaluate_model"]
    log_evaluation_metrics = namespace["log_evaluation_metrics"]
    wandb = namespace["wandb"]

    def train_micro_step(self, batch):
        self.model.train()
        with self.accelerator.accumulate(self.model):
            first, video, language, state, actions, vlm = _move_batch(batch, self.device, self.dtype)
            model = self.model.module if hasattr(self.model, "module") else self.model
            losses = model.training_step(
                first_frame=first,
                video_frames=video,
                state=state,
                actions=actions,
                language_embeddings=language,
                vlm_inputs=vlm,
                return_dict=True,
            )
            self.accelerator.backward(losses["total_loss"])
            if self.accelerator.sync_gradients:
                grad_clip = float(getattr(self.config.training, "grad_clip_norm", 1.0))
                self.accelerator.clip_grad_norm_(self.model.parameters(), grad_clip)
            # AcceleratedOptimizer/DeepSpeed deliberately no-op these calls until
            # the configured accumulation boundary is reached.
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
            updated = bool(self.accelerator.sync_gradients)
        metrics = {key: value.item() if torch.is_tensor(value) else float(value) for key, value in losses.items()}
        return metrics, updated

    def train(self, max_steps, resume_from=None, val_interval=500, reset_scheduler=None):
        if resume_from:
            if reset_scheduler is None:
                reset_scheduler = bool(getattr(self.config.resume, "reset_scheduler", True))
            self.load_checkpoint(resume_from, reset_scheduler=reset_scheduler)

        accumulation = int(self.config.training.get("gradient_accumulation_steps", 1))
        validation_batches = int(getattr(self.config.dataset, "validation_batches", 20))
        logger.info(
            "Starting Chapter-16 Stage3-LoRA for %d optimizer steps (grad_accum=%d, world=%d)",
            max_steps, accumulation, self.world_size,
        )
        started = time.time()
        data_iter = iter(self.train_dataloader)
        epoch = int(getattr(self, "epoch", 0))
        metric_sums: dict[str, float] = {}
        micro_count = 0
        update_started = time.time()
        self.optimizer.zero_grad()

        while self.global_step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if hasattr(self.train_dataloader.sampler, "set_epoch"):
                    self.train_dataloader.sampler.set_epoch(epoch)
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            if batch is None:
                continue

            metrics, updated = train_micro_step(self, batch)
            micro_count += 1
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            if not updated:
                continue

            self.global_step += 1
            self.epoch = epoch
            averaged = {key: value / micro_count for key, value in metric_sums.items()}
            step_time = time.time() - update_started
            metric_sums.clear()
            micro_count = 0
            update_started = time.time()

            if self.global_step % self.log_interval == 0 and self.rank == 0:
                lrs = [group["lr"] for group in self.optimizer.param_groups]
                lr_main = lrs[0] if lrs else 0.0
                lr_wan = lrs[1] if len(lrs) > 1 else lr_main
                logger.info(
                    "Step %d/%d, Loss: %.4f (Video: %.4f, Action: %.4f), "
                    "LR(main/wan): %.2e/%.2e, Time/update: %.2fs",
                    self.global_step, max_steps, averaged["total_loss"],
                    averaged["video_loss"], averaged["action_loss"],
                    lr_main, lr_wan, step_time,
                )
                payload = {
                    **averaged,
                    "learning_rate_main": lr_main,
                    "learning_rate_wan": lr_wan,
                    "step_time": step_time,
                    "epoch": epoch,
                    "global_step": self.global_step,
                }
                if "wandb" in self.report_to:
                    wandb.log(payload)
                if self.tb_writer is not None:
                    for key, value in payload.items():
                        self.tb_writer.add_scalar(f"train/{key}", value, self.global_step)

            if self.global_step % val_interval == 0 and self.val_dataloader is not None:
                if self.rank == 0:
                    values = evaluate_model(
                        self.model, self.val_dataloader, self.accelerator, self.config,
                        num_eval_batches=validation_batches,
                    )
                    logger.info("Held-out validation - Step %d", self.global_step)
                    log_evaluation_metrics(values, self.tb_writer, self.accelerator, self.global_step)
                if dist.is_available() and dist.is_initialized():
                    dist.barrier(device_ids=[torch.cuda.current_device()])

            if self.global_step % self.save_interval == 0:
                self.save_checkpoint()

        if self.rank == 0:
            logger.info(
                "Stage3-LoRA training completed in %.2fs (%d optimizer steps)",
                time.time() - started, self.global_step,
            )
        # All ranks call this. If the final interval was already saved, the
        # rank-symmetric duplicate guard returns on every rank.
        self.save_checkpoint()

    trainer_class.train_step = train_micro_step
    trainer_class.train = train


def main() -> None:
    path = config_argument()
    preflight(path)
    install_motus_imports()
    configure_attention()
    lora_options = configure_lora(path)
    install_g2_dataset_adapter()

    save_full_state = bool(lora_options.get("save_full_state", False))
    trainer_file = MOTUS_ROOT / "train" / "train.py"
    namespace = runpy.run_path(str(trainer_file), run_name="chapter16_motus_official_train")
    trainer_class = namespace["UniDiffuserTrainer"]
    install_accumulating_loop(trainer_class, namespace)

    original_save = trainer_class.save_checkpoint

    def save_checkpoint_once(self, suffix=""):
        marker = (int(self.global_step), str(suffix))
        if getattr(self, "_chapter16_last_saved", None) == marker:
            return
        checkpoint_dir = self.checkpoint_dir / f"checkpoint_step_{self.global_step}{suffix}"

        if save_full_state:
            original_save(self, suffix)
        else:
            self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model)
            adapter_path = save_lora_adapter(unwrapped, checkpoint_dir, self.global_step)
            shutil.copy2(path, checkpoint_dir / "train_config.yaml")
            (checkpoint_dir / "trainer_state.json").write_text(
                json.dumps(
                    {
                        "format": "chapter16-motus-stage3-lora-adapter",
                        "global_step": int(self.global_step),
                        "epoch": int(self.epoch),
                        "full_deepspeed_state": save_full_state,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"LoRA adapter saved: {adapter_path}", flush=True)

        if not save_full_state:
            self.accelerator.wait_for_everyone()
        self._chapter16_last_saved = marker

    original_load = trainer_class.load_checkpoint

    def load_checkpoint_with_adapter(self, checkpoint_path, reset_scheduler=True):
        checkpoint = Path(checkpoint_path)
        adapter = checkpoint / "adapter_model.pt" if checkpoint.is_dir() else checkpoint
        has_full_state = checkpoint.is_dir() and (
            (checkpoint / "latest").exists() or (checkpoint / "pytorch_model").exists()
        )
        if not adapter.is_file() or has_full_state:
            return original_load(self, checkpoint_path, reset_scheduler=reset_scheduler)

        match = re.search(r"step_(\d+)", str(checkpoint))
        self.global_step = int(match.group(1)) if match else 0
        unwrapped = self.accelerator.unwrap_model(self.model)
        _, unexpected = load_lora_adapter(unwrapped, adapter)
        if unexpected:
            raise RuntimeError(f"unexpected LoRA adapter keys: {unexpected[:8]}")
        if self.accelerator.is_main_process:
            print(
                f"LoRA adapter resumed from {adapter} at step {self.global_step}. "
                "Optimizer and scheduler restart for adapter-only resume.",
                flush=True,
            )
        self.accelerator.wait_for_everyone()

    trainer_class.save_checkpoint = save_checkpoint_once
    trainer_class.load_checkpoint = load_checkpoint_with_adapter
    namespace["main"]()


if __name__ == "__main__":
    main()
