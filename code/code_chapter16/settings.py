"""Chapter 16 paths and the Stage-3 settings copied from Motus' public config."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE_ROOT = ROOT.parent
MOTUS_ROOT = Path(os.getenv("MOTUS_ROOT", CODE_ROOT / "Motus")).expanduser().resolve()
WEIGHTS_ROOT = Path(os.getenv("CHAPTER16_WEIGHTS", ROOT / "weights")).expanduser().resolve()
DATA_ROOT = Path(os.getenv("CHAPTER16_DATA", ROOT / "data")).expanduser().resolve()
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
T5_CACHE_ROOT = DATA_ROOT / "cache" / "t5"
CHECKPOINT_ROOT = ROOT / "checkpoints"
OUTPUT_ROOT = ROOT / "outputs"
LOG_ROOT = ROOT / "logs"
TASK_RUNTIME_ROOT = ROOT / "task_runtime"

WAN_ROOT = WEIGHTS_ROOT / "Wan2.2-TI2V-5B"
QWEN_ROOT = WEIGHTS_ROOT / "Qwen3-VL-2B-Instruct"
STAGE2_ROOT = WEIGHTS_ROOT / "Motus"
ROBOTWIN2_ROOT = WEIGHTS_ROOT / "Motus_robotwin2"

COLORS = ("red", "green", "blue")
TASK_NAMES = {color: f"pick_{color}_block" for color in COLORS}
TASK_TEXT = {
    color: f"Pick up the {color} block and place it into the empty box."
    for color in COLORS
}
# Official RoboTwin conversion/deployment adds one scene description before the
# task text.  Keep the same prompt structure, but name this chapter's actual G2
# cameras/embodiment.  Processed metadata, UMT5 and Qwen-VL must see this exact
# same string.
SCENE_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed head camera, a movable left wrist camera, and a movable right wrist camera. "
    "The G2 robot is currently performing the following task: "
)


def stage3_prompt(instruction: str) -> str:
    """Return the single canonical Stage-3 prompt without double-prefixing."""
    text = str(instruction).strip()
    if not text:
        raise ValueError("instruction must not be empty")
    return text if text.startswith(SCENE_PREFIX) else SCENE_PREFIX + text

# Motus configs/robotwin.yaml, adapted only from RoboTwin's 14D embodiment to G2's 16D one.
STATE_DIM = 16
ACTION_DIM = 16
RAW_FPS = 30
GLOBAL_DOWNSAMPLE_RATE = 3
NUM_VIDEO_FRAMES = 8
VIDEO_ACTION_FREQ_RATIO = 2
ACTION_CHUNK_SIZE = NUM_VIDEO_FRAMES * VIDEO_ACTION_FREQ_RATIO
VIDEO_HEIGHT = 384
VIDEO_WIDTH = 320
# Official RoboTwin converter stores a 360x320 T-layout MP4; the official loader
# then aspect-preserving pads it to the 384x320 model input.
ROBOTWIN_VIDEO_HEIGHT = 360
ROBOTWIN_VIDEO_WIDTH = 320
INFERENCE_STEPS = 10

# Two official public sources expose different notions of "training length":
# - Motus_robotwin2 model card: released Stage-3 run used 2,500 clean + 25,000
#   randomized demonstrations and 40,000 optimizer steps.
# - configs/robotwin.yaml: max_steps is a 1,000,000-step upper bound.
# The chapter's default reproduction follows the released checkpoint card and keeps
# the public 1M upper-bound recipe in configs/stage3_g2_public_1m.yaml.
OFFICIAL_CLEAN_EPISODES = 2_500
OFFICIAL_RANDOMIZED_EPISODES = 25_000
OFFICIAL_RELEASE_STEPS = 40_000
OFFICIAL_PUBLIC_MAX_STEPS = 1_000_000
OFFICIAL_BATCH_SIZE_PER_GPU = 8
OFFICIAL_NUM_GPUS = 8

# Practical quota for this chapter's single pick-place task family.  It matches
# the already collected dataset and is intentionally much smaller than Motus'
# 50+ task RoboTwin2 corpus.
CHAPTER16_CLEAN_EPISODES = 150
CHAPTER16_RANDOMIZED_EPISODES = 1_500
CHAPTER16_POSITION_NOISE = 0.01


def ensure_dirs() -> None:
    for path in (
        RAW_ROOT,
        PROCESSED_ROOT,
        T5_CACHE_ROOT,
        CHECKPOINT_ROOT,
        OUTPUT_ROOT,
        LOG_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)
