# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from enum import Enum

from g2_robot.utils.logger import Logger

logger = Logger()


def check_and_fix_env():
    """Check and fix SIM_REPO_ROOT and SIM_ASSETS environment variables."""
    env_root_path = os.getenv("SIM_REPO_ROOT")
    current_dir = Path(__file__).resolve().parent.parent.parent
    default_assets_path = current_dir / "g2_robot" / "assets"

    if not env_root_path:
        # g2_robot/utils/system_utils.py -> 3 levels up = workspace root
        # Keep SIM_REPO_ROOT pointed at g2_robotics for robot_cfg / data_collection.
        env_root_path = current_dir / "g2_robotics"
        os.environ["SIM_REPO_ROOT"] = env_root_path.as_posix()
        logger.warning(f"Warning: env [SIM_REPO_ROOT] empty, will use default: {env_root_path}")
    else:
        logger.info(f"using env SIM_REPO_ROOT={env_root_path}")

    if not os.path.exists(env_root_path):
        os.makedirs(env_root_path, exist_ok=True)

    os.environ["SIM_ASSETS"] = default_assets_path.as_posix()
    logger.info(f"using repo assets path: {default_assets_path}")


def assets_path():
    """Return the g2_robot/assets directory."""
    workspace_root = Path(__file__).resolve().parent.parent.parent
    return (workspace_root / "g2_robot" / "assets").as_posix()


def load_json(json_file):
    if not os.path.exists(json_file):
        raise ValueError("Json file not found: {}".format(json_file))
    with open(json_file) as f:
        return json.load(f)


def generate_new_file_path(dir_path, prefix_name, suffix="json"):
    count = 0
    while os.path.exists(os.path.join(dir_path, f"{prefix_name}_{count:02d}.{suffix}")):
        count += 1
    new_filename = f"{prefix_name}_{count:02d}.{suffix}"
    return os.path.join(dir_path, new_filename)


def TIMENOW():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_local_time():
    tz_utc8 = timezone(timedelta(hours=8))
    utc_now = datetime.now(tz_utc8)
    return str(utc_now)


def ConvertEnum2Int(code: Enum):
    return int(code.value)
