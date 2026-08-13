"""第十三章主程序：看懂指令，识别目标，再由 Nav2 导航过去。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

try:
    from .config import (
        CHAPTER_DIR,
        ControlLimits,
        DEFAULT_OUTPUT_DIR,
        NavigationConfig,
        RobotGeometry,
        PerceptionConfig,
        SensorConfig,
        SimulationConfig,
        TARGET_COLORS,
        VLMConfig,
        YOLO_WORLD_MODEL,
        Velocity2D,
    )
    from .base_controller import G2BaseController
    from .kinematics import SwerveKinematics
    from .local_vlm import keyword_target
    from .perception import YOLOWorldDetector, make_observation_board
    from .ros_bridge import Nav2Bridge, make_standoff_goal
    from .sensors import G2Sensors
    from .simulation import G2Chapter13Simulation
except ImportError:
    from config import (
        CHAPTER_DIR,
        ControlLimits,
        DEFAULT_OUTPUT_DIR,
        NavigationConfig,
        RobotGeometry,
        PerceptionConfig,
        SensorConfig,
        SimulationConfig,
        TARGET_COLORS,
        VLMConfig,
        YOLO_WORLD_MODEL,
        Velocity2D,
    )
    from base_controller import G2BaseController
    from kinematics import SwerveKinematics
    from local_vlm import keyword_target
    from perception import YOLOWorldDetector, make_observation_board
    from ros_bridge import Nav2Bridge, make_standoff_goal
    from sensors import G2Sensors
    from simulation import G2Chapter13Simulation

def _chapter_path(value: str | Path, *, kind: str) -> Path:
    """确保模型和权重来自 code_chapter13，避免静默调用项目外文件。"""
    path = Path(value).expanduser().resolve()
    chapter = CHAPTER_DIR.resolve()
    try:
        path.relative_to(chapter)
    except ValueError as exc:
        raise ValueError(f"{kind} 必须放在本章目录 {chapter} 中，当前为：{path}") from exc
    if kind == "Qwen3-VL 模型":
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Qwen3-VL 模型目录不完整：{path}")
    elif not path.is_file():
        raise FileNotFoundError(f"找不到 {kind}：{path}")
    return path


def parse_args() -> argparse.Namespace:
    defaults = VLMConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", help="例如：请导航到蓝色物体")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--yolo-model", default=str(YOLO_WORLD_MODEL))
    parser.add_argument("--vlm-model", default=str(defaults.model_path))
    parser.add_argument("--vlm-python", default=os.environ.get("VLM_PYTHON", defaults.python))
    parser.add_argument("--skip-vlm", action="store_true", help="仅调试感知/Nav2：按颜色关键词选择")
    parser.add_argument("--perception-only", action="store_true", help="识别和决策后不发送 Nav2 目标")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def run_local_vlm(args, board_path: Path, summaries: list[dict], output_dir: Path) -> dict:
    detections_path = output_dir / "detections.json"
    decision_path = output_dir / "vlm_decision.json"
    detections_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    command = [
        args.vlm_python,
        str(Path(__file__).resolve().with_name("local_vlm.py")),
        "--model", args.vlm_model,
        "--image", str(board_path),
        "--instruction", args.instruction,
        "--detections", str(detections_path),
        "--output", str(decision_path),
    ]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    # VLM 子进程不需要 Isaac ROS 2 Bridge 的动态库，移除后更不容易与 Conda 冲突。
    env["LD_LIBRARY_PATH"] = ":".join(
        item for item in env.get("LD_LIBRARY_PATH", "").split(":")
        if item and "isaacsim.ros2.bridge" not in item
    )
    print("[VLM] 调用本地 Qwen3-VL（不使用网络/API）", flush=True)
    subprocess.run(command, cwd=str(Path(__file__).resolve().parent), env=env, check=True)
    return json.loads(decision_path.read_text(encoding="utf-8"))


def publish_step(simulation, base, bridge, ranges, command: Velocity2D, publish: bool) -> None:
    base.set_velocity(command.vx, command.vy, command.wz)
    base.update(simulation.config.physics_dt)
    if publish:
        bridge.publish(simulation.sim_time, simulation.get_pose2d(), command, ranges)
    simulation.step()


def discover_targets(
    simulation,
    base,
    sensors,
    detector,
    bridge,
    perception_config: PerceptionConfig,
    sensor_config: SensorConfig,
    nav_config: NavigationConfig,
    output_dir: Path,
):
    """先静止观察；若没有看全，再原地转一圈补充目标。"""
    observations = {}
    ranges = np.full(sensor_config.lidar_bins, sensor_config.lidar_max_range, dtype=np.float32)
    publish_every = max(1, simulation.config.physics_hz // nav_config.publish_hz)
    lidar_every = max(1, simulation.config.physics_hz // nav_config.lidar_hz)
    detect_every = max(1, simulation.config.physics_hz // 5)
    settle_steps = int(perception_config.settle_seconds * simulation.config.physics_hz)
    total_steps = int(perception_config.scan_timeout * simulation.config.physics_hz)

    camera_logged = False
    for step in range(total_steps):
        bridge.spin_once()
        if step % lidar_every == 0:
            ranges = sensors.capture_scan()
        if step % detect_every == 0:
            image, depth = sensors.capture_camera()
            if image is not None:
                cv2.imwrite(str(output_dir / "latest_camera.jpg"), image)
                if not camera_logged:
                    finite_depth = depth[np.isfinite(depth) & (depth > 0)]
                    print(
                        f"[相机] image={image.shape}, depth={depth.shape}, "
                        f"valid_depth={len(finite_depth)}", flush=True,
                    )
                    camera_logged = True
                detections = detector.detect(image, depth, sensors.camera)
                for detection in detections:
                    old = observations.get(detection.color)
                    if old is None or detection.confidence > old[0].confidence:
                        observations[detection.color] = (detection, image.copy())
                        print(
                            f"[感知] {detection.color:<6} conf={detection.confidence:.2f} "
                            f"world=({detection.world_xyz[0]:.2f}, {detection.world_xyz[1]:.2f})",
                            flush=True,
                        )
        if all(color in observations for color in TARGET_COLORS):
            print("[感知] 红、蓝、黄三个目标均已找到", flush=True)
            break

        # 先静止等待相机稳定，缺目标时再使用第四章底盘控制原地扫描。
        command = Velocity2D(
            wz=0.0 if step < settle_steps else perception_config.scan_speed
        )
        publish_step(
            simulation, base, bridge, ranges, command, step % publish_every == 0
        )
    base.stop()
    return observations, ranges


def wait_for_nav2(simulation, base, bridge, sensors, ranges, nav_config) -> bool:
    deadline = time.monotonic() + nav_config.nav2_wait_timeout
    publish_every = max(1, simulation.config.physics_hz // nav_config.publish_hz)
    lidar_every = max(1, simulation.config.physics_hz // nav_config.lidar_hz)
    step = 0
    ready_since = None
    print("[Nav2] 等待导航服务器；现在可在终端 2 运行 run_nav2.sh", flush=True)
    while simulation.is_running() and time.monotonic() < deadline:
        bridge.spin_once()
        if bridge.nav2_ready():
            ready_since = ready_since or time.monotonic()
            if time.monotonic() - ready_since >= nav_config.nav2_activation_grace:
                print("[Nav2] navigate_to_pose 已激活", flush=True)
                return True
        else:
            ready_since = None
        if step % lidar_every == 0:
            ranges = sensors.capture_scan()
        publish_step(
            simulation, base, bridge, ranges, Velocity2D(), step % publish_every == 0
        )
        step += 1
    return False


def navigate(simulation, base, bridge, sensors, ranges, goal, nav_config) -> bool:
    bridge.send_goal(goal)
    deadline = time.monotonic() + nav_config.navigation_timeout
    publish_every = max(1, simulation.config.physics_hz // nav_config.publish_hz)
    lidar_every = max(1, simulation.config.physics_hz // nav_config.lidar_hz)
    step = 0
    while simulation.is_running() and time.monotonic() < deadline:
        command = bridge.spin_once()
        if step % lidar_every == 0:
            ranges = sensors.capture_scan()
        publish_step(
            simulation, base, bridge, ranges, command, step % publish_every == 0
        )
        if bridge.navigation_done:
            return bridge.navigation_succeeded
        step += 1
    print("[Nav2] 导航等待超时", flush=True)
    return False


def main() -> None:
    args = parse_args()
    if not args.instruction:
        args.instruction = input("请输入任务（例如：请导航到蓝色物体）：").strip()
    if not args.instruction:
        raise SystemExit("任务不能为空")

    args.yolo_model = str(_chapter_path(args.yolo_model, kind="YOLO-World 权重"))
    if not args.skip_vlm:
        args.vlm_model = str(_chapter_path(args.vlm_model, kind="Qwen3-VL 模型"))
        if not args.vlm_python:
            raise RuntimeError("未指定 VLM Python，请设置 VLM_PYTHON 或使用 --vlm-python")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_config = replace(SimulationConfig(), headless=args.headless)
    sensor_config = SensorConfig()
    perception_config = PerceptionConfig()
    nav_config = NavigationConfig()

    simulation = G2Chapter13Simulation(sim_config)
    sensors = detector = bridge = base = None
    try:
        geometry = RobotGeometry()
        kinematics = SwerveKinematics(geometry.wheel_positions, geometry.wheel_radius)
        base = G2BaseController(simulation.robot, kinematics, ControlLimits())
        sensors = G2Sensors(simulation, sensor_config)
        detector = YOLOWorldDetector(args.yolo_model, perception_config)
        print("[ROS 2] 正在创建 Nav2 bridge...", flush=True)
        bridge = Nav2Bridge(sensor_config, nav_config)
        print("[ROS 2] Nav2 bridge 已创建", flush=True)

        observations, ranges = discover_targets(
            simulation, base, sensors, detector, bridge,
            perception_config, sensor_config, nav_config, output_dir,
        )
        if not observations:
            raise RuntimeError("扫描结束仍未发现红、蓝、黄目标，请检查相机和 YOLO-World")

        board = make_observation_board(observations)
        board_path = output_dir / "target_observation_board.jpg"
        cv2.imwrite(str(board_path), board)
        summaries = [observations[color][0].summary() for color in TARGET_COLORS if color in observations]
        available = {item["color"] for item in summaries}

        if args.skip_vlm:
            target = keyword_target(args.instruction, available)
            decision = {"target": target or "none", "reason": "调试模式：颜色关键词匹配"}
        else:
            try:
                decision = run_local_vlm(args, board_path, summaries, output_dir)
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
                # 明确颜色指令可安全降级；模糊指令绝不猜测。
                target = keyword_target(args.instruction, available)
                if target is None:
                    raise RuntimeError(f"本地 VLM 失败且指令无法安全降级：{exc}") from exc
                decision = {
                    "target": target,
                    "reason": f"Qwen3-VL 运行失败，按明确颜色词安全降级：{exc}",
                    "fallback": True,
                }
                (output_dir / "vlm_decision.json").write_text(
                    json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
                )

        target_color = str(decision.get("target", "none"))
        print(f"[决策] target={target_color}, reason={decision.get('reason', '')}", flush=True)
        if target_color not in observations:
            raise RuntimeError("VLM 没有选择当前可见的有效目标，请把指令说得更明确")
        if args.perception_only:
            print(f"[完成] 感知与 VLM 决策完成，观察板：{board_path}")
            return

        if not wait_for_nav2(simulation, base, bridge, sensors, ranges, nav_config):
            raise RuntimeError("等待 Nav2 超时，请确认 run_nav2.sh 已启动")

        detection = observations[target_color][0]
        goal = make_standoff_goal(
            simulation.get_pose2d(), detection.world_xyz[:2], nav_config.stand_off_distance
        )
        success = navigate(simulation, base, bridge, sensors, ranges, goal, nav_config)
        final_pose = simulation.get_pose2d()
        print(
            f"[完成] success={success}，G2 最终位姿 "
            f"({final_pose.x:.2f}, {final_pose.y:.2f}, {final_pose.yaw:.2f})",
            flush=True,
        )
        if not success:
            raise SystemExit(1)
    except BaseException:
        # Isaac Sim close() 可能在退出阶段吞掉 Python traceback，因此先明确打印。
        import traceback
        traceback.print_exc()
        raise
    finally:
        if base is not None:
            base.stop()
        if bridge is not None:
            bridge.close()
        simulation.close()


if __name__ == "__main__":
    main()
