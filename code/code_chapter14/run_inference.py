"""G2三路视觉 -> π0.5 -> 双臂/夹爪绝对关节控制。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import traceback

import numpy as np

try:
    from .config import ControlConfig, SimulationConfig, TaskConfig
    from .robot import G2Robot
    from .simulation import G2Simulation
    from .task_intent import TaskIntent
    from .vla_client import ChunkRunner, Pi05Client
except ImportError:
    from config import ControlConfig, SimulationConfig, TaskConfig
    from robot import G2Robot
    from simulation import G2Simulation
    from task_intent import TaskIntent
    from vla_client import ChunkRunner, Pi05Client


def main() -> None:
    parser = argparse.ArgumentParser(description="运行G2三色物块入盒任务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--control-waist", action="store_true", help="允许manipulation模型控制腰部5关节")
    parser.add_argument("--single-prompt", action="store_true", help="不拆分任务，一次提示三个物块")
    parser.add_argument(
        "--target-color", choices=("red", "green", "blue"),
        help="只抓取指定颜色并放入盒子，完成后立即结束；默认依次执行三种颜色",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-actions", action="store_true", help="打印模型夹爪原始动作")
    parser.add_argument(
        "--execute-chunk", type=int, default=0,
        help="每次执行多少个模型动作；0表示执行完整动作块（默认，与G2 baseline一致）",
    )
    parser.add_argument("--max-replans", type=int, default=TaskConfig().max_replans)
    args = parser.parse_args()

    sim = None
    try:
        print("[启动 1/4] 创建background.usda和桌面任务...", flush=True)
        sim = G2Simulation(
            SimulationConfig(headless=args.headless),
            TaskConfig(max_replans=args.max_replans),
        )
        robot = G2Robot(sim.robot)
        print("[启动 2/4] 连接π0.5服务...", flush=True)
        client = Pi05Client(args.host, args.port)
        if client.state_dim == 21 and not args.control_waist:
            print("[提示] manipulation模型输入保留腰部状态，但默认只执行前16维；可加--control-waist。", flush=True)
        if client.state_dim not in (16, 21):
            raise ValueError(f"服务state_dim={client.state_dim}不是支持的G2布局")
        if args.control_waist and client.action_dim < 21:
            raise ValueError("当前模型没有21维腰部输出，不能使用--control-waist")

        if args.execute_chunk < 0:
            raise ValueError("--execute-chunk不能小于0")
        control_config = replace(ControlConfig(), execute_chunk=args.execute_chunk)
        runner = ChunkRunner(robot, sim, control_config, args.control_waist)
        print("[启动 3/4] G2移动到数据集初始姿态...", flush=True)
        runner.move_home()
        intent = TaskIntent((args.target_color,)) if args.target_color else TaskIntent()
        task_name = f"{args.target_color}单物块入盒" if args.target_color else "三色物块入盒"
        print(f"[启动 4/4] 开始{task_name}视觉闭环推理...", flush=True)

        last_prompt = None
        for step in range(args.max_replans):
            prompt = intent.full_prompt if args.single_prompt else intent.prompt(sim.task.completed)
            if prompt != last_prompt:
                print(f"[动作意图] {prompt}", flush=True)
                last_prompt = prompt
            images = sim.cameras.capture()
            state = robot.state(include_waist=client.state_dim == 21)
            actions = client.infer(state, images, prompt)
            if args.debug_actions:
                runner.print_grippers(state, actions)
            if args.dry_run:
                delta = actions[0, :14] - state[:14]
                print(f"[DryRun] #{step + 1} {actions.shape=} max|dq|={np.max(np.abs(delta)):.3f}", flush=True)
                runner.hold()
            else:
                runner.execute(actions)
            if intent.is_complete(sim.task.completed):
                print(f"[完成] 第{step + 1}次重规划后完成{task_name}", flush=True)
                break
        else:
            print(f"[结束] 达到最大重规划次数，{task_name}未完成", flush=True)
    except Exception as error:
        print(f"[运行失败] {type(error).__name__}: {error}", flush=True)
        traceback.print_exc()
        raise SystemExit(1) from error
    finally:
        if sim is not None:
            sim.close()


if __name__ == "__main__":
    main()
