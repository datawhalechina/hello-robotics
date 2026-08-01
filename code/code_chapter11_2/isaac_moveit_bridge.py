"""终端 1：加载 Isaac 场景，并把 G2 右臂暴露给 MoveIt 2。"""

import argparse

try:
    from .config import SimulationConfig
    from .simulation import G2MoveItSimulation
    from .trajectory_action_server import IsaacTrajectoryTopicBridge
    from .perception import G2HeadDepthPerception
except ImportError:
    from config import SimulationConfig
    from simulation import G2MoveItSimulation
    from trajectory_action_server import IsaacTrajectoryTopicBridge
    from perception import G2HeadDepthPerception


def main():
    parser = argparse.ArgumentParser(description="G2 Isaac Sim <-> MoveIt 2 bridge")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    sim = G2MoveItSimulation(SimulationConfig(headless=args.headless))
    perception = G2HeadDepthPerception(sim)
    bridge = IsaacTrajectoryTopicBridge(sim.robot, sim, perception)
    try:
        print("[Bridge] 等待 MoveIt 2 规划和执行命令……", flush=True)
        while sim.is_running():
            bridge.update(sim.config.physics_dt)
            sim.step()
    finally:
        bridge.close()
        sim.close()


if __name__ == "__main__":
    main()
