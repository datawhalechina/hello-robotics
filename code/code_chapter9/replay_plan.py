"""不启动仿真，用保存的机器人状态和视觉点云检查完整路径。"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from config import GripperGeometry
from controller import PickController, PlanningError
from kinematics import G2ArmKinematics
from worker import load_grasps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, help='含 robot_state.npz/calibration.json 的运行目录')
    p.add_argument('--pick', default='pick_00')
    args = p.parse_args()
    state_path=args.run/args.pick/'robot_state.npz'
    with np.load(state_path if state_path.exists() else args.run/'robot_state.npz') as f: state = dict(f)
    with np.load(args.run/args.pick/'result.npz') as f: data = dict(f)
    def arm(side):
        return SimpleNamespace(kinematics=G2ArmKinematics(side), get_joint_positions=lambda: state[side].copy())
    sim = SimpleNamespace(T_world_base=state['T_world_base'], T_base_world=np.linalg.inv(state['T_world_base']),
                          arm=arm('right'), left=arm('left'),
                          scene=SimpleNamespace(bounds=state['bounds'], place=state['place'], table_top=float(state['table_z'])))
    geometry = GripperGeometry(**json.loads((args.run/'calibration.json').read_text())['geometry'])
    controller = PickController(sim, geometry)
    for i, g in enumerate(load_grasps(data)):
        try:
            plan = controller.plan(g, data['scene'], data['plane'], data[f'object_{g.object_id}'])
            print(i, str(data['labels'][g.object_id]), 'PLANNED', flush=True)
            break
        except PlanningError as exc:
            print(i, str(data['labels'][g.object_id]), str(exc), flush=True)

if __name__ == '__main__': main()
