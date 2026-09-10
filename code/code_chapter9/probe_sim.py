"""不加载神经网络，验证仿真/相机/夹爪标定；导入本文件不会启动仿真。"""
import argparse
import json
import numpy as np
from config import ROOT, SimulationConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--headless', action='store_true')
    args = p.parse_args()
    from simulation import G2Simulation
    sim = G2Simulation(SimulationConfig(headless=args.headless), ['apple', 'soap', 'can'])
    try:
        cal = sim.calibrate_gripper(ROOT/'assets/g2_gripper.json')
        print('[标定结果]', json.dumps(cal), flush=True)
        frame = sim.camera.capture()
        (ROOT/'outputs').mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ROOT/'outputs/probe_frame.npz', **frame,
                            table_z=sim.scene.table_top, workspace=sim.scene.bounds)
        print('[probe] captured', frame['rgb'].shape, frame['depth'].shape,
              'table', sim.scene.table_top, flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        sim.close()


if __name__ == '__main__':
    main()
