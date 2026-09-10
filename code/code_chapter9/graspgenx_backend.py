"""官方 GraspGenX 真正的推理适配，不用其他夹爪冒充 G2。

读取 simulation 导出的 G2 sweep-volume，使用官方 raw descriptor API。
仅支持 sweep-volume 条件的检查点；不使用其 dummy mesh 做碰撞检测。
"""
import json
import os
import numpy as np
from config import ROOT
from grasp_planner import Grasp

class GraspGenXBackend:
    def __init__(self, calibration=None, device="cuda:0"):
        import torch
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("本章 GraspGenX 官方后端需要 CUDA GPU")
        torch.cuda.set_device(device)
        torch.manual_seed(9)
        path = ROOT/'assets/g2_gripper.json'
        self.cal = calibration or json.loads(path.read_text())
        weights = ROOT/'weights/graspgenx'
        for d in ('gen','dis'):
            if not (weights/'release'/d/'config.yaml').is_file() or not list((weights/'release'/d).glob('*.pth')):
                raise FileNotFoundError(f'缺少 GraspGenX 权重：{weights}/release/{d}')
        # 显式目录避免 import 时自动联网下载外部仓库。扫掠体后端不需要额外夹爪资产。
        os.environ['GRASPGENX_CHECKPOINT_DIR'] = str(weights)
        os.environ['GRASPGENX_GRIPPER_CFG_DIR'] = str(ROOT/'assets')
        from graspgenx.grasp_server import GraspGenXSampler, SWEEP_VOLUME_ONLY_BACKBONES
        from graspgenx.x_grippers import make_sweep_volume_gripper_info
        from graspgenx.utils.checkpoint_io import load_model_cfg
        cfg = load_model_cfg(weights/'release/gen',weights/'release/dis')
        for part in (cfg.diffusion,cfg.discriminator):
            if part.gripper_backbone not in SWEEP_VOLUME_ONLY_BACKBONES:
                raise ValueError('此检查点还需要夹爪点云/TSDF，不能用扫掠体占位符代替')
        keys = ('extents_open','offset_open','extents_mid','offset_mid','gripper_type','fingertip_depth')
        info = make_sweep_volume_gripper_info(**{k:self.cal[k] for k in keys},name='g2_measured')
        self.sampler = GraspGenXSampler(cfg,gripper_info=info)

    def generate(self, points, object_id=0):
        from graspgenx.samplers import run_planner_on_object
        # 严格按官方 demo：点云减均值，生成后平移加回；不能重复居中。
        center = points.mean(axis=0)
        transforms,scores,_,_ = run_planner_on_object((points-center).astype(np.float32),self.sampler,
                                                      planner='diffusion',num_grasps=256,topk_num_grasps=64)
        result = []
        B_G = np.eye(4); B_G[:3,3] = self.cal['base_to_tcp']
        for T, score in zip(transforms,scores):
            T = T.astype(float).copy(); T[:3,3] += center
            T = T @ B_G  # 官方输出 gripper base 位姿 -> 第5章 center_link TCP
            local = (points-T[:3,3]) @ T[:3,:3]
            # 仅对夹指实际接触高度附近估计开口需求，而不是整物体投影。
            contact = local[abs(local[:,2])<.035]
            if len(contact)<12: continue
            lo,hi = np.percentile(contact[:,0],[1,99])
            result.append(Grasp(T,float(hi-lo+.006),float(score),object_id,'graspgenx'))
        return result
