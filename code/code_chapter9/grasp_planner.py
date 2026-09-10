"""可解释的几何抓取生成与统一筛选；GraspGenX 输出也必须经过这里。"""
from dataclasses import dataclass
import numpy as np
from config import GripperGeometry, PlannerConfig
from pointcloud import estimate_pose, validate_transform

@dataclass
class Grasp:
    T: np.ndarray                 # world <- G，G=center_link
    width: float
    score: float
    object_id: int = 0
    source: str = "geometry"


def gripper_collision(T, scene, plane, geom, width=None):
    """夹指和掌部保守盒 vs 观测点云 + 无限桌平面，不等于完整网格碰撞检测。"""
    width = geom.max_width if width is None else width
    q = (np.asarray(scene) - T[:3,3]) @ T[:3,:3]
    m = geom.margin
    finger = ((abs(q[:,0]) >= width/2-m) &
              (abs(q[:,0]) <= width/2+geom.finger_thickness+m) &
              (abs(q[:,1]) <= geom.finger_depth/2+m) &
              (q[:,2] >= geom.finger_z_min-m) & (q[:,2] <= geom.finger_z_max+m))
    palm = ((abs(q[:,0]) <= geom.max_width/2+geom.finger_thickness+m) &
            (abs(q[:,1]) <= .04+m) &
            (q[:,2] >= geom.palm_z_min-m) & (q[:,2] <= geom.palm_z_max+m))
    if np.any(finger | palm):
        return True
    # 检查指尖四角到桌面的有符号距离，防止稀疏深度漏掉桌碰撞。
    corners = np.array([[x,y,z] for x in (-geom.max_width/2-geom.finger_thickness, geom.max_width/2+geom.finger_thickness)
                        for y in (-geom.finger_depth/2, geom.finger_depth/2)
                        for z in (geom.finger_z_min, geom.finger_z_max)])
    world = corners @ T[:3,:3].T + T[:3,3]
    return bool(np.any(world @ plane[:3] + plane[3] < m))


def filter_grasps(candidates, scene, plane, geom=None, cfg=None):
    geom, cfg = geom or GripperGeometry(), cfg or PlannerConfig()
    accepted = []
    for g in candidates:
        try:
            validate_transform(g.T)
        except ValueError:
            continue
        if not np.isfinite([g.width, g.score]).all() or not cfg.min_width <= g.width <= geom.max_width - 2*geom.margin:
            continue
        if -g.T[:3,2] @ plane[:3] < np.cos(np.deg2rad(cfg.max_tilt_deg)):
            continue  # 本章不执行未经验证的侧抓、倒抓
        if any(gripper_collision(_offset(g.T, -d*g.T[:3,2]), scene, plane, geom)
               for d in np.linspace(0, cfg.pregrasp, 13)):
            continue
        accepted.append(g)
    # 按实例轮转，防止一个不可达目标占满候选预算。
    groups = {}
    for g in sorted(accepted, key=lambda g: g.score, reverse=True):
        groups.setdefault(g.object_id, []).append(g)
    balanced = []
    while groups and len(balanced) < cfg.max_candidates:
        for key in list(groups):
            balanced.append(groups[key].pop(0))
            if not groups[key]: del groups[key]
            if len(balanced) == cfg.max_candidates: break
    return balanced


def _offset(T, delta):
    out = T.copy()
    out[:3,3] += delta
    return out


def geometric_grasps(points, plane, object_id=0, geom=None):
    """主轴/短轴、多偏航与多接触高度采样；宽度从点云投影得到，不按类别写死。"""
    geom = geom or GripperGeometry()
    pose = estimate_pose(points, plane)
    main_yaw = np.arctan2(pose.rotation[1,0], pose.rotation[0,0])
    yaws = main_yaw + np.arange(0, 2*np.pi, np.pi/6)
    bottom = pose.center[2] - pose.height/2
    result = []
    for yaw in yaws:
        x = np.array([np.cos(yaw), np.sin(yaw), 0.])
        z = -plane[:3]
        x -= (x @ z)*z
        x /= np.linalg.norm(x)
        R = np.column_stack([x, np.cross(z,x), z])
        projection = (points - pose.center) @ x
        lo, hi = np.percentile(projection, [1,99])
        width = hi-lo + .006
        for fraction in (.55, .72):
            T = np.eye(4)
            T[:3,:3] = R
            T[:3,3] = pose.center + x*((lo+hi)/2)
            T[2,3] = max(bottom + pose.height*fraction, bottom+geom.finger_z_max+.010)
            if T[2,3] > bottom+pose.height+.010:
                continue
            # 小宽度/深一点的夹持优先，分数只是几何启发式而非成功概率。
            score = 1 - width/geom.max_width*.4 - abs(fraction-.55)*.2
            result.append(Grasp(T, float(width), score, object_id))
    return result, pose
