"""同步 RGB + 光轴深度 + K + T_world_camera；不混用 USD 与光学坐标。"""
import numpy as np
from kinematics import matrix_to_quaternion, quaternion_to_matrix


def look_at_optical(eye, target):
    z = np.asarray(target)-eye
    z = z / np.linalg.norm(z)
    x = np.cross(z,[0,0,1.])
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.,0,0])
    x /= np.linalg.norm(x)
    return np.column_stack([x,np.cross(z,x),z])  # x右、y下、z前


class RGBDCamera:
    def __init__(self, sim):
        from isaacsim.sensors.camera import Camera
        self.sim = sim
        target = sim.arm_to_world([.57,.57,-.20])
        eye = target + np.array([.25,-.05,.78])
        self.sensor = Camera('/World/Chapter9/Camera', name='rgbd9', frequency=sim.cfg.rendering_hz, resolution=sim.cfg.resolution)
        self.sensor.set_world_pose(eye, matrix_to_quaternion(look_at_optical(eye,target)), camera_axes='ros')
        self.sensor.set_focal_length(24.)
        self.sensor.set_horizontal_aperture(26.)
        self.sensor.set_vertical_aperture(26.*sim.cfg.resolution[1]/sim.cfg.resolution[0])
        self.sensor.set_clipping_range(.05, 5.)
        self.sensor.initialize()
        self.sensor.add_rgb_to_frame()
        self.sensor.add_distance_to_image_plane_to_frame()

    def capture(self):
        for _ in range(24):
            # 渲染到同一个 frame 后一次性读取；模型推理期间仿真不推进。
            self.sim.step()
            frame = self.sensor.get_current_frame()
            rgba, depth = frame.get('rgb', frame.get('rgba')), frame.get('distance_to_image_plane')
            if rgba is not None and depth is not None and np.asarray(rgba).size and np.asarray(depth).size:
                rgb = np.asarray(rgba)[...,:3].copy()
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb*255,0,255).astype(np.uint8)
                depth = np.asarray(depth).squeeze().astype(np.float32).copy()
                if depth.shape != rgb.shape[:2] or np.isfinite(depth).mean()<.1:
                    continue
                pos, quat = self.sensor.get_world_pose(camera_axes='ros')
                T = np.eye(4); T[:3,:3] = quaternion_to_matrix(quat); T[:3,3] = pos
                return dict(rgb=rgb, depth=depth, K=np.asarray(self.sensor.get_intrinsics_matrix()), T_world_camera=T,
                            frame_id=np.array(self.sim.ticks, dtype=np.int64),
                            sim_time=np.array(self.sim.ticks*self.sim.cfg.physics_dt))
        raise RuntimeError('RGB-D 无有效同步帧；检查 RTX 渲染、相机和 GPU')

    def enable_labels(self):
        self.sensor.add_instance_segmentation_to_frame()
