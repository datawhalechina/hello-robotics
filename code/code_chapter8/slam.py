"""教学版 LiDAR SLAM：漂移里程计、回环检测、ICP 与 SE(2) 位姿图优化。

"""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

try:
    from .config import Map2DConfig, Map3DConfig, SlamConfig
    from .geometry import Pose2D, between, compose, pose_distance, pose_from_vector, wrap_angle
    from .mapping_2d import OccupancyGrid2D
    from .mapping_3d import VoxelMap3D
    from .pointcloud import PointCloud, voxel_sample_indices
except ImportError:
    from config import Map2DConfig, Map3DConfig, SlamConfig
    from geometry import Pose2D, between, compose, pose_distance, pose_from_vector, wrap_angle
    from mapping_2d import OccupancyGrid2D
    from mapping_3d import VoxelMap3D
    from pointcloud import PointCloud, voxel_sample_indices


@dataclass
class Keyframe:
    index: int
    odom_pose: Pose2D
    optimized_pose: Pose2D
    cloud: PointCloud
    scan_xy: np.ndarray
    descriptor: np.ndarray


@dataclass(frozen=True)
class PoseConstraint:
    source: int
    target: int
    measurement: Pose2D
    information: tuple[float, float, float]
    kind: str = "odom"


@dataclass(frozen=True)
class LoopResult:
    source: int
    target: int
    relative_pose: Pose2D
    rmse: float
    inlier_ratio: float
    descriptor_score: float


class DriftedOdometry:
    """从仿真增量构造类似轮速里程计的累计误差。

    true_pose 只用于生成教学数据和误差评估；输出 pose 不会直接复制真值。
    后续接真实机器人时，可把 update() 的输入替换成第四章轮速里程计增量。
    """

    def __init__(self, config: SlamConfig = SlamConfig()) -> None:
        self.config = config
        self.pose = Pose2D()
        self._last_reference: Pose2D | None = None
        self._rng = np.random.default_rng(config.random_seed)

    def reset(self, reference_pose: Pose2D) -> Pose2D:
        self.pose = Pose2D(reference_pose.x, reference_pose.y, reference_pose.yaw)
        self._last_reference = reference_pose
        return self.pose

    def update(self, reference_pose: Pose2D) -> Pose2D:
        if self._last_reference is None:
            return self.reset(reference_pose)
        delta = between(self._last_reference, reference_pose)
        distance = math.hypot(delta.x, delta.y)
        noise = self._rng.normal(0.0, self.config.odom_noise_std, size=3)
        biased = Pose2D(
            x=delta.x * self.config.odom_linear_scale + float(noise[0]),
            y=delta.y * self.config.odom_lateral_scale + float(noise[1]),
            yaw=wrap_angle(
                delta.yaw * self.config.odom_yaw_scale
                + distance * self.config.odom_yaw_bias_per_meter
                + float(noise[2])
            ),
        )
        self.pose = compose(self.pose, biased)
        self._last_reference = reference_pose
        return self.pose


class PoseGraphSLAM:
    """关键帧、回环前端和小规模 SE(2) 位姿图后端。"""

    def __init__(self, config: SlamConfig = SlamConfig()) -> None:
        self.config = config
        self.keyframes: list[Keyframe] = []
        self.constraints: list[PoseConstraint] = []
        self.loop_results: list[LoopResult] = []

    def should_add_keyframe(self, pose: Pose2D) -> bool:
        if not self.keyframes:
            return True
        distance, yaw = pose_distance(self.keyframes[-1].odom_pose, pose)
        return distance >= self.config.keyframe_distance or yaw >= self.config.keyframe_yaw

    def add_keyframe(self, cloud: PointCloud, odom_pose: Pose2D) -> tuple[Keyframe, LoopResult | None]:
        scan_xy = prepare_scan_xy(cloud.xyz)
        optimized_pose = odom_pose
        if self.keyframes:
            previous = self.keyframes[-1]
            odom_increment = between(previous.odom_pose, odom_pose)
            optimized_pose = compose(previous.optimized_pose, odom_increment)
        keyframe = Keyframe(
            index=len(self.keyframes),
            odom_pose=odom_pose,
            optimized_pose=optimized_pose,
            cloud=copy_cloud(cloud),
            scan_xy=scan_xy,
            descriptor=scan_descriptor(scan_xy),
        )
        if self.keyframes:
            previous = self.keyframes[-1]
            self.constraints.append(
                PoseConstraint(
                    source=previous.index,
                    target=keyframe.index,
                    measurement=between(previous.odom_pose, keyframe.odom_pose),
                    information=(80.0, 80.0, 140.0),
                    kind="odom",
                )
            )
        self.keyframes.append(keyframe)

        loop = self._detect_loop(keyframe)
        if loop is not None:
            self.constraints.append(
                PoseConstraint(
                    source=loop.source,
                    target=loop.target,
                    measurement=loop.relative_pose,
                    information=(180.0, 180.0, 260.0),
                    kind="loop",
                )
            )
            self.loop_results.append(loop)
            self.optimize()
        return keyframe, loop

    def _detect_loop(self, current: Keyframe) -> LoopResult | None:
        gap = self.config.loop_min_keyframe_gap
        if current.index < gap:
            return None
        candidates = []
        for candidate in self.keyframes[: current.index - gap + 1]:
            distance, _ = pose_distance(candidate.optimized_pose, current.optimized_pose)
            if distance > self.config.loop_search_radius:
                continue
            score = descriptor_distance(candidate.descriptor, current.descriptor)
            if score <= self.config.loop_descriptor_threshold:
                candidates.append((score, distance, candidate))
        if not candidates:
            return None

        for score, _, candidate in sorted(candidates, key=lambda item: (item[0], item[1]))[:3]:
            initial = between(candidate.optimized_pose, current.optimized_pose)
            relative, rmse, inlier_ratio = icp_2d(
                source=current.scan_xy,
                target=candidate.scan_xy,
                initial_source_to_target=initial,
                max_correspondence=self.config.loop_icp_max_correspondence,
            )
            if (
                rmse <= self.config.loop_icp_max_rmse
                and inlier_ratio >= self.config.loop_icp_min_inlier_ratio
            ):
                return LoopResult(
                    source=candidate.index,
                    target=current.index,
                    relative_pose=relative,
                    rmse=rmse,
                    inlier_ratio=inlier_ratio,
                    descriptor_score=score,
                )
        return None

    def optimize(self) -> None:
        """高斯-牛顿优化；固定第 0 个节点消除规范自由度。"""
        node_count = len(self.keyframes)
        if node_count <= 1:
            return
        poses = [frame.optimized_pose for frame in self.keyframes]
        dimension = 3 * (node_count - 1)
        for _ in range(max(1, self.config.optimize_iterations)):
            hessian = np.zeros((dimension, dimension), dtype=np.float64)
            gradient = np.zeros((dimension,), dtype=np.float64)
            total_error = 0.0
            for edge in self.constraints:
                residual = constraint_residual(poses[edge.source], poses[edge.target], edge.measurement)
                jac_source, jac_target = numerical_jacobians(
                    poses[edge.source], poses[edge.target], edge.measurement
                )
                information = np.diag(np.asarray(edge.information, dtype=np.float64))
                # 回环用 Huber 权重抑制偶发错误匹配。
                norm = float(np.sqrt(residual @ information @ residual))
                robust = min(1.0, 2.5 / max(norm, 1e-12)) if edge.kind == "loop" else 1.0
                information *= robust
                total_error += float(residual @ information @ residual)
                self._accumulate(hessian, gradient, edge.source, edge.target, jac_source, jac_target, information, residual)
            hessian += np.eye(dimension) * 1e-6
            try:
                increment = np.linalg.solve(hessian, -gradient)
            except np.linalg.LinAlgError:
                increment = np.linalg.lstsq(hessian, -gradient, rcond=None)[0]
            for index in range(1, node_count):
                offset = 3 * (index - 1)
                vector = poses[index].as_vector() + increment[offset : offset + 3]
                poses[index] = pose_from_vector(vector)
            if np.max(np.abs(increment), initial=0.0) < 1e-5:
                break
        for frame, pose in zip(self.keyframes, poses):
            frame.optimized_pose = pose

    @staticmethod
    def _accumulate(hessian, gradient, source, target, jac_source, jac_target, information, residual):
        entries = ((source, jac_source), (target, jac_target))
        for node_a, jac_a in entries:
            if node_a == 0:
                continue
            slice_a = slice(3 * (node_a - 1), 3 * node_a)
            gradient[slice_a] += jac_a.T @ information @ residual
            for node_b, jac_b in entries:
                if node_b == 0:
                    continue
                slice_b = slice(3 * (node_b - 1), 3 * node_b)
                hessian[slice_a, slice_b] += jac_a.T @ information @ jac_b

    def raw_path(self) -> list[Pose2D]:
        return [frame.odom_pose for frame in self.keyframes]

    def optimized_path(self) -> list[Pose2D]:
        return [frame.optimized_pose for frame in self.keyframes]

    def save_trajectory(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            stream.write("index,odom_x,odom_y,odom_yaw,optimized_x,optimized_y,optimized_yaw\n")
            for frame in self.keyframes:
                stream.write(
                    f"{frame.index},{frame.odom_pose.x:.6f},{frame.odom_pose.y:.6f},"
                    f"{frame.odom_pose.yaw:.6f},{frame.optimized_pose.x:.6f},"
                    f"{frame.optimized_pose.y:.6f},{frame.optimized_pose.yaw:.6f}\n"
                )


class MappingBackend:
    """把位姿图、3D 地图和 2D 地图组合成一个易调用接口。"""

    def __init__(
        self,
        slam_config: SlamConfig = SlamConfig(),
        map3d_config: Map3DConfig = Map3DConfig(),
        map2d_config: Map2DConfig = Map2DConfig(),
        build_3d: bool = True,
        build_2d: bool = True,
    ) -> None:
        self.slam = PoseGraphSLAM(slam_config)
        self.map3d = VoxelMap3D(map3d_config) if build_3d else None
        self.map2d = OccupancyGrid2D(map2d_config) if build_2d else None

    def add_scan(self, cloud: PointCloud, odom_pose: Pose2D) -> tuple[bool, LoopResult | None]:
        if len(cloud) == 0 or not self.slam.should_add_keyframe(odom_pose):
            return False, None
        keyframe, loop = self.slam.add_keyframe(cloud, odom_pose)
        if loop is None:
            self._integrate(keyframe)
        else:
            self.rebuild_maps()
        return True, loop

    def _integrate(self, keyframe: Keyframe) -> None:
        if self.map3d is not None:
            self.map3d.integrate(keyframe.cloud, keyframe.optimized_pose)
        if self.map2d is not None:
            self.map2d.integrate(keyframe.cloud, keyframe.optimized_pose)

    def rebuild_maps(self) -> None:
        if self.map3d is not None:
            self.map3d.clear()
        if self.map2d is not None:
            self.map2d.clear()
        for keyframe in self.slam.keyframes:
            self._integrate(keyframe)


def copy_cloud(cloud: PointCloud) -> PointCloud:
    return PointCloud(cloud.xyz.copy(), cloud.intensity.copy(), cloud.origin.copy())


def prepare_scan_xy(points_xyz: np.ndarray, max_points: int = 420) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    # 只使用近似竖直结构做平面回环，减少地面/天花板对 ICP 的干扰。
    mask = (points[:, 2] >= 0.15) & (points[:, 2] <= 1.8)
    points = points[mask]
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    selected = voxel_sample_indices(points, 0.18)
    xy = points[selected, :2]
    if len(xy) > max_points:
        xy = xy[np.linspace(0, len(xy) - 1, max_points, dtype=np.int64)]
    return xy


def scan_descriptor(points_xy: np.ndarray, bins: int = 60, max_range: float = 20.0) -> np.ndarray:
    """方位角最小距离直方图；空 bin 记为 1，数值越小表示附近有障碍。"""
    descriptor = np.ones((bins,), dtype=np.float64)
    if len(points_xy) == 0:
        return descriptor
    ranges = np.linalg.norm(points_xy, axis=1)
    angles = np.arctan2(points_xy[:, 1], points_xy[:, 0])
    indices = np.floor((angles + math.pi) / (2.0 * math.pi) * bins).astype(np.int64) % bins
    for index, distance in zip(indices, ranges):
        descriptor[index] = min(descriptor[index], min(float(distance), max_range) / max_range)
    return descriptor


def descriptor_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("描述子维度必须相同")
    return float(np.mean(np.abs(a - b)))


def icp_2d(
    source: np.ndarray,
    target: np.ndarray,
    initial_source_to_target: Pose2D = Pose2D(),
    max_correspondence: float = 0.5,
    max_iterations: int = 18,
) -> tuple[Pose2D, float, float]:
    """点到点 ICP，返回 source 坐标系到 target 坐标系的变换。"""
    source = np.asarray(source, dtype=np.float64).reshape((-1, 2))
    target = np.asarray(target, dtype=np.float64).reshape((-1, 2))
    if len(source) < 12 or len(target) < 12:
        return initial_source_to_target, float("inf"), 0.0
    estimate = initial_source_to_target
    rmse, inlier_ratio = float("inf"), 0.0
    for _ in range(max_iterations):
        transformed = estimate.transform_points(source)
        nearest_indices, distances = nearest_neighbors(transformed, target)
        inliers = distances <= max_correspondence
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = inlier_count / float(len(source))
        if inlier_count < 10:
            return estimate, float("inf"), inlier_ratio
        matched_source = transformed[inliers]
        matched_target = target[nearest_indices[inliers]]
        delta = best_fit_transform_2d(matched_source, matched_target)
        estimate = compose(delta, estimate)
        rmse = float(np.sqrt(np.mean(distances[inliers] ** 2)))
        if math.hypot(delta.x, delta.y) < 1e-4 and abs(delta.yaw) < 1e-4:
            break
    return estimate, rmse, inlier_ratio


def nearest_neighbors(source: np.ndarray, target: np.ndarray, chunk_size: int = 128):
    indices = np.empty((len(source),), dtype=np.int64)
    distances = np.empty((len(source),), dtype=np.float64)
    for start in range(0, len(source), chunk_size):
        chunk = source[start : start + chunk_size]
        squared = np.sum((chunk[:, None, :] - target[None, :, :]) ** 2, axis=2)
        local = np.argmin(squared, axis=1)
        indices[start : start + len(chunk)] = local
        distances[start : start + len(chunk)] = np.sqrt(squared[np.arange(len(chunk)), local])
    return indices, distances


def best_fit_transform_2d(source: np.ndarray, target: np.ndarray) -> Pose2D:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return Pose2D(
        x=float(translation[0]),
        y=float(translation[1]),
        yaw=math.atan2(float(rotation[1, 0]), float(rotation[0, 0])),
    )


def constraint_residual(source: Pose2D, target: Pose2D, measurement: Pose2D) -> np.ndarray:
    predicted = between(source, target)
    return np.array(
        (
            predicted.x - measurement.x,
            predicted.y - measurement.y,
            wrap_angle(predicted.yaw - measurement.yaw),
        ),
        dtype=np.float64,
    )


def numerical_jacobians(source: Pose2D, target: Pose2D, measurement: Pose2D, eps: float = 1e-6):
    base = constraint_residual(source, target, measurement)
    jac_source = np.zeros((3, 3), dtype=np.float64)
    jac_target = np.zeros((3, 3), dtype=np.float64)
    for axis in range(3):
        source_vector = source.as_vector()
        source_vector[axis] += eps
        delta = constraint_residual(pose_from_vector(source_vector), target, measurement) - base
        delta[2] = wrap_angle(delta[2])
        jac_source[:, axis] = delta / eps

        target_vector = target.as_vector()
        target_vector[axis] += eps
        delta = constraint_residual(source, pose_from_vector(target_vector), measurement) - base
        delta[2] = wrap_angle(delta[2])
        jac_target[:, axis] = delta / eps
    return jac_source, jac_target
