"""
Depth camera handler: acquires depth images, computes point clouds,
and provides ROS2 PointCloud2 messages.
"""

import os

import numpy as np


class DepthSensor:
    """Depth camera acquisition and point cloud processing.

    Uses the RobotInterface depth annotators for direct access to
    Isaac Sim's distance_to_image_plane data.
    """

    def __init__(self, robot_interface, camera_id, camera_prim, resolution, max_depth=5.0):
        """
        Args:
            robot_interface: RobotInterface instance with registered cameras
            camera_id: camera identifier string (e.g., "head_front_camera")
            camera_prim: USD prim path of the camera
            resolution: [width, height]
            max_depth: maximum depth in meters to include in point cloud
        """
        self.robot_interface = robot_interface
        self.camera_id = camera_id
        self.camera_prim = camera_prim
        self.width = resolution[0]
        self.height = resolution[1]
        self.max_depth = max_depth

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

    def compute_intrinsics(self):
        """Compute camera intrinsics from the USD camera prim."""
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath(self.camera_prim)
        camera = UsdGeom.Camera(cam_prim)

        focal_length = camera.GetFocalLengthAttr().Get()
        h_aperture = camera.GetHorizontalApertureAttr().Get()
        v_aperture = camera.GetVerticalApertureAttr().Get()

        self.fx = self.width * focal_length / h_aperture
        self.fy = self.height * focal_length / v_aperture
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        print(f"[DepthSensor] Intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, "
              f"cx={self.cx:.1f}, cy={self.cy:.1f}")
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}

    def get_depth_image(self):
        """Get the current depth image.

        Returns:
            np.ndarray: (H, W) float32 depth in meters, or None if unavailable.
        """
        if self.camera_id not in self.robot_interface.depth_annotators:
            print(f"[DepthSensor] No depth annotator for {self.camera_id}")
            return None
        data = self.robot_interface.depth_annotators[self.camera_id].get_data()
        if data is None or data.size == 0:
            return None
        return data.squeeze().astype(np.float32)

    def get_rgb_image(self):
        """Get the current RGB image.

        Returns:
            np.ndarray: (H, W, 3) uint8 RGB, or None if unavailable.
        """
        return self.robot_interface.get_camera_image_rgb(self.camera_id)

    def depth_to_pointcloud(self, depth_image):
        """Convert a depth image to a 3D point cloud.

        Args:
            depth_image: (H, W) float32 depth in meters

        Returns:
            np.ndarray: (N, 3) float32 point cloud in camera frame
        """
        if self.fx is None:
            self.compute_intrinsics()

        h, w = depth_image.shape
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        u, v = np.meshgrid(u, v)

        valid = np.isfinite(depth_image) & (depth_image > 0) & (depth_image < self.max_depth)
        z = depth_image[valid]
        x = (u[valid] - self.cx) * z / self.fx
        y = (v[valid] - self.cy) * z / self.fy

        return np.stack([x, y, z], axis=-1)

    def capture_and_process(self):
        """Capture depth image, convert to point cloud.

        Returns:
            dict with 'depth_image', 'rgb_image', 'pointcloud', 'intrinsics'
            or None if capture failed.
        """
        depth = self.get_depth_image()
        if depth is None:
            print("[DepthSensor] Failed to capture depth image")
            return None

        rgb = self.get_rgb_image()
        points = self.depth_to_pointcloud(depth)

        result = {
            "depth_image": depth,
            "rgb_image": rgb,
            "pointcloud": points,
            "intrinsics": {
                "fx": self.fx, "fy": self.fy,
                "cx": self.cx, "cy": self.cy,
            },
        }
        print(f"[DepthSensor] Captured depth: {depth.shape}, "
              f"point cloud: {points.shape[0]} points")
        return result

    def save_pointcloud_ply(self, points, filepath):
        """Save point cloud to PLY file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            o3d.io.write_point_cloud(filepath, pcd)
        except ImportError:
            with open(filepath, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("end_header\n")
                for p in points:
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        print(f"[DepthSensor] Saved point cloud to {filepath}")

    @staticmethod
    def create_pointcloud2_msg(points, header):
        """Create a sensor_msgs/PointCloud2 message from points."""
        from sensor_msgs.msg import PointCloud2, PointField

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = True
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = points.astype(np.float32).tobytes()
        return msg

    @staticmethod
    def create_depth_colormap(depth_image, max_depth=5.0):
        """Create a colorized depth image for visualization."""
        import cv2

        depth_normalized = np.clip(depth_image / max_depth, 0, 1)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        invalid = ~np.isfinite(depth_image) | (depth_image <= 0)
        colormap[invalid] = [0, 0, 0]
        return colormap
