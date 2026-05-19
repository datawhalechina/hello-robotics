# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""RobotInterface ROS2 node: joint state publishing, TF broadcasting, camera RGB."""

from g2_robot.utils.logger import Logger
from g2_robot.core.constants import MAP_DYNAMIC_TF_NAMES

logger = Logger()

from omni.usd import get_world_transform_matrix
import omni.replicator.core as rep
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.prims import SingleXFormPrim, SingleGeometryPrim, SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path, get_prim_object_type
from g2_robot.utils.usd_utils import *

import numpy as np

from cv_bridge import CvBridge
from pxr import Gf, Sdf, UsdPhysics

from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import TransformStamped, Point, Vector3, Quaternion
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster


class RobotInterface(Node):
    """ROS2 node for joint state publishing, TF broadcasting, and camera RGB.

    Provides:
        - Joint state publishing (/joint_states, /joint_states_ee)
        - Static and dynamic TF broadcasting
        - Camera RGB image publishing via rclpy
        - Articulated object joint state publishing
    """

    def __init__(self):
        super().__init__("geniesim_sensor_node")

        self._sec = 0
        self._nanosec = 0
        self._header = Header(frame_id="base_link")

        self._articulation = None

        self._bridge = CvBridge()

        self._static_tf_tree = None
        self._dynamic_tf_tree = None

        self._enable_ros_pub = True

        # cache
        self.annotators = {}
        self.depth_annotators = {}
        self.parameters = {}
        self.publisher_map = {}

        self.articulated_obj_publishers = []
        self.articulated_objs = []

        # JointState for the main robot
        self._js_msg = JointState()
        self._js_msg.name = []
        self._js_msg.position = []
        self._js_msg.velocity = []
        self._js_msg.effort = []

        # One JointState per articulated object
        self._joint_state_cache = {}

        # Pre-allocated Image messages (one per camera)
        self._img_data_cache = {}
        self._img_msg_cache = {}

        # Pre-built TransformStamped lists for static & dynamic TFs
        self._static_tfs_prebuilt = []
        self._dynamic_tfs_prebuilt = []

        # Pre-built Python containers that never change size
        self._dof_pos_cache = None
        self._dof_vel_cache = None
        self._dof_eff_cache = None

    def disable_ros_pub(self):
        self._enable_ros_pub = False

    def register_joint_state(self, articulation):
        self._articulation = articulation
        self._articulation_dof_names = articulation.dof_names
        self._articulation_pos = self._articulation.get_joint_positions()
        self._articulation_vel = self._articulation.get_joint_velocities()
        self._articulation_eff = self._articulation.get_measured_joint_efforts()

        self.pub_js = self.create_publisher(JointState, "/joint_states", 1)
        self.pub_ee = self.create_publisher(JointState, "/joint_states_ee", 1)

        self._js_msg.name = articulation.dof_names
        N = len(self._js_msg.name)
        self._dof_pos_cache = np.empty(N, dtype=np.float64)
        self._dof_vel_cache = np.empty(N, dtype=np.float64)
        self._dof_eff_cache = np.empty(N, dtype=np.float64)
        self._js_msg.position = [0.0] * N
        self._js_msg.velocity = [0.0] * N
        self._js_msg.effort = [0.0] * N

        self._metadata = articulation._articulation_view._metadata
        self.joint_names_ee = ["idx51_ee_l_joint", "idx91_ee_r_joint"]
        self.joint_indices_ee = 1 + np.array([self._metadata.joint_indices[jn] for jn in self.joint_names_ee])

    def register_articulated_obj(self, articulated_objs):
        for prim_path, articulation in articulated_objs.items():
            self.articulated_objs.append(articulation)
            self.articulated_obj_publishers.append(
                self.create_publisher(JointState, f"/articulated/{prim_path.split('/')[-1]}", 1)
            )

    def register_robot_tf(self, stage, robot_ns):
        robot_ns = robot_ns.replace("/", "")
        if not self._articulation:
            logger.error("register_robot_tf failed before articulation is initialized")
            return
        self.all_joints = []
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                self.all_joints.append(prim)

        self._static_tf_tree = []
        self._dynamic_tf_tree = []
        self.articulat_objects = {}
        self.build_tf_tree(stage, stage.GetPrimAtPath(f"/{robot_ns}/base_link"), None, None)

        def _build_tf_list(tf_tree):
            tfs = []
            for prim, parent in tf_tree:
                tf = TransformStamped()
                tf.header.frame_id = parent.GetName() if parent else "odom"
                if prim:
                    name = prim.GetName()
                    if "link" in name or "Camera" in name:
                        tf.child_frame_id = prim.GetName()
                    else:
                        tf.child_frame_id = str(prim.GetPrimPath()).split("/")[-2]
                    tfs.append(tf)
            return tfs

        self._dynamic_tf_tree.append(
            (stage.GetPrimAtPath(f"/genie/arm_l_end_link"), None)
        )
        self._dynamic_tf_tree.append(
            (stage.GetPrimAtPath(f"/genie/arm_r_end_link"), None)
        )
        self._dynamic_tf_tree.append(
            (stage.GetPrimAtPath(f"/genie/arm_base_link"), None)
        )
        self._dynamic_tf_tree.append(
            (stage.GetPrimAtPath(f"/genie/head_link3/head_front_Camera"), None)
        )
        self._dynamic_tf_tree.append(
            (stage.GetPrimAtPath(f"/genie/gripper_l_base_link/Left_Camera"), None)
        )
        self._dynamic_tf_tree.append(
            (stage.GetPrimAtPath(f"/genie/gripper_r_base_link/Right_Camera"), None)
        )
        for prim in stage.Traverse():
            prim_path = str(prim.GetPrimPath())
            prim_type = get_prim_object_type(prim_path)
            if prim_type == "articulation" and prim_path.startswith("/World/Objects"):
                self.articulat_objects[prim_path] = SingleArticulation(prim_path)
                print("DOF", self.articulat_objects[prim_path].num_dof)
        logger.info(f"record {len(self.articulat_objects)} articulated prim(s) TF:")

        self.register_articulated_obj(self.articulat_objects)

        # enable tf pub
        rigidbody_collider_prims = get_rigidbody_collider_prims(
            robot_name="",
            extra_prim_paths=[],
        )
        logger.info(f"record {len(rigidbody_collider_prims)} prim(s) TF with RigidBody and Collider:")
        for prim in rigidbody_collider_prims:
            if prim.IsValid() and prim.IsActive():
                self.register_obj_tf(prim)

        self._static_tfs_prebuilt = _build_tf_list(self._static_tf_tree)
        self._dynamic_tfs_prebuilt = _build_tf_list(self._dynamic_tf_tree)

        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.dynamic_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster.sendTransform(self._static_tfs_prebuilt)

    def register_obj_tf(self, object_prim):
        self._dynamic_tf_tree.append((object_prim, None))

    def build_tf_tree(self, stage, prim, parent, joint_name):
        prim_name = prim.GetName().split("/")[-1]
        if joint_name in MAP_DYNAMIC_TF_NAMES or "base_link" in prim_name:
            self._dynamic_tf_tree.append((prim, parent))
        else:
            self._static_tf_tree.append((prim, parent))

        for child in self.all_joints:
            joint = UsdPhysics.Joint(child)
            _joint_name = child.GetName().split("/")[-1]
            if "Joint" in _joint_name:
                continue
            first = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
            if first == prim:
                second = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
                if not (
                    any(t[0] == second for t in self._static_tf_tree)
                    or any(t[0] == second for t in self._dynamic_tf_tree)
                ):
                    self.build_tf_tree(stage, second, first, _joint_name)

    def register_camera(self, camera_prim, resolution, every_n_frame):
        try:
            rp = rep.create.render_product(camera_prim, (resolution[0], resolution[1]))
            camera_id = camera_prim.split("/")[-1].lower()
            camera_param = {
                "path": camera_prim,
                "every_n_frame": every_n_frame,
                "resolution": {
                    "width": resolution[0],
                    "height": resolution[1],
                },
                "topic_name": {
                    "rgb": "genie_sim/" + camera_id + "_rgb",
                    "depth": "genie_sim/" + camera_id + "_depth",
                },
                "publish": [
                    "rgb:/" + camera_id + "_rgb",
                    "depth:/" + camera_id + "_depth",
                ],
            }

            if "Fisheye" in camera_prim or "Top" in camera_prim:
                camera_param["publish"] = [
                    "rgb:/" + camera_id + "_rgb",
                ]
            else:
                camera_param["publish"] = [
                    "rgb:/" + camera_id + "_rgb",
                    "depth:/" + camera_id + "_depth",
                ]

            self.parameters[camera_id] = camera_param
            self.publisher_map[camera_id] = self.create_publisher(Image, camera_param["topic_name"]["rgb"], 1)

        except Exception as e:
            logger.warning(f"Failed to register camera {camera_prim}: {e}")
            return

        self.annotators[camera_id] = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotators[camera_id].attach(rp)

        if "Fisheye" not in camera_prim and "Top" not in camera_prim:
            self.depth_annotators[camera_id] = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            self.depth_annotators[camera_id].attach(rp)

        img = Image()
        img.header.frame_id = "camera_optical_frame"
        img.width = resolution[0]
        img.height = resolution[1]
        img.encoding = "rgba8"
        img.step = resolution[0] * 4
        self._img_msg_cache[camera_id] = img
        self._img_data_cache[camera_id] = np.empty((resolution[1], resolution[0], 4), dtype=np.uint8)

    def tick(self, current_time: float, current_step_index: int):
        """Tick all publishers. Call each physics step."""
        self._current_step_index = current_step_index
        self._sec = int(current_time)
        self._nanosec = int((current_time - self._sec) * 1e9)

        self._header.stamp.sec = self._sec
        self._header.stamp.nanosec = self._nanosec

        self.prepare_data()

        if self._enable_ros_pub:
            self.pub_joint_state(self.pub_js, self._articulation)
            self.pub_joint_state_ee(self._articulation)
            self.pub_tf()
            self.pub_articulated_object()
            for cam in self.publisher_map:
                if 0 == self._current_step_index % self.parameters[cam]["every_n_frame"]:
                    self.pub_camera(cam)

    def prepare_data(self):
        if self._articulation is None:
            return

        # joint_state
        self._articulation_pos = self._articulation.get_joint_positions()
        self._articulation_vel = self._articulation.get_joint_velocities()
        self._articulation_eff = self._articulation.get_measured_joint_efforts()

        np.copyto(self._dof_pos_cache, np.asarray(self._articulation_pos, dtype=np.float64))
        np.copyto(self._dof_vel_cache, np.asarray(self._articulation_vel, dtype=np.float64))
        np.copyto(self._dof_eff_cache, np.asarray(self._articulation_eff, dtype=np.float64))

        # tf
        for tf, (prim, parent) in zip(self._dynamic_tfs_prebuilt, self._dynamic_tf_tree):
            tf.header.stamp = self._header.stamp
            matrix = get_world_transform_matrix(prim)
            if parent:
                matrix = matrix * (get_world_transform_matrix(parent).GetInverse())
            translate = matrix.ExtractTranslation()
            orient = matrix.ExtractRotationQuat()
            tf.transform.translation.x = translate[0]
            tf.transform.translation.y = translate[1]
            tf.transform.translation.z = translate[2]
            tf.transform.rotation.x = orient.imaginary[0]
            tf.transform.rotation.y = orient.imaginary[1]
            tf.transform.rotation.z = orient.imaginary[2]
            tf.transform.rotation.w = orient.real

        # cam
        for cam in self.publisher_map:
            if 0 == self._current_step_index % self.parameters[cam]["every_n_frame"]:
                img = self.annotators[cam].get_data()
                if img is None or img.size == 0:
                    continue
                np.copyto(self._img_data_cache[cam], img)

    def publish_transforms(self, tf_tree, broadcaster):
        transforms = []
        if broadcaster == self.static_broadcaster:
            tf = TransformStamped()
            tf.header.frame_id = "map"
            tf.child_frame_id = "odom"
            tf.header.stamp = self._header.stamp
            tf.transform.translation = Vector3(x=0.0, y=0.0, z=0.0)
            tf.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            transforms.append(tf)
        for prim, parent in tf_tree:
            tf = TransformStamped()
            tf.header.frame_id = parent.GetName() if parent else "odom"
            tf.header.stamp = self._header.stamp
            tf.child_frame_id = prim.GetName()
            matrix = get_world_transform_matrix(prim)
            if parent:
                matrix = matrix * (get_world_transform_matrix(parent).GetInverse())
            translate = matrix.ExtractTranslation()
            orient = matrix.ExtractRotationQuat()
            tf.transform.translation = Vector3(x=translate[0], y=translate[1], z=translate[2])
            tf.transform.rotation = Quaternion(
                x=orient.imaginary[0],
                y=orient.imaginary[1],
                z=orient.imaginary[2],
                w=orient.real,
            )
            transforms.append(tf)
        broadcaster.sendTransform(transforms)

    def pub_tf(self):
        if not self._dynamic_tfs_prebuilt:
            return
        self.dynamic_broadcaster.sendTransform(self._dynamic_tfs_prebuilt)

    def pub_joint_state(self, pub, articulation):
        if articulation is None:
            return

        if pub is self.pub_js:
            msg = self._js_msg
            msg.position = self._dof_pos_cache.tolist()
            msg.velocity = self._dof_vel_cache.tolist()
            msg.effort = self._dof_eff_cache.tolist()
        else:
            msg = JointState()
            msg.name = articulation.dof_names
            msg.position = articulation.get_joint_positions().tolist()
            msg.velocity = articulation.get_joint_velocities().tolist()
            msg.effort = articulation.get_measured_joint_efforts().tolist()

        msg.header = self._header
        pub.publish(msg)

    def pub_joint_state_ee(self, articulation):
        if articulation is None:
            return

        eef_6d_forces = articulation.get_measured_joint_forces(self.joint_indices_ee)
        if eef_6d_forces is None:
            return

        msg = JointState()
        msg.header = self._header
        msg.name = []
        for idx, n in enumerate(self.joint_names_ee):
            readings = eef_6d_forces[idx]
            msg.name.append(f"{n}.linear.x")
            msg.name.append(f"{n}.linear.y")
            msg.name.append(f"{n}.linear.z")
            msg.name.append(f"{n}.angular.x")
            msg.name.append(f"{n}.angular.y")
            msg.name.append(f"{n}.angular.z")
            msg.effort.extend(readings.tolist())

        self.pub_ee.publish(msg)

    def pub_camera(self, camera_id):
        if 0 != self._current_step_index % self.parameters[camera_id]["every_n_frame"]:
            return

        try:
            img_msg = self._bridge.cv2_to_imgmsg(
                np.ascontiguousarray(self._img_data_cache[camera_id]),
                encoding="rgba8",
            )

            img = self._img_msg_cache[camera_id]
            img.header.stamp = self._header.stamp
            img.data = img_msg.data
            img.height = img_msg.height
            img.width = img_msg.width
            img.step = img_msg.step
            img.encoding = img_msg.encoding
            img.is_bigendian = img_msg.is_bigendian
            self.publisher_map[camera_id].publish(img)
        except Exception as e:
            print(f"[ERROR] Failed to capture image from {camera_id}: {e}")

    def pub_articulated_object(self):
        for idx, articulation in enumerate(self.articulated_objs):
            self.pub_joint_state(self.articulated_obj_publishers[idx], articulation)

    def get_joint_state_names(self):
        return self._articulation_dof_names

    def get_joint_state_position(self):
        return self._articulation_pos

    def get_joint_state_velocity(self):
        return self._articulation_vel

    def get_joint_state_effort(self):
        return self._articulation_eff

    def get_camera_images_raw(self):
        return self._img_data_cache

    def get_camera_image_raw(self, camera):
        if camera in self._img_data_cache:
            return self._img_data_cache[camera]
        else:
            return np.ndarray()

    def get_camera_image_rgb(self, camera):
        if camera in self._img_data_cache:
            return self._img_data_cache[camera][..., :3]
        else:
            return np.ndarray()

    def get_observation_image(self, dir):
        ret = {}
        if dir == {}:
            for k in self.annotators.keys():
                ret[k] = self.annotators[k].get_data()[..., :3]
        else:
            for k, v in dir.items():
                ret[k] = self.annotators[v].get_data()[..., :3]
        return ret

    def get_observation_depth(self, dir):
        ret = {}
        if dir == {}:
            for k in self.depth_annotators.keys():
                ret[k] = self.depth_annotators[k].get_data().squeeze()
        else:
            for k, v in dir.items():
                if v in self.depth_annotators:
                    ret[k] = self.depth_annotators[v].get_data().squeeze()
        return ret

    def get_joint_state_dict(self):
        return {
            self._articulation.dof_names[i]: self._articulation.get_joint_positions().tolist()[i]
            for i in range(len(self._articulation.dof_names))
        }

    def get_joint_state(self):
        return self._articulation.get_joint_positions().tolist()

    def get_joint_indices_by_name(self, name: str):
        if self._metadata:
            return self._metadata.joint_indices[name]
        else:
            return None

    def get_joint_indices_by_names(self, names: list):
        if self._metadata:
            return 1 + np.array([self._metadata.joint_indices[n] for n in names])
        else:
            return np.array()
