"""固定底盘桌面场景 + 第5章机械臂；不导入其他章节。"""
import json
from dataclasses import asdict
import numpy as np
from config import (ROOT, ROBOT_USD, ROBOT_PRIM_PATH, ARM_BASE_PRIM_PATH,
                    END_EFFECTOR_PRIM_PATHS, HOME_LEFT, HOME_RIGHT, GRIPPER_JOINT,
                    GRIPPER_OPEN, GripperGeometry)
from pointcloud import transform_points

class G2Simulation:
    def __init__(self, cfg, objects):
        if not ROBOT_USD.is_file():
            raise FileNotFoundError(ROBOT_USD)
        from isaacsim import SimulationApp
        # 启动前禁用 Isaac 内置扩展源码监视，不创建不需要的文件监视器。
        self.app = SimulationApp({'headless':cfg.headless, 'renderer':'RaytracedLighting',
                                  'fast_shutdown':True, 'disable_viewport_updates':False,'limit_cpu_threads':12,
                                  'extra_args':['--/app/extensions/fsWatcherEnabled=false']})
        self.cfg, self.ticks = cfg, 0
        self.frame_observer = None  # 可选显示回调，不参与控制/推进额外物理步
        try:
            self._build(objects)
        except BaseException:
            import traceback
            traceback.print_exc()
            self.close(1)
            raise

    def _build(self, objects):
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Sdf, UsdPhysics, UsdLux
        import omni.usd
        from arm_controller import G2ArmController
        from scene import TableScene
        from camera import RGBDCamera
        self.world = World(stage_units_in_meters=1., physics_dt=self.cfg.physics_dt, rendering_dt=1/self.cfg.rendering_hz)
        self.stage = omni.usd.get_context().get_stage()
        self.world.scene.add_default_ground_plane()
        light = UsdLux.DomeLight.Define(self.stage, '/World/Chapter9/Light')
        light.CreateIntensityAttr(1200.)
        add_reference_to_stage(str(ROBOT_USD), ROBOT_PRIM_PATH)
        SingleXFormPrim(ROBOT_PRIM_PATH, position=np.array([0.,0.,-.01]), orientation=np.array([1.,0,0,0]))
        joint = UsdPhysics.FixedJoint.Define(self.stage,'/World/Chapter9/BaseFixed')
        joint.CreateBody1Rel().SetTargets([Sdf.Path(ROBOT_PRIM_PATH+'/base_link')])
        self.world.play()
        for _ in range(90): self.step()
        self.robot = self.world.scene.add(SingleArticulation(ROBOT_PRIM_PATH,'G2_chapter9'))
        self.robot.initialize()
        self.robot.set_solver_position_iteration_count(32)
        self.robot.set_solver_velocity_iteration_count(4)
        self.arm = G2ArmController(self.robot,'right')
        self.left = G2ArmController(self.robot,'left')
        self.gripper_id = list(self.robot.dof_names).index(GRIPPER_JOINT)
        # 初始姿态设置仅用于 reset，不用于抓取或执行路径。
        self.robot.set_joint_positions(HOME_LEFT, joint_indices=self.left.joint_indices)
        self.robot.set_joint_positions(HOME_RIGHT, joint_indices=self.arm.joint_indices)
        self.left.command_positions(HOME_LEFT); self.arm.command_positions(HOME_RIGHT)
        self.command_gripper(GRIPPER_OPEN)
        for _ in range(120): self.step()
        self.T_world_base = self.link_transform(ARM_BASE_PRIM_PATH)
        self.T_base_world = np.linalg.inv(self.T_world_base)
        self.check_fk()
        self.scene = TableScene(self,objects)
        for _ in range(120): self.step()
        self.camera = RGBDCamera(self)
        for _ in range(24): self.step()

    def step(self):
        if not self.app.is_running():
            raise KeyboardInterrupt('仿真窗口已关闭')
        self.world.step(render=True)
        self.ticks += 1
        if self.frame_observer is not None: self.frame_observer()

    def link_transform(self, path):
        # Fabric 运行时应通过 prim API 读取动态位姿，不能用过时的 USD authored transform。
        from isaacsim.core.prims import SingleXFormPrim
        from kinematics import quaternion_to_matrix
        p,q = SingleXFormPrim(path).get_world_pose()
        T = np.eye(4); T[:3,:3] = quaternion_to_matrix(q); T[:3,3] = p
        return T

    def arm_to_world(self, p):
        return transform_points(self.T_world_base,np.asarray(p))

    def command_gripper(self, angle):
        from isaacsim.core.utils.types import ArticulationAction
        self.robot.apply_action(ArticulationAction(joint_positions=np.array([angle]),joint_indices=np.array([self.gripper_id])))

    def grip(self, angle, seconds=.65):
        start = float(self.robot.get_joint_positions()[self.gripper_id])
        for t in np.linspace(0,1,max(2,int(seconds/self.cfg.physics_dt))):
            self.command_gripper(start+(3*t*t-2*t*t*t)*(angle-start))
            self.step()
        for _ in range(30): self.command_gripper(angle); self.step()

    def check_fk(self):
        actual = self.T_base_world @ self.link_transform(END_EFFECTOR_PRIM_PATHS['right'])
        fk = self.arm.forward_kinematics()
        from kinematics import orientation_error
        error = np.linalg.norm(actual[:3,3]-fk.position)
        angle = np.linalg.norm(orientation_error(actual[:3,:3],fk.rotation))
        print(f'[标定] FK vs USD: {error*1000:.2f} mm, {np.rad2deg(angle):.2f} deg',flush=True)
        if error > .012 or angle > .06:
            raise RuntimeError('G2 模型与第五章 FK 不一致，拒绝移动；请核对 tool_transform')

    def calibrate_gripper(self, path):
        """从实际 USD 夹指网格测量开/半开扫掠盒，输出 GraspGenX 条件。

        center_link 的 +X 应为闭合方向。只有无物体接触的 home 位姿才能运行。
        Mesh BBox 是保守近似；记录原始测量方便教学检查。
        """
        from pxr import Usd, UsdGeom
        samples = []
        for angle in (GRIPPER_OPEN,GRIPPER_OPEN/2,0.):
            self.grip(angle)
            T = self.link_transform(END_EFFECTOR_PRIM_PATHS['right'])
            clouds = []
            for side in ('inner','outer'):
                link = f'/genie/gripper_r_{side}_link4'
                # 链接上的可视/碰撞 mesh 局部点，乘实时 link pose。
                prim = self.stage.GetPrimAtPath(link)
                cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                link_usd = np.asarray(cache.GetLocalToWorldTransform(prim)).T
                points = []
                for child in Usd.PrimRange(prim):
                    if child.IsA(UsdGeom.Mesh):
                        vertices = np.asarray(UsdGeom.Mesh(child).GetPointsAttr().Get(),float)
                        local = np.linalg.inv(link_usd) @ np.asarray(cache.GetLocalToWorldTransform(child)).T
                        points.append(transform_points(np.linalg.inv(T) @ self.link_transform(link) @ local, vertices))
                if not points:
                    raise RuntimeError(f'找不到夹指网格 {link}')
                clouds.append(np.concatenate(points))
            clouds.sort(key=lambda p: np.mean(p[:,0]))
            # distal pad 接触区：截取最靠近指尖的 20mm，不使用弯曲连杆的宽度。
            zmax = min(np.max(p[:,2]) for p in clouds)
            pads = [p[p[:,2] > zmax-.020] for p in clouds]
            inner_l, inner_r = np.max(pads[0][:,0]), np.min(pads[1][:,0])
            width = max(0., inner_r-inner_l)
            samples.append(dict(angle=angle,width=float(width),zmax=float(zmax),
                                bounds=[np.array([p.min(0),p.max(0)]).tolist() for p in clouds]))
        self.grip(GRIPPER_OPEN)
        width = samples[0]['width']
        if not .025 < width < .15 or samples[1]['width'] >= width:
            raise RuntimeError(f'夹爪轴向/开闭标定不可信：{samples}')
        bounds = np.asarray(samples[0]['bounds'])
        # 用整段夹指的最窄开口作保守宽度；夹指闭合时指尖会进一步前伸。
        width = float(bounds[1,0,0]-bounds[0,1,0])
        zmax = max(s['zmax'] for s in samples)
        geom = GripperGeometry(max_width=width, finger_thickness=float(np.max(bounds[:,1,0]-bounds[:,0,0])),
                               finger_depth=float(np.max(bounds[:,1,1]-bounds[:,0,1])),
                               finger_z_min=float(bounds[:,0,2].min()),finger_z_max=zmax)
        # 模型基准 B 与 TCP G 同旋转，B 原点是夹爪 base_link 的位置。
        tcp = self.link_transform(END_EFFECTOR_PRIM_PATHS['right'])
        base = self.link_transform('/genie/gripper_r_base_link')
        base_to_tcp = tcp[:3,:3].T @ (tcp[:3,3]-base[:3,3])
        def volume(sample):
            # 开、半开扫掠区位于指尖内侧20mm，坐标相对 B。
            return [max(sample['width'],.002),min(geom.finger_depth,.035),.020], [0.,0.,float(base_to_tcp[2]+sample['zmax']-.010)]
        e0,o0 = volume(samples[0]); e1,o1 = volume(samples[1])
        calibration = dict(geometry=asdict(geom),samples=samples,
                           base_to_tcp=base_to_tcp.tolist(),extents_open=e0,offset_open=o0,extents_mid=e1,offset_mid=o1,
                           gripper_type=1, fingertip_depth=float(base_to_tcp[2]+zmax),
                           note='G2 USD 实测简化扫掠盒；type=1 revolute_2f；非网格级完整夹爪模型')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(calibration,indent=2,ensure_ascii=False))
        return calibration

    def close(self, exit_status=None):
        if getattr(self,'app',None):
            import sys
            import omni.kit.app
            if exit_status is None: exit_status = int(sys.exc_info()[0] is not None)
            # Kit 5.1 的完整插件清理可能卡住；先保存报告/关闭 worker，再快速退出。
            # python.sh 会把非零 Kit 退出码统一转成1，精确状态以 summary.json 为准。
            sys.stdout.flush(); sys.stderr.flush()
            omni.kit.app.get_app().post_quit(exit_status)
            self.app.close(); self.app = None
