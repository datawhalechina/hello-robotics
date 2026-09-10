"""第5章 DLS 位姿 IK + 平滑关节控制，增加抓取状态机与保守路径检查。"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from config import (HOME_RIGHT, GRIPPER_OPEN, GRIPPER_CLOSED, GripperGeometry,
                    JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS, PlannerConfig)
from kinematics import orientation_error
from pointcloud import transform_points, table_height
from grasp_planner import gripper_collision, _offset

class PlanningError(RuntimeError): pass
class ExecutionError(RuntimeError): pass

@dataclass
class PickPlan:
    grasp: object
    approach: list
    descend: list
    lift: list
    transfer: list
    place: list
    retreat: list
    home: list


def _sphere_segment(a,b,spacing=.025):
    return np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/spacing)+1))

class PickController:
    def __init__(self, sim, geometry=None):
        self.sim, self.arm = sim, sim.arm
        self.geom = geometry or GripperGeometry()
        self.cfg = PlannerConfig()

    def _world_fk(self, q):
        pose = self.arm.kinematics.forward(q)
        T = np.eye(4); T[:3,:3]=pose.rotation; T[:3,3]=pose.position
        return self.sim.T_world_base @ T

    def _links(self, q, arm=None):
        arm = arm or self.arm
        _, origins, _ = arm.kinematics._chain(q)
        # 重复的同位置旋转关节不应形成零长度胶囊。
        unique = [origins[0]]
        for p in origins[1:]:
            if np.linalg.norm(p-unique[-1])>.01: unique.append(p)
        return transform_points(self.sim.T_world_base,np.array(unique))

    def _safe(self, q, scene, plane, held=None):
        T = self._world_fk(q)
        if gripper_collision(T,scene,plane,self.geom): return False
        # 手臂近似连杆胶囊；scene 已裁剪到工作台区域，不含机器人自身深度。
        links = self._links(q)
        tree = cKDTree(scene) if len(scene) else None
        for a,b in zip(links[:-1],links[1:]):
            samples = _sphere_segment(a,b)
            if tree and np.any(tree.query(samples)[0]<.032): return False
            on_table = ((samples[:,:2]>self.sim.scene.bounds[0]).all(1)&(samples[:,:2]<self.sim.scene.bounds[1]).all(1))
            if np.any((samples@plane[:3]+plane[3]<.04)&on_table): return False
        left = self._links(self.sim.left.get_joint_positions(),self.sim.left)
        left_points = np.concatenate([_sphere_segment(a,b) for a,b in zip(left[:-1],left[1:])])
        # 第一段在肩部天然靠近另一臂，跳过共同安装区域。
        right_points = np.concatenate([_sphere_segment(a,b) for a,b in zip(links[1:-1],links[2:])])
        if len(right_points) and np.any(cKDTree(left_points).query(right_points)[0]<.065): return False
        if held is not None:
            world = transform_points(T,held)
            if np.any(world@plane[:3]+plane[3] < .004): return False
            # scene 中已剔除目标点云。闭合指间允许物体存在，其他物体不可穿透。
            # 场景点进入整个包围盒都算碰撞，不能只检查八个角点附近。
            local_scene=transform_points(np.linalg.inv(T),scene)
            if np.any(((local_scene>=held.min(0)-.004)&(local_scene<=held.max(0)+.004)).all(1)): return False
        return True

    def _solve(self, T, seed):
        target = self.sim.T_base_world @ T
        seeds = [seed,HOME_RIGHT,(seed+HOME_RIGHT)/2,(JOINT_LOWER_LIMITS+JOINT_UPPER_LIMITS)/2]
        answers=[]
        for s in seeds:
            answer=self.arm.kinematics.inverse(target[:3,3],target[:3,:3],s)
            # 不只相信求解器标记，重新计算最终 FK 位置/真实角度误差。
            actual=self.arm.kinematics.forward(answer.joint_positions)
            if (answer.success and np.linalg.norm(actual.position-target[:3,3])<.004 and
                    np.linalg.norm(orientation_error(target[:3,:3],actual.rotation))<.03):
                answers.append(answer.joint_positions)
                if np.max(abs(answer.joint_positions-seed))<.5: break
        if not answers: raise PlanningError('完整位姿 IK 不可达，未降级成只控制位置')
        return min(answers,key=lambda q: np.linalg.norm(q-seed))

    def _joint_path(self, start, target, scene, plane, held=None):
        samples=np.linspace(start,target,max(2,int(np.max(abs(target-start))/.045)+1))[1:]
        if any(not self._safe(q,scene,plane,held) for q in samples):
            raise PlanningError('关节插值路径碰撞/桌面/另一臂检查不通过')
        return list(samples)

    def _line(self, start_T, target_T, seed, scene, plane, held=None):
        # 接近/抬升/放置姿态固定，只做笛卡尔直线平移。
        result=[]
        for pos in np.linspace(start_T[:3,3],target_T[:3,3],max(2,int(np.linalg.norm(start_T[:3,3]-target_T[:3,3])/.012)+1))[1:]:
            T=target_T.copy(); T[:3,3]=pos
            q=self._solve(T,seed)
            if np.max(abs(q-seed))>.5: raise PlanningError('笛卡尔 IK 分支跳变')
            result.extend(self._joint_path(seed,q,scene,plane,held)); seed=q
        return result

    def plan(self, grasp, scene, plane, object_points):
        """规划全部阶段成功才允许执行；输入仅视觉点云/候选，不使用物体真值。"""
        T=grasp.T; pre=_offset(T,-self.cfg.pregrasp*T[:3,2])
        lift=_offset(T,[0,0,self.cfg.lift])
        target=lift.copy()
        # 托盘上已有物体时用观测点云选空位置，而不是反复放到同一个中心。
        tray_xy=self.sim.scene.place[:2]
        occupied=scene[(abs(scene[:,0]-tray_xy[0])<.12)&(abs(scene[:,1]-tray_xy[1])<.11)&
                       (scene[:,2]>self.sim.scene.table_top+.02)]
        destinations=[tray_xy+offset for offset in ([0.,0.],[-.06,-.05],[-.06,.05],[.06,-.05],[.06,.05])]
        target[:2,3]=(max(destinations,key=lambda p: np.min(np.linalg.norm(occupied[:,:2]-p,axis=1)))
                      if len(occupied) else tray_xy)
        target[2,3]=max(lift[2,3],self.sim.scene.table_top+.23)
        place=target.copy()
        # 释放时保留8cm净空，避免物体底部擦桌；放入低托盘而不是硬插到已知中心。
        place[2,3]=self.sim.scene.place[2]+.08
        # 先停在侧上方观察点，避免夹爪遮住目标；复检后再横移至正上方并下探。
        seed=self.arm.get_joint_positions()
        failures=[]
        for offset in ([-.13,0.,.025],[0.,-.15,.025],[0.,.15,.025]):
            observation=_offset(pre,offset)
            try:
                qp=self._solve(observation,seed)
                approach=self._joint_path(seed,qp,scene,plane)
                to_pre=self._line(observation,pre,qp,scene,plane)
                descend=to_pre+self._line(pre,T,to_pre[-1],scene,plane)
                break
            except PlanningError as exc: failures.append(str(exc))
        else: raise PlanningError('observation/descend: '+'; '.join(failures))
        # 从场景中移除目标的观测点，搬运时该物体成为 held，而不是静态障碍。
        target_tree=cKDTree(object_points)
        background=scene[target_tree.query(scene)[0]>.008]
        # 支撑面先验补齐不可见底面；否则只看到顶面的点云会低估搬运包围盒。
        bottom=object_points.copy(); bottom[:,2]=table_height(plane,bottom[:,:2])
        local=transform_points(np.linalg.inv(T),np.vstack([object_points,bottom]))
        lo,hi=local.min(0),local.max(0)
        held=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        # 初始起升阶段目标尚接触桌面，不用角点-桌面净空约束；后续搬运启用。
        try: rising=self._line(T,lift,descend[-1],background,plane)
        except PlanningError as exc: raise PlanningError(f'lift: {exc}') from exc
        try: transfer=self._line(lift,target,rising[-1],background,plane,held)
        except PlanningError as exc: raise PlanningError(f'transfer: {exc}') from exc
        try: placing=self._line(target,place,transfer[-1],background,plane,held)
        except PlanningError as exc: raise PlanningError(f'place: {exc}') from exc
        try: retreat=self._line(place,target,placing[-1],background,plane)
        except PlanningError as exc: raise PlanningError(f'retreat: {exc}') from exc
        # 放置后返回观测友好的 home；下一帧不能把自身夹爪当成桌面障碍。
        # 将接收区的可能落物范围作为保守占据体，不用理想落点代替真实落物。
        place_cloud=np.array([[x,y,z] for x in np.arange(self.sim.scene.place[0]-.11,self.sim.scene.place[0]+.111,.015)
                              for y in np.arange(self.sim.scene.place[1]-.10,self.sim.scene.place[1]+.101,.015)
                              for z in np.arange(self.sim.scene.table_top,self.sim.scene.table_top+.16,.015)])
        try: home=self._joint_path(retreat[-1],HOME_RIGHT,np.vstack([background,place_cloud]),plane)
        except PlanningError as exc: raise PlanningError(f'home: {exc}') from exc
        return PickPlan(grasp,approach,descend,rising,transfer,placing,retreat,home)

    def move(self, path, seconds):
        if not path: return
        # 已检查路径逐小段执行。每段以第5章的 smoothstep 生成，均不越过已验证段。
        dt=self.sim.cfg.physics_dt
        duration=max(seconds/len(path),2*dt)
        for target in path:
            trajectory=self.arm.make_trajectory(target,duration,dt)
            for q in trajectory:
                self.arm.command_positions(q); self.sim.step()
                actual=self.arm.get_joint_positions()
                if not np.isfinite(actual).all() or np.max(abs(actual-q))>.4:
                    raise ExecutionError('关节跟踪误差过大，停止推进')
        for tick in range(180):
            self.arm.command_positions(path[-1]); self.sim.step()
            if tick >= 24 and np.max(abs(self.arm.get_joint_positions()-path[-1])) < .012: break
        if np.max(abs(self.arm.get_joint_positions()-path[-1]))>.09:
            raise ExecutionError('运动结束后关节未收敛')
        actual=self.sim.link_transform('/genie/gripper_r_center_link')
        target_T=self._world_fk(path[-1])
        if np.linalg.norm(actual[:3,3]-target_T[:3,3])>.015:
            raise ExecutionError(f'实际末端偏差过大 {np.linalg.norm(actual[:3,3]-target_T[:3,3])*1000:.1f}mm，拒绝继续抓取')

    def execute(self, plan, recheck=None, on_stage=None):
        stage=on_stage or (lambda _: None)
        stage("approach / pregrasp recheck")
        self.sim.grip(GRIPPER_OPEN)
        self.move(plan.approach,1.5)
        # 到预抓取位再次观察；目标移动/遮挡时不盲目下探。
        if recheck is not None and not recheck(plan.grasp):
            raise ExecutionError('预抓取复检未通过：物体移动/检测丢失，停止本次抓取')
        stage('descend / close gripper')
        print('[动作] 下探/闭合',flush=True)
        before=self.sim.scene.snapshot_truth()  # 仅用于结果评估，不参与动作生成
        self.move(plan.descend,.8)
        self.sim.grip(GRIPPER_CLOSED)
        stage('lift / verify')
        print('[动作] 抬升',flush=True)
        self.move(plan.lift,1.)
        for _ in range(45): self.sim.step()
        lifted=self.sim.scene.evaluate_lift(before)
        if len(lifted)!=1:
            # 安全退回原抓取位置释放，而不是带着未知状态横移。
            self.move(list(reversed(plan.lift)) + [plan.descend[-1]],1.)
            self.sim.grip(GRIPPER_OPEN)
            raise ExecutionError(f'物理抬升验证失败：抬起{len(lifted)}个物体')
        stage('transfer / place')
        print('[动作] 搬运/放置',flush=True)
        self.move(plan.transfer,1.8)
        if not self.sim.scene.evaluate_lift(before):
            raise ExecutionError('搬运中物体掉落')
        self.move(plan.place,.7)
        self.sim.grip(GRIPPER_OPEN)
        stage("retreat / return home")
        self.move(plan.retreat,.8)
        self.move(plan.home,1.5)
        for _ in range(90): self.sim.step()
        return dict(lifted=lifted[0],placed=self.sim.scene.placed(lifted[0]),
                    gripper_angle=float(self.sim.robot.get_joint_positions()[self.sim.gripper_id]))
