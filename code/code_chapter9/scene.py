"""15 种轻量教学物体与物理桌面。无需下载大型房间或物品资产。"""
import numpy as np
from config import OBJECTS, PICK_SLOTS, TRAY_EXCLUSION_HALF_SIZE
from kinematics import matrix_to_quaternion, rotation_z


def lathe(profile, sides=40):
    """绕 Z 轴旋转截面，构造三角网格（瓶、杯、线轴）。"""
    a = np.arange(sides)*2*np.pi/sides
    points = np.array([[r*np.cos(t), r*np.sin(t), z] for r,z in profile for t in a])
    faces = []
    for j in range(len(profile)-1):
        for i in range(sides):
            u, v = j*sides+i, j*sides+(i+1)%sides
            faces.extend([(u,v,v+sides), (u,v+sides,u+sides)])
    return points, np.asarray(faces)


def mesh_for(spec):
    sx,sy,sz = spec.size
    if spec.shape in ('sphere','ellipsoid','apple'):
        t = np.linspace(-np.pi/2, np.pi/2, 25)
        profile = [(max(1e-6, np.cos(a)/2), np.sin(a)/2) for a in t]
        p,f = lathe(profile)
        if spec.shape == 'apple':
            p[:,2] *= .92 + .08*np.cos(np.arctan2(p[:,1],p[:,0])*5)
        return p*np.array([sx,sy,sz]), f
    if spec.shape == 'box':
        p = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])*.5
        f = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
        return p*np.array(spec.size), f
    profiles = {
        'can': [(0,-.5),(.48,-.5),(.5,-.47),(.5,.47),(.48,.5),(0,.5)],
        'bottle': [(0,-.5),(.47,-.5),(.5,-.46),(.5,.18),(.26,.32),(.26,.45),(.29,.45),(.29,.5),(0,.5)],
        'spool': [(0,-.5),(.5,-.5),(.5,-.38),(.31,-.38),(.31,.38),(.5,.38),(.5,.5),(0,.5)],
        'cup': [(0,-.5),(.40,-.5),(.5,.5),(.44,.5),(.34,-.40),(0,-.40)],
    }
    p,f = lathe(profiles[spec.shape])
    return p*np.array(spec.size), f


def sample_layout(specs, slots, tray_xy, bounds, rng, spread=.022):
    """仅重置/采集使用：整件物体避开托盘、桌边和邻物，不向在线规划泄露真值。

    用 XY 外接圆覆盖任意偏航；返回位置/偏航。无法安全摆放时明确失败，不强行塞入。
    """
    if len(specs) != len(slots) or not 0 <= spread <= .10:
        raise ValueError('摆放数量不一致或 spread 超出 [0, .10] m')
    bounds = np.asarray(bounds)
    poses, placed = [], []
    for spec, slot in zip(specs, rng.permutation(np.asarray(slots))):
        radius = np.linalg.norm(spec.size[:2]) / 2
        for _ in range(500):
            pos = np.asarray(slot, dtype=float).copy()
            pos[:2] += rng.uniform(-spread, spread, 2)
            if np.any(pos[:2]-radius < bounds[0]+.01) or np.any(pos[:2]+radius > bounds[1]-.01):
                continue
            # 与视觉排除框保留 10 mm 间隙；不会靠缩小托盘 ROI 掩盖出生位置错误。
            if np.all(abs(pos[:2]-tray_xy) < TRAY_EXCLUSION_HALF_SIZE+radius+.01):
                continue
            if any(np.linalg.norm(pos[:2]-xy) < radius+r+.012 for xy,r in placed):
                continue
            poses.append((pos, rng.uniform(-np.pi, np.pi)))
            placed.append((pos[:2].copy(), radius))
            break
        else:
            raise ValueError(f'{spec.name} 找不到安全初始位置；调整 PICK_SLOTS/spread/物体尺寸')
    return poses


class TableScene:
    def __init__(self, sim, names):
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.utils.semantics import add_update_semantics
        from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, Vt
        self.sim, self.objects, self.specs = sim, {}, {}
        self.material = PhysicsMaterial('/World/Chapter9/Material', static_friction=1.3, dynamic_friction=1.1, restitution=0.)
        center = sim.arm_to_world([.68,.610,-.26])
        self.table_top = float(center[2]+.04)
        self.bounds = np.array([[center[0]-.46, center[1]-.54], [center[0]+.46, center[1]+.54]])
        sim.world.scene.add(FixedCuboid('/World/Chapter9/Table', name='table9', position=center,
                                       scale=np.array([.92,1.08,.08]), size=1., color=np.array([.43,.45,.48]), physics_material=self.material))
        self.place = sim.arm_to_world([.368,.57,-.50])
        self.place[2] = self.table_top+.004
        # 无高围墙的接收托盘，便于检验释放后物体是否稳定落在指定区域。
        sim.world.scene.add(FixedCuboid('/World/Chapter9/Tray', name='tray9', position=self.place,
                                       scale=np.array([.22,.20,.008]), size=1., color=np.array([.18,.30,.34]), physics_material=self.material))
        self.place[2] += .004
        # 六个位置处于右臂低位俯抓的可达区，而不是仅 TCP 位置可达。
        slots = PICK_SLOTS
        if len(names) > len(slots):
            raise ValueError('每批最多6个物体；用 --objects 选择不同批次')
        for i,name in enumerate(names):
            spec = next((s for s in OBJECTS if s.name==name), None)
            if spec is None:
                raise ValueError(f'未知物体 {name}')
            path = f'/World/Chapter9/Objects/{name}'
            root = UsdGeom.Xform.Define(sim.stage, path)
            mesh = UsdGeom.Mesh.Define(sim.stage, path+'/mesh')
            points, faces = mesh_for(spec)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            mesh.CreateFaceVertexCountsAttr([3]*len(faces))
            mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*spec.color)])
            mesh.CreateDoubleSidedAttr(True)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexDecomposition' if spec.shape=='cup' else 'convexHull')
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.material.material, UsdShade.Tokens.weakerThanDescendants, 'physics')
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(spec.mass)
            add_update_semantics(root.GetPrim(), name)
            pos = sim.arm_to_world([slots[i][0],.57,slots[i][1]])
            pos[2] = self.table_top+spec.size[2]/2+.008
            body = sim.world.scene.add(SingleRigidPrim(path, name='obj_'+name, position=pos, mass=spec.mass))
            body.initialize()
            self.objects[name], self.specs[name] = body, spec
        self.initial = {n: np.asarray(b.get_world_pose()[0]).copy() for n,b in self.objects.items()}

    def randomize(self, rng, spread=.022):
        """只在采集训练数据/重置场景时使用；抓取规划不能读取这些坐标。"""
        poses = sample_layout(list(self.specs.values()), list(self.initial.values()),
                              self.place[:2], self.bounds, rng, spread)
        for (name, body), (pos, yaw) in zip(self.objects.items(), poses):
            pos[2] = self.table_top+self.specs[name].size[2]/2+.008
            body.set_world_pose(pos, matrix_to_quaternion(rotation_z(yaw)))
            body.set_linear_velocity(np.zeros(3)); body.set_angular_velocity(np.zeros(3))

    def evaluate_lift(self, before, min_lift=.045):
        """仅评估，不向 planner 提供真值。"""
        return [n for n,b in self.objects.items() if b.get_world_pose()[0][2]-before[n][2] > min_lift]

    def snapshot_truth(self):
        return {n: np.asarray(b.get_world_pose()[0]).copy() for n,b in self.objects.items()}

    def placed(self, name):
        p = self.objects[name].get_world_pose()[0]
        v = self.objects[name].get_linear_velocity()
        return bool(np.all(abs(p[:2]-self.place[:2]) < [.105,.095]) and
                    self.place[2] < p[2] < self.place[2]+.12 and np.linalg.norm(v)<.03)
