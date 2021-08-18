import pybullet as p
import pybullet_data
from atob.geometry import Cuboid, Sphere
import numpy as np
from pyquaternion import Quaternion
import time
from atob.franka import FrankaRobot
from atob.geometry import SE3
from pathlib import Path


class Bullet:
    def __init__(self, gui=False):
        self.use_gui = gui
        if self.use_gui:
            self.clid = p.connect(p.GUI)
        else:
            self.clid = p.connect(p.DIRECT)
        self.urdf_path = None
        self.robot_id = None
        self.obstacle_ids = []
        self.obstacle_collision_ids = []
        self._link_name_to_index = None
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def __del__(self):
        p.disconnect(self.clid)

    def load_robot(self, urdf_path):
        raise NotImplementedError("Must implement robot loading for specific robot")

    def is_robot_loaded(self):
        return self.robot_id is not None

    def marionette(self, config):
        raise NotImplementedError("Marionette not implemented")

    def reload(self):
        p.disconnect(self.clid)
        self.clid = p.connect(p.GUI)
        self.robot_id = None
        if self.urdf_path is not None:
            self.load_robot(self.urdf_path)
        self.obstacle_ids = []

    @property
    def links(self):
        return [(k, v) for k, v in self._link_name_to_index.items()]

    def link_id(self, name):
        return self._link_name_to_index[name]

    def link_name(self, id):
        return self._index_to_link_name[id]

    @property
    def link_frames(self):
        ret = p.getLinkStates(
            self.robot_id,
            list(range(len(self.links) - 1)),
            computeForwardKinematics=True,
            physicsClientId=self.clid,
        )
        positions = [np.array(r[4]) for r in ret]
        matrices = [
            Quaternion([r[5][3], r[5][0], r[5][1], r[5][2]]).transformation_matrix
            for r in ret
        ]
        frames = {}
        for ii in range(len(positions)):
            matrices[ii][:3, 3] = positions[ii]
            frames[self.link_name(ii)] = matrices[ii]
        return frames

    def load_cuboids(self, cuboids):
        if isinstance(cuboids, Cuboid):
            cuboids = [cuboids]
        ids = []
        for cuboid in cuboids:
            assert isinstance(cuboid, Cuboid)
            kwargs = {}
            if self.use_gui:
                obstacle_visual_id = p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=cuboid.half_extents,
                    rgbaColor=[1, 1, 1, 1],
                    physicsClientId=self.clid,
                )
                kwargs["baseVisualShapeIndex"] = obstacle_visual_id
            obstacle_collision_id = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=cuboid.half_extents,
                physicsClientId=self.clid,
            )
            obstacle_id = p.createMultiBody(
                basePosition=cuboid.center,
                baseOrientation=cuboid.xyzw,
                baseCollisionShapeIndex=obstacle_collision_id,
                physicsClientId=self.clid,
                **kwargs,
            )
            ids.append(obstacle_id)
        self.obstacle_ids.extend(ids)
        self.obstacle_collision_ids.extend(ids)
        if len(ids) == 1:
            return ids[0]
        return ids

    def load_spheres(self, spheres):
        # TODO add visualization logic to sphere
        if isinstance(spheres, Sphere):
            spheres = [spheres]

        ids = []
        for sphere in spheres:
            assert isinstance(sphere, Sphere)
            obstacle_visual_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=sphere.radius,
                rgbaColor=[0, 0, 0, 1],
                physicsClientId=self.clid,
            )
            obstacle_collision_id = p.createCollisionShape(
                shapeType=p.GEOM_SPHERE,
                radius=sphere.radius,
                physicsClientId=self.clid,
            )
            obstacle_id = p.createMultiBody(
                basePosition=sphere.center,
                baseVisualShapeIndex=obstacle_visual_id,
                baseCollisionShapeIndex=obstacle_collision_id,
                physicsClientId=self.clid,
            )
            ids.append(obstacle_id)
        self.obstacle_ids.extend(ids)
        if len(ids) == 1:
            return ids[0]
        return ids

    def clear_obstacle(self, id):
        if id is not None:
            p.removeBody(id, physicsClientId=self.clid)
            self.obstacle_ids = [x for x in self.obstacle_ids if x != id]

    def clear_all_obstacles(self):
        for id in self.obstacle_ids:
            if id is not None:
                p.removeBody(id, physicsClientId=self.clid)
        self.obstacle_ids = []

    def in_collision(self, check_self=False):
        # Step the simulator (only enough for collision detection)
        p.performCollisionDetection(physicsClientId=self.clid)
        if check_self:
            contacts = p.getContactPoints(
                self.robot_id, self.robot_id, physicsClientId=self.clid
            )
            if len(contacts) > 0:
                return True

        # Iterate through all obstacles to check for collisions
        for id in self.obstacle_ids:
            contacts = p.getContactPoints(self.robot_id, id, physicsClientId=self.clid)
            if len(contacts) > 0:
                return True
        return False


class FrankaEnv(Bullet):
    def load_robot(self, urdf_path="franka_panda/panda.urdf"):
        if self.robot_id is not None:
            print("There is already a robot loaded. Removing and reloading")
            p.removeBody(self.robot_id, physicsClientId=self.clid)
        self.robot_id = p.loadURDF(
            urdf_path,
            useFixedBase=True,
            physicsClientId=self.clid,
            flags=p.URDF_USE_SELF_COLLISION,
        )
        self.urdf_path = urdf_path

        # Code snippet borrowed from https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12728
        self._link_name_to_index = {
            p.getBodyInfo(self.robot_id, physicsClientId=self.clid)[0].decode(
                "UTF-8"
            ): -1
        }
        for _id in range(p.getNumJoints(self.robot_id, physicsClientId=self.clid)):
            _name = p.getJointInfo(self.robot_id, _id, physicsClientId=self.clid)[
                12
            ].decode("UTF-8")
            self._link_name_to_index[_name] = _id
        self._index_to_link_name = {}

        for k, v in self._link_name_to_index.items():
            self._index_to_link_name[v] = k

    def ik(self, matrix=None, retries=100):
        self.marionette(FrankaRobot.random_configuration())
        quat = Quaternion(matrix=matrix)
        xyzw = [quat.x, quat.y, quat.z, quat.w]
        position = [matrix[0, 3], matrix[1, 3], matrix[2, 3]]
        lower_limits = [x[0] for x in FrankaRobot.JOINT_LIMITS] + [0, 0]
        upper_limits = [x[1] for x in FrankaRobot.JOINT_LIMITS] + [0.04, 0.04]
        joint_ranges = [x[1] - x[0] for x in FrankaRobot.JOINT_LIMITS] + [0.04, 0.04]
        rest_poses = [0.00, -1.3, 0.00, -2.87, 0.00, 2.00, 0.75, 0, 0]
        for i in range(retries + 1):
            solution = p.calculateInverseKinematics(
                bodyUniqueId=self.robot_id,
                endEffectorLinkIndex=11,
                targetPosition=position,
                targetOrientation=xyzw,
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
            )
            arr = np.array(solution)
            if np.alltrue(arr >= np.array(lower_limits)) and np.alltrue(
                arr <= np.array(upper_limits)
            ):
                self.marionette(solution[:7])
                return solution[:7]
            self.marionette(solution[:7])

        return None

    def collision_free_ik(self, matrix=None, retries=2):
        for i in range(retries + 1):
            sample = self.ik(matrix, retries=20)
            if sample is None:
                continue
            self.marionette(sample)
            collides = self.in_collision(check_self=True)
            if not collides:
                return sample
        return None

    def sample_nearby_ik(self, config, pose, samples=10, failures=10):
        """
        Generates a number of collision-free IK solutions and returns the one
        that is closest in configuration space to the input config. If the
        collision_free_ik call fails a certain number of times, give up and return None
        """
        ik_samples = []
        num_failures = 0
        while len(ik_samples) < samples and num_failures < failures:
            sample = collision_free_ik(matrix=pose)
            if sample is None:
                num_failures += 1
                continue
            ik_samples.append(np.asarray(sample))
        if len(ik_samples) == 0:
            return None
        config = np.asarray(config)
        idx = np.argmin(np.array([np.linalg.norm(config - x) for x in samples]))
        return samples[idx].tolist()

    def marionette(self, config):
        # There is something weird in that the URDFs I load start with joint 1, but the
        # urdfs that bullet loads start with 0. This requires further investigation

        if self.urdf_path == "franka_panda/panda.urdf":
            for i in range(0, 7):
                p.resetJointState(
                    self.robot_id, i, config[i], physicsClientId=self.clid
                )
            # Spread the fingers if they aren't included--prevents self collision
            p.resetJointState(self.robot_id, 9, 0.02, physicsClientId=self.clid)
            p.resetJointState(self.robot_id, 10, 0.02, physicsClientId=self.clid)
        else:
            for i in range(1, 8):
                p.resetJointState(
                    self.robot_id, i, config[i - 1], physicsClientId=self.clid
                )


class FrankaHandEnv(Bullet):
    def load_robot(self):
        urdf_path = Path(__file__).parent.parent / "urdf" / "panda_hand" / "panda.urdf"
        urdf_path = str(urdf_path)
        if self.robot_id is not None:
            print("There is already a robot loaded. Removing and reloading")
            p.removeBody(self.robot_id, physicsClientId=self.clid)
        self.robot_id = p.loadURDF(
            urdf_path,
            useFixedBase=True,
            physicsClientId=self.clid,
            flags=p.URDF_USE_SELF_COLLISION,
        )
        self.urdf_path = urdf_path

        # Code snippet borrowed from https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12728
        self._link_name_to_index = {
            p.getBodyInfo(self.robot_id, physicsClientId=self.clid)[0].decode(
                "UTF-8"
            ): -1
        }
        for _id in range(p.getNumJoints(self.robot_id, physicsClientId=self.clid)):
            _name = p.getJointInfo(self.robot_id, _id, physicsClientId=self.clid)[
                12
            ].decode("UTF-8")
            self._link_name_to_index[_name] = _id
        self._index_to_link_name = {}

        for k, v in self._link_name_to_index.items():
            self._index_to_link_name[v] = k

    def marionette(self, pose, frame):
        assert frame in ["base_frame", "right_gripper", "panda_grasptarget"]
        # Pose is expressed as a transformation from the desired frame to the world
        # But we need to transform it into the base frame

        # TODO maybe there is some way to cache these transforms from the urdf
        # instead of hardcoding them
        if frame == "right_gripper":
            transform = SE3(
                matrix=np.array(
                    [
                        [-0.7071067811865475, 0.7071067811865475, 0, 0],
                        [-0.7071067811865475, -0.7071067811865475, 0, 0],
                        [0, 0, 1, -0.1],
                        [0, 0, 0, 1],
                    ]
                )
            )
            pose = pose @ transform
        elif frame == "panda_grasptarget":
            transform = SE3(
                matrix=np.array(
                    [
                        [0.7071067811865475, 0.7071067811865475, 0, 0],
                        [0.7071067811865475, 0.7071067811865475, 0, 0],
                        [0, 0, 1, -0.105],
                        [0, 0, 0, 1],
                    ]
                )
            )
            pose = pose @ transform

        x, y, z = pose.xyz
        p.resetJointState(self.robot_id, 0, x, physicsClientId=self.clid)
        p.resetJointState(self.robot_id, 1, y, physicsClientId=self.clid)
        p.resetJointState(self.robot_id, 2, z, physicsClientId=self.clid)
        p.resetJointStateMultiDof(
            self.robot_id, 3, pose.so3.xyzw, physicsClientId=self.clid
        )
        # Spread the fingers if they aren't included--prevents self collision
        p.resetJointState(self.robot_id, 4, 0.02, physicsClientId=self.clid)
        p.resetJointState(self.robot_id, 5, 0.02, physicsClientId=self.clid)


class FrankaHandAndArm(Bullet):
    def __init__(self, gui=False):
        self.use_gui = gui
        if self.use_gui:
            self.clid = p.connect(p.GUI)
        else:
            self.clid = p.connect(p.DIRECT)
        self.robot_arm_id = None
        self.robot_hand_id = None
        self.obstacle_ids = []
        self.obstacle_collision_ids = []
        self._link_name_to_index = None
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def load_robot_arm(self, urdf_path="franka_panda/panda.urdf"):
        if self.robot_arm_id is not None:
            print("There is already a robot loaded. Removing and reloading")
            p.removeBody(self.robot_arm_id, physicsClientId=self.clid)
        self.robot_arm_id = p.loadURDF(
            urdf_path,
            useFixedBase=True,
            physicsClientId=self.clid,
            flags=p.URDF_USE_SELF_COLLISION,
        )

    def marionette_arm(self, config):
        # There is something weird in that the URDFs I load start with joint 1, but the
        # urdfs that bullet loads start with 0. This requires further investigation
        for i in range(0, 7):
            p.resetJointState(
                self.robot_arm_id, i, config[i], physicsClientId=self.clid
            )
        # Spread the fingers if they aren't included--prevents self collision
        p.resetJointState(self.robot_arm_id, 9, 0.02, physicsClientId=self.clid)
        p.resetJointState(self.robot_arm_id, 10, 0.02, physicsClientId=self.clid)

    def load_robot_hand(self):
        urdf_path = Path(__file__).parent.parent / "urdf" / "panda_hand" / "panda.urdf"
        urdf_path = str(urdf_path)
        if self.robot_hand_id is not None:
            print("There is already a robot loaded. Removing and reloading")
            p.removeBody(self.robot_hand_id, physicsClientId=self.clid)
        self.robot_hand_id = p.loadURDF(
            urdf_path,
            useFixedBase=True,
            physicsClientId=self.clid,
            flags=p.URDF_USE_SELF_COLLISION,
        )

    def marionette_hand(self, pose, frame):
        assert frame in ["base_frame", "right_gripper", "panda_grasptarget"]
        # Pose is expressed as a transformation from the desired frame to the world
        # But we need to transform it into the base frame

        # TODO maybe there is some way to cache these transforms from the urdf
        # instead of hardcoding them
        if frame == "right_gripper":
            transform = SE3(
                matrix=np.array(
                    [
                        [-0.7071067811865475, 0.7071067811865475, 0, 0],
                        [-0.7071067811865475, -0.7071067811865475, 0, 0],
                        [0, 0, 1, -0.1],
                        [0, 0, 0, 1],
                    ]
                )
            )
            pose = pose @ transform
        elif frame == "panda_grasptarget":
            transform = SE3(
                matrix=np.array(
                    [
                        [0.7071067811865475, 0.7071067811865475, 0, 0],
                        [0.7071067811865475, 0.7071067811865475, 0, 0],
                        [0, 0, 1, -0.105],
                        [0, 0, 0, 1],
                    ]
                )
            )
            pose = pose @ transform

        x, y, z = pose.xyz
        p.resetJointState(self.robot_hand_id, 0, x, physicsClientId=self.clid)
        p.resetJointState(self.robot_hand_id, 1, y, physicsClientId=self.clid)
        p.resetJointState(self.robot_hand_id, 2, z, physicsClientId=self.clid)
        p.resetJointStateMultiDof(
            self.robot_hand_id, 3, pose.so3.xyzw, physicsClientId=self.clid
        )
        # Spread the fingers if they aren't included--prevents self collision
        p.resetJointState(self.robot_hand_id, 4, 0.02, physicsClientId=self.clid)
        p.resetJointState(self.robot_hand_id, 5, 0.02, physicsClientId=self.clid)
