import pybullet as p
import pybullet_data
from atob.geometry import Cuboid, Sphere
import numpy as np
from pyquaternion import Quaternion
import time
from atob.robots import FrankaRobot, FrankaHand
from atob.geometry import SE3
from pathlib import Path


class Bullet:
    def __init__(self, gui=False):
        """
        :param gui: Whether to use a gui to visualize the environment. Only one gui instance allowed
        """
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
        """
        Disconnects the client on destruction
        """
        p.disconnect(self.clid)

    def setCameraPosition(self, yaw, pitch, distance, target):
        p.resetDebugVisualizerCamera(
            distance, yaw, pitch, target, physicsClientId=self.clid
        )

    def printCameraPosition(self):
        params = p.getDebugVisualizerCamera(physicsClientId=self.clid)
        print(f"Yaw: {params[8]}")
        print(f"Pitch: {params[9]}")
        print(f"Distance: {params[10]}")
        print(f"Target: {params[11]}")

    def load_robot(self):
        """
        Generic function to load a robot.
        """
        raise NotImplementedError("Must implement robot loading for specific robot")

    def is_robot_loaded(self):
        """
        Checks whether a robot has been loaded
        """
        return self.robot_id is not None

    def are_obstacles_loaded(self):
        """
        Checks whether there are already obstacles in the scene
        """
        return len(self.obstacle_ids) > 0

    def marionette(self, config):
        """
        Snaps a robot to a state without regard for physics
        """
        raise NotImplementedError("Marionette not implemented")

    def reload(self):
        """
        Reloads everything in the environment
        """
        p.disconnect(self.clid)
        self.clid = p.connect(p.GUI)
        self.robot_id = None
        if self.urdf_path is not None:
            self.load_robot(self.urdf_path)
        self.obstacle_ids = []

    @property
    def links(self):
        """
        :return: The names and bullet ids of all links for the loaded robot
        """
        return [(k, v) for k, v in self._link_name_to_index.items()]

    def link_id(self, name):
        """
        :return: The bullet id corresponding to a specific link name
        """
        return self._link_name_to_index[name]

    def link_name(self, id):
        """
        :return: The name corresponding to a particular bullet id
        """
        return self._index_to_link_name[id]

    @property
    def link_frames(self):
        """
        :return: A dictionary where the link names are the keys
            and the values are the correponding poses as reflected
            by the current state of the environment
        """
        ret = p.getLinkStates(
            self.robot_id,
            list(range(len(self.links) - 1)),
            computeForwardKinematics=True,
            physicsClientId=self.clid,
        )
        frames = {}
        for ii, r in enumerate(ret):
            frames[self.link_name(ii)] = SE3(
                xyz=np.array(r[4]),
                quat=Quaternion([r[5][3], r[5][0], r[5][1], r[5][2]]),
            )
        return frames

    def load_cuboids(self, cuboids):
        """
        Loads cuboids into the environment

        :param cuboids: A list of atob.geometry.Cuboid objects
        """
        if isinstance(cuboids, Cuboid):
            cuboids = [cuboids]
        ids = []
        for cuboid in cuboids:
            assert isinstance(cuboid, Cuboid)
            if cuboid.is_zero_volume():
                continue

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
        """
        Loads a set of spheres into the environment

        :param spheres: A list of atob.geometry.Sphere objects
        """
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
        """
        Removes a specific obstacle from the environment

        :param id: Bullet id of obstacle to remove
        """
        if id is not None:
            p.removeBody(id, physicsClientId=self.clid)
            self.obstacle_ids = [x for x in self.obstacle_ids if x != id]

    def clear_all_obstacles(self):
        """
        Removes all obstacles from bullet environment
        """
        for id in self.obstacle_ids:
            if id is not None:
                p.removeBody(id, physicsClientId=self.clid)
        self.obstacle_ids = []

    def in_collision(self, check_self=False):
        """
        Checks whether the robot is in collision with the environment

        :return: Boolean
        """
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

    def ik(self, pose, retries, frame):
        """
        Calculates ik solution for a specific robot. IK does not currently avoid obstacles
        """
        # TODO should this be moved into the robot's class itself?
        raise NotImplementedError("ik must be implemented for each robot type")

    def _setup_robot(self):
        """
        Internal function for setting up the correspondence between link names and ids.
            This is called internally when loading a robot
        """
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


class FrankaEnv(Bullet):
    def load_robot(self):
        if self.robot_id is not None:
            print("There is already a robot loaded. Removing and reloading")
            p.removeBody(self.robot_id, physicsClientId=self.clid)
        self.robot_id = p.loadURDF(
            FrankaRobot.urdf,
            useFixedBase=True,
            physicsClientId=self.clid,
            flags=p.URDF_USE_SELF_COLLISION,
        )
        self.urdf_path = FrankaRobot.urdf
        self._setup_robot()

    def ik(self, pose, retries=100, frame="panda_grasptarget"):
        """
        Calculates an IK solution (supposedly) constrained to to Franka joint limits

        :param pose: The desired pose expressed as an SE3 object
        :param retries: The number of retries with different seeds.
        :param frame: The desired link name for computing IK
        :return: A configuration corresponding to the requested pose
        """
        # TODO should this function go inside the robot class instead?
        self.marionette(FrankaRobot.random_configuration())
        link_id = self.link_id(frame)
        lower_limits = [x[0] for x in FrankaRobot.JOINT_LIMITS] + [0, 0]
        upper_limits = [x[1] for x in FrankaRobot.JOINT_LIMITS] + [0.04, 0.04]
        joint_ranges = [x[1] - x[0] for x in FrankaRobot.JOINT_LIMITS] + [0.04, 0.04]
        rest_poses = [0.00, -1.3, 0.00, -2.87, 0.00, 2.00, 0.75, 0, 0]
        for i in range(retries + 1):
            solution = p.calculateInverseKinematics(
                bodyUniqueId=self.robot_id,
                endEffectorLinkIndex=link_id,
                targetPosition=pose.xyz,
                targetOrientation=pose.so3.xyzw,
                # lowerLimits=lower_limits,
                # upperLimits=upper_limits,
                # jointRanges=joint_ranges,
                # restPoses=rest_poses,
            )
            arr = np.array(solution)
            if np.alltrue(arr >= np.array(lower_limits)) and np.alltrue(
                arr <= np.array(upper_limits)
            ):
                self.marionette(solution[:7])
                return solution[:7]
            self.marionette(solution[:7])

        return None

    def marionette(self, config):
        for i in range(0, 7):
            p.resetJointState(self.robot_id, i, config[i], physicsClientId=self.clid)
        # Spread the fingers if they aren't included--prevents self collision
        p.resetJointState(self.robot_id, 9, 0.02, physicsClientId=self.clid)
        p.resetJointState(self.robot_id, 10, 0.02, physicsClientId=self.clid)


class FrankaHandEnv(Bullet):
    def load_robot(self):
        if self.robot_id is not None:
            print("There is already a robot loaded. Removing and reloading")
            p.removeBody(self.robot_id, physicsClientId=self.clid)
        self.robot_id = p.loadURDF(
            FrankaHand.urdf,
            useFixedBase=True,
            physicsClientId=self.clid,
            flags=p.URDF_USE_SELF_COLLISION,
        )
        self.urdf_path = FrankaHand.urdf

        # Code snippet borrowed from https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12728
        self._setup_robot()

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


class FrankaHandArmEnv(Bullet):
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
