import numpy as np
from pathlib import Path
from atob.geometry import SE3, SO3
from ikfast_franka_panda import get_fk, get_ik


class FrankaRobot:
    # TODO remove this after making this more general
    JOINT_LIMITS = np.array(
        [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (-0.0175, 3.7525),
            (-2.8973, 2.8973),
        ]
    )
    DOF = 7
    EFF_LIST = set(["panda_link8", "right_gripper", "panda_grasptarget"])
    EFF_T_LIST = {
        ("panda_link8", "right_gripper"): SE3(
            xyz=[0, 0, 0.1], so3=SO3.from_rpy([0, 0, 2.35619449019])
        ),
        ("panda_link8", "panda_grasptarget"): SE3(
            xyz=[0, 0, 0.105], rpy=[0, 0, -0.785398163397]
        ),
    }
    urdf = str(Path(__file__).parent.parent / "urdf" / "franka_panda" / "panda.urdf")

    @staticmethod
    def within_limits(config):
        return np.all(config >= FrankaRobot.JOINT_LIMITS[:, 0]) and np.all(
            config <= FrankaRobot.JOINT_LIMITS[:, 1]
        )

    @staticmethod
    def fk(config, eff_frame="panda_link8"):
        """
        Returns the SE3 frame of the end effector
        """
        assert (
            eff_frame in FrankaRobot.EFF_LIST
        ), "Default FK only calculated for a valid end effector frame"
        pos, rot = get_fk(config)
        mat = np.eye(4)
        mat[:3, :3] = np.asarray(rot)
        mat[:3, 3] = np.asarray(pos)
        if eff_frame == "panda_link8":
            return SE3(matrix=mat)
        elif eff_frame == "right_gripper":
            return (
                SE3(matrix=mat)
                @ FrankaRobot.EFF_T_LIST[("panda_link8", "right_gripper")]
            )
        else:
            return (
                SE3(matrix=mat)
                @ FrankaRobot.EFF_T_LIST[("panda_link8", "panda_grasptarget")]
            )

    @staticmethod
    def ik(pose, panda_link7, eff_frame="panda_link8"):
        """
        :param pose: SE3 pose expressed in specified end effector frame
        :param panda_link7: Value for the joint panda_link7, other IK can be calculated with this joint value set.
            Must be within joint range
        :param eff_frame: Desired end effector frame, must be among [panda_link8, right_gripper, panda_grasptarget]
        :return: Typically 4 solutions to IK
        """
        assert (
            eff_frame in FrankaRobot.EFF_LIST
        ), "IK only calculated for a valid end effector frame"
        if eff_frame == "right_gripper":
            pose = (
                pose @ FrankaRobot.EFF_T_LIST[("panda_link8", "right_gripper")].inverse
            )
        elif eff_frame == "panda_grasptarget":
            pose = (
                pose
                @ FrankaRobot.EFF_T_LIST[("panda_link8", "panda_grasptarget")].inverse
            )
        rot = pose.so3.matrix.tolist()
        pos = pose.xyz
        assert (
            panda_link7 >= FrankaRobot.JOINT_LIMITS[-1, 0]
            and panda_link7 <= FrankaRobot.JOINT_LIMITS[-1, 1]
        ), f"Value for floating joint must be within range {FrankaRobot.JOINT_LIMITS[-1, :].tolist()}"
        solutions = [np.asarray(s) for s in get_ik(pos, rot, [panda_link7])]
        return [
            s
            for s in solutions
            if (
                np.all(s >= FrankaRobot.JOINT_LIMITS[:, 0])
                and np.all(s <= FrankaRobot.JOINT_LIMITS[:, 1])
            )
        ]

    @staticmethod
    def random_configuration():
        limits = FrankaRobot.JOINT_LIMITS
        return (limits[:, 1] - limits[:, 0]) * (np.random.rand(7)) + limits[:, 0]

    @staticmethod
    def random_ik(pose, eff_frame="panda_link8"):
        link7 = FrankaRobot.random_configuration()[-1]
        config = FrankaRobot.random_configuration()
        try:
            return FrankaRobot.ik(pose, config[-1], eff_frame)
        except:
            print(config)
            raise


class FrankaHand:
    JOINT_LIMITS = None
    DOF = 6
    urdf = str(Path(__file__).parent.parent / "urdf" / "panda_hand" / "panda.urdf")

    @staticmethod
    def random_configuration():
        raise NotImplementedError(
            "Random configuration not implemented for Franka Hand"
        )
