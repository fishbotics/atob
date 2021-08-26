import numpy as np
from pathlib import Path

class FrankaRobot:
    # TODO remove this after making this more general
    JOINT_LIMITS = [
        (-2.8973, 2.8973),
        (-1.7628, 1.7628),
        (-2.8973, 2.8973),
        (-3.0718, -0.0698),
        (-2.8973, 2.8973),
        (-0.0175, 3.7525),
        (-2.8973, 2.8973),
    ]
    DOF = 7
    urdf = str(Path(__file__).parent.parent / "urdf" / "franka_panda" / "panda.urdf")

    @staticmethod
    def random_configuration():
        limits = np.array(FrankaRobot.JOINT_LIMITS)
        return (limits[:, 1] - limits[:, 0]) * (np.random.rand(7)) + limits[:, 0]


class FrankaHand:
    JOINT_LIMITS = None
    DOF = 6
    urdf = str(Path(__file__).parent.parent / "urdf" / "panda_hand" / "panda.urdf")

    @staticmethod
    def random_configuration():
        raise NotImplementedError("Random configuration not implemented for Franka Hand"

