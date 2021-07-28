import numpy as np


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

    @staticmethod
    def random_configuration():
        limits = np.array(FrankaRobot.JOINT_LIMITS)
        return (limits[:, 1] - limits[:, 0]) * (np.random.rand(7)) + limits[:, 0]
