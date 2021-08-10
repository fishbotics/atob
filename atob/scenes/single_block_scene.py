import numpy as np
from pyquaternion import Quaternion


def unif_sample(a, b):
    assert a < b, "[a: {}] must be be less than [b: {}]".format(a, b)
    return (b - a) * np.random.random_sample() + a


def unif(center, radius):
    return center + unif_sample(-radius, radius)


class SingleBlockScene:
    def __init__(self):
        pass

    def gen(self):

        # Default the default environment config
        # These defaults are used for the deterministic case
        self.workspace_right = -0.8
        self.workspace_left = 0.8
        self.workspace_bottom = 0.1
        self.workspace_top = 0.8
        self.workspace_back = 0.8
        self.workspace_front = 0.45
        self.rotation_matrix = np.eye(4)
        self.rotation = unif(0, np.pi / 18)
        self.rotation_matrix = self._rotation_matrix()
        self._start = self._random_eff_pose()
        self._target = self._random_eff_pose()
        return True

    @property
    def start(self):
        return self._start

    @property
    def target(self):
        return self._target

    def _random_eff_pose(self):
        transform_matrix = np.eye(4)

        theta = unif(np.pi / 2, np.pi / 4)
        theta = np.pi / 2
        axis = (np.cos(theta), np.sin(theta), 0)
        r = Quaternion(axis=axis, radians=np.pi / 2)
        transform_matrix[:3, :3] = r.rotation_matrix

        transform_matrix[:3, -1] = np.array(
            [
                unif_sample(self.workspace_front, self.workspace_back),
                unif_sample(self.workspace_right, self.workspace_left),
                unif_sample(self.workspace_bottom, self.workspace_top),
            ]
        )

        #  transform_matrix = np.matmul(self.rotation_matrix,transform_matrix)
        transform_matrix = np.matmul(transform_matrix, self.rotation_matrix)
        # TODO clean this: transform into the pybullet target frame rather
        # than the old right gripper frame
        transform_matrix = np.matmul(
            transform_matrix,
            np.array(
                [
                    [np.cos(np.pi), -np.sin(np.pi), 0, 0],
                    [np.sin(np.pi), np.cos(np.pi), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ),
        )
        return transform_matrix

    def _rotation_matrix(self):
        workspace_T_world = np.array(
            [
                [1, 0, 0, -(self.workspace_front + self.workspace_back) / 2],
                [0, 1, 0, -(self.workspace_left + self.workspace_right) / 2],
                [0, 0, 1, -(self.workspace_top + self.workspace_bottom) / 2],
                [0, 0, 0, 1],
            ]
        )
        rotation = np.array(
            [
                [np.cos(self.rotation), -np.sin(self.rotation), 0, 0],
                [np.sin(self.rotation), np.cos(self.rotation), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        world_T_workspace = np.array(
            [
                [1, 0, 0, (self.workspace_front + self.workspace_back) / 2],
                [0, 1, 0, (self.workspace_left + self.workspace_right) / 2],
                [0, 0, 1, (self.workspace_top + self.workspace_bottom) / 2],
                [0, 0, 0, 1],
            ]
        )
        return np.matmul(world_T_workspace, np.matmul(rotation, workspace_T_world))
