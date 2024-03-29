import numpy as np
from copy import deepcopy
from pyquaternion import Quaternion
from geometrout.primitive import Cuboid


def unif_sample(a, b):
    assert a < b, "[a: {}] must be be less than [b: {}]".format(a, b)
    return (b - a) * np.random.random_sample() + a


def unif(center, radius):
    return center + unif_sample(-radius, radius)


# TODO clean this all up--there's a lot of cruft from the old implementation
class CubbyEnvironment:
    def __init__(self, deterministic=True, fix_env=False, fix_start=False):
        self.deterministic = deterministic
        self.fix_env = fix_env
        self.fix_start = fix_start

    def is_blocked(self):
        return self.start_zone != self.target_zone

    def create_unblocked_environment(self):
        assert self.is_blocked()
        env = deepcopy(self)
        if (env.start_zone in [0, 1] and env.target_zone in [2, 3]) or (
            env.start_zone in [2, 3] and env.target_zone in [0, 1]
        ):
            # Remove horizontal divider
            env.middle_shelf_thickness = 0
        if (env.start_zone in [0, 2] and env.target_zone in [1, 3]) or (
            env.start_zone in [1, 3] and env.target_zone in [0, 2]
        ):
            env.center_wall_thickness = 0
        return env

    def gen(self, use_world_rotation=False):

        # Default the default environment config
        # These defaults are used for the deterministic case
        self.cubby_right = -0.7
        self.cubby_left = 0.7
        self.cubby_bottom = 0.2
        self.cubby_top = 0.7
        self.cubby_back = 0.75
        self.cubby_front = 0.55
        self.cubby_mid_h_z = 0.45
        self.cubby_mid_v_y = 0.0
        self.thickness = 0.02
        self.middle_shelf_thickness = 0.02
        self.center_wall_thickness = 0.02
        self.rotation_matrix = np.eye(4)
        self.in_cabinet_rotation = np.pi / 18
        if use_world_rotation:
            self.world_rotation = np.pi / 8
        else:
            self.world_rotation = 0
        self.rotation_matrix = self._rotation_matrix()

        if self.deterministic:
            try:
                self.start_zone += 1
                self.start_zone %= 4
                self.target_zone += 1
                self.target_zone %= 4
            except Exception:
                self.start_zone = 0
                self.target_zone = 1
            self._start = self._deterministic_eff_pose(self.start_zone)
            self._target = self._deterministic_eff_pose(self.target_zone)
        else:
            if not self.fix_env:
                self.cubby_left = unif(self.cubby_left, 0.1)
                self.cubby_right = unif(self.cubby_right, 0.1)
                self.cubby_bottom = unif(self.cubby_bottom, 0.1)
                self.cubby_front = unif(self.cubby_front, 0.1)
                self.cubby_back = self.cubby_front + unif_sample(0.2, 0.35)
                self.cubby_top = unif(self.cubby_top, 0.1)
                self.cubby_mid_h_z = unif(self.cubby_mid_h_z, 0.1)
                self.cubby_mid_v_y = unif(self.cubby_mid_v_y, 0.1)
                self.thickness = unif(self.thickness, 0.01)
                self.in_cabinet_rotation = unif(0, np.pi / 18)
                if use_world_rotation:
                    self.world_rotation = unif(0, np.pi / 2)
                else:
                    self.world_rotation = 0
                self.rotation_matrix = self._rotation_matrix()

            initial_start = np.random.randint(0, 4)
            starts = [x % 4 for x in list(range(initial_start, initial_start + 4))]
            self._start = None
            for s in starts:
                try:
                    self._start = self._random_eff_pose(s)
                    self.start_zone = s
                    break
                except Exception as e:
                    continue
            if self._start is None:
                print("No valid zone for start")
                return False
            targets = [x for x in list(range(4)) if x != self.start_zone]
            np.random.shuffle(targets)
            self._target = None
            for t in targets:
                try:
                    self._target = self._random_eff_pose(t)
                    self.target_zone = t
                    break
                except Exception as e:
                    continue
            if self._target is None:
                print("No valid zone for target")
                return False
        return True

    def check_zone(self, pose_matrix):
        transform_matrix = np.linalg.inv(self.rotation_matrix) @ pose_matrix
        xyz = transform_matrix[:3, -1]
        x, y, z = xyz
        if not (self.cubby_front < x and x < self.cubby_back):
            return -1
        if self.cubby_bottom < z and z < self.cubby_mid_h_z:
            if self.cubby_mid_v_y < y and y < self.cubby_left:
                return 0
            else:
                return 1
        else:
            if self.cubby_mid_v_y < y and y < self.cubby_left:
                return 2
            else:
                return 3

    def _deterministic_eff_pose(self, zone):
        assert self.deterministic

        transform_matrix = np.eye(4)
        r = Quaternion(axis=(0, 1, 0), radians=np.pi / 2)
        transform_matrix[:3, :3] = r.rotation_matrix
        if zone == 0:
            transform_matrix[:3, -1] = np.array(
                [
                    (self.cubby_front + self.cubby_back) / 2,
                    (self.cubby_left + self.cubby_mid_v_y) / 2,
                    (self.cubby_bottom + self.cubby_mid_h_z) / 2,
                ]
            )
        elif zone == 1:
            transform_matrix[:3, -1] = np.array(
                [
                    (self.cubby_front + self.cubby_back) / 2,
                    (self.cubby_right + self.cubby_mid_v_y) / 2,
                    (self.cubby_bottom + self.cubby_mid_h_z) / 2,
                ]
            )
        elif zone == 2:
            transform_matrix[:3, -1] = np.array(
                [
                    (self.cubby_front + self.cubby_back) / 2,
                    (self.cubby_left + self.cubby_mid_v_y) / 2,
                    (self.cubby_top + self.cubby_mid_h_z) / 2,
                ]
            )
        else:
            transform_matrix[:3, -1] = np.array(
                [
                    (self.cubby_front + self.cubby_back) / 2,
                    (self.cubby_right + self.cubby_mid_v_y) / 2,
                    (self.cubby_top + self.cubby_mid_h_z) / 2,
                ]
            )
        transform_matrix = np.matmul(self.rotation_matrix, transform_matrix)
        # TODO clean this: transform into the pybullet target frame rather than the old right gripper frame
        # transform_matrix = np.matmul(
        #     transform_matrix,
        #     np.array(
        #         [
        #             [np.cos(np.pi), -np.sin(np.pi), 0, 0],
        #             [np.sin(np.pi), np.cos(np.pi), 0, 0],
        #             [0, 0, 1, 0],
        #             [0, 0, 0, 1],
        #         ]
        #     ),
        # )
        return transform_matrix

    def _random_eff_pose(self, zone):
        transform_matrix = np.eye(4)

        theta = unif(np.pi / 2, np.pi / 4)
        theta = np.pi / 2
        axis = (np.cos(theta), np.sin(theta), 0)
        r = Quaternion(axis=axis, radians=np.pi / 2)
        transform_matrix[:3, :3] = r.rotation_matrix

        xrearbuf = 0.08
        xfrontbuf = 0.02
        ybuf = 0.12
        zbuf = 0.06

        if zone == 0:
            # lower left
            transform_matrix[:3, -1] = np.array(
                [
                    unif_sample(
                        self.cubby_front + xfrontbuf, self.cubby_back - xrearbuf
                    ),
                    unif_sample(self.cubby_mid_v_y + ybuf, self.cubby_left - ybuf),
                    unif_sample(self.cubby_bottom + zbuf, self.cubby_mid_h_z - zbuf),
                ]
            )
        elif zone == 1:
            # lower right
            transform_matrix[:3, -1] = np.array(
                [
                    unif_sample(
                        self.cubby_front + xfrontbuf, self.cubby_back - xrearbuf
                    ),
                    unif_sample(self.cubby_right + ybuf, self.cubby_mid_v_y - ybuf),
                    unif_sample(self.cubby_bottom + zbuf, self.cubby_mid_h_z - zbuf),
                ]
            )
        elif zone == 2:
            # top left
            transform_matrix[:3, -1] = np.array(
                [
                    unif_sample(
                        self.cubby_front + xfrontbuf, self.cubby_back - xrearbuf
                    ),
                    unif_sample(self.cubby_mid_v_y + ybuf, self.cubby_left - ybuf),
                    unif_sample(self.cubby_mid_h_z + zbuf, self.cubby_top - zbuf),
                ]
            )
        else:
            # top right
            transform_matrix[:3, -1] = np.array(
                [
                    unif_sample(
                        self.cubby_front + xfrontbuf, self.cubby_back - xrearbuf
                    ),
                    unif_sample(self.cubby_right + ybuf, self.cubby_mid_v_y - ybuf),
                    unif_sample(self.cubby_mid_h_z + zbuf, self.cubby_top - zbuf),
                ]
            )

        transform_matrix = np.matmul(self.rotation_matrix, transform_matrix)
        # TODO clean this: transform into the pybullet target frame rather than the old right gripper frame
        # transform_matrix = np.matmul(
        #     transform_matrix,
        #     np.array(
        #         [
        #             [np.cos(np.pi), -np.sin(np.pi), 0, 0],
        #             [np.sin(np.pi), np.cos(np.pi), 0, 0],
        #             [0, 0, 1, 0],
        #             [0, 0, 0, 1],
        #         ]
        #     ),
        # )
        return transform_matrix

    @property
    def start(self):
        if self.fix_start:
            raise Exception(
                "When start position is fixed, start position should be hardcoded and"
                " not queried from environment"
            )
        return self._start

    @property
    def target(self):
        return self._target

    def _rotation_matrix(self):
        cabinet_T_world = np.array(
            [
                [1, 0, 0, -(self.cubby_front + self.cubby_back) / 2],
                [0, 1, 0, -(self.cubby_left + self.cubby_right) / 2],
                [0, 0, 1, -(self.cubby_top + self.cubby_bottom) / 2],
                [0, 0, 0, 1],
            ]
        )
        in_cabinet_rotation = np.array(
            [
                [
                    np.cos(self.in_cabinet_rotation),
                    -np.sin(self.in_cabinet_rotation),
                    0,
                    0,
                ],
                [
                    np.sin(self.in_cabinet_rotation),
                    np.cos(self.in_cabinet_rotation),
                    0,
                    0,
                ],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        world_T_cabinet = np.array(
            [
                [1, 0, 0, (self.cubby_front + self.cubby_back) / 2],
                [0, 1, 0, (self.cubby_left + self.cubby_right) / 2],
                [0, 0, 1, (self.cubby_top + self.cubby_bottom) / 2],
                [0, 0, 0, 1],
            ]
        )
        pivot = np.matmul(
            world_T_cabinet, np.matmul(in_cabinet_rotation, cabinet_T_world)
        )
        return np.matmul(
            np.array(
                [
                    [
                        np.cos(self.world_rotation),
                        -np.sin(self.world_rotation),
                        0,
                        0,
                    ],
                    [
                        np.sin(self.world_rotation),
                        np.cos(self.world_rotation),
                        0,
                        0,
                    ],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ),
            pivot,
        )
        return

    def _unrotated_obstacles(self):
        return {
            "back_wall": {
                "center": [
                    self.cubby_back,
                    (self.cubby_left + self.cubby_right) / 2,
                    self.cubby_top / 2,
                ],
                "dims": [
                    self.thickness,
                    (self.cubby_left - self.cubby_right),
                    self.cubby_top,
                ],
            },
            "bottom_shelf": {
                "center": [
                    (self.cubby_front + self.cubby_back) / 2,
                    (self.cubby_left + self.cubby_right) / 2,
                    self.cubby_bottom,
                ],
                "dims": [
                    self.cubby_back - self.cubby_front,
                    self.cubby_left - self.cubby_right,
                    self.thickness,
                ],
            },
            "top_shelf": {
                "center": [
                    (self.cubby_front + self.cubby_back) / 2,
                    (self.cubby_left + self.cubby_right) / 2,
                    self.cubby_top,
                ],
                "dims": [
                    self.cubby_back - self.cubby_front,
                    self.cubby_left - self.cubby_right,
                    self.thickness,
                ],
            },
            "middle_shelf": {
                "center": [
                    (self.cubby_front + self.cubby_back) / 2,
                    (self.cubby_left + self.cubby_right) / 2,
                    self.cubby_mid_h_z,
                ],
                "dims": [
                    self.cubby_back - self.cubby_front,
                    self.cubby_left - self.cubby_right,
                    self.middle_shelf_thickness,
                ],
            },
            "right_wall": {
                "center": [
                    (self.cubby_front + self.cubby_back) / 2,
                    self.cubby_right,
                    (self.cubby_top + self.cubby_bottom) / 2,
                ],
                "dims": [
                    self.cubby_back - self.cubby_front,
                    self.thickness,
                    (self.cubby_top - self.cubby_bottom) + self.thickness,
                ],
            },
            # Left wall
            "left_wall": {
                "center": [
                    (self.cubby_front + self.cubby_back) / 2,
                    self.cubby_left,
                    (self.cubby_top + self.cubby_bottom) / 2,
                ],
                "dims": [
                    self.cubby_back - self.cubby_front,
                    self.thickness,
                    (self.cubby_top - self.cubby_bottom) + self.thickness,
                ],
            },
            # Center wall
            "center_wall": {
                "center": [
                    (self.cubby_front + self.cubby_back) / 2,
                    self.cubby_mid_v_y,
                    (self.cubby_top + self.cubby_bottom) / 2,
                ],
                "dims": [
                    self.cubby_back - self.cubby_front,
                    self.center_wall_thickness,
                    self.cubby_top - self.cubby_bottom + self.thickness,
                ],
            },
        }

    @property
    def obstacles(self):
        obstacles = self._unrotated_obstacles()
        for name in obstacles:
            center = obstacles[name]["center"]
            new_matrix = np.matmul(
                self.rotation_matrix,
                np.array(
                    [
                        [1, 0, 0, center[0]],
                        [0, 1, 0, center[1]],
                        [0, 0, 1, center[2]],
                        [0, 0, 0, 1],
                    ]
                ),
            )
            obstacles[name]["center"] = [
                float(new_matrix[0, 3]),
                float(new_matrix[1, 3]),
                float(new_matrix[2, 3]),
            ]
            quat = Quaternion(matrix=new_matrix)
            obstacles[name]["quat"] = np.array(
                [
                    quat.w,
                    quat.x,
                    quat.y,
                    quat.z,
                ]
            )
        return [
            Cuboid(center=o["center"], dims=o["dims"], quaternion=o["quat"])
            for o in obstacles.values()
        ]
