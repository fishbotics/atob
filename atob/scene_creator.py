from atob.planner import Planner, FrankaRobot
from atob.geometry import Cuboid
import numpy as np
import time
from datetime import timedelta
import multiprocessing as mp
from multiprocessing import Queue, Process
from ompl.util import noOutputHandler
import os
import h5py
from pathlib import Path
from pyquaternion import Quaternion
from termcolor import colored

PATH_LENGTH = 300
PATHS_PER_FILE = 10000


def log(msg):
    print(colored(msg, "blue"))


def roll_pitch_yaw(m):
    pitch = np.arcsin(-np.clip(m[2, 0], -1, 1))
    if np.abs(m[2, 0]) < 0.999999:
        roll = np.arctan2(m[2, 1], m[2, 2])
        yaw = np.arctan2(m[1, 0], m[0, 0])
    else:
        roll = 0
        yaw = np.arctan2(-m[0, 1], m[1, 1])
    return np.array([roll, pitch, yaw])


def unif_sample(a, b):
    assert a < b, "[a: {}] must be be less than [b: {}]".format(a, b)
    return (b - a) * np.random.random_sample() + a


def unif(center, radius):
    return center + unif_sample(-radius, radius)


# TODO clean this all up--maybe move to another file
class CubbyEnvironment:
    def __init__(self, deterministic=True, fix_env=False, fix_start=False):
        self.deterministic = deterministic
        self.fix_env = fix_env
        self.fix_start = fix_start

    def gen(self):

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
        self.rotation_matrix = np.eye(4)
        self.rotation = 0
        self.rotation = np.pi / 18
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
                self.rotation = unif(0, np.pi / 18)
                self.rotation_matrix = self._rotation_matrix()

            self.start_zone = np.random.randint(0, 4)
            self.target_zone = np.random.randint(0, 4)
            assert self.start_zone < 4
            assert self.target_zone < 4

            for i in range(4):
                try:
                    self._start = self._random_eff_pose((self.start_zone + i) % 4)
                except Exception as e:
                    print(e)
                    if i < 3:
                        continue
                    else:
                        print("No valid zone for start")
                        return False
                else:
                    self.start_zone = (self.start_zone + i) % 4
                    break

            for i in range(4):
                try:
                    self._target = self._random_eff_pose((self.target_zone + i) % 4)
                except:
                    if i < 3:
                        continue
                    else:
                        print("No valid zone for target")
                        return False
                else:
                    self.target_zone = (self.target_zone + i) % 4
                    break
        return True

    def _deterministic_eff_pose(self, zone):
        assert self.deterministic

        transform_matrix = np.eye(4)
        r = Quaternion(axis=(0, 1, 0), radians=np.pi / 2)
        transform_matrix[:3, :3] = r.rotation_matrix
        if zone == 0:
            transform_matrix[:3, -1] = np.array(
                [
                    self.cubby_front,
                    (self.cubby_left + self.cubby_mid_v_y) / 2,
                    (self.cubby_bottom + self.cubby_mid_h_z) / 2,
                ]
            )
        elif zone == 1:
            transform_matrix[:3, -1] = np.array(
                [
                    self.cubby_front,
                    (self.cubby_right + self.cubby_mid_v_y) / 2,
                    (self.cubby_bottom + self.cubby_mid_h_z) / 2,
                ]
            )
        elif zone == 2:
            transform_matrix[:3, -1] = np.array(
                [
                    self.cubby_front,
                    (self.cubby_left + self.cubby_mid_v_y) / 2,
                    (self.cubby_top + self.cubby_mid_h_z) / 2,
                ]
            )
        else:
            transform_matrix[:3, -1] = np.array(
                [
                    self.cubby_front,
                    (self.cubby_right + self.cubby_mid_v_y) / 2,
                    (self.cubby_top + self.cubby_mid_h_z) / 2,
                ]
            )
        transform_matrix = np.matmul(self.rotation_matrix, transform_matrix)
        # TODO clean this: transform into the pybullet target frame rather than the old right gripper frame
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

        #  transform_matrix = np.matmul(self.rotation_matrix,transform_matrix)
        transform_matrix = np.matmul(transform_matrix, self.rotation_matrix)
        # TODO clean this: transform into the pybullet target frame rather than the old right gripper frame
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
        rotation = np.array(
            [
                [np.cos(self.rotation), -np.sin(self.rotation), 0, 0],
                [np.sin(self.rotation), np.cos(self.rotation), 0, 0],
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
        return np.matmul(world_T_cabinet, np.matmul(rotation, cabinet_T_world))

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
                    self.thickness,
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
                    self.thickness,
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


def random_cabinet(planner):
    environment = CubbyEnvironment(deterministic=True)
    # Continually try to regenerate environment until a reasonable solution is found
    while True:
        if environment.gen():
            break
    # Obstacles must be loaded before calling collision free ik
    obstacles = environment.obstacles
    planner.create_scene(obstacles)

    assert (
        environment.cubby_back >= environment.start[0, 3]
    ), "Start is behind rear of cabinet"

    start = planner.bullet.collision_free_ik(environment.start, retries=100)
    if start is None:
        # print("Could not find start configuration")
        return None, None
    goal = planner.bullet.collision_free_ik(environment.target)
    if goal is None:
        # print("Could not find goal configuration")
        return None, None
    try:
        path = planner.plan(
            start=start, goal=goal, interpolate=PATH_LENGTH, max_runtime=2
        )
    except Exception as e:
        print(e)
        return None, None
    if path is None:
        # print("Could not find plan")
        return None, None
    if planner.gui:
        planner.bullet.marionette(start)
        dist = (
            environment.start[:3, 3]
            - planner.bullet.link_frames["panda_grasptarget"][:3, 3]
        )
        time.sleep(3)
        for q in path:
            planner.bullet.marionette(q)
            time.sleep(0.01)
        time.sleep(2)

    # There has to be a reasonable limit on what's feasible for these plans,
    # so we're limiting plans to be under PATH_LENGTH
    if len(path) > PATH_LENGTH:
        return None, None
    return obstacles, path


def process_random_cabinet(seed, file_names):
    np.random.seed(seed)
    planner = Planner(gui=False)
    count = 0
    data = []

    file_names.put(f"data/in_process_data_gen_{seed}.hdf5")
    start_time = time.time()

    with h5py.File(f"data/in_process_data_gen_{seed}.hdf5", "w-") as f:
        paths = f.create_dataset("robot_configurations", (PATHS_PER_FILE, 300, 7))
        dims = f.create_dataset("obstacle_dims", (PATHS_PER_FILE, 7, 3))
        centers = f.create_dataset("obstacle_centers", (PATHS_PER_FILE, 7, 3))
        quats = f.create_dataset("obstacle_quaternions", (PATHS_PER_FILE, 7, 4))

        while count < PATHS_PER_FILE:
            obstacles, path = random_cabinet(planner=planner)
            if obstacles is not None:
                paths[count, :, :] = path
                for ii, obstacle in enumerate(obstacles):
                    dims[count, ii, :] = np.array(obstacle.dims)
                    centers[count, ii, :] = np.array(obstacle.center)
                    quats[count, ii, :] = np.array(obstacle.wxyz)
                count += 1
                if count % 10000 == 0:
                    f.flush()
                    time_elapsed = timedelta(seconds=time.time() - start_time)

                    print(f"PID {os.getpid()}: Completed {count}/{PATHS_PER_FILE} in {time_elapsed}")
            planner.reset()
    return


def naive_block(planner):
    start = FrankaRobot.random_configuration()
    goal = FrankaRobot.random_configuration()
    straight_path = np.linspace(start, goal, num=20)
    blocking_config = straight_path[np.random.randint(7, 13)]
    planner.bullet.marionette(blocking_config)
    frames = planner.bullet.link_frames
    position = frames["panda_link6"][:3, 3]

    obstacle = Cuboid.random(
        dimension_range=[[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]], quaternion=True
    )
    obstacle.center = position
    planner.create_scene([obstacle])
    try:
        path = planner.plan(
            start=start, goal=goal, interpolate=PATH_LENGTH, max_runtime=5
        )
    except Exception:
        return None, None
    if planner.gui:
        planner.bullet.marionette(start)
        time.sleep(3)
        for q in path:
            planner.bullet.marionette(q)
            time.sleep(0.1)
        time.sleep(2)

    # There has to be a reasonable limit on what's feasible for these plans, so we're limiting plans to be under PATH_LENGTH
    if len(path) > PATH_LENGTH:
        return None, None
    return obstacle, path


def double_block(planner):
    start = FrankaRobot.random_configuration()
    goal = FrankaRobot.random_configuration()
    straight_path = np.linspace(start, goal, num=20)
    blocking_config = straight_path[np.random.randint(7, 13)]
    planner.bullet.marionette(blocking_config)
    frames = planner.bullet.link_frames
    position = frames["panda_link6"][:3, 3]

    obstacle = Cuboid.random(
        dimension_range=[[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]], quaternion=True
    )
    obstacle.center = position
    planner.create_scene([obstacle])
    try:
        path = planner.plan(
            start=start, goal=goal, interpolate=PATH_LENGTH, max_runtime=5
        )
    except Exception:
        return None, None

    plan1 = (obstacle, path)
    planner.reset()

    blocking_config = path[np.random.randint(7, 13)]
    planner.bullet.marionette(blocking_config)
    frames = planner.bullet.link_frames
    position = frames["panda_link6"][:3, 3]
    obstacle = Cuboid.random(
        dimension_range=[[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]], quaternion=True
    )
    obstacle.center = position
    planner.create_scene([obstacle])
    try:
        path = planner.plan(
            start=start, goal=goal, interpolate=PATH_LENGTH, max_runtime=5
        )
    except Exception:
        return None, None
    plan2 = (obstacle, path)

    if planner.gui:
        planner.bullet.marionette(start)
        time.sleep(3)
        for q in path:
            planner.bullet.marionette(q)
            time.sleep(0.1)
        time.sleep(2)

    # There has to be a reasonable limit on what's feasible for these plans, so we're limiting plans to be under PATH_LENGTH
    if len(path) > PATH_LENGTH:
        return None, None
    return plan1, plan2


def process_naive_block(seed, file_names):
    np.random.seed(seed)
    planner = Planner(gui=False)
    count = 0
    data = []

    file_names.put(f"data/in_process_data_gen_{seed}.hdf5")

    with h5py.File(f"data/in_process_data_gen_{seed}.hdf5", "w-") as f:
        paths = f.create_dataset("robot_configurations", (PATHS_PER_FILE, 300, 7))
        dims = f.create_dataset("obstacle_dims", (PATHS_PER_FILE, 1, 3))
        centers = f.create_dataset("obstacle_centers", (PATHS_PER_FILE, 1, 3))
        quats = f.create_dataset("obstacle_quaternions", (PATHS_PER_FILE, 1, 4))
        while count < PATHS_PER_FILE:
            obstacle, path = naive_block(planner=planner)
            if obstacle is not None:
                paths[count, :, :] = path
                dims[count, 0, :] = np.array(obstacle.dims)
                centers[count, 0, :] = np.array(obstacle.center)
                quats[count, 0, :] = np.array(obstacle.wxyz)
                count += 1
                if count % 1000 == 0:
                    f.flush()
                    print(f"PID {os.getpid()}: Completed {count}/{PATHS_PER_FILE}")
            planner.reset()
    return


def process_double_block(seed, file_names):
    np.random.seed(seed)
    planner = Planner(gui=False)
    count = 0
    data = []

    file_names.put(f"data/in_process_data_gen_{seed}.hdf5")

    with h5py.File(f"data/in_process_data_gen_{seed}.hdf5", "w-") as f:
        paths = f.create_dataset("robot_configurations", (PATHS_PER_FILE, 300, 7))
        dims = f.create_dataset("obstacle_dims", (PATHS_PER_FILE, 1, 3))
        centers = f.create_dataset("obstacle_centers", (PATHS_PER_FILE, 1, 3))
        quats = f.create_dataset("obstacle_quaternions", (PATHS_PER_FILE, 1, 4))
        while count < PATHS_PER_FILE // 2:
            plan1, plan2 = double_block(planner=planner)
            if plan1 is not None:
                path, obstacle = plan1
                paths[2 * count, :, :] = path
                dims[2 * count, 0, :] = np.array(obstacle.dims)
                centers[2 * count, 0, :] = np.array(obstacle.center)
                quats[2 * count, 0, :] = np.array(obstacle.wxyz)

                path, obstacle = plan2
                paths[2 * count + 1, :, :] = path
                dims[2 * count + 1, 0, :] = np.array(obstacle.dims)
                centers[2 * count + 1, 0, :] = np.array(obstacle.center)
                quats[2 * count + 1, 0, :] = np.array(obstacle.wxyz)

                count += 1
                if count % 500 == 0:
                    f.flush()
                    print(f"PID {os.getpid()}: Completed {2 * count}/{PATHS_PER_FILE}")
            planner.reset()
    return


def main():
    processes = []
    file_names = Queue()
    for i in range(mp.cpu_count() - 4):
        # p = Process(
        #     target=process_naive_block, args=(np.random.randint(1000000), file_names)
        # )
        p = Process(
            target=process_random_cabinet, args=(np.random.randint(1000000), file_names)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    files_to_join = []
    while not file_names.empty():
        files_to_join.append(file_names.get())
    size = PATHS_PER_FILE * len(files_to_join)
    with h5py.File(f"data/train.hdf5", "w") as f:
        paths = f.create_dataset("robot_configurations", (size, 300, 7))
        dims = f.create_dataset("obstacle_dims", (size, 7, 3))
        centers = f.create_dataset("obstacle_centers", (size, 7, 3))
        quats = f.create_dataset("obstacle_quaternions", (size, 7, 4))

        for ii, fi in enumerate(files_to_join):
            with h5py.File(fi, "r") as g:
                paths[ii * PATHS_PER_FILE : (ii + 1) * PATHS_PER_FILE, ...] = g[
                    "robot_configurations"
                ][:, :, :]
                dims[ii * PATHS_PER_FILE : (ii + 1) * PATHS_PER_FILE, ...] = g[
                    "obstacle_dims"
                ][:, :]
                centers[ii * PATHS_PER_FILE : (ii + 1) * PATHS_PER_FILE, ...] = g[
                    "obstacle_centers"
                ][:, :]
                quats[ii * PATHS_PER_FILE : (ii + 1) * PATHS_PER_FILE, ...] = g[
                    "obstacle_quaternions"
                ][:, :]
    for fi in files_to_join:
        os.remove(Path(fi).resolve())
    return True


if __name__ == "__main__":
    noOutputHandler()
    main()
    # file_names = Queue()
    # process_random_cabinet(0, file_names)
    # planner.reset()
    # np.random.seed(0)
    # planner = Planner(gui=False)
    # random_cabinet(planner)
    # planner.reset()
