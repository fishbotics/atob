from atob.planner import FrankaPlanner, FrankaHandPlanner, FrankaRobot
from atob.bullet import FrankaEnv, FrankaHandEnv
from atob.geometry import Cuboid, SE3
from atob.scenes.cubby_scene import CubbyEnvironment
from atob.scenes.single_block_scene import SingleBlockScene
import numpy as np
import time
from datetime import timedelta
import multiprocessing as mp
from multiprocessing import Queue, Process
from ompl.util import noOutputHandler
import os
import h5py
from pathlib import Path
from termcolor import colored

PATH_LENGTH = 300
PATHS_PER_FILE = 10000


def log(msg):
    print(colored(msg, "blue"))


def create_franka_environment(gui):
    bul = FrankaEnv(gui=gui)
    bul.load_robot()
    return bul


def create_franka_hand_environment(gui):
    bul = FrankaHandEnv(gui=gui)
    bul.load_robot()
    return bul


def random_cabinet_eff(planner):
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

    start = SE3(environment.start)
    goal = SE3(environment.target)
    try:
        path = planner.plan(
            start=start, goal=goal, interpolate=PATH_LENGTH, max_runtime=2
        )
    except Exception as e:
        raise e
        print(e)
        return None, None
    if path is None:
        # print("Could not find plan")
        return None, None
    if planner.bullet.use_gui:
        planner.bullet.marionette(start)
        dist = (
            environment.start[:3, 3]
            - planner.bullet.link_frames["panda_grasptarget"][:3, 3]
        )
        time.sleep(5)
        log("MOVING MOVING MOVING")
        for q in path:
            planner.bullet.marionette(q)
            time.sleep(0.01)
        time.sleep(2)

    # There has to be a reasonable limit on what's feasible for these plans,
    # so we're limiting plans to be under PATH_LENGTH
    if len(path) > PATH_LENGTH:
        return None, None
    return obstacles, path


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

    start = planner.bullet.collision_free_ik(environment.start, retries=1000)
    if start is None:
        log("Could not find start configuration")
        return None, None
    multi_goal = [
        planner.bullet.collision_free_ik(environment.target, retries=100)
        for _ in range(20)
    ]
    multi_goal = [g for g in multi_goal if g is not None]
    if len(multi_goal) == 0:
        log("Could not find goal configurations")
        return None, None
    try:
        path = planner.plan(
            start=start, multi_goal=multi_goal, interpolate=PATH_LENGTH, max_runtime=2
        )
    except Exception as e:
        raise e
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
        time.sleep(5)
        log("MOVING MOVING MOVING")
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
    planner = Planner()
    planner.set_environment(create_franka_environment(gui=False))
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

                    print(
                        f"PID {os.getpid()}: Completed {count}/{PATHS_PER_FILE} in {time_elapsed}"
                    )
            planner.reset()
    return


def simple_block(planner):
    environment = SingleBlockScene()
    # Continually try to regenerate environment until a reasonable solution is found
    while True:
        if environment.gen():
            break

    start = planner.bullet.collision_free_ik(environment.start, retries=100)
    if start is None:
        print("Could not find start configuration")
        return None, None
    goal = planner.bullet.sample_nearby_ik(environment.target)
    if goal is None:
        print("Could not find goal configuration")
        return None, None

    straight_path = np.linspace(start, goal, num=20)
    blocking_config = straight_path[np.random.randint(7, 13)]
    planner.bullet.marionette(blocking_config)
    frames = planner.bullet.link_frames
    position = frames["panda_link6"][:3, 3]
    obstacles = [
        Cuboid.random(
            dimension_range=[[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]], quaternion=True
        )
    ]
    obstacles[0].center = position
    planner.create_scene(obstacles)

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
        time.sleep(5)
        for q in path:
            planner.bullet.marionette(q)
            time.sleep(0.01)
        time.sleep(2)

    # There has to be a reasonable limit on what's feasible for these plans,
    # so we're limiting plans to be under PATH_LENGTH
    if len(path) > PATH_LENGTH:
        return None, None
    return obstacles, path


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
            start=start, goal=goal, interpolate=PATH_LENGTH, max_runtime=2
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
            start=start, goal=goal, interpolate=PATH_LENGTH, max_runtime=2
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
            start=start, goal=goal, interpolate=PATH_LENGTH, max_runtime=2
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
    planner = Planner()
    planner.set_environment(create_franka_environment(gui=False))
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
    planner = Planner()
    planner.set_environment(create_franka_environment(gui=False))
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
    # main()

    # planner = FrankaHandPlanner()
    # planner.set_environment(create_franka_hand_environment(gui=True))
    # random_cabinet_eff(planner)

    planner = FrankaPlanner()
    planner.set_environment(create_franka_environment(gui=True))
    random_cabinet(planner)

    # simple_block(planner)
    planner.reset()
