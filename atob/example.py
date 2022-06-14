from atob.planner import FrankaAITStarPlanner
from geometrout.primitive import Cuboid
from robofin.robots import FrankaRobot, FrankaRealRobot
from robofin.bullet import Bullet
from robofin.collision import FrankaSelfCollisionChecker
import time
import numpy as np


def gen_valid_config(sim, robot, selfcc):
    while True:
        q = FrankaRealRobot.random_configuration()
        robot.marionette(q)
        if not sim.in_collision(
            robot, check_self=True
        ) and not selfcc.has_self_collision(q):
            break
    return q


def example1():
    # start = [
    #     -0.53556,
    #     -0.13557598,
    #     2.06197249,
    #     -2.07799268,
    #     0.25589343,
    #     1.91628114,
    #     1.45641799,
    # ]
    # goal = [
    #     -0.89444609,
    #     0.0053469,
    #     -1.21969162,
    #     -2.35934311,
    #     0.56881887,
    #     1.73707844,
    #     2.30240395,
    # ]
    planner = FrankaAITStarPlanner()

    # You can turn the gui off to make this faster/disable visuals
    sim = Bullet(gui=True)
    sim.load_primitives(
        [
            Cuboid(
                center=[0.5, 0, 0.3],
                dims=[0.2, 0.2, 0.5],
                quaternion=[1, 0, 0, 0],
            )
        ]
    )
    franka_arm = sim.load_robot(FrankaRobot)
    selfcc = FrankaSelfCollisionChecker()
    start = gen_valid_config(sim, franka_arm, selfcc)
    goal = gen_valid_config(sim, franka_arm, selfcc)

    planner.load_self_collision_checker(selfcc)
    planner.load_simulation(sim, franka_arm)
    path = planner.plan(
        start=start, goal=goal, min_solution_time=15, max_runtime=20, verbose=True
    )
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)
    path = planner.smooth(path, timesteps=50)
    print(type(path[0]))

    # This just visualizes the final path
    franka_arm.marionette(start)
    time.sleep(1)
    for q in path:
        franka_arm.marionette(q)
        time.sleep(0.1)
    time.sleep(2)


if __name__ == "__main__":
    example1()
