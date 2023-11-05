import time

import numpy as np
from geometrout.primitive import Cuboid
from robofin.bullet import Bullet
from robofin.collision import FrankaCollisionSpheres
from robofin.robots import FrankaRealRobot, FrankaRobot

from atob.planners.arm.aitstar import FrankaAITStar


def gen_valid_config(prismatic_joint, cooo, primitives):
    while True:
        q = FrankaRealRobot.random_configuration()
        if cooo.has_self_collision(q, prismatic_joint):
            continue
        cspheres = cooo.csphere_info(q, prismatic_joint)
        collides = False
        for p in primitives:
            if np.any(p.sdf(cspheres.centers) < cspheres.radii):
                collides = True
                break
        if not collides:
            break
    return q


def example1():
    planner = FrankaAITStar(0.025)

    # You can turn the gui off to make this faster/disable visuals
    cooo = FrankaCollisionSpheres()
    primitives = [
        Cuboid(
            center=np.array([0.5, 0, 0.3]),
            dims=np.array([0.2, 0.2, 0.5]),
            quaternion=np.array([1, 0, 0.0, 0]),
        )
    ]
    start = gen_valid_config(0.025, cooo, primitives)
    goal = gen_valid_config(0.025, cooo, primitives)
    # start = np.array(
    #     [
    #         -0.53556,
    #         -0.13557598,
    #         2.06197249,
    #         -2.07799268,
    #         0.25589343,
    #         1.91628114,
    #         1.45641799,
    #     ]
    # )
    # goal = np.array(
    #     [
    #         -0.89444609,
    #         0.0053469,
    #         -1.21969162,
    #         -2.35934311,
    #         0.56881887,
    #         1.73707844,
    #         2.30240395,
    #     ]
    # )

    planner.load_scene(primitives)

    path = planner.plan(
        start=start, goal=goal, min_solution_time=3, max_runtime=3, verbose=True
    )
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)
    path = planner.smooth(path, timesteps=50)
    print(type(path[0]))

    # This just visualizes the final path
    sim = Bullet(gui=True)
    sim.load_primitives(primitives)
    franka_arm = sim.load_robot(FrankaRobot)
    franka_arm.marionette(start)
    time.sleep(1)
    for q in path:
        franka_arm.marionette(q)
        time.sleep(0.1)
    time.sleep(2)


if __name__ == "__main__":
    example1()
