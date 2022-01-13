from atob.planner import FrankaRRTConnectPlanner
from geometrout.primitive import Cuboid
from robofin.robots import FrankaRobot
from robofin.bullet import Bullet
import time


def example1():
    start = [
        -0.53556,
        -0.13557598,
        2.06197249,
        -2.07799268,
        0.25589343,
        1.91628114,
        1.45641799,
    ]
    goal = [
        -0.89444609,
        0.0053469,
        -1.21969162,
        -2.35934311,
        0.56881887,
        1.73707844,
        2.30240395,
    ]
    planner = FrankaRRTConnectPlanner()

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
    planner.load_simulation(sim, franka_arm)
    path = planner.plan(start=start, goal=goal)

    # This just visualizes the final path
    franka_arm.marionette(start)
    time.sleep(1)
    for q in path:
        franka_arm.marionette(q)
        time.sleep(0.1)
    time.sleep(2)


if __name__ == "__main__":
    example1()
