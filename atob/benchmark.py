from atob.planner import Planner
from atob.geometry import Cuboid
import time

def test1():
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
    planner = Planner(gui=True)
    obstacle = Cuboid(
        [0.5, 0, 0.3], 
        [0.2, 0.2, 0.5], 
        [1, 0, 0, 0],
    )
    planner.create_scene([obstacle])
    path = planner.plan(start=start, goal=goal, interpolate=40)
    planner.bullet.marionette(start)
    time.sleep(3)
    for q in path: 
        planner.bullet.marionette(q)
        time.sleep(0.1)
    time.sleep(2)

if __name__ == '__main__':
    test1()

