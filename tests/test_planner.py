import pytest
from geometrout.primitive import Cuboid
from robofin.robots import FrankaRobot
from robofin.bullet import Bullet
import numpy as np

import atob.planner as planners

# TODO figure out how to set the seed for the planner so
# these are actually reproducible.


class TestFrankaPlanner:
    start = [
        0.042770274971548616,
        -0.42466875881207017,
        0.4689401406579643,
        -2.2141292054250554,
        -2.3899940392083865,
        2.843157294776562,
        -2.8940403297973636,
    ]
    goal = [
        -1.957437616444934,
        1.7191462610744384,
        1.375075910529011,
        -2.317653510823181,
        0.05475706520922774,
        3.0029289414015645,
        -1.0845305881681326,
    ]

    obstacles = [
        Cuboid(
            center=[0.8359762962630319, -0.058462545396288146, 0.3107515412017145],
            dims=[0.02582174706269568, 1.4536968470885963, 0.621503082403429],
            quaternion=[1.0, 0.0, 0.0, 0.0],
        ),
        Cuboid(
            center=[0.652367357543221, -0.058462545396288146, 0.1343172896998709],
            dims=[0.367217877439622, 1.4536968470885963, 0.02582174706269568],
            quaternion=[1.0, 0.0, 0.0, 0.0],
        ),
        Cuboid(
            center=[0.652367357543221, -0.058462545396288146, 0.621503082403429],
            dims=[0.367217877439622, 1.4536968470885963, 0.02582174706269568],
            quaternion=[1.0, 0.0, 0.0, 0.0],
        ),
        Cuboid(
            center=[0.652367357543221, -0.7853109689405863, 0.37791018605164994],
            dims=[0.367217877439622, 0.02582174706269568, 0.5130075397662538],
            quaternion=[1.0, 0.0, 0.0, 0.0],
        ),
        Cuboid(
            center=[0.652367357543221, 0.66838587814801, 0.37791018605164994],
            dims=[0.367217877439622, 0.02582174706269568, 0.5130075397662538],
            quaternion=[1.0, 0.0, 0.0, 0.0],
        ),
        Cuboid(
            center=[0.652367357543221, 0.06633727338173523, 0.37791018605164994],
            dims=[0.367217877439622, 0.02582174706269568, 0.5130075397662538],
            quaternion=[1.0, 0.0, 0.0, 0.0],
        ),
        Cuboid(
            center=[0.652367357543221, -0.058462545396288146, 0.47094690999826516],
            dims=[0.367217877439622, 1.4536968470885963, 0.02582174706269568],
            quaternion=[1.0, 0.0, 0.0, 0.0],
        ),
    ]
    sim = Bullet(gui=False)
    sim.load_primitives(obstacles)
    arm = sim.load_robot(FrankaRobot)

    def test_rrt_exact(self):
        planner = planners.FrankaRRTPlanner()
        planner.load_simulation(self.sim, self.arm)
        plan = planner.plan(self.start, self.goal, max_runtime=10.0, exact=True)
        assert plan is not None
        assert np.allclose(self.start, plan[0], atol=0.01)
        assert np.allclose(self.goal, plan[-1], atol=0.01)

    def test_rrtconnect_exact(self):
        planner = planners.FrankaRRTConnectPlanner()
        planner.load_simulation(self.sim, self.arm)
        plan = planner.plan(self.start, self.goal, max_runtime=10.0, exact=True)
        assert plan is not None
        assert np.allclose(self.start, plan[0], atol=0.01)
        assert np.allclose(self.goal, plan[-1], atol=0.01)

    def test_aitstar_exact(self):
        planner = planners.FrankaAITStarPlanner()
        planner.load_simulation(self.sim, self.arm)
        plan = planner.plan(
            self.start, self.goal, max_runtime=10.0, min_solution_time=5.0, exact=True
        )
        assert plan is not None
        assert np.allclose(self.start, plan[0], atol=0.01)
        assert np.allclose(self.goal, plan[-1], atol=0.01)

    def test_abitstar_exact(self):
        planner = planners.FrankaABITStarPlanner()
        planner.load_simulation(self.sim, self.arm)
        plan = planner.plan(
            self.start, self.goal, max_runtime=10.0, min_solution_time=5.0, exact=True
        )
        assert plan is not None
        assert np.allclose(self.start, plan[0], atol=0.01)
        assert np.allclose(self.goal, plan[-1], atol=0.01)
