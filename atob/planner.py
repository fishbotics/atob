from ompl import base as ob
from ompl import geometric as og

from atob.bullet import Bullet

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

class Planner:
    def __init__(self, urdf_path):
        self.bullet = Bullet(gui=False)
        self.bullet.load_robot(urdf_path)

    def reset(self):
        self.bullet.clear_all_obstacles()

    def create_scene(self, cuboids):
        self.bullet.load_cuboids(cuboids)

    def not_in_collision(self, q):
        self.bullet.marionette(q)
        return not self.bullet.check_collisions()

    def within_range(self, q):
        # TODO Fix this to apply to multiple robots
        for ii in range(FrankaRobot.DOF):
            low, high = FrankaRobot.JOINT_LIMITS[ii]
            if q[ii] < low or q[ii] > high:
                return False
        return True


    def plan(self, start, goal):
        # TODO This currently only works for the Franka because I was too lazy
        # to set it up for other robots
        if not self.withinRange(start):
            raise
        if not self.withinRange(goal):
            raise
        if self.inCollision(start):
            raise
        if self.inCollision(start):

        space = ob.RealVectorStateSpace(FrankaRobot.DOF)
        bounds = ob.RealVectorBounds(FrankaRobot.DOF)

        for ii in range(FrankaRobot.DOF):
            low, high = FrankaRobot.JOINT_LIMITS[ii]
            bounds.setLow(ii, low)
            bounds.setHigh(ii, high)
        space.setBounds(bounds)


        si = ob.SpaceInformation(space)
        checker = ValidityChecker(self.inCollision)
        si.setStateValidityChecker(checker)
        si.setup()


        start_state = ob.State(space)
        goal_state = ob.State(space)

        for ii in range(FrankaRobot.DOF):
            start_state[ii] = start[ii]
            goal_state[ii] = goal[ii]

        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start, goal)
        pdef.setOptimizationObjective(allocateObjective(si, objectiveType))
        optimizingPlanner = allocatePlanner(si, plannerType)
        optimizingPlanner.setProblemDefinition(pdef)
        optimizingPlanner.setup()


        si.setStartAndGoalStates(start, goal)
        solved = ss.solve(1.0)
        if solved:
            # try to shorten the path
            ss.simplifySolution()
            # print the simplified path
            return ss.getSolutionPath()
        return False

