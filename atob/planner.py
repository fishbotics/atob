from ompl import base as ob
from ompl import geometric as og

from atob.bullet import Bullet
from atob.errors import ConfigurationError, CollisionError


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


def path_as_python(path, dof):
    pypath = []
    path_length = path.getStateCount()
    return [[path.getState(ii)[j] for j in range(dof)] for ii in range(path_length)]


class Planner:
    def __init__(self, urdf_path):
        self.bullet = Bullet(gui=False)
        self.bullet.load_robot(urdf_path)
        self._scene_created = False

    def reset(self):
        self.bullet.clear_all_obstacles()
        self._scene_created = False

    def create_scene(self, cuboids):
        self.bullet.load_cuboids(cuboids)
        self._scene_created = True

    def _not_in_collision(self, q):
        self.bullet.marionette(q)
        return not self.bullet.in_collision()

    def check_within_range(self, q):
        # TODO Fix this to apply to multiple robots
        for ii in range(FrankaRobot.DOF):
            low, high = FrankaRobot.JOINT_LIMITS[ii]
            if q[ii] < low or q[ii] > high:
                raise ConfigurationError(ii, q[ii], FrankaRobot.JOINT_LIMITS[ii])
        return

    def plan(self, start, goal, max_runtime=5.0, interpolate=False):
        # TODO This currently only works for the Franka because I was too lazy
        # to set it up for other robots

        if not self._scene_created:
            raise Exception("Scene not set up yet. Load scene before planning")
        # Verify start state is valid
        self.check_within_range(start)
        if not self._not_in_collision(start):
            raise CollisionError(start)

        # Verify goal state is valid
        self.check_within_range(goal)
        if not self._not_in_collision(goal):
            raise CollisionError(goal)

        # Define the state space
        space = ob.RealVectorStateSpace(FrankaRobot.DOF)

        # Set the boundaries on the state space via the joint limits
        bounds = ob.RealVectorBounds(FrankaRobot.DOF)
        for ii in range(FrankaRobot.DOF):
            low, high = FrankaRobot.JOINT_LIMITS[ii]
            bounds.setLow(ii, low)
            bounds.setHigh(ii, high)
        space.setBounds(bounds)

        # Space information is an object that wraps the planning space itself, as well as
        # other properties about the space. Most notably, it has a reference to
        # a collision checker able to verify a specific configuration
        space_information = ob.SpaceInformation(space)

        # Sets the validity checker as a function. This can also be a class if there are
        # additional methods that need to be defined, for example to use in an optimizing object
        space_information.setStateValidityChecker(
            ob.StateValidityCheckerFn(self._not_in_collision)
        )
        space_information.setup()

        # Copy the start and goal states into the OMPL representation
        start_state = ob.State(space)
        goal_state = ob.State(space)
        for ii in range(FrankaRobot.DOF):
            start_state[ii] = start[ii]
            goal_state[ii] = goal[ii]

        # Define a planning problem on the planning space with the given collision checker
        pdef = ob.ProblemDefinition(space_information)
        pdef.setStartAndGoalStates(start_state, goal_state)

        # The planning problem needs to know what it's optimizing for
        pdef.setOptimizationObjective(
            ob.PathLengthOptimizationObjective(space_information)
        )

        # Set up the actual planner and give it the problem
        optimizing_planner = og.BITstar(space_information)
        optimizing_planner.setProblemDefinition(pdef)
        optimizing_planner.setup()

        solved = optimizing_planner.solve(max_runtime)
        if solved:
            if interpolate:
                path.interpolate()
            return path_as_python(pdef.getSolutionPath())
        return False
