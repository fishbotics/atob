import random
import time

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from robofin.kinematics.collision import franka_arm_collides_fast
from robofin.robot_constants import FrankaConstants, RealFrankaConstants

from atob.caelan_smoothing import smooth_cubic
from atob.errors import CollisionError, ConfigurationError
from atob.planners.base import Planner


def random_indices(N):
    # First sample two indices without replacement
    idx0 = random.randint(0, N - 1)
    # This is a little trick to just not pick the same index twice
    idx1 = random.randint(0, N - 2)
    if idx1 >= idx0:
        idx1 += 1
    # Reset the variable names to be in order
    idx0, idx1 = (idx0, idx1) if idx1 > idx0 else (idx1, idx0)
    return idx0, idx1


def path_as_python(path, dof):
    path_length = path.getStateCount()
    return [[path.getState(ii)[j] for j in range(dof)] for ii in range(path_length)]


def steer_to(start, end, threshold=0.1):
    """
    I don't like the name steer_to but other people use it so whatever
    """
    # Check which joint has the largest movement
    which_joint = np.argmax(np.abs(end - start))
    num_steps = int(np.ceil(np.abs(end[which_joint] - start[which_joint]) / 0.1))
    return np.linspace(start, end, num=num_steps)


class FrankaArmBase(Planner):
    def __init__(self, prismatic_joint, buffer, real=True):
        super().__init__()
        self.prismatic_joint = prismatic_joint
        self.buffer = buffer
        if real:
            self.robot_constants = RealFrankaConstants
        else:
            self.robot_constants = FrankaConstants

    def _not_in_collision(self, q):
        current_time = time.time()
        collision_free = not franka_arm_collides_fast(
            q, self.prismatic_joint, self.cooo, self.scene_obstacle_arrays, self.buffer
        )
        total_time = time.time() - current_time
        self.total_collision_checking_time += total_time
        self.collision_check_counts += 1
        return collision_free

    def check_within_range(self, q):
        for ii in range(self.robot_constants.DOF):
            low, high = self.robot_constants.JOINT_LIMITS[ii]
            if q[ii] < low or q[ii] > high:
                return False
        return True

    def setup_problem(self, start, goal):
        self.last_problem_stats = {}
        if not self._loaded_environment:
            raise Exception("Scene not set up yet. Load scene before planning")
        # Verify start state is valid
        if not self.check_within_range(start):
            raise ConfigurationError(start, self.robot_constants.JOINT_LIMITS)
        if not self._not_in_collision(start):
            raise CollisionError(start)

        # Verify goal state is valid
        if not self.check_within_range(goal):
            raise ConfigurationError(goal, self.robot_constants.JOINT_LIMITS)
        if not self._not_in_collision(goal):
            raise CollisionError(goal)

        # Define the state space
        space = ob.RealVectorStateSpace(self.robot_constants.DOF)

        # Set the boundaries on the state space via the joint limits
        bounds = ob.RealVectorBounds(self.robot_constants.DOF)
        for ii in range(self.robot_constants.DOF):
            low, high = self.robot_constants.JOINT_LIMITS[ii]
            # TODO don't commit this change without thinking it through
            # ideally starts and goals should never be at the joint limits
            bounds.setLow(ii, low)
            bounds.setHigh(ii, high)
        space.setBounds(bounds)

        # Space information is an object that wraps the planning space itself, as well as
        # other properties about the space. Most notably, it has a reference to
        # a collision checker able to verify a specific configuration
        space_information = ob.SpaceInformation(space)

        # Sets the validity checker as a function. This can also be a class if there are
        # additional methods that need to be defined, for example to use in an optimizing object
        def collision_checker(q):
            state = np.zeros(7)
            for ii in range(7):
                state[ii] = q[ii]
            return self._not_in_collision(state)

        space_information.setStateValidityChecker(
            ob.StateValidityCheckerFn(collision_checker)
        )
        space_information.setup()

        # Define a planning problem on the planning space with the given collision checker
        pdef = ob.ProblemDefinition(space_information)

        # Copy the start and goal states into the OMPL representation
        start_state = ob.State(space)
        for ii in range(self.robot_constants.DOF):
            start_state[ii] = start[ii]
        pdef.addStartState(start_state)

        goal_state = ob.GoalState(space_information)
        gstate = ob.State(space)
        for ii in range(self.robot_constants.DOF):
            gstate[ii] = goal[ii]
        goal_state.setState(gstate)
        pdef.setGoal(goal_state)

        return space_information, pdef

    @staticmethod
    def check_solution(pdef, exact=True, verbose=False):
        if not pdef.hasSolution():
            if verbose:
                print("Could not find path")
            return None
        if exact and not pdef.hasExactSolution():
            if verbose:
                print("Could not find exact path")
            return None
        return pdef.getSolutionPath()

    def communicate_solve_info(self):
        elapsed_time = self.last_problem_stats["solve_time"]
        print(f"Planning time: {elapsed_time}")
        print(
            "Average collision check time:"
            f" {self.total_collision_checking_time / self.collision_check_counts}"
        )
        print(f"Total collision check time: {self.total_collision_checking_time}")

    def postprocess_path(
        self,
        path,
        space_information,
        shortcut_strategy,
        spline,
        interpolate,
        verbose=False,
    ):
        assert path is not None, "Cannot postprocess path that is None"
        for strategy in shortcut_strategy:
            assert strategy in ["ompl", "python"]
        simplifier = og.PathSimplifier(space_information)
        # TODO maybe can remove the OMPL post-processing now that separate
        # post-processing is implemented. This requires investigation
        if len(shortcut_strategy) > 0:
            self.last_problem_stats["shortcut_time"] = {}
        if "ompl" in shortcut_strategy:
            start_time = time.time()
            simplifier.shortcutPath(path)
            self.last_problem_stats["shortcut_time"]["ompl"] = time.time() - start_time
        if spline:
            start_time = time.time()
            simplifier.smoothBSpline(path)
            self.last_problem_stats["ompl_spline_time"] = time.time() - start_time
        if verbose:
            print(f"Smoothing time: {self.last_problem_stats['ompl_spline_time']}")
        if interpolate > 0:
            start_time = time.time()
            path.interpolate(interpolate)
            self.last_problem_stats["interpolation_time"] = time.time() - start_time
        path = path_as_python(path, self.robot_constants.DOF)
        if "python" in shortcut_strategy:
            start_time = time.time()
            path = self.shortcut(path, max_iterations=len(path))
            self.last_problem_stats["shortcut_time"]["python"] = (
                time.time() - start_time
            )
        return path

    def shortcut(self, path, max_iterations=50):
        path = np.asarray(path)
        indexed_path = list(zip(path, range(len(path))))
        checked_pairs = set()
        for _ in range(max_iterations):
            idx0, idx1 = random_indices(len(indexed_path))
            start, idx_start = indexed_path[idx0]
            end, idx_end = indexed_path[idx1]
            # Skip if this pair was already checked
            if (idx_start, idx_end) in checked_pairs:
                continue

            # The collision check resolution should never be smaller
            # than the original path was, so use the indices from the original
            # path to determine how many collision checks to do
            shortcut_path = steer_to(start, end)
            good_path = True
            # Check the shortcut
            for q in shortcut_path:
                if not (self.check_within_range(q) and self._not_in_collision(q)):
                    good_path = False
                    break
            if good_path:
                indexed_path = indexed_path[: idx0 + 1] + indexed_path[idx1:]

            # Add the checked pair into the record to avoid duplicates
            checked_pairs.add((idx_start, idx_end))

        # TODO move this into a test suite instead of a runtime check
        assert np.allclose(path[0], indexed_path[0][0])
        assert np.allclose(path[-1], indexed_path[-1][0])
        return [p[0] for p in indexed_path]

    def smooth(self, path, num_timesteps=None, fixed_timestep=None):
        assert (num_timesteps is None) != (
            fixed_timestep is None
        ), "Must either set num_timesteps or fixed_timestep"
        # TODO this code needs to be cleaned up
        curve = smooth_cubic(
            path,
            lambda q: not self._not_in_collision(q),
            np.radians(3) * np.ones(7),
            self.robot_constants.VELOCITY_LIMIT,
            self.robot_constants.ACCELERATION_LIMIT,
        )
        if fixed_timestep is None:
            assert num_timesteps is not None  # Needed for typechecking
            fixed_timestep = (curve.x[-1] - curve.x[0]) / (num_timesteps - 1)
        else:
            num_timesteps = int(
                np.ceil(1 + (curve.x[-1] - curve.x[0]) / fixed_timestep)
            )
        return [curve(fixed_timestep * i) for i in range(num_timesteps)]
