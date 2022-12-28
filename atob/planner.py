from ompl import base as ob
from ompl import geometric as og
import logging

from robofin.robots import FrankaRobot, FrankaRealRobot
from atob.errors import ConfigurationError, CollisionError
from atob.caelan_smoothing import smooth_cubic

from geometrout.transform import SE3
import time
import numpy as np
import random

from pyquaternion import Quaternion
from scipy.interpolate import CubicHermiteSpline


def steer_to(start, end, sim, robot, threshold=0.1):
    """
    I don't like the name steer_to but other people use it so whatever
    """
    # Check which joint has the largest movement
    which_joint = np.argmax(np.abs(end - start))
    num_steps = int(np.ceil(np.abs(end[which_joint] - start[which_joint]) / 0.1))
    return np.linspace(start, end, num=num_steps)


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
    pypath = []
    path_length = path.getStateCount()
    return [[path.getState(ii)[j] for j in range(dof)] for ii in range(path_length)]


def pose_path_as_python(path):
    pypath = []
    path_length = path.getStateCount()
    for ii in range(path_length):
        pose = path.getState(ii)
        rot = pose.rotation()
        quat = Quaternion([rot.w, rot.x, rot.y, rot.z])
        xyz = np.array([pose.getX(), pose.getY(), pose.getZ()])
        pypath.append(SE3(xyz=xyz, quaternion=quat))
    return pypath


class Planner:
    def __init__(self):
        self._loaded_environment = False
        self.total_collision_checking_time = 0
        self.collision_check_counts = 0
        self.collision_radius = 0.0

    def reset(self):
        self.sim.clear_all_obstacles()
        self._loaded_environment = False
        self.collision_check_counts = 0

    def load_simulation(self, sim, sim_robot):
        self.sim = sim
        self.sim_robot = sim_robot
        self._loaded_environment = True

    def _not_in_collision(self, q):
        current_time = time.time()
        self.sim_robot.marionette(q)
        ret = not self.sim.in_collision(
            self.sim_robot, self.collision_radius, check_self=True
        )
        total_time = time.time() - current_time
        self.total_collision_checking_time += total_time
        self.collision_check_counts += 1
        return ret

    def plan(
        self,
        start,
        goal=None,
        multi_goal=None,
        max_runtime=1.0,
        interpolate=0,
        verbose=False,
    ):
        raise NotImplementedError("Plan must be implemented in child classes")


class FrankaHandPlanner(Planner):
    def _not_in_collision(self, q, frame):
        current_time = time.time()
        self.sim_robot.marionette(q, frame)
        ret = not self.sim.in_collision(self.sim_robot)
        total_time = time.time() - current_time
        self.total_collision_checking_time += total_time
        self.collision_check_counts += 1
        return ret

    def plan(
        self,
        start,
        goal,
        search_bounds=[[-1, -1, -1], [1, 1, 1]],
        frame="right_gripper",
        max_runtime=1.0,
        interpolate=0,
        verbose=False,
    ):
        # TODO This currently only works for the Franka because I was too lazy
        # to set it up for other robots

        if not self._loaded_environment:
            raise Exception("Scene not set up yet. Load scene before planning")

        # Verify start state is valid
        # For hand, we don't have a sense for joint ranges, so just check collisions
        if not self._not_in_collision(start, frame):
            raise CollisionError(start)

        # Verify goal state is valid
        if not self._not_in_collision(goal, frame):
            raise CollisionError(goal)

        # Define the state space, which is SE3 because it's just the end effector
        space = ob.SE3StateSpace()

        # Set bounds for R^3 part of the state space
        bounds = ob.RealVectorBounds(3)
        for ii in range(3):
            bounds.setLow(ii, search_bounds[0][ii])
            bounds.setHigh(ii, search_bounds[1][ii])
        space.setBounds(bounds)

        # Space information is an object that wraps the planning space itself, as well as
        # other properties about the space. Most notably, it has a reference to
        # a collision checker able to verify a specific configuration
        space_information = ob.SpaceInformation(space)

        def check_collision(q):
            rot = q.rotation()
            quat = Quaternion([rot.w, rot.x, rot.y, rot.z])
            xyz = np.array([q.getX(), q.getY(), q.getZ()])
            return self._not_in_collision(SE3(xyz=xyz, quaternion=quat), frame)

        # Sets the validity checker as a function. This can also be a class if there are
        # additional methods that need to be defined, for example to use in an optimizing object
        space_information.setStateValidityChecker(
            ob.StateValidityCheckerFn(check_collision)
        )
        space_information.setup()

        # Define a planning problem on the planning space with the given collision checker
        pdef = ob.ProblemDefinition(space_information)

        # Copy the start and goal states into the OMPL representation
        start_state = ob.State(space)
        mutable_start_state = start_state.get()
        x, y, z = start.xyz
        mutable_start_state.setX(x)
        mutable_start_state.setY(y)
        mutable_start_state.setZ(z)
        mutable_start_rotation = mutable_start_state.rotation()
        (
            mutable_start_rotation.x,
            mutable_start_rotation.y,
            mutable_start_rotation.z,
            mutable_start_rotation.w,
        ) = start.so3.xyzw
        pdef.addStartState(start_state)

        goal_state = ob.GoalState(space_information)
        gstate = ob.State(space)
        mutable_gstate = gstate.get()
        x, y, z = goal.xyz
        mutable_gstate.setX(x)
        mutable_gstate.setY(y)
        mutable_gstate.setZ(z)
        mutable_goal_rotation = mutable_gstate.rotation()
        (
            mutable_goal_rotation.x,
            mutable_goal_rotation.y,
            mutable_goal_rotation.z,
            mutable_goal_rotation.w,
        ) = goal.so3.xyzw
        goal_state.setState(gstate)
        pdef.setGoal(goal_state)

        # The planning problem needs to know what it's optimizing for
        pdef.setOptimizationObjective(
            ob.PathLengthOptimizationObjective(space_information)
        )

        # Set up the actual planner and give it the problem
        optimizing_planner = og.ABITstar(space_information)
        optimizing_planner.setUseKNearest(False)

        optimizing_planner.setProblemDefinition(pdef)
        optimizing_planner.setup()

        start_time = time.time()
        # termination_condition = ob.CostConvergenceTerminationCondition(pdef)
        # solved = optimizing_planner.solve(termination_condition)
        solved = optimizing_planner.solve(max_runtime)
        if verbose:
            print(f"Planning time: {time.time() - start_time}")
            print(
                f"Average collision check time: {self.total_collision_checking_time / self.collision_check_counts}"
            )
            print(f"Total collision check time: {self.total_collision_checking_time}")
        if not solved:
            return None
        path = pdef.getSolutionPath()

        simplifier = og.PathSimplifier(space_information)
        start_time = time.time()
        simplifier.shortcutPath(path)
        simplifier.smoothBSpline(path)
        if verbose:
            print(f"Smoothing time: {time.time() - start_time}")

        if interpolate > 0:
            path.interpolate(interpolate)
        path = pose_path_as_python(path)
        return path


class FrankaAITStarHandPlanner(FrankaHandPlanner):
    def plan(
        self,
        start,
        goal,
        search_bounds=[[-1, -1, -1], [1, 1, 1]],
        frame="right_gripper",
        max_runtime=1.0,
        interpolate=0,
        verbose=False,
    ):
        # TODO This currently only works for the Franka because I was too lazy
        # to set it up for other robots

        if not self._loaded_environment:
            raise Exception("Scene not set up yet. Load scene before planning")

        # Verify start state is valid
        # For hand, we don't have a sense for joint ranges, so just check collisions
        if not self._not_in_collision(start, frame):
            raise CollisionError(start)

        # Verify goal state is valid
        if not self._not_in_collision(goal, frame):
            raise CollisionError(goal)

        # Define the state space, which is SE3 because it's just the end effector
        space = ob.SE3StateSpace()

        # Set bounds for R^3 part of the state space
        bounds = ob.RealVectorBounds(3)
        for ii in range(3):
            bounds.setLow(ii, search_bounds[0][ii])
            bounds.setHigh(ii, search_bounds[1][ii])
        space.setBounds(bounds)

        # Space information is an object that wraps the planning space itself, as well as
        # other properties about the space. Most notably, it has a reference to
        # a collision checker able to verify a specific configuration
        space_information = ob.SpaceInformation(space)

        def check_collision(q):
            rot = q.rotation()
            quat = Quaternion([rot.w, rot.x, rot.y, rot.z])
            xyz = np.array([q.getX(), q.getY(), q.getZ()])
            return self._not_in_collision(SE3(xyz=xyz, quaternion=quat), frame)

        # Sets the validity checker as a function. This can also be a class if there are
        # additional methods that need to be defined, for example to use in an optimizing object
        space_information.setStateValidityChecker(
            ob.StateValidityCheckerFn(check_collision)
        )
        space_information.setup()

        # Define a planning problem on the planning space with the given collision checker
        pdef = ob.ProblemDefinition(space_information)

        # Copy the start and goal states into the OMPL representation
        start_state = ob.State(space)
        mutable_start_state = start_state.get()
        x, y, z = start.xyz
        mutable_start_state.setX(x)
        mutable_start_state.setY(y)
        mutable_start_state.setZ(z)
        mutable_start_rotation = mutable_start_state.rotation()
        (
            mutable_start_rotation.x,
            mutable_start_rotation.y,
            mutable_start_rotation.z,
            mutable_start_rotation.w,
        ) = start.so3.xyzw
        pdef.addStartState(start_state)

        goal_state = ob.GoalState(space_information)
        gstate = ob.State(space)
        mutable_gstate = gstate.get()
        x, y, z = goal.xyz
        mutable_gstate.setX(x)
        mutable_gstate.setY(y)
        mutable_gstate.setZ(z)
        mutable_goal_rotation = mutable_gstate.rotation()
        (
            mutable_goal_rotation.x,
            mutable_goal_rotation.y,
            mutable_goal_rotation.z,
            mutable_goal_rotation.w,
        ) = goal.so3.xyzw
        goal_state.setState(gstate)
        pdef.setGoal(goal_state)

        # The planning problem needs to know what it's optimizing for
        pdef.setOptimizationObjective(
            ob.PathLengthOptimizationObjective(space_information)
        )

        # Set up the actual planner and give it the problem
        optimizing_planner = og.AITstar(space_information)

        optimizing_planner.setProblemDefinition(pdef)
        optimizing_planner.setup()

        start_time = time.time()
        # termination_condition = ob.CostConvergenceTerminationCondition(pdef)
        # solved = optimizing_planner.solve(termination_condition)
        solved = optimizing_planner.solve(max_runtime)
        if verbose:
            print(f"Planning time: {time.time() - start_time}")
            print(
                f"Average collision check time: {self.total_collision_checking_time / self.collision_check_counts}"
            )
            print(f"Total collision check time: {self.total_collision_checking_time}")
        if not solved:
            return None
        path = pdef.getSolutionPath()

        simplifier = og.PathSimplifier(space_information)
        start_time = time.time()
        simplifier.shortcutPath(path)
        simplifier.smoothBSpline(path)
        if verbose:
            print(f"Smoothing time: {time.time() - start_time}")

        if interpolate > 0:
            path.interpolate(interpolate)
        path = pose_path_as_python(path)
        return path


class FrankaArmPlanner(Planner):
    def __init__(self, real=True, sphere_self_collision_checker=True):
        self._loaded_environment = False
        self.total_collision_checking_time = 0
        self.collision_check_counts = 0
        self.sphere_self_collision_checker = sphere_self_collision_checker
        if real:
            self.robot_type = FrankaRealRobot
        else:
            self.robot_type = FrankaRobot

    def load_self_collision_checker(self, checker):
        self.self_collision_checker = checker

    def _not_in_collision(self, q):
        current_time = time.time()
        self.sim_robot.marionette(q)
        if self.sphere_self_collision_checker:
            ret = not (
                self.sim.in_collision(self.sim_robot, check_self=True)
                or self.self_collision_checker.has_self_collision(q)
            )
        else:
            ret = not self.sim.in_collision(self.sim_robot, check_self=True)
        total_time = time.time() - current_time
        self.total_collision_checking_time += total_time
        self.collision_check_counts += 1
        return ret

    def check_within_range(self, q):
        for ii in range(self.robot_type.DOF):
            low, high = self.robot_type.JOINT_LIMITS[ii]
            if q[ii] < low or q[ii] > high:
                return False
        return True

    def setup_problem(self, start, goal):
        self.last_problem_stats = {}
        if not self._loaded_environment:
            raise Exception("Scene not set up yet. Load scene before planning")
        # Verify start state is valid
        if not self.check_within_range(start):
            raise ConfigurationError(start, self.robot_type.JOINT_LIMITS)
        if not self._not_in_collision(start):
            raise CollisionError(start)

        # Verify goal state is valid
        if not self.check_within_range(goal):
            raise ConfigurationError(goal, self.robot_type.JOINT_LIMITS)
        if not self._not_in_collision(goal):
            raise CollisionError(goal)

        # Define the state space
        space = ob.RealVectorStateSpace(self.robot_type.DOF)

        # Set the boundaries on the state space via the joint limits
        bounds = ob.RealVectorBounds(self.robot_type.DOF)
        for ii in range(self.robot_type.DOF):
            low, high = self.robot_type.JOINT_LIMITS[ii]
            # TODO don't commit this change without thinking it through--ideally starts and goals should never be at the joint limits
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
        for ii in range(self.robot_type.DOF):
            start_state[ii] = start[ii]
        pdef.addStartState(start_state)

        goal_state = ob.GoalState(space_information)
        gstate = ob.State(space)
        for ii in range(self.robot_type.DOF):
            gstate[ii] = goal[ii]
        goal_state.setState(gstate)
        pdef.setGoal(goal_state)

        return space_information, pdef

    def check_solution(self, pdef, exact=True, verbose=False):
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
            f"Average collision check time: {self.total_collision_checking_time / self.collision_check_counts}"
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
        path = path_as_python(path, self.robot_type.DOF)
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
            shortcut_path = steer_to(start, end, self.sim, self.sim_robot)
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

    def smooth(self, path, timesteps):
        # TODO this code needs to be cleaned up
        curve = smooth_cubic(
            path,
            lambda q: not self._not_in_collision(q),
            np.radians(3) * np.ones(7),
            self.robot_type.VELOCITY_LIMIT,
            self.robot_type.ACCELERATION_LIMIT,
        )
        ts = (curve.x[-1] - curve.x[0]) / (timesteps - 1)
        return [curve(ts * i) for i in range(timesteps)]


class FrankaRRTStarPlanner(FrankaArmPlanner):
    def plan(
        self,
        start,
        goal,
        max_runtime=1.0,
        min_solution_time=1.0,
        exact=True,
        interpolate=0,
        shortcut_strategy=["python"],
        spline=True,
        verbose=False,
    ):
        space_information, pdef = self.setup_problem(start, goal)

        # TODO This currently only works for the Franka because I was too lazy
        # to set it up for other robots

        # The planning problem needs to know what it's optimizing for
        pdef.setOptimizationObjective(
            ob.PathLengthOptimizationObjective(space_information)
        )

        # Set up the actual planner and give it the problem
        optimizing_planner = og.RRTstar(space_information)

        optimizing_planner.setProblemDefinition(pdef)
        optimizing_planner.setup()

        start_time = time.time()
        # TODO Fix this after getting a response on https://github.com/ompl/ompl/issues/866
        if exact:
            solved = optimizing_planner.solve(
                ob.plannerOrTerminationCondition(
                    ob.plannerAndTerminationCondition(
                        ob.timedPlannerTerminationCondition(min_solution_time),
                        ob.exactSolnPlannerTerminationCondition(pdef),
                    ),
                    ob.timedPlannerTerminationCondition(max_runtime),
                )
            )
        else:
            optimizing_planner.solve(max_runtime)
        self.last_problem_stats["solve_time"] = time.time() - start_time
        if verbose:
            self.communicate_solve_info()
        path = self.check_solution(pdef, exact, verbose)
        if path is None:
            return None
        return self.postprocess_path(
            path,
            space_information,
            shortcut_strategy,
            spline=spline,
            interpolate=0,
            verbose=verbose,
        )


class FrankaAITStarPlanner(FrankaArmPlanner):
    def plan(
        self,
        start,
        goal,
        max_runtime=1.0,
        min_solution_time=1.0,
        exact=False,
        interpolate=0,
        shortcut_strategy=["python"],
        spline=True,
        verbose=False,
    ):
        space_information, pdef = self.setup_problem(start, goal)

        # TODO This currently only works for the Franka because I was too lazy
        # to set it up for other robots

        # The planning problem needs to know what it's optimizing for
        pdef.setOptimizationObjective(
            ob.PathLengthOptimizationObjective(space_information)
        )

        # Set up the actual planner and give it the problem
        optimizing_planner = og.AITstar(space_information)

        optimizing_planner.setProblemDefinition(pdef)
        optimizing_planner.setup()

        start_time = time.time()
        # TODO Fix this after getting a response on https://github.com/ompl/ompl/issues/866
        if exact:
            solved = optimizing_planner.solve(
                ob.plannerOrTerminationCondition(
                    ob.plannerOrTerminationCondition(
                        ob.plannerAndTerminationCondition(
                            ob.timedPlannerTerminationCondition(min_solution_time),
                            ob.exactSolnPlannerTerminationCondition(pdef),
                        ),
                        ob.timedPlannerTerminationCondition(max_runtime),
                    ),
                    ob.CostConvergenceTerminationCondition(pdef),
                )
            )
        else:
            optimizing_planner.solve(max_runtime)
        self.last_problem_stats["solve_time"] = time.time() - start_time
        if verbose:
            self.communicate_solve_info()
        path = self.check_solution(pdef, exact, verbose)
        if path is None:
            return None
        return self.postprocess_path(
            path,
            space_information,
            shortcut_strategy,
            spline=spline,
            interpolate=0,
            verbose=verbose,
        )


class FrankaABITStarPlanner(FrankaArmPlanner):
    def plan(
        self,
        start,
        goal,
        max_runtime=1.0,
        min_solution_time=1.0,
        exact=False,
        interpolate=0,
        shortcut_strategy=["python"],
        spline=True,
        verbose=False,
    ):
        space_information, pdef = self.setup_problem(start, goal)

        # The planning problem needs to know what it's optimizing for
        pdef.setOptimizationObjective(
            ob.PathLengthOptimizationObjective(space_information)
        )

        # Set up the actual planner and give it the problem
        optimizing_planner = og.ABITstar(space_information)
        optimizing_planner.setUseKNearest(False)

        optimizing_planner.setProblemDefinition(pdef)
        optimizing_planner.setup()

        start_time = time.time()
        if exact:
            solved = optimizing_planner.solve(
                ob.plannerOrTerminationCondition(
                    ob.plannerAndTerminationCondition(
                        ob.timedPlannerTerminationCondition(min_solution_time),
                        ob.exactSolnPlannerTerminationCondition(pdef),
                    ),
                    ob.timedPlannerTerminationCondition(max_runtime),
                )
            )
        else:
            optimizing_planner.solve(max_runtime)
        self.last_problem_stats["solve_time"] = time.time() - start_time
        if verbose:
            self.communicate_solve_info()
        path = self.check_solution(pdef, exact, verbose)
        if path is None:
            return None
        return self.postprocess_path(
            path,
            space_information,
            shortcut_strategy,
            spline=spline,
            interpolate=0,
            verbose=verbose,
        )


class FrankaRRTConnectPlanner(FrankaArmPlanner):
    def plan(
        self,
        start,
        goal,
        max_runtime=1.0,
        exact=True,
        interpolate=0,
        shortcut_strategy=["python"],
        spline=True,
        verbose=False,
    ):
        space_information, pdef = self.setup_problem(start, goal)

        # The planning problem needs to know what it's optimizing for
        # pdef.setOptimizationObjective(
        #     ob.PathLengthOptimizationObjective(space_information)
        # )

        # Set up the actual planner and give it the problem
        planner = og.RRTConnect(space_information)

        planner.setProblemDefinition(pdef)
        planner.setup()

        start_time = time.time()
        planner.solve(max_runtime)

        self.last_problem_stats["solve_time"] = time.time() - start_time
        if verbose:
            self.communicate_solve_info()
        path = self.check_solution(pdef, exact, verbose)
        if path is None:
            return None
        return self.postprocess_path(
            path,
            space_information,
            shortcut_strategy,
            spline=spline,
            interpolate=0,
            verbose=verbose,
        )


class FrankaRRTPlanner(FrankaArmPlanner):
    def plan(
        self,
        start,
        goal,
        max_runtime=1.0,
        exact=False,
        interpolate=0,
        shortcut_strategy=["python"],
        spline=True,
        verbose=False,
    ):
        space_information, pdef = self.setup_problem(start, goal)

        # The planning problem needs to know what it's optimizing for
        pdef.setOptimizationObjective(
            ob.PathLengthOptimizationObjective(space_information)
        )

        # Set up the actual planner and give it the problem
        planner = og.RRT(space_information)

        planner.setProblemDefinition(pdef)
        planner.setup()

        start_time = time.time()
        planner.solve(max_runtime)
        self.last_problem_stats["solve_time"] = time.time() - start_time
        if verbose:
            self.communicate_solve_info()
        path = self.check_solution(pdef, exact, verbose)
        if path is None:
            return None
        return self.postprocess_path(
            path,
            space_information,
            shortcut_strategy,
            spline=spline,
            interpolate=0,
            verbose=verbose,
        )
