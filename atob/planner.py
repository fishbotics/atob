from ompl import base as ob
from ompl import geometric as og
import logging

from robofin.robots import FrankaRobot
from atob.errors import ConfigurationError, CollisionError

from geometrout.transform import SE3
import time
import numpy as np
import random

from pyquaternion import Quaternion
from scipy.interpolate import CubicHermiteSpline


def quickest_two_ramp(qi_n, qj_n, vi_n, vj_n, vmax_n, amax_n):
    polynomial = np.polynomial.Polynomial(
        [(vi_n ** 2 - vj_n ** 2) / (2 * amax_n) + (qi_n - qj_n), 2 * vi_n, amax_n]
    )
    max_value = abs(vmax_n) / abs(amax_n)
    roots = polynomial.roots()
    roots = roots[np.isreal(roots)]
    roots = roots[roots >= 0]
    roots = roots[roots <= abs(vmax_n) / abs(amax_n)]
    roots = roots[abs(vi_n + roots * amax_n) <= vmax_n + 1e-6]
    if len(roots) == 0:
        return None
    T = 2 * np.min(roots) + (vi_n - vj_n) / amax_n
    return T if T >= 0 else None


def quickest_three_stage(qi_n, qj_n, vi_n, vj_n, vmax_n, amax_n):
    tp1 = (vmax_n - vi_n) / amax_n
    tl = (vj_n ** 2 + vi_n ** 2 - 2 * vmax_n ** 2) / (2 * vmax_n * amax_n) + (
        qj_n - qi_n
    ) / vmax_n
    tp2 = (vj_n - vmax_n) / -amax_n  # Difference from Hauser
    if tp1 < 0 or tl < 0 or tp2 < 0:
        return None
    return tp1 + tl + tp2


def black_magic_duration_bound_1d(qi_n, qj_n, vi_n, vj_n, vmax_n, amax_n):
    candidates = [
        quickest_two_ramp(qi_n, qj_n, vi_n, vj_n, vmax_n, amax_n),
        quickest_two_ramp(qi_n, qj_n, vi_n, vj_n, -vmax_n, -amax_n),
        quickest_three_stage(qi_n, qj_n, vi_n, vj_n, vmax_n, amax_n),
        quickest_three_stage(qi_n, qj_n, vi_n, vj_n, -vmax_n, -amax_n),
    ]
    candidates = [t for t in candidates if t is not None]
    if not candidates:
        return None
    T = min(t for t in candidates)
    return max(1e-3, T)


def black_magic_duration_bound(qi, qj, vi, vj, vmax, amax):
    durations = [
        black_magic_duration_bound_1d(qi_n, qj_n, vi_n, vj_n, vmax_n, amax_n)
        for qi_n, qj_n, vi_n, vj_n, vmax_n, amax_n in zip(qi, qj, vi, vj, vmax, amax)
    ]
    if any(d is None for d in durations):
        return None
    return max(durations)


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
        pypath.append(SE3(xyz=xyz, quat=quat))
    return pypath


class Planner:
    def __init__(self):
        self._loaded_environment = False
        self.total_collision_checking_time = 0
        self.collision_check_counts = 0

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
        ret = not self.sim.in_collision(self.sim_robot, check_self=True)
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
        ret = not self.sim.in_collision(self.sim_robot, check_self=True)
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
            return self._not_in_collision(SE3(xyz=xyz, quat=quat), frame)

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
            return self._not_in_collision(SE3(xyz=xyz, quat=quat), frame)

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
    def check_within_range(self, q):
        for ii in range(FrankaRobot.DOF):
            low, high = FrankaRobot.JOINT_LIMITS[ii]
            if q[ii] < low or q[ii] > high:
                return False
        return True

    def setup_problem(self, start, goal):
        if not self._loaded_environment:
            raise Exception("Scene not set up yet. Load scene before planning")
        # Verify start state is valid
        if not self.check_within_range(start):
            raise ConfigurationError(start, FrankaRobot.JOINT_LIMITS)
        if not self._not_in_collision(start):
            raise CollisionError(start)

        # Verify goal state is valid
        if not self.check_within_range(goal):
            raise ConfigurationError(goal, FrankaRobot.JOINT_LIMITS)
        if not self._not_in_collision(goal):
            raise CollisionError(goal)

        # Define the state space
        space = ob.RealVectorStateSpace(FrankaRobot.DOF)

        # Set the boundaries on the state space via the joint limits
        bounds = ob.RealVectorBounds(FrankaRobot.DOF)
        for ii in range(FrankaRobot.DOF):
            low, high = FrankaRobot.JOINT_LIMITS[ii]
            bounds.setLow(ii, low + 1e-3)
            bounds.setHigh(ii, high - 1e-3)
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
        for ii in range(FrankaRobot.DOF):
            start_state[ii] = start[ii]
        pdef.addStartState(start_state)

        goal_state = ob.GoalState(space_information)
        gstate = ob.State(space)
        for ii in range(FrankaRobot.DOF):
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

    def communicate_solve_info(self, elapsed_time):
        print(f"Planning time: {elapsed_time}")
        print(
            f"Average collision check time: {self.total_collision_checking_time / self.collision_check_counts}"
        )
        print(f"Total collision check time: {self.total_collision_checking_time}")

    def postprocess_path(
        self, path, space_information, shortcut, spline, interpolate, verbose=False
    ):
        assert path is not None, "Cannot postprocess path that is None"
        simplifier = og.PathSimplifier(space_information)
        start_time = time.time()
        # TODO maybe can remove the OMPL post-processing now that separate
        # post-processing is implemented. This requires investigation
        if shortcut:
            simplifier.shortcutPath(path)
        if spline:
            simplifier.smoothBSpline(path)
        if verbose:
            print(f"Smoothing time: {time.time() - start_time}")
        if interpolate > 0:
            path.interpolate(interpolate)
        path = path_as_python(path, FrankaRobot.DOF)
        if shortcut:
            path = self.shortcut(path, max_iterations=len(path))
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

    def is_curve_in_collision(self, curve):
        start, end = curve.x[0], curve.x[-1]
        timestep = 1e-2
        for t in np.arange(start, end, timestep):
            q = curve(t)
            if not (self.check_within_range(q) and self._not_in_collision(q)):
                return True
        return False

    def smooth(self, path, timesteps, max_iterations=50):
        path = np.asarray(path)
        amax = FrankaRobot.ACCELERATION_LIMIT
        vmax = FrankaRobot.ACCELERATION_LIMIT
        if len(path) <= 2:
            logging.info("Cannot smooth path with length less than 3")
            # return path
        durations = np.zeros(len(path))
        for idx in range(len(path) - 1):
            qi, qj = path[idx], path[idx + 1]
            qh = path[0] if idx == 0 else path[idx - 1]
            qk = path[-1] if idx == len(path) - 2 else path[idx + 2]
            velocity = 0.5 * (qj - qh)
            acc1 = (qj - qi) - (qi - qh)
            acc2 = (qh - qj) - (qj - qi)
            v_duration = np.max(np.abs(velocity) / vmax)
            a1_duration = np.max(np.sqrt(np.abs(acc1) / amax)) if idx > 0 else 0.0
            a2_duration = (
                np.max(np.sqrt(np.abs(acc2) / amax)) if idx < len(path) - 2 else 0.0
            )
            max_duration = np.max([v_duration, a1_duration, a2_duration])
            durations[idx + 1] = max(durations[idx + 1], max_duration)
        timestamps = np.cumsum(durations)
        velocities = np.zeros_like(path)
        curve = CubicHermiteSpline(timestamps, path, dydx=velocities)
        for _ in range(max_iterations):
            timestamps = curve.x
            durations[1:] = np.array(
                [qj - qi for qi, qj in zip(timestamps[:-1], timestamps[1:])]
            )
            positions = np.array([curve(t) for t in timestamps])
            velocities = np.array([curve(t, nu=1) for t in timestamps])

            # Pick two random times (in order)
            t1, t2 = np.random.uniform(0, timestamps[-1], 2)
            t1, t2 = (t1, t2) if t1 <= t2 else (t2, t1)
            position_chunk = np.array([curve(t1), curve(t2)])
            velocity_chunk = np.array([curve(t1, nu=1), curve(t2, nu=1)])
            assert np.alltrue(velocity_chunk <= vmax + 1e-5)

            # TODO Re-read https://motion.cs.illinois.edu/papers/icra10-smoothing.pdf
            # to see if there is a cleaner implementation
            min_new_duration = black_magic_duration_bound(
                position_chunk[0],
                position_chunk[1],
                velocity_chunk[0],
                velocity_chunk[1],
                vmax,
                amax,
            )

            # This is an alternative strategy but leads to jerkier motion
            # v_duration = np.max(
            #     np.abs(position_chunk[1] - position_chunk[0]) / vmax
            # )
            # a_duration = np.max(
            #     np.sqrt(np.abs(velocity_chunk[1] - velocity_chunk[0]) / amax)
            # )

            # min_new_duration = np.max([v_duration, a_duration])
            if min_new_duration is None:
                logging.info(
                    "New duration invalid with smoothing selections--moving on"
                )
                continue

            # max_new_duration = min(max_new_duration, ramp_new_duration)
            new_duration = np.random.uniform(min_new_duration, t2 - t1)

            # The last index before t1
            idx1 = np.argmax(timestamps > t1) - 1
            # The first index after t2
            idx2 = np.argmin(timestamps < t2)
            assert idx2 > idx1
            timestamp_chunk = [t1, t1 + new_duration]
            curve_chunk = CubicHermiteSpline(
                timestamp_chunk, position_chunk, dydx=velocity_chunk
            )
            assert (
                curve.x[-1] - curve.x[0] > t2 - t1
            ), "Something is wrong with curve calculation"
            if self.is_curve_in_collision(curve_chunk):
                continue
            position_chunk = np.array([curve_chunk(x) for x in curve_chunk.x])
            velocity_chunk = np.array([curve_chunk(x, nu=1) for x in curve_chunk.x])
            duration_chunk = np.array(
                [t1 - timestamps[idx1]]
                + [x - curve_chunk.x[0] for x in curve_chunk.x[1:]]
                + [timestamps[idx2] - t2]
            )
            # duration chunk is longer than the other chunks
            durations = np.concatenate(
                (durations[: idx1 + 1], duration_chunk, durations[idx2 + 1 :])
            )
            times = np.cumsum(durations)
            positions = np.concatenate(
                (positions[: idx1 + 1], position_chunk, positions[idx2:])
            )
            velocities = np.concatenate(
                (velocities[: idx1 + 1], velocity_chunk, velocities[idx2:])
            )
            curve = CubicHermiteSpline(times, positions, dydx=velocities)
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
        shortcut=True,
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
        if verbose:
            self.communicate_solve_info(time.time() - start_time)
        path = self.check_solution(pdef, exact, verbose)
        if path is None:
            return None
        return self.postprocess_path(
            path,
            space_information,
            shortcut,
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
        shortcut=True,
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
        if verbose:
            self.communicate_solve_info(time.time() - start_time)
        path = self.check_solution(pdef, exact, verbose)
        if path is None:
            return None
        return self.postprocess_path(
            path,
            space_information,
            shortcut,
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
        shortcut=True,
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
        if verbose:
            self.communicate_solve_info(time.time() - start_time)
        path = self.check_solution(pdef, exact, verbose)
        if path is None:
            return None
        return self.postprocess_path(
            path,
            space_information,
            shortcut,
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
        shortcut=True,
        spline=True,
        verbose=False,
    ):
        space_information, pdef = self.setup_problem(start, goal)

        # The planning problem needs to know what it's optimizing for
        pdef.setOptimizationObjective(
            ob.PathLengthOptimizationObjective(space_information)
        )

        # Set up the actual planner and give it the problem
        planner = og.RRTConnect(space_information)

        planner.setProblemDefinition(pdef)
        planner.setup()

        start_time = time.time()
        planner.solve(max_runtime)

        if verbose:
            self.communicate_solve_info(time.time() - start_time)
        path = self.check_solution(pdef, exact, verbose)
        if path is None:
            return None
        return self.postprocess_path(
            path,
            space_information,
            shortcut,
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
        shortcut=True,
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
        if verbose:
            self.communicate_solve_info(time.time() - start_time)
        path = self.check_solution(pdef, exact, verbose)
        if path is None:
            return None
        return self.postprocess_path(
            path,
            space_information,
            shortcut,
            spline=spline,
            interpolate=0,
            verbose=verbose,
        )
