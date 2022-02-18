from ompl import base as ob
from ompl import geometric as og

from robofin.robots import FrankaRobot
from atob.errors import ConfigurationError, CollisionError

from geometrout.transform import SE3
import time
import numpy as np

from pyquaternion import Quaternion


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
        # TODO Fix this to apply to multiple robots
        for ii in range(FrankaRobot.DOF):
            low, high = FrankaRobot.JOINT_LIMITS[ii]
            if q[ii] < low or q[ii] > high:
                raise ConfigurationError(ii, q[ii], FrankaRobot.JOINT_LIMITS[ii])
        return

    def setup_problem(self, start, goal):
        if not self._loaded_environment:
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
        if shortcut:
            simplifier.shortcutPath(path)
        if spline:
            simplifier.smoothBSpline(path)
        if verbose:
            print(f"Smoothing time: {time.time() - start_time}")
        if interpolate > 0:
            path.interpolate(interpolate)
        path = path_as_python(path, FrankaRobot.DOF)
        return path


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
