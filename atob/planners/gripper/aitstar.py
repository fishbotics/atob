import time

import numpy as np
from geometrout import SE3
from ompl import base as ob
from ompl import geometric as og

from atob.errors import CollisionError, ConfigurationError
from atob.planners.gripper.base import FrankaGripperBase, pose_path_as_python


class FrankaGripperAITStar(FrankaGripperBase):
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
            xyz = np.array([q.getX(), q.getY(), q.getZ()])
            return self._not_in_collision(
                SE3(xyz, np.array([rot.w, rot.x, rot.y, rot.z])), frame
            )

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
                "Average collision check time:"
                f" {self.total_collision_checking_time / self.collision_check_counts}"
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
