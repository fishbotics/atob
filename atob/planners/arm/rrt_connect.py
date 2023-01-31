import time

from ompl import base as ob
from ompl import geometric as og

from atob.planners.arm.base import FrankaArmBase


class FrankaRRTConnectPlanner(FrankaArmBase):
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
