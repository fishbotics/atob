import time

from ompl import base as ob
from ompl import geometric as og

from atob.planners.arm.base import FrankaArmBase


class FrankaABITStarPlanner(FrankaArmBase):
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
