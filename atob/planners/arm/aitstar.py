import time

from ompl import base as ob
from ompl import geometric as og

from atob.planners.arm.base import FrankaArmBase

# class EefPathOptimizationObjective(ob.OptimizationObjective):
#
#     def __init__(si):
#         pass
#
#     def stateCost(s):
#         pass
#
#     def motionCost(s):
#         pass
#
#     def motionCostHeuristic(s1, s2):
#         pass
#
#     def
#
class FrankaAITStar(FrankaArmBase):
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
