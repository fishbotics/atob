import time


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
