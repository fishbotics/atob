import time

from geometrout.primitive import (
    Cuboid,
    CuboidArray,
    Cylinder,
    CylinderArray,
    Sphere,
    SphereArray,
)
from robofin.kinematics.collision import FrankaCollisionSpheres


class Planner:
    def __init__(self):
        self._loaded_environment = False
        self.total_collision_checking_time = 0
        self.collision_check_counts = 0
        self.collision_radius = 0.0
        self.cooo = FrankaCollisionSpheres()
        self.scene_obstacle_arrays = []

    def reset(self):
        self.scene_obstacle_arrays = []
        self._loaded_environment = False
        self.collision_check_counts = 0

    def load_scene(self, primitives):
        spheres = []
        cuboids = []
        cylinders = []
        for o in primitives:
            if isinstance(o, Sphere):
                spheres.append(o)
            elif isinstance(o, Cuboid):
                cuboids.append(o)
            elif isinstance(o, Cylinder):
                cylinders.append(o)
            else:
                raise NotImplementedError(f"Unsupport primitive of type {type(o)}")

        self.scene_obstacle_arrays.extend(
            [
                SphereArray(spheres),
                CuboidArray(cuboids),
                CylinderArray(cylinders),
            ]
        )
        self._loaded_environment = True

    def _not_in_collision(self, q, prismatic_joint):
        raise NotImplementedError(
            "Collision function must be implemented in subclasses"
        )

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
