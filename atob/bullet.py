import pybullet as p
from atob.geometry import Cuboid, Sphere
import time


class Bullet:
    def __init__(self, gui=False):
        if gui:
            self.clid = p.connect(p.GUI)
        else:
            self.clid = p.connect(p.DIRECT)
        self.urdf_path = None
        self.obstacle_ids = []

    def __del__(self):
        p.disconnect(self.clid)

    def reload(self):
        p.disconnect(self.clid)
        self.clid = p.connect(p.GUI)
        if self.urdf_path is not None:
            self.load_robot(self.urdf_path)
        self.obstacle_ids = []

    def load_robot(self, path):
        self.robot_id = p.loadURDF(path, useFixedBase=True, physicsClientId=self.clid)
        self.urdf_path = path

    def load_cuboids(self, cuboids):
        if isinstance(cuboids, Cuboid):
            cuboids = [cuboids]
        ids = []
        for cuboid in cuboids:
            assert isinstance(cuboid, Cuboid)
            # obstacle_visual_id = p.createVisualShape(
            #     shapeType=p.GEOM_BOX,
            #     halfExtents=cuboid.half_extents,
            #     rgbaColor=[1, 1, 1, 1],
            #     physicsClientId=self.clid,
            # )
            obstacle_collision_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=cuboid.half_extents,
                physicsClientId=self.clid,
            )
            obstacle_id = p.createMultiBody(
                basePosition=cuboid.center,
                baseOrientation=cuboid.xyzw,
                # baseVisualShapeIndex=obstacle_visual_id,
                baseCollisionShapeIndex=obstacle_collision_id,
                physicsClientId=self.clid,
            )
            ids.append(obstacle_id)
        self.obstacle_ids.extend(ids)
        if len(ids) == 1:
            return ids[0]
        return ids

    def load_spheres(self, spheres):
        if isinstance(spheres, Sphere):
            spheres = [spheres]

        ids = []
        for sphere in spheres:
            assert isinstance(sphere, Sphere)
            # obstacle_visual_id = p.createVisualShape(
            #     shapeType=p.GEOM_SPHERE,
            #     radius=sphere.radius,
            #     rgbaColor=[0, 0, 0, 1],
            #     physicsClientId=self.clid,
            # )
            obstacle_collision_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=sphere.radius,
                physicsClientId=self.clid,
            )
            obstacle_id = p.createMultiBody(
                basePosition=sphere.center,
                # baseVisualShapeIndex=obstacle_visual_id,
                baseCollisionShapeIndex=obstacle_collision_id,
                physicsClientId=self.clid,
            )
            ids.append(obstacle_id)
        self.obstacle_ids.extend(ids)
        if len(ids) == 1:
            return ids[0]
        return ids

    def clear_obstacle(self, id):
        if id is not None:
            p.removeBody(id, physicsClientId=self.clid)
            self.obstacle_ids = [x for x in self.obstacle_ids if x != id]

    def clear_all_obstacles(self):
        for id in self.obstacle_ids:
            if id is not None:
                p.removeBody(id, physicsClientId=self.clid)
        self.obstacle_ids = []

    def in_collision(self):
        # Step the simulator (only enough for collision detection)
        p.performCollisionDetection(physicsClientId=self.clid)

        # Iterate through all obstacles to check for collisions
        for id in self.obstacle_ids:
            contacts = p.getContactPoints(self.robot_id, id, physicsClientId=self.clid)
            if len(contacts) > 0:
                return True
        return False

    def marionette(self, config):
        for i in range(1, 8):
            p.resetJointState(
                self.robot_id, i, config[i - 1], physicsClientId=self.clid
            )
