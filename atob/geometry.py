from pyquaternion import Quaternion
import numpy as np


class Cuboid:
    def __init__(self, center, dims, quaternion):
        """
        Parameters
        ----------
        center: np.array([x, y, z])
        dims: np.array([x, y, z])
        quaternion: np.array([w, x, y, z])
        """

        # Note that the input type of these is arrays but I'm still casting.
        # This is because its easier to just case to numpy arrays than it is to
        # check for type
        self._center = np.asarray(center)
        self._dims = np.asarray(dims)
        self._quaternion = Quaternion(np.asarray(quaternion))

    @property
    def xyzw(self):
        return self._quaternion.vector.tolist() + [self._quaternion.scalar]

    @property
    def wxyz(self):
        return [self._quaternion.scalar] + self._quaternion.vector.tolist()

    @property
    def center(self):
        return self._center.tolist()

    @property
    def half_extents(self):
        return (self._dims / 2).tolist()


class Sphere:
    def __init__(self, center, radius):
        self._center = np.asarray(center)
        self._radius = radius

    @property
    def center(self):
        return self._center.tolist()

    @property
    def radius(self):
        return self._radius
