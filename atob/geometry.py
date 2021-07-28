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

    @classmethod
    def random(cls, center_range=None, dimension_range=None, quaternion=False):
        if center_range is not None:
            center_range = np.asarray(center_range)
            assert center_range.shape == (2, 3), "Center range should be passed in as numpy array with 2x3 dimension where first row is the low end of each dimension's range and second row is the high end"
            
            center = (center_range[1, :] - center_range[0, :]) * np.random.rand(3) + center_range[0, :]
        else:
            center = np.array([0, 0, 0])
        if dimension_range is not None:
            dimension_range = np.asarray(dimension_range)
            assert dimension_range.shape == (2, 3), "Dimension range should be passed in as numpy array with 2x3 dimension where first row is the low end of each dimension's range and second row is the high end"
            dims = (dimension_range[1, :] - dimension_range[0, :]) * np.random.rand(3) + dimension_range[0, :]
        else:
            dims = np.array([1, 1, 1])
        if quaternion:
            quaternion = Quaternion.random()
        else:
            quaternion = Quaternion([1, 0, 0, 0])

        return cls(center, dims, quaternion)
        


    @property
    def xyzw(self):
        return self._quaternion.vector.tolist() + [self._quaternion.scalar]

    @property
    def wxyz(self):
        return [self._quaternion.scalar] + self._quaternion.vector.tolist()

    @property
    def center(self):
        return self._center.tolist()

    @center.setter
    def center(self, value):
        self._center = np.asarray(value)

    @property
    def dims(self):
        return self._dims.tolist()

    @dims.setter
    def dims(self, value):
        self._dims = np.asarray(value)

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
