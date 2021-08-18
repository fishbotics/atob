from pyquaternion import Quaternion
import numpy as np


class SO3:
    def __init__(self, quat):
        self._quat = quat

    @property
    def transformation_matrix(self):
        return self._quat.transformation_matrix

    @property
    def xyzw(self):
        return self._quat.vector.tolist() + [self._quat.scalar]

    @property
    def wxyz(self):
        return [self._quat.scalar] + self._quat.vector.tolist()


class SE3:
    def __init__(self, matrix=None, xyz=None, quat=None):
        assert bool(matrix is None) != bool(xyz is None and quat is None)
        if matrix is not None:
            self._xyz = matrix[:3, 3]
            self._so3 = SO3(Quaternion(matrix=matrix))
        else:
            assert isinstance(quat, Quaternion)
            self._xyz = np.asarray(xyz)
            self._so3 = SO3(quat)

    def __matmul__(self, other):
        return SE3(matrix=self.matrix @ other.matrix)

    def inv(self):
        return SE3(matrix=np.linalg.inv(self.matrix))

    @property
    def matrix(self):
        m = self._so3.transformation_matrix
        m[:3, 3] = self.xyz
        return m

    @property
    def so3(self):
        return self._so3

    @property
    def xyz(self):
        return self._xyz.tolist()

    @classmethod
    def from_ompl(cls, pose):
        rot = pose.rotation()
        xyz = np.array([pose.getX(), pose.getY(), pose.getZ()])
        return cls(xyz=xyz, quat=Quaternion([rot.w, rot.x, rot.y, rot.z]))


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
            assert center_range.shape == (
                2,
                3,
            ), "Center range should be passed in as numpy array with 2x3 dimension where first row is the low end of each dimension's range and second row is the high end"

            center = (center_range[1, :] - center_range[0, :]) * np.random.rand(
                3
            ) + center_range[0, :]
        else:
            center = np.array([0, 0, 0])
        if dimension_range is not None:
            dimension_range = np.asarray(dimension_range)
            assert dimension_range.shape == (
                2,
                3,
            ), "Dimension range should be passed in as numpy array with 2x3 dimension where first row is the low end of each dimension's range and second row is the high end"
            dims = (dimension_range[1, :] - dimension_range[0, :]) * np.random.rand(
                3
            ) + dimension_range[0, :]
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
    def pose(self):
        return SE3(xyz=self._center, quat=self._quaternion)

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
