from pyquaternion import Quaternion
import numpy as np


class SO3:
    """
    A generic class defining a 3D orientation. Mostly a wrapper around quaternions
    """

    def __init__(self, quat):
        """
        :param quat: Quaternion
        """
        self._quat = quat

    @classmethod
    def from_rpy(cls, rpy):
        """
        Convert roll-pitch-yaw coordinates to a 3x3 homogenous rotation matrix.

        The roll-pitch-yaw axes in a typical URDF are defined as a
        rotation of ``r`` radians around the x-axis followed by a rotation of
        ``p`` radians around the y-axis followed by a rotation of ``y`` radians
        around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles. See
        Wikipedia_ for more information.

        .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

        :param rpy: The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).
        :return: An SO3 object
        """
        coords = np.asanyarray(rpy, dtype=np.float64)
        c3, c2, c1 = np.cos(rpy)
        s3, s2, s1 = np.sin(rpy)

        matrix = np.array(
            [
                [c1 * c2, (c1 * s2 * s3) - (c3 * s1), (s1 * s3) + (c1 * c3 * s2)],
                [c2 * s1, (c1 * c3) + (s1 * s2 * s3), (c3 * s1 * s2) - (c1 * s3)],
                [-s2, c2 * s3, c2 * c3],
            ],
            dtype=np.float64,
        )
        return SO3(Quaternion(matrix=matrix))

    @property
    def inverse(self):
        """
        :return: The inverse of the orientation
        """
        return SO3(self._quat.inverse)

    @property
    def transformation_matrix(self):
        return self._quat.transformation_matrix

    @property
    def xyzw(self):
        """
        :return: A list representation of the quaternion as xyzw
        """
        return self._quat.vector.tolist() + [self._quat.scalar]

    @property
    def wxyz(self):
        """
        :return: A list representation of the quaternion as wxyz
        """
        return [self._quat.scalar] + self._quat.vector.tolist()

    @property
    def matrix(self):
        """
        :return: The matrix representation of the orientation
        """
        return self._quat.rotation_matrix


class SE3:
    """
    A generic class defining a 3D pose with some helper functions for easy conversions
    """

    def __init__(self, matrix=None, xyz=None, quat=None, so3=None, rpy=None):
        assert bool(matrix is None) != bool(
            xyz is None and (bool(quat is None) ^ bool(so3 is None) ^ bool(rpy is None))
        )
        if matrix is not None:
            self._xyz = matrix[:3, 3]
            self._so3 = SO3(Quaternion(matrix=matrix))
        else:
            self._xyz = np.asarray(xyz)
            if quat is not None:
                assert isinstance(quat, Quaternion)
                self._so3 = SO3(quat)
            elif rpy is not None:
                self._so3 = SO3.from_rpy(rpy)
            else:
                self._so3 = so3

    def __matmul__(self, other):
        """
        Allows for numpy-style matrix multiplication using `@`
        """
        return SE3(matrix=self.matrix @ other.matrix)

    @property
    def inverse(self):
        """
        :return: The inverse transformation
        """
        return SE3(xyz=-self._xyz, so3=self._so3.inverse)

    @property
    def matrix(self):
        """
        :return: The internal matrix representation
        """
        m = self._so3.transformation_matrix
        m[:3, 3] = self.xyz
        return m

    @property
    def so3(self):
        """
        :return: The representation of orientation
        """
        return self._so3

    @property
    def xyz(self):
        """
        :return: The translation vector
        """
        return self._xyz.tolist()

    @classmethod
    def from_ompl(cls, pose):
        """
        Constructs SE3 object from an ompl SE3 state type

        :param: OMPL SE3StateSpace object
        :return: SE3 object
        """
        rot = pose.rotation()
        xyz = np.array([pose.getX(), pose.getY(), pose.getZ()])
        return cls(xyz=xyz, quat=Quaternion([rot.w, rot.x, rot.y, rot.z]))

    @classmethod
    def from_unit_axes(cls, origin, x, y, z):
        """
        Constructs SE3 object from unit axes indicating direction and an origin

        :param: OMPL SE3StateSpace object
        :return: SE3 object
        """
        assert np.isclose(np.dot(x, y), 0)
        assert np.isclose(np.dot(x, z), 0)
        assert np.isclose(np.dot(y, z), 0)
        assert np.isclose(np.linalg.norm(x), 1)
        assert np.isclose(np.linalg.norm(y), 1)
        assert np.isclose(np.linalg.norm(z), 1)
        m = np.eye(4)
        m[:3, 0] = x
        m[:3, 1] = y
        m[:3, 2] = z
        m[:3, 2] = origin
        return cls(matrix=m)


class Cuboid:
    def __init__(self, center, dims, quaternion):
        """
        :param center: np.array([x, y, z])
        :param dims: np.array([x, y, z])
        :param quaternion: np.array([w, x, y, z])
        """

        # Note that the input type of these is arrays but I'm still casting.
        # This is because its easier to just case to numpy arrays than it is to
        # check for type
        self._center = np.asarray(center)
        self._dims = np.asarray(dims)
        self._quaternion = Quaternion(np.asarray(quaternion))

    @classmethod
    def random(cls, center_range=None, dimension_range=None, quaternion=False):
        """
        Creates a random cuboid within the given ranges
        :param center_range: If given, represents the uniform range from which to draw a center.
            Should be np.array with dimension 2x3. First row is lower limit, second row is upper limit
            If not given, center defaults to [0, 0, 0]
        :param dimension_range: If given, represents the uniform range from which to draw a center.
            Should be np.array with dimension 2x3. First row is lower limit, second row is upper limit
            If not given, center defaults to [1, 1, 1]
        :param quaternion: If True, will give a random orientation to cuboid.
            If not given, will be set as the identity
        :return: Cuboid object drawn from specified uniform distribution
        """
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

    def is_zero_volume(self):
        return np.isclose(self._dims, 0).any()

    @property
    def xyzw(self):
        """
        :return: The quaternion of the cuboid, represented as x, y, z, w
        """
        return self._quaternion.vector.tolist() + [self._quaternion.scalar]

    @property
    def wxyz(self):
        """
        :return: The quaternion of the cuboid, represented at w, x, y, z
        """
        return [self._quaternion.scalar] + self._quaternion.vector.tolist()

    @property
    def pose(self):
        """
        :return: The pose of the cuboid as an SE3 object
        """
        return SE3(xyz=self._center, quat=self._quaternion)

    @property
    def center(self):
        """
        :return: The center of the object as a list
        """
        return self._center.tolist()

    @center.setter
    def center(self, value):
        """
        Set the center of the cuboid

        :param value: The new center of the cuboid
        """
        self._center = np.asarray(value)

    @property
    def dims(self):
        """
        :return: The dimensions of the cuboid as a list
        """
        return self._dims.tolist()

    @dims.setter
    def dims(self, value):
        """
        Set the dimensions of the cuboid

        :param value: The new desired dimensions of the cuboid
        """
        self._dims = np.asarray(value)

    @property
    def half_extents(self):
        """
        :return: The half-dimensions of the cuboid. This is necessary for some interfaces.
        """
        return (self._dims / 2).tolist()


class Sphere:
    def __init__(self, center, radius):
        """
        Constructs an internal sphere representation

        :param center: The center of the sphere as a list of numpy array
        :param radius: The radius of the sphere as a number
        """
        self._center = np.asarray(center)
        self._radius = radius

    @property
    def center(self):
        """
        :return: The center of the sphere as a list
        """
        return self._center.tolist()

    @property
    def radius(self):
        """
        :return: The radius of the sphere
        """
        return self._radius
