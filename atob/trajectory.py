import math
import bisect
import itertools
import numpy as np
from robofin.robots import FrankaRobot

"""
This file was inspired by Klampt: http://motion.pratt.duke.edu/klampt/pyklampt_docs/_modules/klampt/model/trajectory.html
It's job is to reparametrize a trajectory for even smoothing
"""


class SplineTrajectory:
    def __init__(self, times, milestones):
        self.times, self.milestones, self.velocities = self.from_spline(
            times, milestones
        )

    def from_spline(self, times, milestones):
        velocities = []
        if len(milestones) == 1:
            velocities.append(np.zeros(milestones.shape))
        elif len(milestones) == 2:
            _s = times[1] - times[0]
            s = 1.0 / _s if _s != 0 else 0
            v = s * (milestones[1] - milestones[0])
            velocities.extend([v, v])
        else:
            N = len(milestones)
            timeiter = zip(range(0, N - 2), range(1, N - 1), range(2, N))
            for i in range(1, N - 1):
                dt = times[i] - times[i - 1]
                s = (
                    1.0 / (times[i + 1] - times[i - 1])
                    if times[i + 1] - times[i - 1] != 0
                    else 0
                )
                v = s * (milestones[i + 1] - milestones[i - 1])
                for j, (a, x, b) in enumerate(
                    zip(milestones[i - 1], milestones[i], milestones[i + 1])
                ):
                    if x <= min(a, b) or x >= max(a, b):
                        v[j] = 0.0
                    elif (v[j] < 0 and x - v[j] * dt / 3 >= a) or (
                        v[j] > 0 and x - v[j] * dt / 3.0 <= a
                    ):
                        v[j] = 3.0 / dt * (x - a)
                    elif (v[j] < 0 and x + v[j] * dt / 3 < b) or (
                        v[j] > 0 and x + v[j] * dt / 3.0 > b
                    ):
                        v[j] = 3.0 / dt * (b - x)
                velocities.append(v)
            x2 = milestones[1] - velocities[0] / 3
            x1 = x2 - (milestones[1] - milestones[0]) / 3
            v0 = 3 * (x1 - milestones[0])

            xn_2 = milestones[-2] + velocities[-1] / 3
            xn_1 = xn_2 + (milestones[-1] - milestones[-2]) / 3
            vn = 3 * (milestones[-1] - xn_1)
            velocities = [v0] + velocities + [vn]
        return times, milestones, velocities

    def eval(self, t):
        assert (
            self.velocities is not None
        ), "Cannot resample trajectory without setting velocities"
        i, u = self.get_segment(t)
        if i < 0:
            return self.milestones[0], self.velocities[0]
        elif i + 1 >= len(self.milestones):
            return self.milestones[-1], self.velocities[-1]
        return self.interpolate(
            (self.milestones[i], self.velocities[i]),
            (self.milestones[i + 1], self.velocities[i + 1]),
            u,
            self.times[i + 1] - self.times[i],
        )

    def eval_config(self, t):
        x, dx = self.eval(t)
        return x

    def get_segment(self, t):
        assert len(self.times) != 0, "Cannot have empty trajectory!"
        if len(self.times) == 1:
            return (-1, 0)
        if t >= self.times[-1]:
            return (len(self.milestones) - 1, 0)
        if t <= self.times[0]:
            return (0, 0)
        i = bisect.bisect_right(self.times, t)
        assert i > 0 and i < len(self.times), f"Invalid time index {t} in {self.times}"
        if i == 0:
            return (-1, 0)
        u = (t - self.times[i - 1]) / (self.times[i] - self.times[i - 1])
        assert u >= 0 and u <= 1
        return i - 1, u

    def interpolate(self, a, b, u, dt):
        assert len(a[0]) == len(a[1])
        assert len(a[0]) == len(b[0])
        assert len(a[1]) == len(b[1])
        x1, v1 = a[0], dt * a[1]
        x2, v2 = b[0], dt * b[1]
        x = self.hermite_eval(x1, v1, x2, v2, u)
        dx = self.hermite_deriv(x1, v1, x2, v2, u) / dt
        return x, dx

    @staticmethod
    def hermite_eval(x1, v1, x2, v2, u):
        # Make sure everything is the same shape
        assert x1.shape == v1.shape
        assert x1.shape == x2.shape
        assert x2.shape == v2.shape
        return (
            (2 * u ** 3 - 3 * u ** 2 + 1.0) * x1
            + (-2 * u ** 3 + 3 * u ** 2) * x2
            + (u ** 3 - 2 * u ** 2 + u) * v1
            + (u ** 3 - u ** 2) * v2
        )

    @staticmethod
    def hermite_deriv(x1, v1, x2, v2, u):
        # Make sure everything is the same shape
        assert x1.shape == v1.shape
        assert x1.shape == x2.shape
        assert x2.shape == v2.shape
        return (
            (6.0 * u ** 2 - 6.0 * u) * (x1 - x2)
            + (3.0 * u ** 2 - 4.0 * u + 1) * v1
            + (3.0 * u ** 2 - 2.0 * u) * v2
        )


class Trajectory:
    def __init__(self, times, milestones, velocities=None):
        self.times = times
        self.milestones = milestones
        self.velocities = velocities

    @classmethod
    def from_path(cls, path, length=None, dt=None, robot_type=FrankaRobot):
        assert bool(dt is None) != bool(length is None)

        vmax = robot_type.VELOCITY_LIMIT
        amax = robot_type.ACCELERATION_LIMIT
        _durations = np.array([0.0] * (len(path) - 1))
        for i in range(len(path) - 1):
            ## In the sequence it goes p, q, r, n
            q, r = path[i], path[i + 1]
            p = q if i == 0 else path[i - 1]
            n = r if i == len(path) - 2 else path[i + 2]
            v = 0.5 * (r - p)
            a1 = (r - q) - (q - p)
            a2 = (n - r) - (r - q)

            # This stuff can probably be done with Numpy operations
            for j in range(len(v)):
                if abs(v[j]) > vmax[j] * _durations[i]:
                    _durations[i] = abs(v[j]) / vmax[j]
            if i > 0:
                for j in range(len(a1)):
                    if abs(a1[j]) > amax[j] * _durations[i] ** 2:
                        _durations[i] = math.sqrt(abs(a1[j]) / amax[j])
            if i < len(path) - 2:
                for j in range(len(a2)):
                    if abs(a2[j]) > amax[j] * _durations[i] ** 2:
                        _durations[i] = math.sqrt(abs(a2[j]) / amax[j])

        spline_times = [0] + list(itertools.accumulate(_durations))
        spline = SplineTrajectory(np.asarray(spline_times), path)
        total_distance = spline_times[-1]
        if total_distance == 0:
            print("There is no movement in the path")
            return None

        if dt is not None:
            N = int(math.ceil(total_distance / dt))
            assert N > 0
        else:
            # length is not None:
            N = length - 1

        dt = total_distance / N
        assert dt > 0

        times = np.asarray(
            [0.0] + [float(i) / N * total_distance for i in range(1, N + 1)]
        )
        milestones = [spline.milestones[0]] + [spline.eval_config(t) for t in times[1:]]
        trajectory = cls(times, milestones)

        scaling = 0.0
        vscaling = 0.0
        aLimitingTime = 0
        vLimitingTime = 0
        for i in range(N):
            q, r = trajectory.milestones[i], trajectory.milestones[i + 1]
            p = q if i == 0 else trajectory.milestones[i - 1]
            v = 0.5 * (r - p)
            a = ((r - q) - (q - p)) / dt ** 2
            for x, lim in zip(v, vmax):
                if abs(x) > lim * vscaling:
                    vscaling = abs(x) / lim
                    vLimitingTime = i
            if i == 0:
                continue
            for x, lim in zip(a, amax):
                if abs(x) > lim * scaling ** 2:
                    scaling = math.sqrt(abs(x) / lim)
                    aLimitingTime = i
        scaling = max(vscaling, scaling)
        trajectory.times = scaling * trajectory.times
        return trajectory
