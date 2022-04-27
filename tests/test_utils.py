import pytest

assert False
from robofin.robots import FrankaRobot
import torch
from limp.utils import normalize_franka_joints, unnormalize_franka_joints


def test_normalize_torch():
    lower_limits = torch.as_tensor(FrankaRobot.JOINT_LIMITS[:, 0])
    lower_normalized = normalize_franka_joints(lower_limits)
    assert torch.allclose(lower_normalized, -1.0 * torch.ones(1))

    upper_limits = torch.as_tensor(FrankaRobot.JOINT_LIMITS[:, 0])
    upper_normalized = normalize_franka_joints(upper_limits)
    assert torch.allclose(upper_normalized, torch.ones(1))


def test_unnormalize_torch():
    assert torch.allclose(
        unnormalize_franka_joints(-1.0 * torch.ones(7)),
        torch.as_tensor(FrankaRobot.JOINT_LIMITS[:, 0]),
    )
    assert torch.allclose(
        unnormalize_franka_joints(1.0 * torch.ones(7)),
        torch.as_tensor(FrankaRobot.JOINT_LIMITS[:, 1]),
    )
