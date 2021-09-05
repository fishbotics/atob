import numpy as np


def collision_free_ik(robot, bullet_env, pose, frame, retries=100):
    """
    Uses naive sampling to generate a collision free IK solution

    :param bullet_env: Environment that subclasses atob.bullet.Bullet
    :param pose: The desired pose for the end effector
    :param frame: The frame for which you want the pose to match the input
    :return: If solution is found, returns a configuration. If not, returns None
    """
    for i in range(retries + 1):
        samples = robot.random_ik(pose, frame)
        for sample in samples:
            bullet_env.marionette(sample)
            if not bullet_env.in_collision(check_self=False):
                return sample
    return None


def sample_nearby_ik(robot, config, pose, frame, samples=10, failures=10):
    """
    Generates a number of collision-free IK solutions and returns the one
    that is closest in configuration space to the input config. If the
    collision_free_ik call fails a certain number of times, give up and return None

    :param bullet_env: Environment that subclasses atob.bullet.Bullet
    :param config: The input configuration which the IK solution should be close to
    :param pose: The desired pose
    :param frame: The frame for which you want the pose to match the input
    :param samples: The maximum number of IK samples to draw
    :param failures: The maximum number of failures allowed when drawing samples
        before giving up
    :return: A state space if a solution is found, otherwise returns None
    """
    ik_samples = []
    num_failures = 0
    while len(ik_samples) < samples and num_failures < failures:
        samples = collision_free_ik(robot, pose, frame)
        if len(sample) == 0:
            num_failures += 1
            continue
        ik_samples.extend([np.asarray(sample) for s in samples])
    if len(ik_samples) == 0:
        return None
    config = np.asarray(config)
    idx = np.argmin(np.array([np.linalg.norm(config - x) for x in ik_samples]))
    return ik_samples[idx].tolist()
