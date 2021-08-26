def collision_free_ik(bullet_env, pose, retries=2, frame):
    """
    Uses naive sampling to generate a collision free IK solution

    :param bullet_env: Environment that subclasses atob.bullet.Bullet
    :param pose: The desired pose for the end effector
    :param frame: The frame for which you want the pose to match the input
    :return: If solution is found, returns a configuration. If not, returns None
    """
    for i in range(retries + 1):
        sample = bullet_env.ik(pose, retries=20, frame)
        if sample is None:
            continue
        bullet_env.marionette(sample)
        collides = bullet_env.in_collision(check_self=True)
        if not collides:
            return sample
    return None


def sample_nearby_ik(bullet_env, config, pose, frame, samples=10, failures=10):
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
        sample = collision_free_ik(bullet_env, matrix=pose)
        if sample is None:
            num_failures += 1
            continue
        ik_samples.append(np.asarray(sample))
    if len(ik_samples) == 0:
        return None
    config = np.asarray(config)
    idx = np.argmin(np.array([np.linalg.norm(config - x) for x in samples]))
    return samples[idx].tolist()
