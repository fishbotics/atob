class ConfigurationError(Exception):
    def __init__(self, ii, joint, limits):
        self.ii = ii
        self.joint = joint
        self.limits = limits

    def __str__(self):
        return f"Joint number {self.ii}, with value {self.joint} found to be outside the limits for that joint, {self.limits}"


class CollisionError(Exception):
    def __init__(self, q):
        self.q = q

    def __str__(self):
        return f"Configuration with value {q} in collision in environment"
