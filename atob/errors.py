class ConfigurationError(Exception):
    def __init__(self, configuration, limits):
        # If the message printed has nones, this means that something
        # is wrong with the place that threw this exception
        self.ii = None
        self.joint = None
        self.limits = None
        for ii in range(len(configuration)):
            low, high = limits[ii]
            if configuration[ii] < low or configuration[ii] > high:
                self.ii = ii
                self.joint = configuration[ii]
                self.limits = limits
                break

    def __str__(self):
        return f"Joint number {self.ii}, with value {self.joint} found to be outside the limits for that joint, {self.limits}"


class CollisionError(Exception):
    def __init__(self, q):
        self.q = q

    def __str__(self):
        return f"Configuration with value {self.q} in collision in environment"
