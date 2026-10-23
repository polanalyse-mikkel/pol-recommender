class NotFittedError(Exception):
    """Raised when a recommendation is requested but the model is not fitted."""

    pass


class UserNotFittedError(Exception):
    """Raised when a recommendation is requested for a user not present in training data."""

    pass
