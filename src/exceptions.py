from typing import Optional


class CollisionFormatUnrecognizedError(Exception):
    """
    An exception raised when the collision format cannot be recognised or is not
    supported by this library.
    """

    def __init__(self, message: Optional[str]):
        if message is None:
            super().__init__(
                "unrecognized image format: this library can only handle Collision Data "
            )
        else:
            super().__init__(message)


class IncompatibleShapeError(Exception):
    """
    An exception raised when the shape of the supplied array or dataframe is
    incompatible with the expected shape as required by the callee.
    """

    def __init__(self, expected_shape: str, actual_shape: str, *args: object):
        super().__init__(
            f"incompatible shape: expected `{expected_shape}` but got `{actual_shape}` instead",
            *args)
