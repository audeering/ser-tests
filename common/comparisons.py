import typing

import numpy as np


def abs_gt(x: float, threshold: float) -> bool:
    r"""Compare absolute value is greater than threshold."""
    return np.abs(x) > threshold


def abs_lt(x: float, threshold: float) -> bool:
    r"""Compare absolute value is less than threshold."""
    return np.abs(x) < threshold


def distribution_in_region(
        x: float,
        threshold: typing.Tuple[float, float],
) -> bool:
    r"""Compare if the percentage of samples per class is in given range.

    Args:
        x: percentage of samples per class,
        threshold: lower and upper limit for percentage of samples

    Returns:
        ``True`` if the test has passed

    """
    return (x > threshold[0] and x < threshold[1])


def gt_or_nan(x: float, threshold: float) -> typing.Optional[bool]:
    if np.isnan(x):
        return None
    return x > threshold


def abs_lt_or_nan(x: float, threshold: float) -> typing.Optional[bool]:
    if np.isnan(x):
        return None
    return np.abs(x) < threshold
