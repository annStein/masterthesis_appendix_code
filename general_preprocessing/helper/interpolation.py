import numpy as np


def interpolate_nan_values(array, method='linear'):
    """
    this method chooses the interpolation method
    """
    if method == "linear":
        return interpolate_nan_values_linear(array)
    else:
        raise Exception("No valid interpolation method")


def interpolate_nan_values_linear(array):
    result = np.copy(array).astype(float)
    nans, values = np.isnan(result), lambda l: l.nonzero()[0]
    if len(result[nans]) != len(result):
        result[nans] = np.interp(values(nans), values(~nans), result[~nans])
    return result
