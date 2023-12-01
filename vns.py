__version__ = "0.0.1"

import vaex as vx


def mat_to_vx(mat):
    """Convert a numpy array from scipy.io.loadmat to a vaex DataFrame.

    Parameters
    ----------
    mat : numpy.ndarray
        A numpy array from scipy.io.loadmat.

    Returns
    -------
    vaex.dataframe.DataFrame
        A vaex DataFrame.

    """
    return vx.from_arrays(time_focused=mat["PDS"].item()[0])
