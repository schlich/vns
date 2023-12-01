import numpy as np

from vns import mat_to_vx


def test_numpy_array_from_scipy_io_loadmat_to_vaex_dataframe():
    ndarray_after_loadmat = np.ndarray(
        shape=(),
        dtype=[("time_focused", object)],
        buffer=np.array([1.0, 2.0, 3.0]),
    )

    assert mat_to_vx(ndarray_after_loadmat).shape == (3, 1)
