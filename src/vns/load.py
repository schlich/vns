import dask.bag as db
import numpy as np
import scipy


def mat_data(path: Path) -> np.ndarray:
    return scipy.io.loadmat(
        path,
        squeeze_me=True,
    )["PDS"]

