__version__ = "0.0.1"

from datatree import DataTree
import numpy as np
import xarray as xr


def extinction_learning(datatree: DataTree) -> xr.Dataset:
    """Produce extinction learning results."""
    return xr.DataArray(data=np.ndarray(shape=(3)), dims=("Trial"))
