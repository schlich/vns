__version__ = "0.0.1"

import datetime
import os

import numpy as np
import pandas as pd
import scipy
import xarray as xr
from datatree import DataTree


def extinction_learning(datatree: DataTree) -> xr.Dataset:
    """Produce extinction learning results."""
    return xr.DataArray(data=np.ndarray(shape=(3)), dims=("Trial"))


def date_from_filename(filename: str) -> datetime.datetime:
    """Convert filename to datetime object.

    Args:
    ----
        filename (str): The filename to convert.

    Returns:
    -------
        datetime.datetime: The converted datetime object.
    """
    return datetime.datetime.strptime(
        filename[12:-4],
        "%d_%m_%Y_%H_%M",
    ).replace(tzinfo=datetime.UTC)


def session_attrs(filename: str) -> dict:
    """Extract session attributes from a file.

    Args:
    ----
        filename (str): The name of the file.

    Returns:
    -------
        dict: The extracted session attributes.
    """
    c = scipy.io.loadmat(
        "data/BFINAC_VNS/" + filename,
        squeeze_me=True,
    )["c"]
    return {
        field_name: data
        for field_name, data in {
            field_name: c.item()[i] for i, field_name in enumerate(c.dtype.names)
        }.items()
        if not isinstance(data, np.ndarray)
    }


def sessions() -> pd.DataFrame:
    """Retrieve session data as a DataFrame."""
    return pd.DataFrame.from_records(
        [session_attrs(filename) for filename in os.listdir("data/BFINAC_VNS")],
        index=pd.Index(
            [
                date_from_filename(filename)
                for filename in os.listdir("data/BFINAC_VNS")
            ],
            name="datetime",
        ),
    )
