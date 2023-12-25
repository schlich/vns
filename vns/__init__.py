__version__ = "0.0.1"

import datetime
import os

import pandas as pd
import scipy

# from datatree import DataTree
# import numpy as np
# import xarray as xr


# def extinction_learning(datatree: DataTree) -> xr.Dataset:
#     """Produce extinction learning results."""
#     return xr.DataArray(data=np.ndarray(shape=(3)), dims=("Trial"))


def date_from_filename(filename):
    return datetime.datetime.strptime(
        filename[12:-4],
        "%d_%m_%Y_%H_%M",
    )

def session_attrs(filename):
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


def sessions():
    pd.DataFrame.from_records(
    [session_attrs(filename) for filename in os.listdir("data/BFINAC_VNS")],
    index=pd.Index(
        [date_from_filename(filename) for filename in os.listdir("data/BFINAC_VNS")],
        name="datetime",
    ),
)
