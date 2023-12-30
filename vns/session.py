__all__ = ["Session"]

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import scipy
import xarray as xr
from datatree import DataTree


def mat_data(path: Path) -> np.ndarray:
    return scipy.io.loadmat(
        path,
        squeeze_me=True,
    )


def start_dt(matfile_path: Path) -> pd.Timestamp:
    return pd.to_datetime(
        matfile_path.stem.split("_", maxsplit=1)[1],
        format="%d_%m_%Y_%H_%M",
    )


class Session:
    """A group of consecutive trials."""

    def __init__(self, matfile_path: Path):
        """Construct a Session object from a path to its .mat file."""
        pds_data = mat_data(matfile_path)["PDS"]
        self.pds_data = pds_data
        self.data = pd.DataFrame(
            {
                "fractals": pds_data["fractals"].item(),
            },
            index=pd.Index(
                pds_data["trialnumber"].item(),
                name="Trial Number",
            ),
        )

    sub_arrays = (
        "EyeJoy",
        "onlineEye",
        "onlineLickForce",
        "samplesBlinkLogical",
        "img1",
        "imgfeedback",
        "spikes",
        "sptimes",
    )

    session_columns = (
        "targetacquisitionthreshold",
        "ITI_dur",
    )

    def column_measures(self):
        return self.data_struct("PDS").dtype.names

    def fractals(self) -> xr.DataArray:
        return pd.Series(
            self.data_struct("PDS")["fractals"].item(),
            name="fractals",
            dtype="category",
        ).to_xarray()

    def img(self):
        return self.data_struct("c")

    def data_array(self, struct, measure, dtype) -> xr.DataArray:
        pds_data = self.data_struct(struct)
        return pd.Series(
            pds_data[measure].item(),
            name=measure,
            dtype=dtype,
            index=pd.Index(pds_data["trialnumber"], name="trial"),
        ).to_xarray()

    @property
    def dataset(self) -> xr.Dataset:
        field_dtypes = {
            "fractals": "category",
            "targAngle": float,
            "targAmp": float,
            "goodtrial": bool,
            "fixreq": bool,
            "datapixxtime": float,
            "trialstarttime": float,
            "timefpon": float,
            "timefpoff": float,
            "windowchosen": bool,
            "timetargetoff": float,
            "feedid": "category",
            "TrialTypeSave": "category",
            "timefpabort": float,
            "repeatflag": bool,
            "monkeynotinitiated": bool,
        }
        return xr.Dataset(
            data_vars={
                field: self.data_array("PDS", field, dtype)
                for field, dtype in field_dtypes.items()
            },
            coords={"trial": self.data_array("PDS", "trialnumber", int)},
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.dataset.to_dataframe()

    @property
    def datatree(self) -> DataTree:
        return DataTree(name=self.matfile_path.stem, data=self.dataset)

    def eyejoy(self) -> pd.DataFrame:
        eyejoy_data = pd.DataFrame.from_records(
            self.data_struct("PDS")["EyeJoy"].item(),
        ).rename(
            columns={0: "x", 1: "y", 4: "t"},
        )
        eyejoy_data.index.name = "trial"
        trial_t = eyejoy_data[["x", "y", "t"]].explode(column=["x", "y", "t"])
        trial_t["dt"] = pd.to_timedelta(trial_t["t"], unit="s")
        return trial_t.set_index("dt", append=True)[["x", "y"]]

    def plot(self):
        return px.line(
            pd.DataFrame({"trial": [], "proportion_looking": []}),
            x="trial",
            y="proportion_looking",
            title=self.matfile_path.stem,
            template="plotly_white",
        )
