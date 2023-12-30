from pathlib import Path

import pandas as pd
import plotly.express as px
import scipy
import xarray as xr

__all__ = ["Experiment"]


class Experiment:
    """All data from an experiment."""

    def __repr__(self):
        return "Experiment('BFINAC_VNS')"

    def sessions(self):
        paths = Path("/workspaces/vns/data/BFINAC_VNS").glob("*.mat")
        return xr.Dataset(
            {
                "trials": (
                    "session",
                    xr.DataArray(
                        scipy.io.loadmat(
                            "/workspaces/vns/data/BFINAC_VNS/BFnovelinac_01_02_2019_15_03.mat",
                            squeeze_me=True,
                        )["PDS"]["trialnumber"].item(),
                        dims=("trial"),
                    ),
                ),
            },
            coords={
                "session": xr.DataArray(
                    pd.Index(
                        sorted(
                            [
                                pd.to_datetime(
                                    path.stem.split("_", maxsplit=1)[1],
                                    format="%d_%m_%Y_%H_%M",
                                )
                                for path in paths
                            ],
                        ),
                        name="start_date",
                    ),
                ),
            },
        )

    def plot(self):
        return px.line(
            self.data,
            x="Trial",
            y="Proportion Looking",
            template="plotly_white",
            facet_col="Day",
        )
