from __future__ import annotations

__version__ = "0.0.1"

import datetime
import os
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandera as pa
import plotly.express as px
import scipy
import xarray as xr
from matplotlib import animation
from matplotlib.collections import LineCollection

if TYPE_CHECKING:
    from datatree import DataTree
    from pandera.typing import DataFrame, Index, Series


fields = {
    "fractals": str,
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
    "feedid": str,
    "TrialTypeSave": str,
    "timefpabort": float,
    "repeatflag": bool,
    "monkeynotinitiated": bool,
}


class EyeJoy(pa.DataFrameModel):
    x: Series[float]
    y: Series[float]
    t: Index[float] = pa.Field(ge=0, check_name=True)


class ExperimentSchema(pa.DataFrameModel):
    start_time: Index[datetime.datetime]
    n_trials: Series[int]

class Trial:
    def __init__(self, eyejoy: DataFrame[EyeJoy] | None = None):
        if eyejoy is None:
            eyejoy = pd.read_parquet("/workspaces/vns/data/xy_1_1.parquet")
        self.eyejoy = eyejoy

    def plot(self):
        eyejoy = self.eyejoy.reset_index()
        return px.line(
            eyejoy,
            x="t",
            y=["x", "y"],
            template="plotly_white",
        )

    def animate(self):
        fig, ax = plt.subplots()
        data = self.eyejoy

        def anim_func(current_t: int):
            return ax.add_collection(LineCollection(segments=[line(x, 10)]))

        return animation.FuncAnimation(
            fig,
            anim_func,
            blit=True,
        ).save("data/ANIMATE.html", writer="html")


class Session:
    def __init__(
        self,
        data_file: Path,
    ):
        self.data_file = data_file

    @property
    def start_time(self) -> pd.Timestamp:
        """The start time of the session.

        Returns
        -------
            pd.Timestamp: The start time of the session.
        """
        return pd.to_datetime(
            Path(self.data_file).stem.split("_", maxsplit=1)[1],
            format="%d_%m_%Y_%H_%M",
        )

    @property
    def source_filetype(self) -> str:
        return self.data_file.suffix

    def matfile_path(self) -> Path:
        return Path(
            f"data/BFINAC_VNS/mat/BFnovelinac_{self.start_time.strftime('%d_%m_%Y_%H_%M')}.mat",
        )

    def parquet_path(self) -> Path:
        return Path(
            f"data/BFINAC_VNS/parquet/BFnovelinac_{self.start_time.strftime('%d_%m_%Y_%H_%M')}.parquet",
        )

    def n_trials(self) -> int:
        return len(self.trials)

    def get_trials(self):
        data = mat_data(self.matfile_path())["PDS"]

        def extract_fields(data: DataFrame) -> DataFrame:

            return pd.DataFrame(
                {
                    field_name: pd.Series(
                        data[field_name],
                        index=data["trialnumber"],
                        dtype=field_dtype,
                    )
                    for field_name, field_dtype in fields.items()
                },
            )

        return {
            "data": extract_fields(data),
            # "trials": (Trial(session=session) for session in sessions),
        }

    def to_parquet(self):
        self.trials().to_parquet(self.parquet_path())

    def __repr__(self):
        return f"<Session {self.start_time.strftime('%Y-%m-%d %H:%M')}>"

    def __lt__(self, other: Session):
        return self.start_time < other.start_time

    def plot(self):
        trials = self.trials()
        return px.bar(trials)

    @classmethod
    def matpath2date(cls, matfile_path: Path) -> datetime.datetime:
        return


class Experiment:
    """A collection of sessions from a single experiment.

    Attributes
    ----------
        sessions: A list of sessions objects from the experiment.

    """

    def __init__(
        self,
        sessions: list[Session],
    ):
        exp_data = data_dir / label / "mat"
        self.mat_files = Path(exp_data).glob("*.mat")
        self.label = label

    def __repr__(self):
        return f"<Experiment {self.label}>"


    def sessions(self):
        return sorted(
            Session(matfile_path=matfile_path) for matfile_path in self.mat_files
        )

    def start_times(self):
        return [session.start_time for session in self.sessions()]

    def trial_counts(self):
        return [len(session.n_trials()) for session in self.sessions()]

    def data(self) -> DataFrame[ExperimentSchema]:
        sessions = self.sessions()
        return pd.DataFrame(
            [
                pd.Series(
                    [session.n_trials() for session in sessions],
                    index = pd.Index([session.start_time for session in sessions], name="start_time"),
                    name="n_trials",
                    dtype=int,
                ),
            ]
        ).pipe(DataFrame[ExperimentSchema])

    def summary(self):
        return pd.DataFrame(
            {
                "start_time": [session.start_time for session in self.sessions()],
                "n_trials": [len(session.trials()) for session in self.sessions()],
            },
        )

    def plot(self):
        return px.scatter(
            self.summary(),
            x="start_time",
            y="n_trials",
            template="plotly_white",
        )


def extinction_learning(datatree: DataTree) -> xr.Dataset:
    """Produce extinction learning results."""
    return xr.DataArray(data=np.ndarray(shape=(3)), dims=("Trial"))


def date_from_filename(filepath: Path) -> datetime.datetime:
    """Convert filename to datetime object.

    Args:
    ----
        filename (str): The filename to convert.

    Returns:
    -------
        datetime.datetime: The converted datetime object.
    """
    return pd.Timestamp(
        datetime.datetime.strptime(
            str(Path(filepath).stem)[12:],
            "%d_%m_%Y_%H_%M",
        ),
    )


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
