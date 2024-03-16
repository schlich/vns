from __future__ import annotations

__version__ = "0.0.1"

import datetime
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandera as pa
import plotly.express as px
import scipy.io as scipy_io
import xarray as xr
from matplotlib import animation
from matplotlib.patches import Ellipse
from pandera.typing import DataFrame
from pydantic import BaseModel

if TYPE_CHECKING:
    from datatree import DataTree
    from pandera.typing import Index, Series


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


class Trial(BaseModel):
    session: Session
    id: int

    def animate(self):
        eyejoy = self.session.get_trials()["EyeJoy"]

        t_trial = DataFrame[EyeJoy](eyejoy.item()[self.id]).T.rename(
            columns={0: "x", 1: "y", 4: "t"},
        )[["x", "y", "t"]]
        df = t_trial.reset_index().rename(columns={"index": "i"})
        downsampled = df.groupby(df.index // 100).mean()

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_box_aspect(1)
        ax.set_title(f"Session={self.session.label}, Trial={self.id}")

        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])

        t_display = ax.text(2, -5.5, "t=0.0s")

        fixation_point = Ellipse((0, 0), 0.5, 0.5, color="black", alpha=0.3)

        fp = ax.add_patch(fixation_point)

        scat = ax.scatter(
            downsampled.loc[0, "x"],
            downsampled.loc[0, "y"],
        )

        def update(frame: int) -> tuple[Any, Any, Any]:
            time_fp_on = 0.758133
            time_fp_off = 2.041467
            scat.set_offsets((downsampled.loc[frame, "x"], downsampled.loc[frame, "y"]))
            t_display.set_text(f"time={frame/10}s")
            r = 0 if frame / 10 < time_fp_on or frame / 10 > time_fp_off else 0.5
            fp.set_width(r)
            fp.set_height(r)

            return (scat, t_display, fp)

        return animation.FuncAnimation(
            fig,
            update,
            frames=len(downsampled) - 1,
            repeat=True,
        )


class Session(BaseModel):
    data_filepath: Path
    # trials = list[Trial]

    @property
    def filetype(self):
        return self.data_filepath.suffix

    @property
    def label(self):
        return self.data_filepath.stem

    @property
    def start_time(self) -> pd.Timestamp:
        return pd.to_datetime(
            Path(self.data_filepath).stem.split("_", maxsplit=1)[1],
            format="%d_%m_%Y_%H_%M",
        )

    def get_trials(self):
        return scipy_io.loadmat(str(self.data_filepath), squeeze_me=True)["PDS"]

    def __lt__(self, other: Session):
        return self.start_time < other.start_time


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
                    index=pd.Index(
                        [session.start_time for session in sessions],
                        name="start_time",
                    ),
                    name="n_trials",
                    dtype=int,
                ),
            ],
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
