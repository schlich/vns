from __future__ import annotations

__version__ = "0.0.1"

import datetime
import logging
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

# class Cursor:

#     def __init__(self, agent: .DataFrame, x_line: matplotlib.lines.Line2D, y_line: matplotlib.lines.Line2D):


def mat_data(path: Path) -> np.ndarray:
    return scipy.io.loadmat(
        path,
        squeeze_me=True,
    )


class EyeJoy(pa.DataFrameModel):
    x: Series[float]
    y: Series[float]
    t: Index[float] = pa.Field(ge=0, check_name=True)


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
        start_time: datetime.datetime | None = None,
        matfile_path: Path | None = None,
        trials: list[Trial] | None = None,
    ):
        self.start_time = start_time or pd.to_datetime(
            Path(matfile_path).stem.split("_", maxsplit=1)[1],
            format="%d_%m_%Y_%H_%M",
        )
        self.trials = trials or self.get_trials()

    def matfile_path(self) -> Path:
        return Path(
            f"data/BFINAC_VNS/mat/BFnovelinac_{self.start_time.strftime('%d_%m_%Y_%H_%M')}.mat",
        )

    def parquet_path(self) -> Path:
        return Path(
            f"data/BFINAC_VNS/parquet/BFnovelinac_{self.start_time.strftime('%d_%m_%Y_%H_%M')}.parquet",
        )

    def get_trials(self):
        # parquet_path = self.parquet_path()
        # if parquet_path.exists():
        #     return pd.read_parquet(parquet_path)
        data = mat_data(self.matfile_path())["PDS"]
        return pd.DataFrame(
            pd.Series(
                data["fractals"],
                name="fractal",
                index=data["trialnumber"],
                dtype=str,
            ),
            pd.Series(
                data["targAngle"],
                name="targAngle",
                index=data["trialnumber"],
                dtype=float,
            ),
        )

    def to_parquet(self):
        self.trials().to_parquet(self.parquet_path())

    def __repr__(self):
        return f"<Session {self.start_time.strftime("%Y-%m-%d %H:%M")}>"

    def __lt__(self, other: Session):
        return self.start_time < other.start_time

    def plot(self):
        trials = self.trials()
        return px.bar(trials)


class Experiment:
    """A collection of sessions from a single experiment.

    Attributes
    ----------
        sessions: A list of sessions objects from the experiment.

    """

    def __init__(
        self,
        label: str | None = None,
        sessions: list[Session] | None = None,
        data_dir: Path | None = None,
    ):
        if data_dir is None:
            data_dir = Path(os.environ.get("XDG_DATA_HOME", "data"))
        if label is None:
            label = "BFINAC_VNS"
        exp_data = data_dir / label / "mat"
        mat_files = Path(exp_data).glob("*.mat")
        logging.info(mat_files)
        if sessions is None:
            sessions = sorted(
                Session(matfile_path=matfile_path)
                for matfile_path in Path(exp_data).glob("*.mat")
            )
        self.sessions = sessions
        self.label = label

    def __repr__(self):
        return f"<Experiment {self.label}>"

    def summary(self):
        return pd.DataFrame(
            {
                "start_time": [session.start_time for session in self.sessions],
                "n_trials": [len(session.trials()) for session in self.sessions],
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
