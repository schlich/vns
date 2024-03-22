from __future__ import annotations

__version__ = "0.0.1"

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import polars as pl
import scipy
from dagster import Definitions, asset
from icecream import ic
from matplotlib import animation
from matplotlib.patches import Ellipse
from pandera.api.polars.model import DataFrameModel
from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

fields = {
    "trialnumber": int,
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


class EyeJoy(DataFrameModel):
    x: float
    y: float
    t: float


class CursorSample(BaseModel):
    x: float
    y: float

    def t(
        self,
        session_start: datetime,
        trial_number: int,
        sample_number: int,
    ) -> datetime:
        return session_start + timedelta(
            seconds=trial_number * 10 + sample_number / 1000,
        )


class Trial(BaseModel):
    path: Path

    @property
    def id(self) -> int:
        return int(self.path.stem)

    def eyejoy(self):
        return pl.scan_parquet(self.path.with_suffix(".parquet"))

    def replace_cursor_data(self):
        mat_path = self.path.parent.with_suffix(".mat")
        matfile = scipy.io.loadmat(
            mat_path,
            squeeze_me=True,
        )
        eyejoy = pl.DataFrame(matfile["PDS"]["EyeJoy"].item()[self.id].T).rename(
            {
                "column_0": "x",
                "column_1": "y",
                "column_2": "d",
                "column_3": "?",
                "column_4": "t",
            },
        )
        eyejoy.write_parquet(self.path.with_suffix(".parquet"))

    def trim_trailing_zeros(self):
        eyejoy = pl.read_parquet(self.path.with_suffix(".parquet")).filter(
            pl.col("t") != 0.0,
        )
        eyejoy.write_parquet(self.path.with_suffix(".parquet"))

    def get_column(self, *, column: str):
        return pl.DataFrame(
            scipy.io.loadmat(
                self.path.parent.with_suffix(".mat"),
                squeeze_me=True,
            )["PDS"][column].item(),
        )

    def add_column_to_parquet(self, column_to_append: pl.DataFrame):
        old_df = pl.read_parquet(self.path / "trials.parquet")
        new_df = pl.concat([old_df, column_to_append], how="horizontal")
        new_df.write_parquet(self.path / "trials.parquet")

    def drop_column_from_parquet(self, column_to_drop: str):
        parquet_path = self.path.with_suffix(".parquet")
        data = pl.read_parquet(parquet_path)
        if column_to_drop in data.columns:
            pl.read_parquet(parquet_path).drop([column_to_drop]).write_parquet(
                parquet_path,
            )

    def animate(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_box_aspect(1)
        ax.set_title(f"Trial={self.id}")

        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])

        t_display = ax.text(2, -5.5, "t=0.0s")

        fixation_point = Ellipse((0, 0), 0.5, 0.5, color="black", alpha=0.3)

        fp = ax.add_patch(fixation_point)

        eyejoy = self.eyejoy()

        ic(eyejoy)

        downsampled_eyejoy = (
            eyejoy
            .sort(by="t")
            .collect()
            .group_by_dynamic(index_column="t", every="100ms")
            .agg(
                    pl.col("x").mean(),
                    pl.col("y").mean(),
                )
        )

        scat = ax.scatter(
            downsampled_eyejoy[0, "x"],
            downsampled_eyejoy[0, "y"],
        )

        def update(frame: int) -> tuple[Any, Any, Any]:
            fixation_point_interval = (0.758133, 2.041467)
            scat.set_offsets(
                (
                    downsampled_eyejoy[frame, "x"],
                    downsampled_eyejoy[frame, "y"],
                ),
            )
            t_frame = frame/10
            t_display.set_text(f"time={t_frame}s")
            if fixation_point_interval[0] < t_frame < fixation_point_interval[1]:
                r = 0
            else:
                r = 0.5
            fp.set_width(r)
            fp.set_height(r)

            return (scat, t_display, fp)

        return animation.FuncAnimation(
            fig,
            update,
            frames=len(self.eyejoy().count().collect()) - 1,
            repeat=True,
        )


class Session(BaseModel):
    path: Path

    def get_trials(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.path / "trials.parquet")

    @property
    def datetime(self) -> datetime:
        return datetime.strptime(
            str(self.path).split("_", maxsplit=1)[1],
            "%d_%m_%Y_%H_%M",
        ).astimezone()


class Experiment(BaseModel):
    label: str
    sessions: list[Session]

@asset
def trials() -> pl.DataFrame:
    return pl.read_parquet("data/trials.parquet")

definitions = Definitions(
    assets=[trials]
)