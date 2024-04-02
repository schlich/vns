from datetime import timedelta
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import patito as pt
import polars as pl
import scipy
from matplotlib.artist import Artist
from matplotlib.patches import Rectangle
from pandera.api.polars.model import DataFrameModel
from pydantic import BaseModel


class CursorSample(DataFrameModel):
    x: float
    y: float
    t: timedelta


class TrialData(pt.Model):
    trialnumber: int
    fractals: int
    targAngle: float
    targAmp: float
    goodtrial: bool
    fixreq: bool
    datapixxtime: float
    trialstarttime: float
    timefpon: float
    timefpoff: float
    windowchosen: bool
    timetargetoff: float
    feedid: int
    TrialTypeSave: int
    timefpabort: float
    repeatflag: bool
    monkeynotinitiated: bool


class Trial(BaseModel):
    path: Path

    def id(self):
        return int(self.path.stem)

    def get_field(self, field: str) -> Any:
        return (
            pl.read_parquet(self.path.parent / "trials.parquet").item(self.id(), field)
            * 10
        )

    def cursor(self):
        return (
            pl.scan_parquet(
                self.path.with_suffix(".parquet"),
            )
            .gather_every(100)
            .select("x", "y", "t")
            .collect()
        )

    def mat2parquet(self):
        mat_path = self.path.parent.with_suffix(".mat")
        matfile = scipy.io.loadmat(
            mat_path,
            squeeze_me=True,
        )
        eyejoy = pl.DataFrame(matfile["PDS"]["EyeJoy"].item()[self.id() - 1].T).rename(
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
        return self.cursor().filter(
            pl.col("t") != 0.0,
        )

    def animate(self):
        fig, (axis_left, axis_right) = plt.subplots(ncols=2)

        cursor = self.trim_trailing_zeros()

        axis_left.set_xlim(left=-6, right=6)
        axis_left.set_ylim(bottom=-6, top=6)
        axis_right.set_xlim(left=0, right=len(cursor))
        axis_right.set_ylim(bottom=-6, top=6)

        cursor_sample = axis_left.scatter(
            x=cursor[0, "x"],
            y=cursor[0, "y"],
        )
        time_fp_on = self.get_field("timefpon")
        time_fp_off = self.get_field("timefpoff")
        time_target_off = self.get_field("timetargetoff")
        time_fp_abort = self.get_field("timefpabort")

        axis_right.fill(
            [time_fp_on] * 2 + [time_fp_off] * 2, [-0.5, 0.5, 0.5, -0.5], color="0.9"
        )

        axis_right.plot([time_target_off] * 2, [-6, 6], color="0.5")
        axis_right.plot([time_fp_abort] * 2, [-6, 6], color="r")

        axis_right.plot(
            range(len(cursor)),
            cursor[["x", "y"]],
        )

        fixation_point = Rectangle(
            xy=(-0.5, -0.5),
            width=0,
            height=0,
        )

        axis_left.add_patch(fixation_point)

        def update(frame) -> tuple[Artist, Artist]:
            cursor_sample.set_offsets(
                (
                    cursor[frame, "x"],
                    cursor[frame, "y"],
                ),
            )

            fp_on = int(time_fp_on < frame / 10 < time_fp_off)

            fixation_point.set_width(fp_on)
            fixation_point.set_height(fp_on)

            return (cursor_sample, fixation_point)

        return animation.FuncAnimation(
            fig,
            update,
            frames=len(cursor),
            repeat=True,
        )
