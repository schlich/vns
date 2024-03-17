from __future__ import annotations

__version__ = "0.0.1"

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandera.polars as pa
import polars as pl
import scipy
from matplotlib import animation
from matplotlib.patches import Ellipse
from pydantic import BaseModel

if TYPE_CHECKING:
    from pandera.typing import Index, Series


fields = {
    "trialnumber": int,
    "fractals": str,
    "targAngle": float,
    "targAmp": float,
    "goodtrial": bool,
    "fixreq": bool,
    # "datapixxtime": float,
    # "trialstarttime": float,
    "timefpon": float,
    "timefpoff": float,
    # "windowchosen": bool,
    "timetargetoff": float,
    # "feedid": str,
    "TrialTypeSave": str,
    "timefpabort": float,
    "repeatflag": bool,
    # "monkeynotinitiated": bool,
}


class EyeJoy(pa.DataFrameModel):
    x: float
    y: float
    t: float


class CursorSample(BaseModel):
    x: float
    y: float
    t: float


class ExperimentSchema(pa.DataFrameModel):
    start_time: Index[datetime.datetime]
    n_trials: Series[int]


class Trial(BaseModel):
    id: int
    fractal: int

    def eyejoy(self) -> EyeJoy:
        return (
            pl.read_parquet(
                f"../data/BFINAC_VNS/BFnovelinac_31_01_2019_16_11/{self.id}.parquet",
            )
            .rename({"column_0": "x", "column_1": "y"})
            .select(["x", "y"])
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

        scat = ax.scatter(
            self.eyejoy.loc[0, "x"],
            self.eyejoy.loc[0, "y"],
        )

        def update(frame: int) -> tuple[Any, Any, Any]:
            time_fp_on = 0.758133
            time_fp_off = 2.041467
            x = self.eyejoy.loc[frame, "x"]
            y = self.eyejoy.loc[frame, "y"]
            scat.set_offsets((x, y))
            t_display.set_text(f"time={frame/10000}s")
            r = 0 if frame / 10000 < time_fp_on or frame / 10000 > time_fp_off else 0.5
            fp.set_width(r)
            fp.set_height(r)

            return (scat, t_display, fp)

        return animation.FuncAnimation(
            fig,
            update,
            frames=len(self.eyejoy) - 1,
            repeat=True,
        )


def parse_filename(label: str):
    return datetime.datetime.strptime(
        label.split("_", maxsplit=1)[1],
        "%d_%m_%Y_%H_%M",
    ).astimezone()


def get_sessions(matfiles: list[Path]):
    return {
        parse_filename(path.stem).isoformat(): parse_filename(path.stem).strftime(
            "%Y-%m-%d %H:%M",
        )
        for path in matfiles
    }


def mat2parquet(mat_path: Path, remove: bool | None = None):
    new_folder = Path(mat_path.parent / mat_path.stem)
    new_folder.mkdir(exist_ok=True)
    parquet_path = new_folder / "trials.parquet"
    if not parquet_path.exists():
        pds_data = scipy.io.loadmat(str(mat_path), squeeze_me=True)["PDS"]
        trials = pl.DataFrame(
            pl.Series(
                name=field,
                values=pds_data[field].item(),
            )
            for field in fields
        )
        trials.write_parquet(parquet_path)
    else:
        trials = pl.read_parquet(parquet_path)
    for trial_id in trials["trialnumber"]:
        eyejoy_path = new_folder / f"{trial_id}.parquet"
        if not eyejoy_path.exists():
            pl.DataFrame(
                scipy.io.loadmat(
                    mat_path,
                    squeeze_me=True,
                )["PDS"]["EyeJoy"]
                .item()[trial_id - 1]
                .T,
            ).rename({"column_0": "x", "column_1": "y"}).select(
                ["x", "y"],
            ).write_parquet(
                eyejoy_path,
            )
    if remove:
        mat_path.unlink()


class Session(BaseModel):
    datetime: datetime.datetime

    def get_trials(self) -> pl.DataFrame:
        base_path = Path("/workspaces/vns/data/BFINAC_VNS")
        session_simpledate = self.datetime.strftime("%Y-%m-%d_%H-%M")
        parquet_path = Path(base_path / session_simpledate).with_suffix(".parquet")
        return pl.read_parquet(parquet_path)


class Experiment(BaseModel):
    label: str
    sessions: list[Session]
