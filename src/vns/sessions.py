from datetime import datetime
from pathlib import Path

import plotly.express as px
import polars as pl
import scipy
from pydantic import BaseModel

fields = {
    "trialnumber": int,
    "fractals": int,
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
    "feedid": int,
    "TrialTypeSave": int,
    "timefpabort": float,
    "repeatflag": bool,
    "monkeynotinitiated": bool,
}


class Session(BaseModel):
    path: Path

    def trials(self):
        return pl.scan_parquet(self.path / "trials.parquet")

    def trial_paths(self):
        return (
            self.path / str(trial)
            for trial in self.trials().select("trialnumber").collect().to_series()
        )

    def datetime(self):
        return (
            datetime.strptime(
                str(self.path.stem).split("_", maxsplit=1)[1],
                "%d_%m_%Y_%H_%M",
            )
            .astimezone()
            .strftime("%Y-%m-%d %H:%M")
        )

    def mat2polars(self):
        mat_data = scipy.io.loadmat(
            str(self.path),
            squeeze_me=True,
        )["PDS"]
        return pl.concat(
            (
                pl.Series(
                    name=column, values=mat_data[column].item(), dtype=dtype
                ).to_frame()
                for column, dtype in fields.items()
            ),
            how="horizontal",
        )

    def du(self):
        return sum(file.stat().st_size for file in self.path.rglob("*"))


class Experiment(BaseModel):
    path: Path

    def sessions(self):
        if Path(self.path / "sessions.parquet").exists():
            return pl.read_parquet(self.path / "sessions.parquet")
        else:
            sessions_data = pl.DataFrame(
                pl.Series(
                    "path", tuple(session.stem for session in self.path.glob("*/"))
                ),
            )
            sessions_data.write_parquet(self.path / "sessions.parquet")
            return sessions_data

    def du(self):
        return sum(file.stat().st_size for file in self.path.rglob("*"))

    def plot(self):
        return px.bar(self.sessions(), x="session start", y=["size", "n trials"])
