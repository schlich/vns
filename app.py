from datetime import datetime
from pathlib import Path

import altair as alt
import holoviews as hv
import panel as pn
import param
import polars as pl


class Trial(param.Parameterized):
    session_paths = sorted(Path("data/BFINAC_VNS/parquet").glob("*/trials.parquet"))
    sessions = {
        datetime.strptime(
            str(Path(session_path).parent.stem).split("_", maxsplit=1)[1],
            "%d_%m_%Y_%H_%M",
        ).astimezone(): session_path
        for session_path in session_paths
    }

    session = param.Selector(objects=dict(sorted(sessions.items())))
    trial = param.Integer(default=1)

    def cursor(self):
        return (
            pl.scan_parquet(
                Path(self.session).parent / f"{self.trial}.parquet",
            )
            .select("x", "y")
            .collect()
        )

    def session_data(self):
        return pl.read_parquet(self.session)

    def view(self):
        return (
            self.cursor().select("x", "y").plot().opts(tools=["vline"])
            * self.fixation_period()
            * hv.VLine(self.timetargetoff()).opts(color="black")
            * hv.VLine(self.timeoutcome()).opts(color="blue")
        )

    def fractal(self):
        return self.session_data()[self.trial, "fractals"]

    def target_angle(self):
        return self.session_data()[self.trial, "targAngle"]

    def timefpon(self):
        return self.session_data()[self.trial, "timefpon"] * 1000

    def timefpoff(self):
        return self.session_data()[self.trial, "timefpoff"] * 1000

    def timefpabort(self):
        return self.session_data()[self.trial, "timefpabort"] * 1000

    def timeoutcome(self):
        return self.session_data()[self.trial, "timeoutcome"] * 1000

    def timetargetoff(self):
        return self.session_data()[self.trial, "timetargetoff"] * 1000

    def heatmap(self):
        data = self.cursor()
        return hv.HexTiles(data)

    def cursorpath(self):
        return hv.Path(self.cursor().to_dict())

    def session_date(self):
        return (
            datetime.strptime(
                str(Path(self.session).parent.stem).split("_", maxsplit=1)[1],
                "%d_%m_%Y_%H_%M",
            )
            .astimezone()
            .strftime("%Y-%m-%d %H:%M")
        )

    def fixation_period(self):
        time_fp_off = self.timefpoff()
        if time_fp_off > 0:
            end = time_fp_off
            color = "green"
        else:
            end = self.timefpabort()
            color = "red"
        return hv.VSpan(self.timefpon(), end).opts(
            color=color,
        )

    def rolling(self):
        data = self.session_data().set_sorted("trialnumber")
        return hv.Curve(
            data.rolling(index_column="trialnumber", period="20i").agg(
                pl.col("goodtrial").sum() / pl.count()
            )
        )

    def successrate(self):
        data = self.session_data().set_sorted("trialnumber")
        return data.select(pl.col("goodtrial").sum() / pl.count())


class Cursor(param.Parameterized):
    position = param.XYCoordinates(default=(0, 0))


trial = Trial()


def view(frame):
    t = frame * 10
    rect_data = pl.DataFrame(
        {
            "x": [-1],
            "x2": [1],
            "y": [-1],
            "y2": [1],
        }
    ).to_pandas()
    data = trial.cursor().slice(t, 1).to_pandas()
    if trial.timefpon() < t < max((trial.timefpoff(), trial.timefpabort())):
        return alt.Chart(data).mark_point().encode(
            x=alt.X("x").scale(domain=(-10, 10)),
            y=alt.Y("y").scale(domain=(-10, 10)),
        ) + alt.Chart(rect_data).mark_rect(color="grey", opacity=0.5).encode(
            x="x:Q",
            x2="x2:Q",
            y="y:Q",
            y2="y2:Q",
        )
    else:
        return (
            alt.Chart(data)
            .mark_point()
            .encode(
                x=alt.X("x").scale(domain=(-10, 10)),
                y=alt.Y("y").scale(domain=(-10, 10)),
            )
        )


frame = pn.widgets.Player(start=1, end=1000, interval=10, loop_policy="loop")

cursor_anim = pn.bind(view, frame)
frame_value = pn.bind(lambda t: f"{t / 100}s", frame)

pn.Row(
    trial.param.session,
    trial.param.trial,
).servable()
pn.Row(
    cursor_anim,
    trial.view,
).servable()
pn.Row(frame, frame_value).servable()
pn.Row(trial.rolling, trial.successrate).servable()
