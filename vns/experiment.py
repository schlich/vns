from pathlib import Path

import pandas as pd
import plotly.express as px
import scipy
import xarray as xr

__all__ = ["Experiment"]

from vns.session import Session

class Experiment:

    def __init__(self, label, filetype="parquet"):
        self.path = Path("data") / label / filetype
        self.sessions = (Session(session) for session in self.path.glob(f"*.{filetype}"))

    def mat2parquet(self):
        list(session.to_parquet() for session in self.sessions)