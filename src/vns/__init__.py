from __future__ import annotations

__version__ = "0.0.1"

import os
import tarfile
from pathlib import Path

import httpx
import polars as pl
import scipy
from dagster import Definitions, asset

from vns import sessions, trials

__all__ = ["trials", "sessions"]


@asset
def raw_data():
    with open(os.environ["XDG_DATA_HOME"], mode="wb") as tar:
        tar.write(
            httpx.get(
                "https://wustl.box.com/shared/static/2jmet2tj9jfkfyrsgb2cvxx3wof4zo4f.gz"
            ).content
        )


@asset
def matlab_files(raw_data):
    with tarfile.open(raw_data, "r:gz") as tar:
        tar.extractall(filter="data")
    matlab_filepaths = Path(os.environ["XDG_DATA_HOME"]).glob("*.mat")
    return (scipy.io.loadmat(path) for path in matlab_filepaths)


@asset
def trials_asset() -> pl.DataFrame:
    return pl.read_parquet("data/trials.parquet")


@asset
def sessions_asset() -> pl.DataFrame:
    return pl.concat(
        [
            pl.read_parquet(session / "trials.parquet")
            for session in Path("data/BFINAC_VNS").glob("*.")
        ]
    )


definitions = Definitions(assets=[raw_data, matlab_files, trials_asset, sessions_asset])
