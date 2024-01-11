from pathlib import Path

import pandas as pd
from vns import Session


def test_session_start_time(tmpdir: Path):
    file_path = (
        tmpdir / "BFnovelinac" / "mat" / Path("BFnovelinac_01_02_2019_15_03.mat")
    )
    session = Session(file_path)
    assert session.start_time == pd.Timestamp("2019-02-01 15:03:00")
