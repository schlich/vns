from pathlib import Path

import pandas as pd

from vns import Experiment, Session


def test_session_dates():
    start_time = pd.Timestamp("2019-02-01 15:03:00")

    session = Session(start_time)
    experiment = Experiment(sessions=[session])
    assert [session.start_time for session in experiment.sessions] == [start_time]


def test_session_data_read(tmpdir: Path):
    matfile_path = tmpdir / "mat" / "BFnovelinac_01_02_2019_15_03.mat"
    assert Session(matfile_path=matfile_path).start_time == pd.Timestamp(
        "2019-02-01 15:03:00",
    )


def test_experiment_from_label(tmpdir):
    exp_dir = tmpdir / "BFnovelinac"
    exp_dir.mkdir()
    exp_mat_dir = exp_dir / "mat"
    exp_mat_dir.mkdir()
    Path(exp_mat_dir / "BFnovelinac_01_02_2019_15_03.mat").touch()
    experiment = Experiment(label="BFnovelinac", data_dir=tmpdir)
    assert [session.start_time for session in experiment.sessions] == [
        pd.Timestamp("2019-02-01 15:03:00"),
    ]
