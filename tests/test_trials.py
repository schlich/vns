from pathlib import Path

import numpy as np
import pandas as pd

from vns import Session, Trial


def test_session_load_trials():
    trials = [Trial()]
    session = Session(start_time=pd.Timestamp("2019-02-01 15:03:00"), trials=trials)
    assert session.trials == trials


def test_trials_eyejoy():
    eyejoy = pd.DataFrame({"x": [9, 8, 7], "y": [4, 5, 6], "t": [1, 2, 3]})
    trial = Trial(eyejoy=eyejoy)
    pd.testing.assert_frame_equal(eyejoy, trial.eyejoy)


def test_trial_load_numpy(tmpdir):
    trial_np_dir = Path(tmpdir)
    eyejoy = pd.DataFrame(
        {
            "x": [9, 8, 7],
            "y": [4, 5, 6],
        },
        index=pd.Index([1, 2, 3], name="t"),
    )
    np_path = trial_np_dir / "eyejoy.npz"
    np.savez(
        np_path,
        t=eyejoy.index.to_numpy(),
        pos=eyejoy[["x", "y"]].to_numpy(),
    )
    trial = Trial(eyejoy=eyejoy)
    pd.testing.assert_frame_equal(eyejoy, trial.eyejoy)
