import pandas as pd

from vns import Session, Trial


def test_session_load_trials():
    trials = [Trial()]
    session = Session(start_time=pd.Timestamp("2019-02-01 15:03:00"), trials=trials)
    assert session.trials == trials


def test_trials_eyejoy():
    eyejoy_data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    trial = Trial(eyejoy_data=eyejoy_data)
    pd.testing.assert_frame_equal(trial.eyejoy, eyejoy_data)
