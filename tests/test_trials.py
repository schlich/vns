from pathlib import Path

import numpy as np
import pandas as pd
from vns import trials


def test_trials_eyejoy():
    eyejoy = pd.DataFrame({"x": [9, 8, 7], "y": [4, 5, 6], "t": [1, 2, 3]})
    trial = trials.Trial(eyejoy=eyejoy)
    pd.testing.assert_frame_equal(eyejoy, trial.eyejoy)


def test_trial_load_numpy(tmp_path: Path):
    trial_np_dir = Path(tmp_path)
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
    trial = trials.Trial(eyejoy=eyejoy)
    pd.testing.assert_frame_equal(eyejoy, trial.eyejoy)
