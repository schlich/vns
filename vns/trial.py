import pandas as pd
import zarr

from vns.session import Session


def _eyejoy(pds_data):
    fields = ["x", "y", "pupil", "t", "t0"]
    df = pd.DataFrame.from_records(
        pds_data["EyeJoy"].item(),
        columns=fields,
    ).explode(fields)
    df.index.name = "trial"
    df["t0"] = pd.to_timedelta(df["t0"], unit="s")
    return df.set_index("t0", append=True)


class Trial:
    def __init__(self, session: Session, trial_number: int):
        pds_data = session.pds_data
        trial = pds_data["trialnumber"].item()
        zarr.save("data/trial1.zarr", trial)
        self.trial = trial

    def eyejoy(self):
        return _eyejoy(self.pds_data)
