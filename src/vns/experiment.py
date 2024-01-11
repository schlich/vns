from __future__ import annotations

import os
from pathlib import Path

__all__ = ["Experiment"]

from vns.session import Session


class Experiment:
    def __init__(self, label: str, data_dir: Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path(os.environ.get("XDG_DATA_HOME", "data"))
        exp_data = data_dir / label
        parquet_path = exp_data / "parquet"
        matfile_path = exp_data / "mat"

        data_files = parquet_path.glob("*.parquet") or matfile_path.glob("*.mat")
        self.sessions = (Session(session) for session in data_files)

    def __len__(self):
        return len(range(self.sessions["Date"]))

    def mat2parquet(self):
        return [session.to_parquet() for session in self.sessions]
