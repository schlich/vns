# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_experiment.ipynb.

# %% auto 0
__all__ = ['Experiment']

# %% ../nbs/00_experiment.ipynb 2
class Experiment:
    """All data from an experiment."""

    def __init__(self: "Experiment", zarr_path: Path) -> None:
        """Load data from a zarr file."""
        self.data = dt.open_datatree(zarr_path, engine="zarr")
