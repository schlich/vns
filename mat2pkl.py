import pickle
from pathlib import Path

from icecream import ic
import scipy

matfiles = Path("BFINAC_VNS").glob("*.mat")

for matfile in matfiles:
    stem = matfile.stem
    with open(f"data/{stem}.pkl", "wb") as f:
        pickle.dump(scipy.io.loadmat(matfile), f)

