import pandas as pd
from ydata_profiling import ProfileReport

trials = pd.read_parquet("data/trials+latency.parquet.gzip")[
    ["outcome_latency", "target_amplitude"]
]


ProfileReport(trials).to_file("src/vns/assets/profile_report.html")
