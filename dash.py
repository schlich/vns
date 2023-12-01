import json
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, dash_table

with Path("data/trial_codes.json").open("r") as f:
    trial_codes = json.load(f)

with Path("data/vns_sessions.json").open("r") as f:
    vns_sessions = json.load(f)

trials = pd.read_parquet("data/trials+latency.parquet.gzip").rename(
    columns={"outcome_latency": "Outcome Latency"},
)[["Outcome Latency", "trial_type"]]

trials["Trial Type"] = trials["trial_type"].astype(str).map(trial_codes)

sessions = pd.read_parquet("data/sessions.parquet.gzip")["Session"]
inverse_vns_sessions = {v: k for k, vs in vns_sessions.items() for v in vs}
sessions = sessions.astype(str).map(inverse_vns_sessions).fillna("neither")

trials = trials.join(sessions)
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

pre_surgery_sessions = trials[trials["Session"] == "neither"]

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        figure=px.ecdf(
                            trials,
                            x="Outcome Latency",
                            template="plotly_white",
                        ),
                    ),
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        figure=px.scatter(
                            sessions,
                            x=sessions.index,
                            y="Session",
                            color="Session",
                            template="plotly_white",
                        ),
                    ),
                ),
                dbc.Col(
                    dcc.Graph(
                        figure=px.scatter(
                            trials,
                            y="Trial Type",
                            x="Outcome Latency",
                            template="plotly_white",
                        ),
                    ),
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    dash_table.DataTable(
                        trials.to_dict("records"),
                        columns=[{"name": i, "id": i} for i in trials.columns],
                    ),
                ),
            ],
        ),
    ],
)

# app.layout = html.Iframe(
#     src="assets/profile_report.html",
#     style={"height": "1080px", "width": "100%"},
# )

if __name__ == "__main__":
    app.run_server(debug=True)
