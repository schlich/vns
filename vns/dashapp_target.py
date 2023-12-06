import dash_bootstrap_components as dbc
import pandas as pd
import pandera as pa
import plotly.express as px
from dash import Dash, dcc
from pandera.typing import DataFrame, Series

class Session(pa.DataFrameModel):
    phase: Series[int] = pa.Field()
779
class SessionBlock(7797)


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    dcc.Graph(
        figure=px.bar(
            pd.DataFrame(
                {
                    "context": [],
                    r"% Freezing": [],
                    "Day": [],
                    "Epoch": [],
                    "Treatment": [],
                },
            ),
            title="Retrieval",
            x="context",
            y=r"% Freezing",
            color="Treatment",
            facet_col="Epoch",
            barmode="group",
            template="plotly_white",
        ),
    ),
)

if __name__ == "__main__":
    app.run_server(debug=True)
