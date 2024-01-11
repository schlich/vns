import datetime

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel
from shiny import render
from shiny.express import input, ui


class Session(BaseModel):
    start: datetime.datetime


ui.input_select(id="session", label="Session", choices=("1","2","3"))

ui.input_select(id="trial", label="Trial", choices=("1","2","3"))

@render.text
def txt():
    return f"Session:{input.session()}, Trial:{input.trial()}"

@render.plot
def plot():
    eyejoy = pd.read_parquet("data/eyejoy.parquet")
    print(eyejoy.columns)
    plt.scatter(eyejoy["t"], eyejoy["x"])
