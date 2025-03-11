# Import required libraries
import numpy as np
import pandas as pd
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px
import os
import pickle
import mlflow
import dash_bootstrap_components as dbc
from dash import Dash

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

global EncodedLabel
EncodedLabel = {
    "Diesel": 0,
    "Petrol": 1,
    "Automatic": 0,
    "Manual": 1,
    "Dealer": [0, 0],
    "Individual": [1, 0],
    "Trustmark Dealer": [0, 1],
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
}

owner = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]
fuel = ["Diesel", "Petrol"]

seller_type = ["Individual", "Dealer", "Trustmark Dealer"]

year = list(range(1886, 2024, 1))

transmission = ["Manual", "Automatic"]

X_tran_stat = {
    "year": [2013.8134256055364, 2015.0, 2017, 4.039062067605816],
    "fuel": [0.4528719723183391, 0.0, 0, 0.4978084455072879],
    "seller_type": [0.8898269896193771, 1.0, 1, 0.3952063053306725],
    "transmission": [0.8690657439446366, 1.0, 1, 0.33735178726916865],
    "owner": [1.453287197231834, 1.0, 1, 0.707741211934925],
    "engine": [1464.542271562767, 1248.0, 1248.0, 507.5802774640794],
    "max_power": [91.86694112627987, 82.85, 74.0, 35.92656163539812],
}

# print(pickle.__version__)
filename = "model/"
best_model = mlflow.pyfunc.load_model(filename)
best_model = best_model.get_raw_model()

loaded_model = pickle.load(open("old_model/car-price.model", "rb"))


def preprocess(v: dict):
    r = []
    for i, j in v.items():
        # value = None
        if (j == None) and (i in ["engine", "max_power"]):
            value = X_tran_stat[i][1]
        elif (j == None) and i == "seller_type":
            value = EncodedLabel["Individual"]
        elif j == None:
            value = X_tran_stat[i][2]
        else:
            if i == "year":
                value = (j - 1886) / (2024 - 1886)
            elif i in ["engine", "max_power"]:
                value = (j - X_tran_stat[i][0]) / X_tran_stat[i][3]
            else:
                value = EncodedLabel[j]
        if i == "seller_type":
            r.append(value[0])
            r.append(value[1])
        else:
            r.append(value)
    return r


def old_preprocess(v: dict):
    r = []
    for i, j in v.items():
        # value = None
        if (j == None) and (i in ["engine", "max_power"]):
            value = X_tran_stat[i][1]
        elif (j == None) and i == "seller_type":
            value = EncodedLabel["Individual"]
        elif j == None:
            value = X_tran_stat[i][2]
        else:
            if i == "year":
                value = j
            elif i in ["engine", "max_power"]:
                value = (j - X_tran_stat[i][0]) / X_tran_stat[i][3]
            else:
                value = EncodedLabel[j]
        if i == "seller_type":
            r.append(value[0])
            r.append(value[1])
        else:
            r.append(value)
    return r


# Create a dash application
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    dcc.Tabs(
        [
            dcc.Tab(
                label="New Model",
                children=[
                    html.H2(
                        children="Car Price Prediction",
                        style={
                            "textAlign": "center",
                            "color": "#503D36",
                            "font-size": 40,
                        },
                    ),
                    dbc.Stack(
                        dcc.Textarea(
                            id="explainer",
                            value="""This is machine learning model to predict the selling price of the car. There are seven features you can input. They are the year the car is produced, fuel type of the car, how many times the car is traded(owner), how the car is traded (seller type), the transmission type, the engine size measured in cc and the max power measured in bhp. After inputting, click the predict button. The output can be seen in the bottom. Although the accuracy of the new model performance may be lower than the old model, the new model is more sensitive to feature values. This is good for predicting slightly varied inputs.""",
                            style={
                                "width": "100%",
                                "height": 80,
                                "whiteSpace": "pre-line",
                            },
                            readOnly=True,
                        )
                    ),
                    dbc.Stack(
                        [
                            dcc.Dropdown(
                                id="year-dropdown",
                                options=year,
                                # value='ALL',
                                placeholder="Select model year",
                                searchable=True,
                                style={
                                    "marginRight": "10px",
                                    "margin-top": "10px",
                                    "width": "100%",
                                },
                            ),
                            dcc.Dropdown(
                                id="fuel-dropdown",
                                options=fuel,
                                # value='ALL',
                                placeholder="Select fuel type",
                                searchable=True,
                                style={
                                    "marginRight": "10px",
                                    "margin-top": "10px",
                                    "width": "100%",
                                },
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    id="seller_type-dropdown",
                                    options=seller_type,
                                    # value='ALL',
                                    placeholder="Select seller type",
                                    searchable=True,
                                    style={
                                        "marginRight": "10px",
                                        "margin-top": "10px",
                                        "width": "100%",
                                    },
                                )
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    id="transmission-dropdown",
                                    options=transmission,
                                    # value='ALL',
                                    placeholder="Select transmission type",
                                    searchable=True,
                                    style={
                                        "marginRight": "10px",
                                        "margin-top": "10px",
                                        "width": "100%",
                                    },
                                )
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    id="owner-dropdown",
                                    options=owner,
                                    # value='ALL',
                                    placeholder="Select owner type",
                                    searchable=True,
                                    style={
                                        "marginRight": "10px",
                                        "margin-top": "10px",
                                        "width": "100%",
                                    },
                                )
                            ),
                        ],
                    ),
                    dbc.Stack(
                        [
                            # html.Br(),
                            html.Div(
                                dcc.Input(
                                    id="engine",
                                    # options=year,
                                    # value='ALL',
                                    type="number",
                                    placeholder="Enter Engine Size in CC",
                                    style={
                                        "marginRight": "10px",
                                        "margin-top": "10px",
                                        "width": "100%",
                                        "height": 30,
                                    },
                                )
                            ),
                            # html.Br(),
                            html.Div(
                                dcc.Input(
                                    id="max_power",
                                    # options=year,
                                    # value='ALL',
                                    type="number",
                                    placeholder="Enter maximum power (bhp)",
                                    style={
                                        "marginRight": "10px",
                                        "margin-top": "10px",
                                        "width": "100%",
                                        "height": 30,
                                    },
                                )
                            ),
                        ]
                    ),
                    html.Div(
                        html.Button(
                            "Predict",
                            id="predict-val",
                            n_clicks=0,
                            style={
                                "marginRight": "10px",
                                "margin-top": "10px",
                                "width": "100%",
                                "height": 50,
                                "background-color": "white",
                                "color": "black",
                            },
                        ),
                    ),
                    html.Br(),
                    dcc.Textarea(
                        id="prediction",
                        value="This is where you can see the predicted car price after clicking the predict button",
                        style={
                            "width": "100%",
                            "height": 40,
                            "whiteSpace": "pre-line",
                            "font-size": "1.5em",
                            "textAlign": "center",
                            "color": "#503D36",
                        },
                        readOnly=True,
                    ),
                ],
            ),
            dcc.Tab(
                label="Old Model",
                children=[
                    html.H1(
                        "Car Price Prediction",
                        style={
                            "textAlign": "center",
                            "color": "#503D36",
                            "font-size": 40,
                        },
                    ),
                    html.Div(
                        dcc.Dropdown(
                            id="old-year-dropdown",
                            options=year,
                            # value='ALL',
                            placeholder="Select model year",
                            searchable=True,
                            style={
                                "marginRight": "10px",
                                "margin-top": "10px",
                                "width": 200,
                            },
                        )
                    ),
                    html.Div(
                        dcc.Dropdown(
                            id="old-fuel-dropdown",
                            options=fuel,
                            # value='ALL',
                            placeholder="Select fuel type",
                            searchable=True,
                            style={
                                "marginRight": "10px",
                                "margin-top": "10px",
                                "width": 200,
                            },
                        )
                    ),
                    html.Div(
                        dcc.Dropdown(
                            id="old-seller_type-dropdown",
                            options=seller_type,
                            # value='ALL',
                            placeholder="Select seller type",
                            searchable=True,
                            style={
                                "marginRight": "10px",
                                "margin-top": "10px",
                                "width": 200,
                            },
                        )
                    ),
                    html.Div(
                        dcc.Dropdown(
                            id="old-transmission-dropdown",
                            options=transmission,
                            # value='ALL',
                            placeholder="Select transmission type",
                            searchable=True,
                            style={
                                "marginRight": "10px",
                                "margin-top": "10px",
                                "width": 200,
                            },
                        )
                    ),
                    html.Div(
                        dcc.Dropdown(
                            id="old-owner-dropdown",
                            options=owner,
                            # value='ALL',
                            placeholder="Select owner type",
                            searchable=True,
                            style={
                                "marginRight": "10px",
                                "margin-top": "10px",
                                "width": 200,
                            },
                        )
                    ),
                    html.Br(),
                    html.Div(
                        dcc.Input(
                            id="old-engine",
                            # options=year,
                            # value='ALL',
                            type="number",
                            placeholder="Enter Engine Size in CC",
                            style={
                                "marginRight": "10px",
                                "margin-top": "10px",
                                "width": 200,
                            },
                        )
                    ),
                    html.Br(),
                    html.Div(
                        dcc.Input(
                            id="old-max_power",
                            # options=year,
                            # value='ALL',
                            type="number",
                            placeholder="Enter maximum power (bhp)",
                            style={
                                "marginRight": "10px",
                                "margin-top": "10px",
                                "width": 200,
                            },
                        )
                    ),
                    html.Br(),
                    html.Div(id="old-feature"),
                    html.Br(),
                    html.Div(
                        html.Button(
                            "Predict",
                            id="old-predict-val",
                            n_clicks=0,
                            style={
                                "marginRight": "10px",
                                "margin-top": "10px",
                                "width": 200,
                                "height": 50,
                                "background-color": "white",
                                "color": "black",
                            },
                        ),
                    ),
                    html.Br(),
                    html.Div(id="old-prediction"),
                ],
            ),
        ]
    )
)


@app.callback(
    Output(component_id="prediction", component_property="value"),
    [
        Input(component_id="predict-val", component_property="n_clicks"),
        #  Input(component_id='feature', component_property='children')
        # Input(component_id='feature', component_property='children'),
        Input(component_id="fuel-dropdown", component_property="value"),
        Input(component_id="seller_type-dropdown", component_property="value"),
        Input(component_id="transmission-dropdown", component_property="value"),
        Input(component_id="owner-dropdown", component_property="value"),
        Input(component_id="year-dropdown", component_property="value"),
        Input(component_id="engine", component_property="value"),
        Input(component_id="max_power", component_property="value"),
    ],
)
def predict(click, f, s, t, o, y, e, m):
    v = {
        "year": y,
        "fuel": f,
        "seller_type": s,
        "transmission": t,
        "owner": o,
        "engine": e,
        "max_power": m,
    }
    r = preprocess(v)
    sample = np.array([r])
    if click == 0:
        global c
        c = click
        result = "This is where you can see the predicted car price after clicking the predict button"
    elif click != c:
        c = click
        predicted_car_price = float(str(np.exp(best_model.predict(sample)[0])))
        result = f"The prediction for this car is {predicted_car_price:.2f}"
    else:
        result = "This is where you can see the predicted car price after clicking the predict button"
    return result


@app.callback(
    Output(component_id="old-feature", component_property="children"),
    [
        Input(component_id="old-fuel-dropdown", component_property="value"),
        Input(component_id="old-seller_type-dropdown", component_property="value"),
        Input(component_id="old-transmission-dropdown", component_property="value"),
        Input(component_id="old-owner-dropdown", component_property="value"),
        Input(component_id="old-year-dropdown", component_property="value"),
        Input(component_id="old-engine", component_property="value"),
        Input(component_id="old-max_power", component_property="value"),
    ],
)
def feature(f, s, t, o, y, e, m):
    return f"The feature vector is {[y, f, s,t, o, e, m]}"


@app.callback(
    Output(component_id="old-prediction", component_property="children"),
    [
        Input(component_id="old-predict-val", component_property="n_clicks"),
        Input(component_id="old-fuel-dropdown", component_property="value"),
        Input(component_id="old-seller_type-dropdown", component_property="value"),
        Input(component_id="old-transmission-dropdown", component_property="value"),
        Input(component_id="old-owner-dropdown", component_property="value"),
        Input(component_id="old-year-dropdown", component_property="value"),
        Input(component_id="old-engine", component_property="value"),
        Input(component_id="old-max_power", component_property="value"),
    ],
)
def predict(click, f, s, t, o, y, e, m):
    v = {
        "year": y,
        "fuel": f,
        "seller_type": s,
        "transmission": t,
        "owner": o,
        "engine": e,
        "max_power": m,
    }
    r = preprocess(v)
    sample = np.array([r])
    if click == 0:
        global c
        c = click
        result = ""
    elif click != c:
        c = click
        predicted_car_price = np.exp(loaded_model.predict(sample))[0]
        result = f"The prediction for this car is {predicted_car_price:.2f}"
    else:
        result = ""
    return result


# Run the app
if __name__ == "__main__":
    app.run_server()
