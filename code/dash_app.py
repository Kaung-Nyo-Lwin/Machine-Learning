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
# from dash import dash_bootstrap_components as dbc

# # Read the airline data into pandas dataframe
# df = pd.read_csv(os.path.join(os.getcwd(),"Cars.csv"))
# # max_payload = spacex_df['Payload Mass (kg)'].max()
# # min_payload = spacex_df['Payload Mass (kg)'].min()
# df.dropna(inplace=True)
# name = sorted(df['name'].str.split(" ").str[0].unique())
# fuel = sorted(df['fuel'].unique())
# seller_type = sorted(df['seller_type'].unique())
# transmission = sorted(df['transmission'].unique())
# owner = sorted(df['owner'].unique())
# engine = sorted(df['engine'].unique())
# seats = sorted(df['seats'].unique())

# import inference
# fuel = inference.fuel
# seller_type = inference.seller_type
# transmission = inference.transmission
# owner = inference.owner.keys
global EncodedLabel
EncodedLabel = {'Diesel': 0,
                'Petrol': 1,
                'Automatic': 0,
                'Manual': 1,
                'Dealer': [0,0],
                'Individual': [1,0],
                'Trustmark Dealer': [0,1],
                'First Owner' : 1,
                'Second Owner' : 2,
                'Third Owner' : 3,
                'Fourth & Above Owner' : 4
                }

owner = ['First Owner' ,
        'Second Owner' ,
        'Third Owner' ,
        'Fourth & Above Owner'
]
fuel = ['Diesel', 'Petrol']

seller_type = ['Individual', 'Dealer', 'Trustmark Dealer']

year = list(range(1886,2024,1))

transmission = ['Manual', 'Automatic']

X_tran_stat = {'year': [2013.8134256055364, 2015.0, 2017, 4.039062067605816], 
               'fuel': [0.4528719723183391, 0.0, 0, 0.4978084455072879], 
               'seller_type': [0.8898269896193771, 1.0, 1, 0.3952063053306725], 
               'transmission': [0.8690657439446366, 1.0, 1, 0.33735178726916865], 
               'owner': [1.453287197231834, 1.0, 1, 0.707741211934925], 
               'engine': [1464.542271562767, 1248.0, 1248.0, 507.5802774640794], 
               'max_power': [91.86694112627987, 82.85, 74.0, 35.92656163539812]}



loaded_model = pickle.load(open('model/car-price.model', 'rb'))

    
def preprocess(v : dict):
    r = []
    for i,j in v.items():
        # value = None
        if (j == None) and (i in ['engine','max_power']):
            value = X_tran_stat[i][1]
        elif (j == None) and i == 'seller_type' :
            value = EncodedLabel['Individual']
        elif j == None :
            value = X_tran_stat[i][2]
        else:
            if i == 'year' :
                value = j
            elif i in ['engine','max_power']:
                value = (j - X_tran_stat[i][0]) / X_tran_stat[i][3]                
            else:
                value = EncodedLabel[j]
        if i == 'seller_type':
            r.append(value[0])
            r.append(value[1])
        else:
            r.append(value)
    return r

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[html.H1('Car Price Prediction',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                # dcc.Dropdown(id='site-dropdown',...)
                                # html.Br(),
                                # html.H2('Features to input',
                                #         style={'textAlign': 'center', 'color': '#503D36',
                                #                'font-size': 30}),
                                html.Div(dcc.Dropdown(id='year-dropdown',
                                                options=year,
                                                # value='ALL',
                                                placeholder="Select model year",
                                                searchable=True,
                                                style={'marginRight': '10px', 'margin-top': '10px', 'width': 200}
                                                )),
                                
                                html.Div(dcc.Dropdown(id='fuel-dropdown',
                                                options=fuel,
                                                # value='ALL',
                                                placeholder="Select fuel type",
                                                searchable=True,
                                                style={'marginRight': '10px', 'margin-top': '10px', 'width': 200}
                                                )),                                 
                                 html.Div(dcc.Dropdown(id='seller_type-dropdown',
                                                options=seller_type,
                                                # value='ALL',
                                                placeholder="Select seller type",
                                                searchable=True,
                                                style={'marginRight': '10px', 'margin-top': '10px', 'width': 200}
                                                )),   
                                html.Div(dcc.Dropdown(id='transmission-dropdown',
                                                options=transmission,
                                                # value='ALL',
                                                placeholder="Select transmission type",
                                                searchable=True,
                                                style={'marginRight': '10px', 'margin-top': '10px', 'width': 200}
                                                )), 
                                html.Div(dcc.Dropdown(id='owner-dropdown',
                                                options=owner,
                                                # value='ALL',
                                                placeholder="Select owner type",
                                                searchable=True,
                                                style={'marginRight': '10px', 'margin-top': '10px', 'width': 200}
                                                )), 

                                html.Br(),
                                html.Div(dcc.Input(id='engine',
                                                # options=year,
                                                # value='ALL',
                                                type = 'number',
                                                placeholder="Enter Engine Size in CC",
                                                style={'marginRight': '10px', 'margin-top': '10px', 'width': 200}
                                                )),
                                
                                html.Br(),
                                html.Div(dcc.Input(id='max_power',
                                                # options=year,
                                                # value='ALL',
                                                type = 'number',
                                                placeholder="Enter maximum power (bhp)",
                                                style={'marginRight': '10px', 'margin-top': '10px', 'width': 200}
                                                )),
                                html.Br(),
                                html.Div(id='feature'),

                                html.Br(),
                                html.Div(                                        
                                        html.Button('Predict', 
                                                    id='predict-val', 
                                                    n_clicks=0,
                                                    style={'marginRight': '10px', 
                                                           'margin-top': '10px', 
                                                           'width': 200,
                                                           'height': 50,
                                                           'background-color': 'white',
                                                           'color': 'black'
                                                            }
                                                    ),                                        
                                        ),
                                
                                html.Br(),
                                html.Div(id='prediction'),
                                                   
  
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
@app.callback(Output(component_id='feature', component_property='children'),
                [
                    Input(component_id='fuel-dropdown', component_property='value'),
                    Input(component_id='seller_type-dropdown', component_property='value'),
                    Input(component_id='transmission-dropdown', component_property='value'),
                    Input(component_id='owner-dropdown', component_property='value'),
                    Input(component_id='year-dropdown', component_property='value'),
                    Input(component_id='engine', component_property='value'),
                    Input(component_id='max_power', component_property='value'),
                 
                 ]
                )

def feature(f,s,t,o,y,e,m): 
    return f"The feature vector is {[y, f, s,t, o, e, m]}"

# @app.callback(Output(component_id='predict-val', component_property='value'),
#                 [
#                     Input(component_id='fuel-dropdown', component_property='value'),
#                     Input(component_id='seller_type-dropdown', component_property='value'),
#                     Input(component_id='transmission-dropdown', component_property='value'),
#                     Input(component_id='owner-dropdown', component_property='value'),
#                     Input(component_id='year-dropdown', component_property='value'),
#                     Input(component_id='engine', component_property='value'),
#                     Input(component_id='max_power', component_property='value'),
#                 ]
#             )
# def feature(f,s,t,o,y,e,m): 
#     # f = EncodedLabel[f]
#     # s = EncodedLabel[s]
#     # t = EncodedLabel[t]
#     # o = EncodedLabel[o]
#     v = {'year' : y, 
#          'fuel': f, 
#          'seller_type' : s,
#          'transmission' : t, 
#          'owner' : o, 
#          'engine' : e, 
#          'max_power' : m
#     }
#     r = preprocess(v)
#     sample = np.array([r])
#     return f"{sample}"

# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(Output(component_id='prediction', component_property='children'),
                [Input(component_id='predict-val', component_property = 'n_clicks'),
                #  Input(component_id='feature', component_property='children')
                # Input(component_id='feature', component_property='children'),
                Input(component_id='fuel-dropdown', component_property='value'),
                Input(component_id='seller_type-dropdown', component_property='value'),
                Input(component_id='transmission-dropdown', component_property='value'),
                Input(component_id='owner-dropdown', component_property='value'),
                Input(component_id='year-dropdown', component_property='value'),
                Input(component_id='engine', component_property='value'),
                Input(component_id='max_power', component_property='value'),
                ]
                )

def predict(click,f,s,t,o,y,e,m):
    # feature = feature[23:-1].split(",")
    # feature[1] = EncodedLabel[str(feature[1])]
    # feature[2] = EncodedLabel[feature[2]]
    # # feature[3] = EncodedLabel[feature[3]]
    # # feature[4] = EncodedLabel[feature[4]] 
    # feature = feature[23:-1].split(",")
    # y, f, s,t, o, e, m = feature
    v = {'year' : y, 
         'fuel': f, 
         'seller_type' : s,
         'transmission' : t, 
         'owner' : o, 
         'engine' : e, 
         'max_power' : m
    }
    r = preprocess(v)
    sample = np.array([r])
    if click == 0:
        global c 
        c = click
        # predicted_car_price = np.exp(loaded_model.predict(sample))[0]
        result = ''
    elif click != c:
        c = click
        predicted_car_price = np.exp(loaded_model.predict(sample))[0]
        result = f"The prediction for this car is {predicted_car_price:.2f}"
    else:
        result = ''
    return result
    # return f"{click}"


# Run the app
if __name__ == '__main__':
    app.run_server()
