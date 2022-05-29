from asyncio.windows_events import NULL
import dash
from dash import dcc, Dash
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import dash_bootstrap_components as dbc
import base64
from bubbly.bubbly import bubbleplot
import textwrap
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------------------------------------------
# Calling the app
app = Dash(external_stylesheets=[dbc.themes.GRID])

# ----------------------------------------------------------------------------------------------------------------------
# Main python code here (dataset reading and predictions)

# Mockup Dataset 1
mdf_one = px.data.stocks()

# Mockup Dataset 2
mdf_two = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

# ----------------------------------------------------------------------------------------------------------------------
# Dashboard Components

# Graph one mock up
graph_one_mockup = go.Figure(px.line(mdf_one, x='date', y="GOOG"))

# Graph two mock up
graph_two_mockup = go.Figure(data=[go.Candlestick(x=mdf_two['Date'],
                open=mdf_two['AAPL.Open'],
                high=mdf_two['AAPL.High'],
                low=mdf_two['AAPL.Low'],
                close=mdf_two['AAPL.Close'])])

# Table one mock up
table_one_mockup = go.Figure(data=[go.Table(header=dict(values=['Crypto', 'Open',
                                                                'Close', 'High', 'Low',
                                                                'Historical Max', 'Historical Min', 'Volume']),
                 cells=dict(values=[['BTC', 'ETH', 'LUNA1', 'SOL', 'ADA', 'AVAX', 'MATIC', 'ATOM', 'LINK','AXS'],
                                    [97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04],
                                    [97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04],
                                    [97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04],
                                    [97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04],
                                    [97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04],
                                    [97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04],
                                    [97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04]]))
                     ])

# Table two mock up
table_two_mockup = go.Figure(data=[go.Table(header=dict(values=['Cryptocurrency', 'Model', 'Prediction Day 1', 'Prediction Day 2']),
                 cells=dict(values=[['BTC', 'ETH', 'LUNA1', 'SOL', 'ADA', 'AVAX', 'MATIC', 'ATOM', 'LINK', 'AXS'],
                                    ['Random Forest', 'Support Vector Regressor', 'Linear Regressor',
                                     'Neural Network Regressor', 'ARIMA BOX JENKINS', 'XGBRegressor',
                                     'ARIMA BOX JENKINS', 'Linear Regression', 'ARIMA BOX JENKINS', 'ARIMA BOX JENKINS'],
                                    [97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04],
                                    [97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04, 97.04]]))
                     ])

# Year Slider
year_slide = dcc.RangeSlider(2017, 2022,
                             value=[2017,2022],
                             id='year_slide',
                             tooltip={"placement": "bottom", "always_visible": True},
                             step=1,
                             allowCross=False,
                             marks={2017:{'label':'2017'}, 2022:{'label':'2022'}}
                             )

# Buttons 2nd Row
button_groups_one = html.Div(
    [
        dbc.ButtonGroup(
            [dbc.Button("ADA"), dbc.Button("LINK")],
            size="lg",
            className="me-1",
            style={'padding':'10px 10px 10px 10px'}
        ),
        dbc.ButtonGroup(
            [dbc.Button("ATOM"),dbc.Button("LUNA1")],
            size="lg",
            className="me-1",
        ),
        dbc.ButtonGroup(
            [dbc.Button("AVAX"),dbc.Button("MATIC")],
            size="lg",
            className="me-1",
        ),
        dbc.ButtonGroup(
            [dbc.Button("AXS"),dbc.Button("SOL")],
            size="lg",
            className="me-1",
        ),
        dbc.ButtonGroup(
            [dbc.Button("BTC"),dbc.Button("ETH")],
            size="lg",
            className="me-1",
        )
    ]
)

# Dropdown menu
dropdown_year = dcc.Dropdown(
    id='year_dropdown',
    className = "dropdown",
    options=['2017','2018','2019','2020','2021','2022'],
    value='2012',
    multi=False,
    placeholder="Select year",
    clearable = False,
    style={"background-color" : "white",'padding':'0px 20px 0px 20px'}
)

# Month Slider
month_slide = dcc.RangeSlider(1, 12,
                             value=[1,12],
                             id='month_slide',
                             tooltip={"placement": "bottom", "always_visible": True},
                             step=1,
                             allowCross=False,
                             marks={1:{'label':'January'},12:{'label':'December'}}
                             )





# ----------------------------------------------------------------------------------------------------------------------
# App layout
app.layout = dbc.Container([

    # 1st Row - header
    dbc.Row([

        html.H1("BC5 Cryptocurrencies Dashboard",
                style={'letter-spacing': '1.5px','font-weight': 'bold','text-transform': 'uppercase', 'text-align': 'center', 'padding':'15px'}),

        html.H2("Nova IMS - Business Cases #5",
                style={'margin-bottom': '5px', 'text-align': 'right'}),

        html.H3("Celso Endres m20200739 | Gabriel Souza m20210598 | Luiz Vizeu m20210554 | Rogério Paulo m20210597",
                style={'margin-bottom': '5px','margin-top': '5px', 'text-align': 'right'})]),

    # 2nd Row
    dbc.Row([

        dbc.Col(html.Div([

            html.H2("Control Panel for Period View", style={'text-align': 'center', 'padding':'15px'}),
            html.H4("Select the year range:", style={'text-align': 'center'}),
            year_slide,
            html.H4("Select the currency:", style={'text-align': 'center','padding':'15px 0px 0px 0px'}),
            button_groups_one
        ],
            style={'box-shadow':'1px 1px 3px lightgray', "background-color" : "white", 'height':'521.42px'}),
            width=2,
            style={'padding':'2px 0px 15px 15px'}),

        dbc.Col(html.Div([

            html.H2("Period View", style={'text-align': 'left', 'padding':'15px 15px'}),
            dcc.Graph(id="graph_one",figure=graph_one_mockup)
        ],
            style={'box-shadow':'1px 1px 3px lightgray', "background-color" : "white"}),
            width=10,
            style={'padding':'2px 15px 15px 15px'})
    ]),

    # 3rd Row
    dbc.Row([

        dbc.Col(html.Div([

            html.H2("Control Panel for Month/Day View", style={'text-align': 'center', 'padding':'15px'}),
            html.H4("Select the year:", style={'text-align': 'center'}),
            dropdown_year,
            html.H4("Select month range:", style={'text-align': 'center','padding':'15px 0px 0px 0px'}),
            month_slide
        ],
            style={'box-shadow':'1px 1px 3px lightgray', "background-color" : "white", 'height':'971.42px'}),
            width=2,
            style={'padding':'2px 0px 15px 15px'}),

        dbc.Col(html.Div([

            html.H2("Month/Day View", style={'text-align': 'left', 'padding':'15px 15px'}),
            dcc.Graph(id="graph_two",figure=graph_two_mockup),
            dcc.Graph(id="table_one", figure=table_one_mockup,
                      style={'box-shadow': '1px 1px 3px lightgray', "background-color": "white"})
        ],
            style={'box-shadow':'1px 1px 3px lightgray', "background-color" : "white"}),
            width=10,
            style={'padding':'2px 15px 15px 15px'})
    ]),

    # 4th Row

    dbc.Col(html.Div([

        dbc.Row([
            html.H2("Close Price Prediction and Forecasting", style={'text-align': 'left', 'padding': '15px 15px',"background-color" : "white"}),
            dcc.Graph(id="table_two", figure=table_two_mockup,
                      style={"background-color": "white"}),

            dbc.Row([

                dbc.Col(html.Div([
                    dcc.Graph(id="temp_one", figure=graph_one_mockup),
                    dcc.Graph(id="temp_two", figure=graph_one_mockup),
                    dcc.Graph(id="temp_three", figure=graph_one_mockup),
                    dcc.Graph(id="temp_four", figure=graph_one_mockup),
                    dcc.Graph(id="temp_five", figure=graph_one_mockup)
                ],

                                 style={"background-color" : "white"}),
                        style={'padding':'2px 15px 15px 15px'}),

                dbc.Col(html.Div([
                    dcc.Graph(id="temp_six", figure=graph_one_mockup),
                    dcc.Graph(id="temp_seven", figure=graph_one_mockup),
                    dcc.Graph(id="temp_eight", figure=graph_one_mockup),
                    dcc.Graph(id="temp_nine", figure=graph_one_mockup),
                    dcc.Graph(id="temp_ten", figure=graph_one_mockup)
                ],
                                 style={"background-color" : "white"}),
                        style={'padding':'2px 15px 15px 15px'})

            ])

        ],
            style={'padding':'2px 15px 15px 15px',"background-color": "white"}
        )
],
        style={'box-shadow':'1px 1px 3px lightgray', "background-color" : "white"}),
        style={'padding':'2px 15px 15px 15px'}
    )

],

    # Container
    fluid = True,
    style = {'background-color': '#F2F2F2','font-family': 'sans-serif','color': '#606060','font-size': '14px'}

)

# ----------------------------------------------------------------------------------------------------------------------
# Callbacks

# ----------------------------------------------------------------------------------------------------------------------
# Running the app
if __name__ == '__main__':
    app.run_server(debug=True)