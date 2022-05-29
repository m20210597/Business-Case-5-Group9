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
import yfinance as yf
import datetime

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

# Main dataset
cryptocurrencies = ['ADA-USD', 'ATOM-USD', 'AVAX-USD', 'AXS-USD', 'BTC-USD', 'ETH-USD', 'LINK-USD', 'LUNA1-USD', 'MATIC-USD', 'SOL-USD']
data = yf.download(cryptocurrencies, period = '2190d', interval = '1d')

# Storing each indicator in separately dataframe
df_open = data['Open'].reset_index()
df_close = data['Close'].reset_index()
df_adj_close = data['Adj Close'].reset_index()
df_high = data['High'].reset_index()
df_low = data['Low'].reset_index()
df_volume = data['Volume'].reset_index()

## Gabriel

# Code here

## Gabriel

# ----------------------------------------------------------------------------------------------------------------------
# Dashboard Components

# Graph one
graph_one_mockup = go.Figure(px.line(df_close, x='Date', y="BTC-USD"))

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
year_slide = dcc.RangeSlider(min(df_open['Date']).year, max(df_open['Date']).year,
                             value=[min(df_open['Date']).year,max(df_open['Date']).year],
                             id='year_slide',
                             tooltip={"placement": "bottom", "always_visible": True},
                             step=1,
                             allowCross=False,
                             marks={min(df_open['Date']).year:{'label':str(min(df_open['Date']).year)}, max(df_open['Date']).year:{'label':str(max(df_open['Date']).year)}}
                             )

# Currency Dropdown One
dropdown_currency_one = dcc.Dropdown(
    id='dropdown_currency_one',
    className="dropdown",
    options=cryptocurrencies,
    value='ADA-USD',
    multi=False,
    placeholder="Select currency",
    clearable = False,
    style={"background-color" : "white",'padding':'0px 20px 0px 20px'}
)

# Currency Dropdown Two
dropdown_currency_two = dcc.Dropdown(
    id='dropdown_currency_two',
    className="dropdown",
    options=cryptocurrencies,
    value='ADA-USD',
    multi=False,
    placeholder="Select currency",
    clearable = False,
    style={"background-color" : "white",'padding':'0px 20px 0px 20px'}
)

# Dropdown menu
dropdown_year = dcc.Dropdown(
    id='year_dropdown',
    className = "dropdown",
    options=[*range(min(df_open['Date']).year, max(df_open['Date']).year+1, 1)],
    value=max(df_open['Date']).year,
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

day_range = days = datetime.timedelta(1)

# Dropdown menu for dates
dropdown_date = dcc.Dropdown(
    id='date_dropdown',
    className="dropdown",
    options=pd.to_datetime(df_open['Date']).dt.date,
    value=pd.to_datetime(max(df_open['Date'])).date()-day_range,
    multi=False,
    placeholder="Select date",
    clearable=False,
    style={"background-color" : "white",'padding':'0px 20px 0px 20px'}
)

# Dropdown menu for indicator
dropdown_indicator = dcc.Dropdown(
    id='indicator_dropdown',
    className="dropdown",
    options=['Open','Close','Adj Close','High','Low','Volume'],
    value='Close',
    multi=False,
    placeholder="Select date",
    clearable=False,
    style={"background-color" : "white",'padding':'0px 20px 0px 20px'}
)


# ----------------------------------------------------------------------------------------------------------------------
# App layout

server = app.server


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

            html.H2("Control Panel for Price Over Time", style={'text-align': 'center', 'padding':'15px'}),
            html.H4("Select the year range:", style={'text-align': 'center'}),
            year_slide,
            html.H4("Select the currency:", style={'text-align': 'center','padding':'15px 0px 0px 0px'}),
            dropdown_currency_one,
            html.H4("Select the indicator:", style={'text-align': 'center','padding':'15px 0px 0px 0px'}),
            dropdown_indicator
        ],
            style={'box-shadow':'1px 1px 3px lightgray', "background-color" : "white", 'height':'521.42px'}),
            width=2,
            style={'padding':'2px 0px 15px 15px'}),

        dbc.Col(html.Div([

            html.H2("Cryptocurrency Price Over Time", style={'text-align': 'left', 'padding':'15px 15px'}),
            dcc.Graph(id="graph_one")
        ],
            style={'box-shadow':'1px 1px 3px lightgray', "background-color" : "white"}),
            width=10,
            style={'padding':'2px 0px 15px 15px'})
    ]),

    # 3rd Row
    dbc.Row([

        dbc.Col(html.Div([

            html.H2("Control Panel for Candlestick Chart", style={'text-align': 'center', 'padding':'15px'}),
            html.H4("Select the year:", style={'text-align': 'center'}),
            dropdown_year,
            html.H4("Select month range:", style={'text-align': 'center','padding':'15px 0px 0px 0px'}),
            month_slide,
            html.H4("Select the currency:", style={'text-align': 'center','padding':'15px 0px 0px 0px'}),
            dropdown_currency_two,
            html.H4("Select a specific date:", style={'text-align': 'center','padding':'15px 0px 0px 0px'}),
            dropdown_date
        ],
            style={'box-shadow':'1px 1px 3px lightgray', "background-color" : "white", 'height':'1321.42px'}),
            width=2,
            style={'padding':'2px 0px 15px 15px'}),

        dbc.Col(html.Div([

            html.H2("Candlestick Chart", style={'text-align': 'left', 'padding':'15px 15px'}),
            dcc.Graph(id="graph_two"),
            dcc.Graph(id="table_one",style={'box-shadow': '1px 1px 3px lightgray', "background-color": "white"})
        ],
            style={'box-shadow':'1px 1px 3px lightgray', "background-color" : "white"}),
            width=10,
            style={'padding':'2px 0px 15px 15px'})
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

# 2nd Row --------------------------------------------------------------------------------------------------------------
@app.callback(
    Output(component_id='graph_one', component_property='figure'),
    [Input('year_slide', 'value'),
     Input('dropdown_currency_one', 'value'),
     Input('indicator_dropdown', 'value')]
)

def second_row(year_range, currency, indicator):

    initial_year = format(year_range)[1:5]
    final_year = format(year_range)[6:11]

    initial_date = datetime.datetime(int(initial_year), 1, 1)
    final_date = datetime.datetime(int(final_year), 12, 31)

    df_temp = data[indicator].reset_index()
    dataset = df_temp.loc[(df_temp["Date"] >= initial_date) & (df_temp["Date"] <= final_date)]

    graph_one_figure = go.Figure(px.line(dataset, x='Date', y=str(currency),
                                         title=str('Cryptocurrency: ' + currency)
                                         ))

    graph_one_figure.update_layout(
        xaxis=None,
        yaxis=dict(title='Price - USD'),
    )

    return graph_one_figure


# 3rd Row --------------------------------------------------------------------------------------------------------------
@app.callback(
    [Output(component_id='graph_two', component_property='figure'),
     Output(component_id='table_one', component_property='figure')],
    [Input('year_dropdown', 'value'),
     Input('dropdown_currency_two', 'value'),
     Input('month_slide', 'value'),
     Input('date_dropdown', 'value')]
)

def third_row(year, currency, month, date):

    step_one = format(month).replace('[','')
    step_two = step_one.replace(', ','|')
    step_three = step_two.replace(']','')
    step_four = step_three.split('|',1)
    initial_month = int(step_four[0])
    final_month = int(step_four[1])

    days = datetime.timedelta(1)

    if final_month+1>12:
        initial_date = datetime.datetime(year, initial_month, 1)
        final_date = datetime.datetime(year+1, 1, 1)-days
    else:
        initial_date = datetime.datetime(year, initial_month, 1)
        final_date = datetime.datetime(year, final_month+1, 1)-days

    open = df_open.loc[(df_open["Date"] >= initial_date) & (df_open["Date"] <= final_date)]
    high = df_high.loc[(df_high["Date"] >= initial_date) & (df_high["Date"] <= final_date)]
    low = df_low.loc[(df_low["Date"] >= initial_date) & (df_low["Date"] <= final_date)]
    close = df_close.loc[(df_close["Date"] >= initial_date) & (df_close["Date"] <= final_date)]

    graph_two_figure = go.Figure(data=[go.Candlestick(x=open['Date'],
                                                      open=open[currency],
                                                      high=high[currency],
                                                      low=low[currency],
                                                      close=close[currency],)]
                                 )

    graph_two_figure.update_layout(
        title=str('Cryptocurrency: ' + currency),
        xaxis=None,
        yaxis=dict(title='Price - USD'),
        height=800
    )


    # First Table -------------------------------------------------------

    tb_open = round(df_open[df_open['Date'] == date], 2)
    tb_close = round(df_close[df_close['Date'] == date], 2)
    tb_high = round(df_high[df_high['Date'] == date], 2)
    tb_low = round(df_low[df_low['Date'] == date], 2)
    tb_volume = round(df_volume[df_volume['Date'] == date], 0)

    tb_first_row = [tb_open['BTC-USD'], tb_open['ETH-USD'], tb_open['LUNA1-USD'],
                    tb_open['SOL-USD'], tb_open['ADA-USD'], tb_open['AVAX-USD'],
                    tb_open['MATIC-USD'], tb_open['ATOM-USD'], tb_open['LINK-USD'],
                    tb_open['AXS-USD']]

    tb_second_row = [tb_close['BTC-USD'], tb_close['ETH-USD'], tb_close['LUNA1-USD'],
                    tb_close['SOL-USD'], tb_close['ADA-USD'], tb_close['AVAX-USD'],
                    tb_close['MATIC-USD'], tb_close['ATOM-USD'], tb_close['LINK-USD'],
                    tb_close['AXS-USD']]

    tb_third_row = [tb_high['BTC-USD'], tb_high['ETH-USD'], tb_high['LUNA1-USD'],
                    tb_high['SOL-USD'], tb_high['ADA-USD'], tb_high['AVAX-USD'],
                    tb_high['MATIC-USD'], tb_high['ATOM-USD'], tb_high['LINK-USD'],
                    tb_high['AXS-USD']]

    tb_fourth_row = [tb_low['BTC-USD'], tb_low['ETH-USD'], tb_low['LUNA1-USD'],
                    tb_low['SOL-USD'], tb_low['ADA-USD'], tb_low['AVAX-USD'],
                    tb_low['MATIC-USD'], tb_low['ATOM-USD'], tb_low['LINK-USD'],
                    tb_low['AXS-USD']]

    tb_fifth_row = [np.nanmax(round(df_high['BTC-USD'],2)), np.nanmax(round(df_high['ETH-USD'],2)), np.nanmax(round(df_high['LUNA1-USD'],2)),
                    np.nanmax(round(df_high['SOL-USD'],2)), np.nanmax(round(df_high['ADA-USD'],2)), np.nanmax(round(df_high['AVAX-USD'],2)),
                    np.nanmax(round(df_high['MATIC-USD'],2)), np.nanmax(round(df_high['ATOM-USD'],2)), np.nanmax(round(df_high['LINK-USD'],2)),
                    np.nanmax(round(df_high['AXS-USD'],2))]

    tb_sixth_row = [np.nanmin(round(df_high['BTC-USD'],2)), np.nanmin(round(df_high['ETH-USD'],2)), np.nanmin(round(df_high['LUNA1-USD'],2)),
                    np.nanmin(round(df_high['SOL-USD'],2)), np.nanmin(round(df_high['ADA-USD'],2)), np.nanmin(round(df_high['AVAX-USD'],2)),
                    np.nanmin(round(df_high['MATIC-USD'],2)), np.nanmin(round(df_high['ATOM-USD'],2)), np.nanmin(round(df_high['LINK-USD'],2)),
                    np.nanmin(round(df_high['AXS-USD'],2))]

    tb_seventh_row = [tb_volume['BTC-USD'], tb_volume['ETH-USD'], tb_volume['LUNA1-USD'],
                    tb_volume['SOL-USD'], tb_volume['ADA-USD'], tb_volume['AVAX-USD'],
                    tb_volume['MATIC-USD'], tb_volume['ATOM-USD'], tb_volume['LINK-USD'],
                    tb_volume['AXS-USD']]

    table_one_figure = go.Figure(data=[go.Table(header=dict(values=['Crypto', 'Open',
                                                                'Close', 'High', 'Low',
                                                                'Historical Max', 'Historical Min', 'Volume']),
                 cells=dict(values=[['BTC', 'ETH', 'LUNA1', 'SOL', 'ADA', 'AVAX', 'MATIC', 'ATOM', 'LINK','AXS'],
                                    tb_first_row,
                                    tb_second_row,
                                    tb_third_row,
                                    tb_fourth_row,
                                    tb_fifth_row,
                                    tb_sixth_row,
                                    tb_seventh_row]))
                     ])


    return graph_two_figure, table_one_figure



# ----------------------------------------------------------------------------------------------------------------------
# Running the app
if __name__ == '__main__':
    app.run_server(debug=True)