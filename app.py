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

#common packages
import glob
from math import ceil, pi, sqrt
import os
from itertools import product

import statsmodels.api as sm

import holidays
import itertools

#dataviz
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
sns.set()
import graphviz
import matplotlib.cm as cm

#algorithms for data preparation and preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
# !pip install ta
import ta
from ta import add_all_ta_features
from sklearn.feature_selection import RFE

#Modeling and Assessment
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error as MSE, r2_score, mean_absolute_percentage_error as MAPE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neural_network import MLPRegressor

#Time Series and Modeling
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.tools import diff
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------------------------------------------
# Calling the app
app = Dash(external_stylesheets=[dbc.themes.GRID])

# ----------------------------------------------------------------------------------------------------------------------
# Main python code here (dataset reading and predictions)

# Main dataset import
cryptocurrencies = ['ADA-USD', 'ATOM-USD', 'AVAX-USD', 'AXS-USD', 'BTC-USD', 'ETH-USD', 'LINK-USD', 'LUNA1-USD', 'MATIC-USD', 'SOL-USD']
data = yf.download(cryptocurrencies, period = '2190d', interval = '1d')

# Storing each indicator in separately dataframe
df_open = data['Open'].reset_index()
df_close = data['Close'].reset_index()
df_adj_close = data['Adj Close'].reset_index()
df_high = data['High'].reset_index()
df_low = data['Low'].reset_index()
df_volume = data['Volume'].reset_index()


# Time Series Data Preparation and Preprocessing

# Creating a list with all the currencies
list_of_currencys = df_volume.iloc[:,1:].columns.to_list()

# Creating the datasets for each currency
df = {}

for currency in list_of_currencys:
    df[currency] = pd.DataFrame()

    # retrieving open price
    df1 = df_open[['Date', currency]].copy()
    # filtering only non-null records
    df1 = df1[~df1[currency].isnull()].copy()
    # renaming column ETH-USD to open, which means the Open price for the currency
    df1.rename(columns={currency: "open"}, inplace=True)

    # retrieving close price
    df2 = df_close[['Date', currency]]
    # filtering only non-null records
    df2 = df2[~df2[currency].isnull()].copy()
    # renaming column ETH-USD to close, which means the Open price for the currency
    df2.rename(columns={currency: "close"}, inplace=True)

    # retrieving adj_close price
    df3 = df_adj_close[['Date', currency]]
    # filtering only non-null records
    df3 = df3[~df3[currency].isnull()].copy()
    # renaming column ETH-USD to adj_close, which means the adj_close price for the currency
    df3.rename(columns={currency: "adj_close"}, inplace=True)

    # retrieving highest price
    df4 = df_high[['Date', currency]]
    # filtering only non-null records
    df4 = df4[~df4[currency].isnull()].copy()
    # renaming column ETH-USD to high, which means the highest price for the currency
    df4.rename(columns={currency: "high"}, inplace=True)

    # retrieving lowest price
    df5 = df_low[['Date', currency]]
    # filtering only non-null records
    df5 = df5[~df5[currency].isnull()].copy()
    # renaming column ETH-USD to df5, which means the lowest price for the currency
    df5.rename(columns={currency: "low"}, inplace=True)

    # retrieving Volume
    df6 = df_volume[['Date', currency]]
    # filtering only non-null records
    df6 = df6[~df6[currency].isnull()].copy()
    # renaming column ETH-USD to Volume, which means the Volume for the currency
    df6.rename(columns={currency: "volume"}, inplace=True)

    name = str(currency)

    # merging dataframes into a single dataframe
    temp_2 = pd.merge(df1, df2, left_on='Date', right_on='Date', how='left')
    temp_3 = pd.merge(temp_2, df3, left_on='Date', right_on='Date', how='left')
    temp_4 = pd.merge(temp_3, df4, left_on='Date', right_on='Date', how='left')
    temp_5 = pd.merge(temp_4, df5, left_on='Date', right_on='Date', how='left')
    temp_6 = pd.merge(temp_5, df6, left_on='Date', right_on='Date', how='left')
    df[currency] = temp_6.copy()
    df[currency]['Date'] = pd.to_datetime(df[currency]['Date'])
    df[currency]['volume'] = df[currency]['volume'].astype('Int64')


# Time Series Model and Assessment

# ADA-USD
##creating a df to predict the crypto currency
dfada = df['ADA-USD'].copy()

# Creating a new feature for better representing day-wise values
dfada['mean'] = (dfada['low'] + dfada['high'])/2

# Cleaning the data for any NaN or Null fields
dfada = dfada.dropna()

# Creating a copy for applying shift
dataset_for_prediction = dfada.copy()
dataset_for_prediction['Actual']=dataset_for_prediction['close'].shift()
dataset_for_prediction=dataset_for_prediction.dropna()

# date time typecast
dataset_for_prediction['Date'] =pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index= dataset_for_prediction['Date']

# normalizing the exogeneous variables
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['volume']])  #['low', 'high', 'open', 'adj_close', 'volume', 'mean']
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X=scaled_input
X.rename(columns={0:'Volume'}, inplace=True)

# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output =pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y=scaler_output
y.rename(columns={0:'Observed Data'}, inplace= True)
y.index=dataset_for_prediction.index

# train-test split (cannot shuffle in case of time series)
train_X, train_y = X[:-7].dropna(), y[:-7].dropna()
test_X, test_y = X[-9:].dropna(), y[-8:].dropna()

# Init the best SARIMAX model
model = SARIMAX(
    train_y,
    exog=train_X,
    order=(1,1,1),
    seasonal_order =(2, 1, 0, 6)
)

# training the model
results = model.fit(disp=0)

# get predictions
predictions = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)

# setting up for plots
act = pd.DataFrame(scaler_output[-30:])
predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
predictions['Actual'] = act['Observed Data']
predictions.rename(columns={0:'Pred', 'predicted_mean':'Pred'}, inplace=True)

# post-processing inverting normalization
testPredict = sc_out.inverse_transform(predictions[['Pred']])
testActual = sc_out.inverse_transform(predictions[['Actual']])

# Out of sample forecast
pred = results.get_prediction(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)
pred_ci = pred.conf_int()

# print RMSE and MAPE
ada_rmse = sqrt(MSE(testActual, testPredict))
ada_mape = MAPE(testActual, testPredict)

#forecast
fcst = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X).to_frame()
fcst2 = sc_out.inverse_transform(fcst)
#storing the predictions in a dataframe
ada_predictions = pd.DataFrame(fcst2, index = fcst.index, columns = ['price'])


# ATOM-USD

#creating a df to predict the crypto currency
dfatom = df['ATOM-USD'].copy()

# Creating a new feature for better representing day-wise values
dfatom['mean'] = (dfatom['low'] + dfatom['high'])/2

# Cleaning the data for any NaN or Null fields
dfatom = dfatom.dropna()

# Creating a copy for applying shift
dataset_for_prediction = dfatom.copy()
dataset_for_prediction['Actual']=dataset_for_prediction['close'].shift()
dataset_for_prediction=dataset_for_prediction.dropna()

# date time typecast
dataset_for_prediction['Date'] =pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index= dataset_for_prediction['Date']

# normalizing the exogeneous variables
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['volume']])  #['low', 'high', 'open', 'adj_close', 'volume', 'mean']
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X=scaled_input
X.rename(columns={0:'Volume'}, inplace=True)

# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output =pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y=scaler_output
y.rename(columns={0:'Observed Data'}, inplace= True)
y.index=dataset_for_prediction.index

# train-test split (cannot shuffle in case of time series)
train_X, train_y = X[:-7].dropna(), y[:-7].dropna()
test_X, test_y = X[-9:].dropna(), y[-8:].dropna()

# Init the best SARIMAX model
model = SARIMAX(
    train_y,
    exog=train_X,
    order=(2,1,0),
    seasonal_order =(2, 1, 0, 6)
)

# training the model
results = model.fit(disp=0)

# get predictions
predictions = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)

# setting up for plots
act = pd.DataFrame(scaler_output[-30:])
predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
predictions['Actual'] = act['Observed Data']
predictions.rename(columns={0:'Pred', 'predicted_mean':'Pred'}, inplace=True)

# post-processing inverting normalization
testPredict = sc_out.inverse_transform(predictions[['Pred']])
testActual = sc_out.inverse_transform(predictions[['Actual']])

# Out of sample forecast
pred = results.get_prediction(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)
pred_ci = pred.conf_int()

#limits the predictions to zero if it is a negative output
testPredict = testPredict.clip(min=0)

# print RMSE and MAPE
atom_rmse = sqrt(MSE(testActual, testPredict))
atom_mape = MAPE(testActual, testPredict)

#forecast
fcst = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X).to_frame()
fcst2 = sc_out.inverse_transform(fcst)
#storing the predictions in a dataframe
atom_predictions = pd.DataFrame(fcst2, index = fcst.index, columns = ['price'])
atom_predictions.tail(3)


#AVAX-USD

#creating a df to predict the crypto currency
dfavax = df['AVAX-USD'].copy()

# Creating a new feature for better representing day-wise values
dfavax['mean'] = (dfavax['low'] + dfavax['high'])/2
dfavax = dfavax[2:].copy()

# Cleaning the data for any NaN or Null fields
dfavax = dfavax.dropna()

# Creating a copy for applying shift
dataset_for_prediction = dfavax.copy()
dataset_for_prediction['Actual']=dataset_for_prediction['close'].shift()
dataset_for_prediction=dataset_for_prediction.dropna()

# date time typecast
dataset_for_prediction['Date'] =pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index= dataset_for_prediction['Date']

# normalizing the exogeneous variables
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['volume']])  #['low', 'high', 'open', 'adj_close', 'volume', 'mean']
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X=scaled_input
X.rename(columns={0:'Volume'}, inplace=True)

# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output =pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y=scaler_output
y.rename(columns={0:'Observed Data'}, inplace= True)
y.index=dataset_for_prediction.index

# train-test split (cannot shuffle in case of time series)
train_X, train_y = X[:-7].dropna(), y[:-7].dropna()
test_X, test_y = X[-9:].dropna(), y[-8:].dropna()

# Init the best SARIMAX model
model = SARIMAX(
    train_y,
    exog=train_X,
    order=(1,1,0),
    seasonal_order =(2, 1, 0, 6)
)

# training the model
results = model.fit(disp=0)

# get predictions
predictions = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)

# setting up for plots
act = pd.DataFrame(scaler_output[-30:])
predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
predictions['Actual'] = act['Observed Data']
predictions.rename(columns={0:'Pred', 'predicted_mean':'Pred'}, inplace=True)

# post-processing inverting normalization
testPredict = sc_out.inverse_transform(predictions[['Pred']])
testActual = sc_out.inverse_transform(predictions[['Actual']])

# Out of sample forecast
pred = results.get_prediction(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)
pred_ci = pred.conf_int()

#limits the predictions to zero if it is a negative output
testPredict = testPredict.clip(min=0)

# print RMSE and MAPE
avax_rmse = sqrt(MSE(testActual, testPredict))
avax_mape = MAPE(testActual, testPredict)

#forecast
fcst = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X).to_frame()
fcst2 = sc_out.inverse_transform(fcst)
#storing the predictions in a dataframe
avax_predictions = pd.DataFrame(fcst2, index = fcst.index, columns = ['price'])
avax_predictions.tail(3)


# AXS-USD
#creating a df to predict the crypto currency
dfaxs = df['AXS-USD'].copy()

# Creating a new feature for better representing day-wise values
dfaxs['mean'] = (dfaxs['low'] + dfaxs['high'])/2
dfaxs.head(3)

# Cleaning the data for any NaN or Null fields
dfaxs = dfaxs.dropna()

# Creating a copy for applying shift
dataset_for_prediction = dfaxs.copy()
dataset_for_prediction['Actual']=dataset_for_prediction['close'].shift()
dataset_for_prediction=dataset_for_prediction.dropna()

# date time typecast
dataset_for_prediction['Date'] =pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index= dataset_for_prediction['Date']

# normalizing the exogeneous variables
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['volume']])  #['low', 'high', 'open', 'adj_close', 'volume', 'mean']
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X=scaled_input
X.rename(columns={0:'Volume'}, inplace=True)

# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output =pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y=scaler_output
y.rename(columns={0:'Observed Data'}, inplace= True)
y.index=dataset_for_prediction.index

# train-test split (cannot shuffle in case of time series)
train_X, train_y = X[:-7].dropna(), y[:-7].dropna()
test_X, test_y = X[-9:].dropna(), y[-8:].dropna()

# Init the best SARIMAX model
model = SARIMAX(
    train_y,
    exog=train_X,
    order=(0,1,0),
    seasonal_order =(2, 1, 0, 6)
)

# training the model
results = model.fit(disp=0)

# get predictions
predictions = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)

# setting up for plots
act = pd.DataFrame(scaler_output[-30:])
predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
predictions['Actual'] = act['Observed Data']
predictions.rename(columns={0:'Pred', 'predicted_mean':'Pred'}, inplace=True)

# post-processing inverting normalization
testPredict = sc_out.inverse_transform(predictions[['Pred']])
testActual = sc_out.inverse_transform(predictions[['Actual']])

# Out of sample forecast
pred = results.get_prediction(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)
pred_ci = pred.conf_int()

#limits the predictions to zero if it is a negative output
testPredict = testPredict.clip(min=0)

# print RMSE and MAPE
axs_rmse = sqrt(MSE(testActual, testPredict))
axs_mape = MAPE(testActual, testPredict)

#forecast
fcst = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X).to_frame()
fcst2 = sc_out.inverse_transform(fcst)
#storing the predictions in a dataframe
axs_predictions = pd.DataFrame(fcst2, index = fcst.index, columns = ['price'])
axs_predictions.tail(3)


# BTC-USD
##creating a df to predict the crypto currency
dfbtc = df['BTC-USD'].copy()

# Creating a new feature for better representing day-wise values
dfbtc['mean'] = (dfbtc['low'] + dfbtc['high'])/2

# Cleaning the data for any NaN or Null fields
dfbtc = dfbtc.dropna()

# Creating a copy for applying shift
dataset_for_prediction = dfbtc.copy()
dataset_for_prediction['Actual']=dataset_for_prediction['close'].shift()
dataset_for_prediction=dataset_for_prediction.dropna()

# date time typecast
dataset_for_prediction['Date'] =pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index= dataset_for_prediction['Date']

# normalizing the exogeneous variables
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['volume']])  #['low', 'high', 'open', 'adj_close', 'volume', 'mean']
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X=scaled_input
X.rename(columns={0:'Volume'}, inplace=True)

# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output =pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y=scaler_output
y.rename(columns={0:'Observed Data'}, inplace= True)
y.index=dataset_for_prediction.index

# train-test split (cannot shuffle in case of time series)
train_X, train_y = X[:-7].dropna(), y[:-7].dropna()
test_X, test_y = X[-9:].dropna(), y[-8:].dropna()

# Init the best SARIMAX model
model = SARIMAX(
    train_y,
    exog=train_X,
    order=(0,1,0),
    seasonal_order =(2, 1, 0, 6)
)

# training the model
results = model.fit(disp=0)

# get predictions
predictions = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)

# setting up for plots
act = pd.DataFrame(scaler_output[-30:])
predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
predictions['Actual'] = act['Observed Data']
predictions.rename(columns={0:'Pred', 'predicted_mean':'Pred'}, inplace=True)

# post-processing inverting normalization
testPredict = sc_out.inverse_transform(predictions[['Pred']])
testActual = sc_out.inverse_transform(predictions[['Actual']])

# Out of sample forecast
pred = results.get_prediction(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)
pred_ci = pred.conf_int()

#limits the predictions to zero if it is a negative output
testPredict = testPredict.clip(min=0)

# print RMSE and MAPE
btc_rmse = sqrt(MSE(testActual, testPredict))
btc_mape = MAPE(testActual, testPredict)

#forecast
fcst = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X).to_frame()
fcst2 = sc_out.inverse_transform(fcst)
#storing the predictions in a dataframe
btc_predictions = pd.DataFrame(fcst2, index = fcst.index, columns = ['price'])


# ETH_USD
#creating a df to predict the crypto currency
dfeth = df['ETH-USD'].copy()

# Creating a new feature for better representing day-wise values
dfeth['mean'] = (dfeth['low'] + dfeth['high'])/2

# Cleaning the data for any NaN or Null fields
dfeth = dfeth.dropna()

# Creating a copy for applying shift
dataset_for_prediction = dfeth.copy()
dataset_for_prediction['Actual']=dataset_for_prediction['close'].shift()
dataset_for_prediction=dataset_for_prediction.dropna()

# date time typecast
dataset_for_prediction['Date'] =pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index= dataset_for_prediction['Date']

# normalizing the exogeneous variables
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['volume']])  #['low', 'high', 'open', 'adj_close', 'volume', 'mean']
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X=scaled_input
X.rename(columns={0:'Volume'}, inplace=True)

# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output =pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y=scaler_output
y.rename(columns={0:'Observed Data'}, inplace= True)
y.index=dataset_for_prediction.index

# train-test split (cannot shuffle in case of time series)
train_X, train_y = X[:-7].dropna(), y[:-7].dropna()
test_X, test_y = X[-9:].dropna(), y[-8:].dropna()

# Init the best SARIMAX model
model = SARIMAX(
    train_y,
    exog=train_X,
    order=(1,1,0),
    seasonal_order =(2, 1, 0, 6)
)

# training the model
results = model.fit(disp=0)

# get predictions
predictions = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)

# setting up for plots
act = pd.DataFrame(scaler_output[-30:])
predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
predictions['Actual'] = act['Observed Data']
predictions.rename(columns={0:'Pred', 'predicted_mean':'Pred'}, inplace=True)

# post-processing inverting normalization
testPredict = sc_out.inverse_transform(predictions[['Pred']])
testActual = sc_out.inverse_transform(predictions[['Actual']])

# Out of sample forecast
pred = results.get_prediction(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)
pred_ci = pred.conf_int()

#limits the predictions to zero if it is a negative output
testPredict = testPredict.clip(min=0)

# print RMSE and MAPE
eth_rmse = sqrt(MSE(testActual, testPredict))
eth_mape = MAPE(testActual, testPredict)

#forecast
fcst = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X).to_frame()
fcst2 = sc_out.inverse_transform(fcst)
#storing the predictions in a dataframe
eth_predictions = pd.DataFrame(fcst2, index = fcst.index, columns = ['price'])


# LINK-USD
#creating a df to predict the crypto currency
dflink = df['LINK-USD'].copy()

# Creating a new feature for better representing day-wise values
dflink['mean'] = (dflink['low'] + dflink['high'])/2

# Cleaning the data for any NaN or Null fields
dflink = dflink.dropna()

# Creating a copy for applying shift
dataset_for_prediction = dflink.copy()
dataset_for_prediction['Actual']=dataset_for_prediction['close'].shift()
dataset_for_prediction=dataset_for_prediction.dropna()

# date time typecast
dataset_for_prediction['Date'] =pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index= dataset_for_prediction['Date']

# normalizing the exogeneous variables
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['volume']])  #['low', 'high', 'open', 'adj_close', 'volume', 'mean']
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X=scaled_input
X.rename(columns={0:'Volume'}, inplace=True)

# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output =pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y=scaler_output
y.rename(columns={0:'Observed Data'}, inplace= True)
y.index=dataset_for_prediction.index

# train-test split (cannot shuffle in case of time series)
train_X, train_y = X[:-7].dropna(), y[:-7].dropna()
test_X, test_y = X[-9:].dropna(), y[-8:].dropna()

# Init the best SARIMAX model
model = SARIMAX(
    train_y,
    exog=train_X,
    order=(1,1,1),
    seasonal_order =(2, 1, 0, 6)
)

# training the model
results = model.fit(disp=0)

# get predictions
predictions = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)

# setting up for plots
act = pd.DataFrame(scaler_output[-30:])
predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
predictions['Actual'] = act['Observed Data']
predictions.rename(columns={0:'Pred', 'predicted_mean':'Pred'}, inplace=True)

# post-processing inverting normalization
testPredict = sc_out.inverse_transform(predictions[['Pred']])
testActual = sc_out.inverse_transform(predictions[['Actual']])

# Out of sample forecast
pred = results.get_prediction(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)
pred_ci = pred.conf_int()

#limits the predictions to zero if it is a negative output
testPredict = testPredict.clip(min=0)

# print RMSE and MAPE
link_rmse = sqrt(MSE(testActual, testPredict))
link_mape = MAPE(testActual, testPredict)

#forecast
fcst = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X).to_frame()
fcst2 = sc_out.inverse_transform(fcst)
#storing the predictions in a dataframe
link_predictions = pd.DataFrame(fcst2, index = fcst.index, columns = ['price'])

# LUNA1-USD
#creating a df to predict the crypto currency
dfluna1 = df['LUNA1-USD'].copy()

# Creating a new feature for better representing day-wise values
dfluna1['mean'] = (dfluna1['low'] + dfluna1['high'])/2

# Cleaning the data for any NaN or Null fields
dfluna1 = dfluna1.dropna()

# Creating a copy for applying shift
dataset_for_prediction = dfluna1.copy()
dataset_for_prediction['Actual']=dataset_for_prediction['close'].shift()
dataset_for_prediction=dataset_for_prediction.dropna()

# date time typecast
dataset_for_prediction['Date'] =pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index= dataset_for_prediction['Date']

# normalizing the exogeneous variables
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['volume']])  #['low', 'high', 'open', 'adj_close', 'volume', 'mean']
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X=scaled_input
X.rename(columns={0:'Volume'}, inplace=True)

# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output =pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y=scaler_output
y.rename(columns={0:'Observed Data'}, inplace= True)
y.index=dataset_for_prediction.index

# train-test split (cannot shuffle in case of time series)
train_X, train_y = X[:-7].dropna(), y[:-7].dropna()
test_X, test_y = X[-9:].dropna(), y[-8:].dropna()

# Init the best SARIMAX model
model = SARIMAX(
    train_y,
    exog=train_X,
    order=(0,1,0),
    seasonal_order =(3, 1, 0, 6)
)

# training the model
results = model.fit(disp=0)

# get predictions
predictions = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)

# setting up for plots
act = pd.DataFrame(scaler_output[-30:])
predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
predictions['Actual'] = act['Observed Data']
predictions.rename(columns={0:'Pred', 'predicted_mean':'Pred'}, inplace=True)

# post-processing inverting normalization
testPredict = sc_out.inverse_transform(predictions[['Pred']])
testActual = sc_out.inverse_transform(predictions[['Actual']])

# Out of sample forecast
pred = results.get_prediction(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)
pred_ci = pred.conf_int()

#limits the predictions to zero if it is a negative output
testPredict = testPredict.clip(min=0)

# print RMSE and MAPE
luna1_rmse = sqrt(MSE(testActual, testPredict))
luna1_mape = MAPE(testActual, testPredict)

#forecast
fcst = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X).to_frame()
fcst2 = sc_out.inverse_transform(fcst)
#storing the predictions in a dataframe
luna1_predictions = pd.DataFrame(fcst2, index = fcst.index, columns = ['price'])


#MATIC-USD
#creating a df to predict the crypto currency
dfmatic = df['MATIC-USD'].copy()

# Creating a new feature for better representing day-wise values
dfmatic['mean'] = (dfmatic['low'] + dfmatic['high'])/2

# Cleaning the data for any NaN or Null fields
dfmatic = dfmatic.dropna()

# Creating a copy for applying shift
dataset_for_prediction = dfmatic.copy()
dataset_for_prediction['Actual']=dataset_for_prediction['close'].shift()
dataset_for_prediction=dataset_for_prediction.dropna()

# date time typecast
dataset_for_prediction['Date'] =pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index= dataset_for_prediction['Date']

# normalizing the exogeneous variables
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['volume']])  #['low', 'high', 'open', 'adj_close', 'volume', 'mean']
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X=scaled_input
X.rename(columns={0:'Volume'}, inplace=True)

# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output =pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y=scaler_output
y.rename(columns={0:'Observed Data'}, inplace= True)
y.index=dataset_for_prediction.index

# train-test split (cannot shuffle in case of time series)
train_X, train_y = X[:-7].dropna(), y[:-7].dropna()
test_X, test_y = X[-9:].dropna(), y[-8:].dropna()

# Init the best SARIMAX model
model = SARIMAX(
    train_y,
    exog=train_X,
    order=(2,1,2),
    seasonal_order =(2, 1, 0, 6)
)

# training the model
results = model.fit(disp=0)

# get predictions
predictions = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)

# setting up for plots
act = pd.DataFrame(scaler_output[-30:])
predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
predictions['Actual'] = act['Observed Data']
predictions.rename(columns={0:'Pred', 'predicted_mean':'Pred'}, inplace=True)

# post-processing inverting normalization
testPredict = sc_out.inverse_transform(predictions[['Pred']])
testActual = sc_out.inverse_transform(predictions[['Actual']])

# Out of sample forecast
pred = results.get_prediction(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)
pred_ci = pred.conf_int()

#limits the predictions to zero if it is a negative output
testPredict = testPredict.clip(min=0)

# print RMSE and MAPE
matic_rmse = sqrt(MSE(testActual, testPredict))
matic_mape = MAPE(testActual, testPredict)

#forecast
fcst = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X).to_frame()
fcst2 = sc_out.inverse_transform(fcst)
#storing the predictions in a dataframe
matic_predictions = pd.DataFrame(fcst2, index = fcst.index, columns = ['price'])


# SOL-USD
#creating a df to predict the crypto currency
dfsol = df['SOL-USD'].copy()

# Creating a new feature for better representing day-wise values
dfsol['mean'] = (dfsol['low'] + dfsol['high'])/2

# Cleaning the data for any NaN or Null fields
dfsol = dfsol.dropna()

# Creating a copy for applying shift
dataset_for_prediction = dfsol.copy()
dataset_for_prediction['Actual']=dataset_for_prediction['close'].shift()
dataset_for_prediction=dataset_for_prediction.dropna()

# date time typecast
dataset_for_prediction['Date'] =pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index= dataset_for_prediction['Date']

# normalizing the exogeneous variables
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['volume']])  #['low', 'high', 'open', 'adj_close', 'volume', 'mean']
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X=scaled_input
X.rename(columns={0:'Volume'}, inplace=True)

# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output =pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y=scaler_output
y.rename(columns={0:'Observed Data'}, inplace= True)
y.index=dataset_for_prediction.index

# train-test split (cannot shuffle in case of time series)
train_X, train_y = X[:-7].dropna(), y[:-7].dropna()
test_X, test_y = X[-9:].dropna(), y[-8:].dropna()

# Init the best SARIMAX model
model = SARIMAX(
    train_y,
    exog=train_X,
    order=(0,1,0),
    seasonal_order =(2, 1, 0, 6)
)

# training the model
results = model.fit(disp=0)

# get predictions
predictions = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)

# setting up for plots
act = pd.DataFrame(scaler_output[-30:])
predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
predictions['Actual'] = act['Observed Data']
predictions.rename(columns={0:'Pred', 'predicted_mean':'Pred'}, inplace=True)

# post-processing inverting normalization
testPredict = sc_out.inverse_transform(predictions[['Pred']])
testActual = sc_out.inverse_transform(predictions[['Actual']])

# Out of sample forecast
pred = results.get_prediction(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X)
pred_ci = pred.conf_int()

#limits the predictions to zero if it is a negative output
testPredict = testPredict.clip(min=0)

# print RMSE and MAPE
sol_rmse = sqrt(MSE(testActual, testPredict))
sol_mape = MAPE(testActual, testPredict)

#forecast
fcst = results.predict(start= len(train_y), end= len(train_y)+len(test_y), exog = test_X).to_frame()
fcst2 = sc_out.inverse_transform(fcst)
#storing the predictions in a dataframe
sol_predictions = pd.DataFrame(fcst2, index = fcst.index, columns = ['price'])

# Time Series Predictions Summary
# #creating dataframes for each currency to summarize the predictions

ada_predictions.index.name = 'Date'
ada = ada_predictions.rename(columns={'price': 'ADA-USD'}).reset_index()

atom_predictions.index.name = 'Date'
atom = atom_predictions.rename(columns={'price': 'ATOM-USD'}).reset_index()

avax_predictions.index.name = 'Date'
avax = avax_predictions.rename(columns={'price': 'AVAX-USD'}).reset_index()

axs_predictions.index.name = 'Date'
axs = axs_predictions.rename(columns={'price': 'AXS-USD'}).reset_index()

btc_predictions.index.name = 'Date'
btc = btc_predictions.rename(columns={'price': 'BTC-USD'}).reset_index()

eth_predictions.index.name = 'Date'
eth = eth_predictions.rename(columns={'price': 'ETH-USD'}).reset_index()

link_predictions.index.name = 'Date'
link = link_predictions.rename(columns={'price': 'LINK-USD'}).reset_index()

luna1_predictions.index.name = 'Date'
luna1 = luna1_predictions.rename(columns={'price': 'LUNA1-USD'}).reset_index()

matic_predictions.index.name = 'Date'
matic = matic_predictions.rename(columns={'price': 'MATIC-USD'}).reset_index()

sol_predictions.index.name = 'Date'
sol = sol_predictions.rename(columns={'price': 'SOL-USD'}).reset_index()

temp_2 = pd.merge(ada, atom, left_on='Date', right_on='Date', how='inner')
temp_3 = pd.merge(temp_2, avax, left_on='Date', right_on='Date', how='inner')
temp_4 = pd.merge(temp_3, axs, left_on='Date', right_on='Date', how='inner')
temp_5 = pd.merge(temp_4, btc, left_on='Date', right_on='Date', how='inner')
temp_6 = pd.merge(temp_5, eth, left_on='Date', right_on='Date', how='inner')
temp_7 = pd.merge(temp_6, link, left_on='Date', right_on='Date', how='inner')
temp_8 = pd.merge(temp_7, luna1, left_on='Date', right_on='Date', how='inner')
temp_9 = pd.merge(temp_8, matic, left_on='Date', right_on='Date', how='inner')
final_predictions = pd.merge(temp_9, sol, left_on='Date', right_on='Date', how='inner')

df_pred_final = final_predictions[-2:].copy()

df_val_final = final_predictions[:-2].copy()



# ----------------------------------------------------------------------------------------------------------------------
# Dashboard Components

# Table two mock up
table_two_mockup = go.Figure(data=[go.Table(header=dict(values=['Cryptocurrency', 'Model', 'Prediction Day 1', 'Prediction Day 2']),
                 cells=dict(values=[['BTC', 'ETH', 'LUNA1', 'SOL', 'ADA', 'AVAX', 'MATIC', 'ATOM', 'LINK', 'AXS'],
                                    ['Random Forest', 'Support Vector Regressor', 'Linear Regressor',
                                     'Neural Network Regressor', 'ARIMA BOX JENKINS', 'XGBRegressor',
                                     'ARIMA BOX JENKINS', 'Linear Regression', 'ARIMA BOX JENKINS', 'ARIMA BOX JENKINS'],
                                    [round(df_pred_final['BTC-USD'].iloc[0],2), round(df_pred_final['ETH-USD'].iloc[0],2),
                                     round(df_pred_final['LUNA1-USD'].iloc[0],2), round(df_pred_final['SOL-USD'].iloc[0],2),
                                     round(df_pred_final['ADA-USD'].iloc[0],2), round(df_pred_final['AVAX-USD'].iloc[0],2),
                                     round(df_pred_final['MATIC-USD'].iloc[0],2), round(df_pred_final['ATOM-USD'].iloc[0],2),
                                     round(df_pred_final['LINK-USD'].iloc[0],2), round(df_pred_final['AXS-USD'].iloc[0],2)],
                                    [round(df_pred_final['BTC-USD'].iloc[1],2), round(df_pred_final['ETH-USD'].iloc[1],2),
                                     round(df_pred_final['LUNA1-USD'].iloc[1],2), round(df_pred_final['SOL-USD'].iloc[1],2),
                                     round(df_pred_final['ADA-USD'].iloc[1],2), round(df_pred_final['AVAX-USD'].iloc[1],2),
                                     round(df_pred_final['MATIC-USD'].iloc[1],2), round(df_pred_final['ATOM-USD'].iloc[1],2),
                                     round(df_pred_final['LINK-USD'].iloc[1],2), round(df_pred_final['AXS-USD'].iloc[1],2)]]))
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
    value='BTC-USD',
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
    value='BTC-USD',
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

# Setting the Dataset for the prediction graphs
# i_date = datetime.datetime(max(df_close['Date']).year-1, max(df_close['Date']).month, 1)
day_range_two = datetime.timedelta(90)
i_date = max(df_open['Date'])-day_range_two
f_date = max(df_close['Date'])
df_close_pred = df_close.loc[(df_close["Date"] >= i_date) & (df_close["Date"] <= f_date)]

# Prediction graph one - BTC-USD
pred_graph_btc = go.Figure(px.line(df_close_pred, x='Date', y='BTC-USD',title='Cryptocurrency: BTC-USD'))
pred_graph_btc.update_layout(xaxis=None,yaxis=dict(title='Price - USD'))
pred_graph_btc.add_trace(go.Scatter(x=df_val_final['Date'], y=df_val_final['BTC-USD'], mode='lines', name='Validation'))
pred_graph_btc.add_trace(go.Scatter(x=df_pred_final['Date'], y=df_pred_final['BTC-USD'], mode='lines', name='Prediction'))

# Prediction graph one - ETH-USD
pred_graph_eth = go.Figure(px.line(df_close_pred, x='Date', y='ETH-USD',title='Cryptocurrency: ETH-USD'))
pred_graph_eth.update_layout(xaxis=None,yaxis=dict(title='Price - USD'))
pred_graph_eth.add_trace(go.Scatter(x=df_val_final['Date'], y=df_val_final['ETH-USD'], mode='lines', name='Validation'))
pred_graph_eth.add_trace(go.Scatter(x=df_pred_final['Date'], y=df_pred_final['ETH-USD'], mode='lines', name='Prediction'))

# Prediction graph one - LUNA1-USD
pred_graph_luna = go.Figure(px.line(df_close_pred, x='Date', y='LUNA1-USD',title='Cryptocurrency: LUNA1-USD'))
pred_graph_luna.update_layout(xaxis=None,yaxis=dict(title='Price - USD'))
pred_graph_luna.add_trace(go.Scatter(x=df_val_final['Date'], y=df_val_final['LUNA1-USD'], mode='lines', name='Validation'))
pred_graph_luna.add_trace(go.Scatter(x=df_pred_final['Date'], y=df_pred_final['LUNA1-USD'], mode='lines', name='Prediction'))

# Prediction graph one - SOL-USD
pred_graph_sol = go.Figure(px.line(df_close_pred, x='Date', y='SOL-USD',title='Cryptocurrency: SOL-USD'))
pred_graph_sol.update_layout(xaxis=None,yaxis=dict(title='Price - USD'))
pred_graph_sol.add_trace(go.Scatter(x=df_val_final['Date'], y=df_val_final['SOL-USD'], mode='lines', name='Validation'))
pred_graph_sol.add_trace(go.Scatter(x=df_pred_final['Date'], y=df_pred_final['SOL-USD'], mode='lines', name='Prediction'))

# Prediction graph one - ADA-USD
pred_graph_ada = go.Figure(px.line(df_close_pred, x='Date', y='ADA-USD',title='Cryptocurrency: ADA-USD'))
pred_graph_ada.update_layout(xaxis=None,yaxis=dict(title='Price - USD'))
pred_graph_ada.add_trace(go.Scatter(x=df_val_final['Date'], y=df_val_final['ADA-USD'], mode='lines', name='Validation'))
pred_graph_ada.add_trace(go.Scatter(x=df_pred_final['Date'], y=df_pred_final['ADA-USD'], mode='lines', name='Prediction'))

# Prediction graph one - AVAX-USD
pred_graph_avax = go.Figure(px.line(df_close_pred, x='Date', y='AVAX-USD',title='Cryptocurrency: AVAX-USD'))
pred_graph_avax.update_layout(xaxis=None,yaxis=dict(title='Price - USD'))
pred_graph_avax.add_trace(go.Scatter(x=df_val_final['Date'], y=df_val_final['AVAX-USD'], mode='lines', name='Validation'))
pred_graph_avax.add_trace(go.Scatter(x=df_pred_final['Date'], y=df_pred_final['AVAX-USD'], mode='lines', name='Prediction'))

# Prediction graph one - MATIC-USD
pred_graph_matic = go.Figure(px.line(df_close_pred, x='Date', y='MATIC-USD',title='Cryptocurrency: MATIC-USD'))
pred_graph_matic.update_layout(xaxis=None,yaxis=dict(title='Price - USD'))
pred_graph_matic.add_trace(go.Scatter(x=df_val_final['Date'], y=df_val_final['MATIC-USD'], mode='lines', name='Validation'))
pred_graph_matic.add_trace(go.Scatter(x=df_pred_final['Date'], y=df_pred_final['MATIC-USD'], mode='lines', name='Prediction'))

# Prediction graph one - ATOM-USD
pred_graph_atom = go.Figure(px.line(df_close_pred, x='Date', y='ATOM-USD',title='Cryptocurrency: ATOM-USD'))
pred_graph_atom.update_layout(xaxis=None,yaxis=dict(title='Price - USD'))
pred_graph_atom.add_trace(go.Scatter(x=df_val_final['Date'], y=df_val_final['ATOM-USD'], mode='lines', name='Validation'))
pred_graph_atom.add_trace(go.Scatter(x=df_pred_final['Date'], y=df_pred_final['ATOM-USD'], mode='lines', name='Prediction'))

# Prediction graph one - LINK-USD
pred_graph_link = go.Figure(px.line(df_close_pred, x='Date', y='LINK-USD',title='Cryptocurrency: LINK-USD'))
pred_graph_link.update_layout(xaxis=None,yaxis=dict(title='Price - USD'))
pred_graph_link.add_trace(go.Scatter(x=df_val_final['Date'], y=df_val_final['LINK-USD'], mode='lines', name='Validation'))
pred_graph_link.add_trace(go.Scatter(x=df_pred_final['Date'], y=df_pred_final['LINK-USD'], mode='lines', name='Prediction'))

# Prediction graph one - AXS-USD
pred_graph_axs = go.Figure(px.line(df_close_pred, x='Date', y='AXS-USD',title='Cryptocurrency: AXS-USD'))
pred_graph_axs.update_layout(xaxis=None,yaxis=dict(title='Price - USD'))
pred_graph_axs.add_trace(go.Scatter(x=df_val_final['Date'], y=df_val_final['AXS-USD'], mode='lines', name='Validation'))
pred_graph_axs.add_trace(go.Scatter(x=df_pred_final['Date'], y=df_pred_final['AXS-USD'], mode='lines', name='Prediction'))



# ----------------------------------------------------------------------------------------------------------------------
server = app.server
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
            html.H4("Select a specific date:", style={'text-align': 'center','padding':'550px 0px 0px 0px'}),
            dropdown_date
        ],
            style={'box-shadow':'1px 1px 3px lightgray', "background-color" : "white", 'height':'1321.42px'}),
            width=2,
            style={'padding':'2px 0px 15px 15px'}),

        dbc.Col(html.Div([

            html.H2("Candlestick Chart & Specific Date Table", style={'text-align': 'left', 'padding':'15px 15px'}),
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
                    dcc.Graph(id="pred_graph_btc", figure=pred_graph_btc),
                    dcc.Graph(id="pred_graph_eth", figure=pred_graph_eth),
                    dcc.Graph(id="pred_graph_luna", figure=pred_graph_luna),
                    dcc.Graph(id="pred_graph_sol", figure=pred_graph_sol),
                    dcc.Graph(id="pred_graph_avax", figure=pred_graph_avax)
                ],

                                 style={"background-color" : "white"}),
                        style={'padding':'2px 15px 15px 15px'}),

                dbc.Col(html.Div([
                    dcc.Graph(id="pred_graph_ada", figure=pred_graph_ada),
                    dcc.Graph(id="pred_graph_matic", figure=pred_graph_matic),
                    dcc.Graph(id="pred_graph_atom", figure=pred_graph_atom),
                    dcc.Graph(id="pred_graph_link", figure=pred_graph_link),
                    dcc.Graph(id="pred_graph_axs", figure=pred_graph_axs)
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