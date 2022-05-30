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

# Storing each indicator in separately dataframe for visuals only
data_visual = data.copy()
df_open_visual = data['Open'].reset_index()
df_close_visual = data['Close'].reset_index()
df_adj_close_visual = data['Adj Close'].reset_index()
df_high_visual = data['High'].reset_index()
df_low_visual = data['Low'].reset_index()
df_volume_visual = data['Volume'].reset_index()

# Storing each indicator in separately dataframe
df_open = data['Open'].reset_index()
df_close = data['Close'].reset_index()
df_adj_close = data['Adj Close'].reset_index()
df_high = data['High'].reset_index()
df_low = data['Low'].reset_index()
df_volume = data['Volume'].reset_index()

# Creating a list with all the currencies
list_of_currencys = df_volume.iloc[:, 1:].columns.to_list()

# In[6]:


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

# <hr>
# <a class="anchor" id="ts_modeling">
#
# # 4.0 Time Series Model and Assessment
#
# </a>

# <hr>
# <a class="anchor" id="BTC-USD">
#
# ## BTC-USD
#
# </a>

# In[7]:


##creating a df to predict the crypto currency
dfbtc = df['BTC-USD'].copy()

# In[8]:


# Creating a new feature for better representing day-wise values
dfbtc['mean'] = (dfbtc['low'] + dfbtc['high']) / 2

# In[9]:


# Cleaning the data for any NaN or Null fields
dfbtc = dfbtc.dropna()

# In[10]:


# Creating a copy for applying shift
dataset_for_prediction = dfbtc.copy()
dataset_for_prediction['Actual'] = dataset_for_prediction['close'].shift()
dataset_for_prediction = dataset_for_prediction.dropna()

# In[11]:


# date time typecast
dataset_for_prediction['Date'] = pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index = dataset_for_prediction['Date']

# In[12]:


# normalizing the exogeneous variables
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(
    dataset_for_prediction[['volume']])  # ['low', 'high', 'open', 'adj_close', 'volume', 'mean']
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X = scaled_input
X.rename(columns={0: 'Volume'}, inplace=True)

# In[13]:


# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output = pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y = scaler_output
y.rename(columns={0: 'Observed Data'}, inplace=True)
y.index = dataset_for_prediction.index

# In[14]:


# train-test split (cannot shuffle in case of time series)
train_X, train_y = X[:-7].dropna(), y[:-7].dropna()
test_X, test_y = X[-9:].dropna(), y[-8:].dropna()

# In[15]:


# Init the best SARIMAX model
model = SARIMAX(
    train_y,
    exog=train_X,
    order=(0, 1, 0),
    seasonal_order=(2, 1, 0, 6)
)

# In[16]:


# training the model
results = model.fit()

# In[17]:


# get predictions
predictions = results.predict(start=len(train_y), end=len(train_y) + len(test_y), exog=test_X)

# In[18]:


# forecast
fcst = results.predict(start=len(train_y), end=len(train_y) + len(test_y), exog=test_X).to_frame()
fcst2 = sc_out.inverse_transform(fcst)
# storing the predictions in a dataframe
btc_predictions = pd.DataFrame(fcst2, index=fcst.index, columns=['price'])

# <hr>
# <a class="anchor" id="ETH-USD">
#
# ## ETH-USD
#
# </a>

# In[19]:


# creating a df to predict the crypto currency
dfeth = df['ETH-USD'].copy()

# In[20]:


# Creating a new feature for better representing day-wise values
dfeth['mean'] = (dfeth['low'] + dfeth['high']) / 2

# In[21]:


# Cleaning the data for any NaN or Null fields
dfeth = dfeth.dropna()

# In[22]:


# Creating a copy for applying shift
dataset_for_prediction = dfeth.copy()
dataset_for_prediction['Actual'] = dataset_for_prediction['close'].shift()
dataset_for_prediction = dataset_for_prediction.dropna()

# In[23]:


# date time typecast
dataset_for_prediction['Date'] = pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index = dataset_for_prediction['Date']

# In[24]:


# normalizing the exogeneous variables
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(
    dataset_for_prediction[['volume']])  # ['low', 'high', 'open', 'adj_close', 'volume', 'mean']
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X = scaled_input
X.rename(columns={0: 'Volume'}, inplace=True)

# In[25]:


# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output = pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y = scaler_output
y.rename(columns={0: 'Observed Data'}, inplace=True)
y.index = dataset_for_prediction.index

# In[26]:


# train-test split (cannot shuffle in case of time series)
train_X, train_y = X[:-7].dropna(), y[:-7].dropna()
test_X, test_y = X[-9:].dropna(), y[-8:].dropna()

# In[27]:


# Init the best SARIMAX model
model = SARIMAX(
    train_y,
    exog=train_X,
    order=(1, 1, 0),
    seasonal_order=(2, 1, 0, 6)
)

# In[28]:


# training the model
results = model.fit()

# In[29]:


# get predictions
predictions = results.predict(start=len(train_y), end=len(train_y) + len(test_y), exog=test_X)

# In[30]:


# forecast
fcst = results.predict(start=len(train_y), end=len(train_y) + len(test_y), exog=test_X).to_frame()
fcst2 = sc_out.inverse_transform(fcst)
# storing the predictions in a dataframe
eth_predictions = pd.DataFrame(fcst2, index=fcst.index, columns=['price'])

# <hr>
# <a class="anchor" id="LINK-USD">
#
# ## LINK-USD
#
# </a>

# In[31]:


# creating a df to predict the crypto currency
dflink = df['LINK-USD'].copy()

# In[32]:


# Creating a new feature for better representing day-wise values
dflink['mean'] = (dflink['low'] + dflink['high']) / 2

# In[33]:


# Cleaning the data for any NaN or Null fields
dflink = dflink.dropna()

# In[34]:


# Creating a copy for applying shift
dataset_for_prediction = dflink.copy()
dataset_for_prediction['Actual'] = dataset_for_prediction['close'].shift()
dataset_for_prediction = dataset_for_prediction.dropna()

# In[35]:


# date time typecast
dataset_for_prediction['Date'] = pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index = dataset_for_prediction['Date']

# In[36]:


# normalizing the exogeneous variables
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(
    dataset_for_prediction[['volume']])  # ['low', 'high', 'open', 'adj_close', 'volume', 'mean']
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X = scaled_input
X.rename(columns={0: 'Volume'}, inplace=True)

# In[37]:


# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output = pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y = scaler_output
y.rename(columns={0: 'Observed Data'}, inplace=True)
y.index = dataset_for_prediction.index

# In[38]:


# train-test split (cannot shuffle in case of time series)
train_X, train_y = X[:-7].dropna(), y[:-7].dropna()
test_X, test_y = X[-9:].dropna(), y[-8:].dropna()

# In[39]:


# Init the best SARIMAX model
model = SARIMAX(
    train_y,
    exog=train_X,
    order=(1, 1, 1),
    seasonal_order=(2, 1, 0, 6)
)

# In[40]:


# training the model
results = model.fit()

# In[41]:


# get predictions
predictions = results.predict(start=len(train_y), end=len(train_y) + len(test_y), exog=test_X)

# In[42]:


# forecast
fcst = results.predict(start=len(train_y), end=len(train_y) + len(test_y), exog=test_X).to_frame()
fcst2 = sc_out.inverse_transform(fcst)
# storing the predictions in a dataframe
link_predictions = pd.DataFrame(fcst2, index=fcst.index, columns=['price'])

# <hr>
# <a class="anchor" id="MATIC-USD">
#
# ## MATIC-USD
#
# </a>

# In[43]:


# creating a df to predict the crypto currency
dfmatic = df['MATIC-USD'].copy()

# In[44]:


# Creating a new feature for better representing day-wise values
dfmatic['mean'] = (dfmatic['low'] + dfmatic['high']) / 2

# In[45]:


# Cleaning the data for any NaN or Null fields
dfmatic = dfmatic.dropna()

# In[46]:


# Creating a copy for applying shift
dataset_for_prediction = dfmatic.copy()
dataset_for_prediction['Actual'] = dataset_for_prediction['close'].shift()
dataset_for_prediction = dataset_for_prediction.dropna()

# In[47]:


# date time typecast
dataset_for_prediction['Date'] = pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index = dataset_for_prediction['Date']

# In[48]:


# normalizing the exogeneous variables
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(
    dataset_for_prediction[['volume']])  # ['low', 'high', 'open', 'adj_close', 'volume', 'mean']
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X = scaled_input
X.rename(columns={0: 'Volume'}, inplace=True)

# In[49]:


# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output = pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y = scaler_output
y.rename(columns={0: 'Observed Data'}, inplace=True)
y.index = dataset_for_prediction.index

# In[50]:


# train-test split (cannot shuffle in case of time series)
train_X, train_y = X[:-7].dropna(), y[:-7].dropna()
test_X, test_y = X[-9:].dropna(), y[-8:].dropna()

# In[51]:


# Init the best SARIMAX model
model = SARIMAX(
    train_y,
    exog=train_X,
    order=(2, 1, 2),
    seasonal_order=(2, 1, 0, 6)
)

# In[52]:


# training the model
results = model.fit()

# In[53]:


# get predictions
predictions = results.predict(start=len(train_y), end=len(train_y) + len(test_y), exog=test_X)

# In[54]:


# forecast
fcst = results.predict(start=len(train_y), end=len(train_y) + len(test_y), exog=test_X).to_frame()
fcst2 = sc_out.inverse_transform(fcst)
# storing the predictions in a dataframe
matic_predictions = pd.DataFrame(fcst2, index=fcst.index, columns=['price'])

# <hr>
# <a class="anchor" id="SOL-USD">
#
# ## SOL-USD
#
# </a>

# In[55]:


# creating a df to predict the crypto currency
dfsol = df['SOL-USD'].copy()

# In[56]:


# Creating a new feature for better representing day-wise values
dfsol['mean'] = (dfsol['low'] + dfsol['high']) / 2

# In[57]:


# Cleaning the data for any NaN or Null fields
dfsol = dfsol.dropna()

# In[58]:


# Creating a copy for applying shift
dataset_for_prediction = dfsol.copy()
dataset_for_prediction['Actual'] = dataset_for_prediction['close'].shift()
dataset_for_prediction = dataset_for_prediction.dropna()

# In[59]:


# date time typecast
dataset_for_prediction['Date'] = pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index = dataset_for_prediction['Date']

# In[60]:


# normalizing the exogeneous variables
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(
    dataset_for_prediction[['volume']])  # ['low', 'high', 'open', 'adj_close', 'volume', 'mean']
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X = scaled_input
X.rename(columns={0: 'Volume'}, inplace=True)

# In[61]:


# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output = pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y = scaler_output
y.rename(columns={0: 'Observed Data'}, inplace=True)
y.index = dataset_for_prediction.index

# In[62]:


# train-test split (cannot shuffle in case of time series)
train_X, train_y = X[:-7].dropna(), y[:-7].dropna()
test_X, test_y = X[-9:].dropna(), y[-8:].dropna()

# In[63]:


# Init the best SARIMAX model
model = SARIMAX(
    train_y,
    exog=train_X,
    order=(0, 1, 0),
    seasonal_order=(2, 1, 0, 6)
)

# In[64]:


# training the model
results = model.fit()

# In[65]:


# get predictions
predictions = results.predict(start=len(train_y), end=len(train_y) + len(test_y), exog=test_X)

# In[66]:


# forecast
fcst = results.predict(start=len(train_y), end=len(train_y) + len(test_y), exog=test_X).to_frame()
fcst2 = sc_out.inverse_transform(fcst)
# storing the predictions in a dataframe
sol_predictions = pd.DataFrame(fcst2, index=fcst.index, columns=['price'])

# <hr>
# <a class="anchor" id="ts_predictions">
#
# ##  Time Series Predictions Summary
#
# </a>

# In[67]:


# #creating dataframes for each currency to summarize the predictions

btc_predictions.index.name = 'Date'
btc = btc_predictions.rename(columns={'price': 'BTC-USD'}).reset_index()

eth_predictions.index.name = 'Date'
eth = eth_predictions.rename(columns={'price': 'ETH-USD'}).reset_index()

link_predictions.index.name = 'Date'
link = link_predictions.rename(columns={'price': 'LINK-USD'}).reset_index()

matic_predictions.index.name = 'Date'
matic = matic_predictions.rename(columns={'price': 'MATIC-USD'}).reset_index()

sol_predictions.index.name = 'Date'
sol = sol_predictions.rename(columns={'price': 'SOL-USD'}).reset_index()

# In[68]:


temp_a = pd.merge(btc, link, left_on='Date', right_on='Date', how='inner')
temp_b = pd.merge(temp_a, matic, left_on='Date', right_on='Date', how='inner')
temp_c = pd.merge(temp_b, eth, left_on='Date', right_on='Date', how='inner')
final_predictions = pd.merge(temp_c, sol, left_on='Date', right_on='Date', how='inner')

# In[69]:


final_predictions

# In[70]:


# creating the dataframe to store the predictions for D+1 and D+2 of each currency
df_pred_final = final_predictions[-2:].copy()

# In[71]:


df_pred_final

# In[72]:


# creating the dataframe to store the predictions for the validation data
df_val_final = final_predictions[:-2].copy()

# In[73]:


df_val_final

# <hr>
# <a class="anchor" id="ml">
#
# # 5.0 Machine Learning
#
# </a>

# <hr>
# <a class="anchor" id="data_prep_ml">
#
# # 6.0 Data Preparation and Preprocessing
#
# </a>

# In[74]:


# importing data from yahoo finance lib
data = yf.download(cryptocurrencies, period='496d', interval='1d')

# In[75]:


# storing each indicator in separately dataframe
df_open = data['Open'].reset_index()

df_close = data['Close'].reset_index()

df_adj_close = data['Adj Close'].reset_index()

df_high = data['High'].reset_index()

df_low = data['Low'].reset_index()

df_volume = data['Volume'].reset_index()

# In[76]:


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

    # Adding Three new rows to the dataset
    df[currency] = df[currency].append(
        pd.DataFrame({'Date': pd.date_range(start=df[currency].Date.iloc[-1], periods=4, freq='D', closed='right')}))
    df[currency].reset_index(inplace=True, drop=True)

    # Feature Engineering
    df[currency]['year'] = pd.DatetimeIndex(df[currency]['Date']).year
    df[currency]['quarter'] = pd.DatetimeIndex(df[currency]['Date']).quarter
    df[currency]['month'] = pd.DatetimeIndex(df[currency]['Date']).month
    df[currency]['week_number_year'] = pd.DatetimeIndex(df[currency]['Date']).week
    df[currency]['day_of_the_week'] = pd.DatetimeIndex(df[currency]['Date']).weekday
    df[currency]['day'] = pd.DatetimeIndex(df[currency]['Date']).day

# #### Creating the Target

# In[77]:


for currency in list_of_currencys:
    df[currency]['target_close'] = df[currency]['close']

# #### Shifting the Data

# In[78]:


for currency in list_of_currencys:
    df_temp_shift = df[currency][['open', 'close', 'adj_close', 'high', 'low', 'volume']].shift(+3)
    df_temp_target = df[currency][
        ['Date', 'year', 'quarter', 'month', 'week_number_year', 'day_of_the_week', 'day', 'target_close']]
    df_temp_final = pd.concat([df_temp_shift, df_temp_target], axis=1)
    # df[currency] = df_temp_final
    df[currency] = df_temp_final.iloc[3:, :]
    df[currency].reset_index(inplace=True, drop=True)

# #### Creating new features

# In[79]:


# Preserving the original datasets
df_original = {}
for currency in list_of_currencys:
    df_original[currency] = df[currency].copy()

# In[80]:


# Adding Technical Analysis Features
for currency in list_of_currencys:
    ta.add_all_ta_features(df[currency], "open", "high", "low", "close", "volume", fillna=False)

# In[81]:


# Dropping the features trend_psar_up and trend_psar_down due to the quantity of NaN values
for currency in list_of_currencys:
    df[currency].drop(['trend_psar_up', 'trend_psar_down'], axis=1, inplace=True)

# In[82]:


# Changing the types of volume_adi and volume_obv from object to float64
for currency in list_of_currencys:
    df[currency] = df[currency].astype({'volume_adi': 'float64', 'volume_obv': 'float64'})

# In[83]:


# Drop the missing NaN values from the new features

list_columns_to_drop_NaN = df['ADA-USD'].columns.tolist()
list_columns_to_drop_NaN.remove('target_close')

for currency in list_of_currencys:
    df[currency].dropna(axis=0, how='any', subset=list_columns_to_drop_NaN, inplace=True)

# In[84]:


# Reseting indexes and Checking new features
for currency in list_of_currencys:
    df[currency].reset_index(inplace=True, drop=True)

# #### Feature Selection

# In[85]:


# Creating copies of the datasets for the purpose of feature selection
df_FS = {}
for currency in list_of_currencys:
    df_FS[currency] = df[currency].copy()

# In[86]:


# Removing Missing Values in the target (last three rows - new dates to be predicted)
for currency in list_of_currencys:
    df_FS[currency].dropna(axis=0, inplace=True)

# In[87]:


# iterating the minimun date to define the initial date for train datasets
dt_imin = []
for currency in list_of_currencys:
    dt_imin.append(df_FS[currency]['Date'].min())
dt_min = min(dt_imin)

# In[88]:


# iterating the maximum date to define the size of the split
dt_imax = []
for currency in list_of_currencys:
    dt_imax.append(df_FS[currency]['Date'].max())
dt_max = min(dt_imax)

# In[89]:


# calculates the date interval - size of the series
date_interval = (dt_max - dt_min) / np.timedelta64(1, 'D')

# measuring the split size
split_size = date_interval * 0.7

# In[92]:


# calculates the split date
split_date = dt_min + datetime.timedelta(days=split_size)
split_date = split_date.strftime('%Y-%m-%d')

# calculates the initial date
init_date = dt_min.strftime('%Y-%m-%d')

# In[93]:


# defining the features for each currency based on the RFE feature selection
feat_ada = ['momentum_kama']

feat_atom = ['volatility_bbm', 'trend_ema_slow', 'trend_visual_ichimoku_a']

feat_avax = ['volatility_bbl', 'volatility_kcl', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
             'trend_ema_fast', 'trend_ichimoku_a', 'trend_ichimoku_b', 'momentum_tsi', 'momentum_kama']

feat_axs = ['close', 'adj_close', 'volume_vwap', 'volatility_kcl', 'trend_macd_signal', 'trend_ema_slow', 'others_cr']

feat_luna = ['close']

# <hr>
# <a class="anchor" id="ml_modeling">
#
# # 7.0 Machine Learning Model and Assessment
#
# </a>

# <hr>
# <a class="anchor" id="linear">
#
# ## Linear Regression
#
# </a>

# In[94]:


# Using LR to predict results for AVAX-USD
df_pred = {}
summary_LR = pd.DataFrame(index=['New_pred_1', 'New_pred_2'])

# Dataset splitting parameters
initial_Date = init_date
splitting_Date = split_date

for currency in list_of_currencys:
    # List of features to be used
    feat_list = feat_avax

    # Splitting the dataset
    X = df[currency][feat_list]
    y = df[currency]['target_close']
    initial_index = df[currency].index[df[currency]['Date'] == initial_Date].tolist()
    splitt_index = df[currency].index[df[currency]['Date'] == splitting_Date].tolist()
    X_train = X[initial_index[0]:splitt_index[0]]
    X_val = X[splitt_index[0]:-3]
    y_train = y[initial_index[0]:splitt_index[0]]
    y_val = y[splitt_index[0]:-3]
    X_new_pred = X[-3:]

    #     # Splitting the dataset - classic approach
    #     X = df[currency].drop(['Date','target_close'], axis = 1)
    #     y = df[currency]['target_close']
    #     X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.1, random_state=0, stratify=None, shuffle=False)

    # Training the model and running the predictions
    model = LinearRegression()
    model.fit(X_train, y_train)
    model_pred = model.predict(X_val)
    model_pred_df = pd.DataFrame(data=model_pred, index=X_val.index, columns=['pred'])

    # Running predictions for the training dataset for overfitting checking purpose
    model_pred_train = model.predict(X_train)
    model_pred_train_df = pd.DataFrame(data=model_pred_train, index=X_train.index, columns=['pred'])
    model_pred_train_df = pd.concat([df[currency], model_pred_train_df], axis=1)

    # Predicting the new data
    model_new_pred = model.predict(X_new_pred)
    model_new_pred_df = pd.DataFrame(data=model_new_pred, index=X_new_pred.index, columns=['pred'])
    model_new_pred_df = pd.concat([df[currency], model_new_pred_df], axis=1)

    # RMSE and R-squared
    rmse_val = np.sqrt(MSE(y_val, model_pred))
    mape_val = MAPE(y_val, model_pred)
    rmse_train = np.sqrt(MSE(y_train, model_pred_train))
    mape_train = MAPE(y_train, model_pred_train)

    # Concatenating the predictions to the original Dataset
    df_pred[currency] = pd.concat([df[currency], model_pred_df], axis=1)

    summary_LR[currency] = [round(model_new_pred_df[-3:-2]['pred'].tolist()[0], 2),
                            round(model_new_pred_df[-2:-1]['pred'].tolist()[0], 2)]

# In[95]:


# calling the predictions from Linear Regression
lr_predictions = summary_LR['AVAX-USD'].reset_index(drop=True)

# calling the predictions from ARIMA
df_arima = df_pred_final.reset_index(drop=True)

# storing the predictions in result_a
result_a = pd.concat([df_arima, lr_predictions], axis=1)

# storing the validation predictions for the chosen currencies
lr_avax = df_pred['AVAX-USD'][['Date', 'target_close']][-10:-3]
lr_avax = lr_avax.rename(columns={'target_close': 'AVAX-USD'})

# In[96]:


# Using LR to predict results for LUNA1-USD
df_pred = {}
summary_LR = pd.DataFrame(index=['New_pred_1', 'New_pred_2'])

# Dataset splitting parameters
initial_Date = init_date
splitting_Date = split_date

for currency in list_of_currencys:
    # List of features to be used
    feat_list = feat_luna

    # Splitting the dataset
    X = df[currency][feat_list]
    y = df[currency]['target_close']
    initial_index = df[currency].index[df[currency]['Date'] == initial_Date].tolist()
    splitt_index = df[currency].index[df[currency]['Date'] == splitting_Date].tolist()
    X_train = X[initial_index[0]:splitt_index[0]]
    X_val = X[splitt_index[0]:-3]
    y_train = y[initial_index[0]:splitt_index[0]]
    y_val = y[splitt_index[0]:-3]
    X_new_pred = X[-3:]

    #     # Splitting the dataset - classic approach
    #     X = df[currency].drop(['Date','target_close'], axis = 1)
    #     y = df[currency]['target_close']
    #     X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.1, random_state=0, stratify=None, shuffle=False)

    # Training the model and running the predictions
    model = LinearRegression()
    model.fit(X_train, y_train)
    model_pred = model.predict(X_val)
    model_pred_df = pd.DataFrame(data=model_pred, index=X_val.index, columns=['pred'])

    # Running predictions for the training dataset for overfitting checking purpose
    model_pred_train = model.predict(X_train)
    model_pred_train_df = pd.DataFrame(data=model_pred_train, index=X_train.index, columns=['pred'])
    model_pred_train_df = pd.concat([df[currency], model_pred_train_df], axis=1)

    # Predicting the new data
    model_new_pred = model.predict(X_new_pred)
    model_new_pred_df = pd.DataFrame(data=model_new_pred, index=X_new_pred.index, columns=['pred'])
    model_new_pred_df = pd.concat([df[currency], model_new_pred_df], axis=1)

    # RMSE and R-squared
    rmse_val = np.sqrt(MSE(y_val, model_pred))
    mape_val = MAPE(y_val, model_pred)
    rmse_train = np.sqrt(MSE(y_train, model_pred_train))
    mape_train = MAPE(y_train, model_pred_train)

    # Concatenating the predictions to the original Dataset
    df_pred[currency] = pd.concat([df[currency], model_pred_df], axis=1)

    summary_LR[currency] = [round(model_new_pred_df[-3:-2]['pred'].tolist()[0], 2),
                            round(model_new_pred_df[-2:-1]['pred'].tolist()[0], 2)]

# In[97]:


# calling the predictions from Linear Regression
lr2_predictions = summary_LR['LUNA1-USD'].reset_index(drop=True)

# storing the predictions in result
result_b = pd.concat([result_a, lr2_predictions], axis=1)

# storing the validation predictions for the chosen currencies
lr_luna = df_pred['LUNA1-USD'][['Date', 'target_close']][-10:-3]
lr_luna = lr_luna.rename(columns={'target_close': 'LUNA1-USD'})

# <hr>
# <a class="anchor" id="rand_forest">
#
# ## Random Forest
#
# </a>

# In[98]:


df_pred = {}
summary_RF = pd.DataFrame(index=['New_pred_1', 'New_pred_2'])

# Dataset splitting parameters
initial_Date = init_date
splitting_Date = split_date

for currency in list_of_currencys:
    # List of features to be used
    feat_list = feat_ada

    # Splitting the dataset
    X = df[currency][feat_list]
    y = df[currency]['target_close']
    initial_index = df[currency].index[df[currency]['Date'] == initial_Date].tolist()
    splitt_index = df[currency].index[df[currency]['Date'] == splitting_Date].tolist()
    X_train = X[initial_index[0]:splitt_index[0]]
    X_val = X[splitt_index[0]:-3]
    y_train = y[initial_index[0]:splitt_index[0]]
    y_val = y[splitt_index[0]:-3]
    X_new_pred = X[-3:]

    #     # Splitting the dataset - classic approach
    #     X = df[currency].drop(['Date','target_close'], axis = 1)
    #     y = df[currency]['target_close']
    #     X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.1, random_state=0, stratify=None, shuffle=False)

    # Training the model and running the predictions
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    model_pred = model.predict(X_val)
    model_pred_df = pd.DataFrame(data=model_pred, index=X_val.index, columns=['pred'])

    # Running predictions for the training dataset for overfitting checking purpose
    model_pred_train = model.predict(X_train)
    model_pred_train_df = pd.DataFrame(data=model_pred_train, index=X_train.index, columns=['pred'])
    model_pred_train_df = pd.concat([df[currency], model_pred_train_df], axis=1)

    # Predicting the new data
    model_new_pred = model.predict(X_new_pred)
    model_new_pred_df = pd.DataFrame(data=model_new_pred, index=X_new_pred.index, columns=['pred'])
    model_new_pred_df = pd.concat([df[currency], model_new_pred_df], axis=1)

    # RMSE and R-squared
    rmse_val = np.sqrt(MSE(y_val, model_pred))
    mape_val = MAPE(y_val, model_pred)
    rmse_train = np.sqrt(MSE(y_train, model_pred_train))
    mape_train = MAPE(y_train, model_pred_train)

    # Concatenating the predictions to the original Dataset
    df_pred[currency] = pd.concat([df[currency], model_pred_df], axis=1)

    summary_RF[currency] = [round(model_new_pred_df[-3:-2]['pred'].tolist()[0], 2),
                            round(model_new_pred_df[-2:-1]['pred'].tolist()[0], 2)]

# In[99]:


# calling the predictions from Random Forest
rf_predictions = summary_RF['ADA-USD'].reset_index(drop=True)

# storing the predictions in result
result_c = pd.concat([result_b, rf_predictions], axis=1)

# storing the validation predictions for the chosen currencies
rf_ada = df_pred['ADA-USD'][['Date', 'target_close']][-10:-3]
rf_ada = rf_ada.rename(columns={'target_close': 'ADA-USD'})

# <hr>
# <a class="anchor" id="svm">
#
# ## Support Vector Machines
#
# </a>

# In[100]:


df_pred = {}
summary_SVM = pd.DataFrame(index=['New_pred_1', 'New_pred_2'])

# Dataset splitting parameters
initial_Date = init_date
splitting_Date = split_date

for currency in list_of_currencys:
    # List of features to be used
    feat_list = feat_atom

    # Splitting the dataset
    X = df[currency][feat_list]
    y = df[currency]['target_close']
    initial_index = df[currency].index[df[currency]['Date'] == initial_Date].tolist()
    splitt_index = df[currency].index[df[currency]['Date'] == splitting_Date].tolist()
    X_train = X[initial_index[0]:splitt_index[0]]
    X_val = X[splitt_index[0]:-3]
    y_train = y[initial_index[0]:splitt_index[0]]
    y_val = y[splitt_index[0]:-3]
    X_new_pred = X[-3:]

    #     # Splitting the dataset - classic approach
    #     X = df[currency].drop(['Date','target_close'], axis = 1)
    #     y = df[currency]['target_close']
    #     X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.1, random_state=0, stratify=None, shuffle=False)

    # Training the model and running the predictions
    model = svm.SVR()
    model.fit(X_train, y_train)
    model_pred = model.predict(X_val)
    model_pred_df = pd.DataFrame(data=model_pred, index=X_val.index, columns=['pred'])

    # Running predictions for the training dataset for overfitting checking purpose
    model_pred_train = model.predict(X_train)
    model_pred_train_df = pd.DataFrame(data=model_pred_train, index=X_train.index, columns=['pred'])
    model_pred_train_df = pd.concat([df[currency], model_pred_train_df], axis=1)

    # Predicting the new data
    model_new_pred = model.predict(X_new_pred)
    model_new_pred_df = pd.DataFrame(data=model_new_pred, index=X_new_pred.index, columns=['pred'])
    model_new_pred_df = pd.concat([df[currency], model_new_pred_df], axis=1)

    # RMSE and R-squared
    rmse_val = np.sqrt(MSE(y_val, model_pred))
    mape_val = MAPE(y_val, model_pred)
    rmse_train = np.sqrt(MSE(y_train, model_pred_train))
    mape_train = MAPE(y_train, model_pred_train)

    # Concatenating the predictions to the original Dataset
    df_pred[currency] = pd.concat([df[currency], model_pred_df], axis=1)

    summary_SVM[currency] = [round(model_new_pred_df[-3:-2]['pred'].tolist()[0], 2),
                             round(model_new_pred_df[-2:-1]['pred'].tolist()[0], 2)]

# In[101]:


# calling the predictions from Random Forest
svm_predictions = summary_SVM['ATOM-USD'].reset_index(drop=True)

# storing the predictions in result
result_d = pd.concat([result_c, svm_predictions], axis=1)

# storing the validation predictions for the chosen currencies
svm_atom = df_pred['ATOM-USD'][['Date', 'target_close']][-10:-3]
svm_atom = svm_atom.rename(columns={'target_close': 'ATOM-USD'})

# <hr>
# <a class="anchor" id="neural">
#
# ## Neural Network Regressor
#
# </a>

# In[102]:


df_pred = {}
summary_NNR = pd.DataFrame(index=['New_pred_1', 'New_pred_2'])

# Dataset splitting parameters
initial_Date = init_date
splitting_Date = split_date

for currency in list_of_currencys:
    # List of features to be used
    feat_list = feat_axs

    # Splitting the dataset
    X = df[currency][feat_list]
    y = df[currency]['target_close']
    initial_index = df[currency].index[df[currency]['Date'] == initial_Date].tolist()
    splitt_index = df[currency].index[df[currency]['Date'] == splitting_Date].tolist()
    X_train = X[initial_index[0]:splitt_index[0]]
    X_val = X[splitt_index[0]:-3]
    y_train = y[initial_index[0]:splitt_index[0]]
    y_val = y[splitt_index[0]:-3]
    X_new_pred = X[-3:]

    #     # Splitting the dataset - classic approach
    #     X = df[currency].drop(['Date','target_close'], axis = 1)
    #     y = df[currency]['target_close']
    #     X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.1, random_state=0, stratify=None, shuffle=False)

    # Training the model and running the predictions
    model = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100, 100, 100), max_iter=100, learning_rate_init=0.0001,
                         learning_rate='adaptive')
    model.fit(X_train, y_train)
    model_pred = model.predict(X_val)
    model_pred_df = pd.DataFrame(data=model_pred, index=X_val.index, columns=['pred'])

    # Running predictions for the training dataset for overfitting checking purpose
    model_pred_train = model.predict(X_train)
    model_pred_train_df = pd.DataFrame(data=model_pred_train, index=X_train.index, columns=['pred'])
    model_pred_train_df = pd.concat([df[currency], model_pred_train_df], axis=1)

    # Predicting the new data
    model_new_pred = model.predict(X_new_pred)
    model_new_pred_df = pd.DataFrame(data=model_new_pred, index=X_new_pred.index, columns=['pred'])
    model_new_pred_df = pd.concat([df[currency], model_new_pred_df], axis=1)

    # RMSE and R-squared
    rmse_val = np.sqrt(MSE(y_val, model_pred))
    mape_val = MAPE(y_val, model_pred)
    rmse_train = np.sqrt(MSE(y_train, model_pred_train))
    mape_train = MAPE(y_train, model_pred_train)

    # Concatenating the predictions to the original Dataset
    df_pred[currency] = pd.concat([df[currency], model_pred_df], axis=1)

    summary_NNR[currency] = [round(model_new_pred_df[-3:-2]['pred'].tolist()[0], 2),
                             round(model_new_pred_df[-2:-1]['pred'].tolist()[0], 2)]

# In[103]:


# calling the predictions from Random Forest
nnr_predictions = summary_NNR['AXS-USD'].reset_index(drop=True)

# storing the predictions for D+1 and D+2 in the dataframe df_pred_dashboard
df_pred_final = pd.concat([result_d, nnr_predictions], axis=1)

# storing the validation predictions for the chosen currencies
nnr_axs = df_pred['AXS-USD'][['Date', 'target_close']][-10:-3]
nnr_axs = nnr_axs.rename(columns={'target_close': 'AXS-USD'})

# In[104]:


# storing the validation predictions for the chosen currencies
temp_val_a = pd.merge(df_val_final, lr_avax, left_on='Date', right_on='Date', how='left')
temp_val_b = pd.merge(temp_val_a, lr_luna, left_on='Date', right_on='Date', how='left')
temp_val_c = pd.merge(temp_val_b, rf_ada, left_on='Date', right_on='Date', how='left')
temp_val_d = pd.merge(temp_val_c, svm_atom, left_on='Date', right_on='Date', how='left')
temp_val_final = pd.merge(temp_val_d, nnr_axs, left_on='Date', right_on='Date', how='left')

# filling missing values with the mean for each column
fill_mean = lambda col: col.fillna(col.mean())
df_val_final = temp_val_final.apply(fill_mean, axis=0)

# In[105]:


# final validation dataset to be used on Dashboard
df_val_final

# In[106]:


# final predictions dataset to be used on Dashboard
df_pred_final




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
year_slide = dcc.RangeSlider(min(df_open_visual ['Date']).year, max(df_open_visual ['Date']).year,
                             value=[min(df_open_visual ['Date']).year,max(df_open_visual ['Date']).year],
                             id='year_slide',
                             tooltip={"placement": "bottom", "always_visible": True},
                             step=1,
                             allowCross=False,
                             marks={min(df_open_visual ['Date']).year:{'label':str(min(df_open_visual ['Date']).year)}, max(df_open_visual ['Date']).year:{'label':str(max(df_open_visual ['Date']).year)}}
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
    options=[*range(min(df_open_visual ['Date']).year, max(df_open_visual ['Date']).year+1, 1)],
    value=max(df_open_visual ['Date']).year,
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

day_range = datetime.timedelta(2)

# Dropdown menu for dates
dropdown_date = dcc.Dropdown(
    id='date_dropdown',
    className="dropdown",
    options=pd.to_datetime(df_open_visual ['Date']).dt.date,
    value=pd.to_datetime(max(df_open_visual ['Date'])).date()-day_range,
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

    df_temp = data_visual [indicator].reset_index()
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

    open = df_open_visual.loc[(df_open_visual["Date"] >= initial_date) & (df_open_visual["Date"] <= final_date)]
    high = df_high_visual.loc[(df_high_visual["Date"] >= initial_date) & (df_high_visual["Date"] <= final_date)]
    low = df_low_visual.loc[(df_low_visual["Date"] >= initial_date) & (df_low_visual["Date"] <= final_date)]
    close = df_close_visual.loc[(df_close_visual["Date"] >= initial_date) & (df_close_visual["Date"] <= final_date)]

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

    tb_open = round(df_open_visual[df_open_visual['Date'] == date], 2)
    tb_close = round(df_close_visual[df_close_visual['Date'] == date], 2)
    tb_high = round(df_high_visual[df_high_visual['Date'] == date], 2)
    tb_low = round(df_low_visual[df_low_visual['Date'] == date], 2)
    tb_volume = round(df_volume_visual[df_volume_visual['Date'] == date], 0)

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

    tb_fifth_row = [np.nanmax(round(df_high_visual['BTC-USD'],2)), np.nanmax(round(df_high_visual['ETH-USD'],2)), np.nanmax(round(df_high_visual['LUNA1-USD'],2)),
                    np.nanmax(round(df_high_visual['SOL-USD'],2)), np.nanmax(round(df_high_visual['ADA-USD'],2)), np.nanmax(round(df_high_visual['AVAX-USD'],2)),
                    np.nanmax(round(df_high_visual['MATIC-USD'],2)), np.nanmax(round(df_high_visual['ATOM-USD'],2)), np.nanmax(round(df_high_visual['LINK-USD'],2)),
                    np.nanmax(round(df_high_visual['AXS-USD'],2))]

    tb_sixth_row = [np.nanmin(round(df_high_visual['BTC-USD'],2)), np.nanmin(round(df_high_visual['ETH-USD'],2)), np.nanmin(round(df_high_visual['LUNA1-USD'],2)),
                    np.nanmin(round(df_high_visual['SOL-USD'],2)), np.nanmin(round(df_high_visual['ADA-USD'],2)), np.nanmin(round(df_high_visual['AVAX-USD'],2)),
                    np.nanmin(round(df_high_visual['MATIC-USD'],2)), np.nanmin(round(df_high_visual['ATOM-USD'],2)), np.nanmin(round(df_high_visual['LINK-USD'],2)),
                    np.nanmin(round(df_high_visual['AXS-USD'],2))]

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