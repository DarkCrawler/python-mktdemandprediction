# This file contains the code for evaluation of models apart of the selected ones
# Methods contain: Naive method | Rolling average | exponential smoothing | Holt's double smoothing
# Author : Tushar Bisht

# library
import pandas as pd
from plotly.offline import plot
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import Holt
import numpy as nm

#<<change file path>>
data = pd.read_csv("/Users/sg0220142/codebase/as-hackday/repo/predictors/data/us_201401-201803.txt", sep='\t')

oddata = data[(data['ORG'] == 'ABE') & (data['DST'] == 'SFO')]  # selecting a bigger market
oddata['monthyear'] = data["YEAR"].map(str) + '-' + data["MONTH"].map(str)
oddata = oddata.sort_values(['YEAR', 'MONTH'])

# selecting just the total pax and the month year labels to work the models on
oddata = pd.DataFrame(oddata, columns=['monthyear', 'Total_apax'])

# selecting the in and out data:::
indata = oddata[(data['YEAR'] < 2017)]
indata = indata.reset_index()
indata = indata.drop("index", axis=1)

outdata = oddata[data['YEAR'] >= 2017]  # year 2017 and 2018 becomes the sample data to test the model
outdata = outdata.reset_index()
outdata = outdata.drop("index", axis=1)

# implementing naive methid
y_naive = indata.copy()
y_naive.loc[0, 'naive'] = y_naive.loc[0, 'Total_apax']
for i in range(1, len(y_naive)):
    y_naive.loc[i, 'naive'] = y_naive.loc[i - 1, 'Total_apax']

rms_naive = sqrt(mean_squared_error(y_naive.Total_apax, y_naive.naive))

print ("RMS value for naive method ::::", rms_naive)
trace_trim_data = go.Scatter(x=y_naive['monthyear'], y=y_naive['Total_apax'], mode='lines', name='indate')
trace_out_rolling = go.Scatter(x=y_naive['monthyear'], y=y_naive['naive'], mode='lines', name='naiveforecast')

data_naive = [trace_trim_data, trace_out_rolling]
layout = go.Layout(
    title="Naive method :::" + str(rms_naive)
)
fig = go.Figure(data=data_naive,layout=layout)
plot(fig, filename="naive.html")


# applying exponential smoothing
y_expo = indata.copy()
alpha = 0.2
y_expo.loc[0, 'expo'] = y_expo.loc[0, 'Total_apax']
for i in range(1, len(y_expo)):
    y_expo.loc[i, 'expo'] = alpha * y_expo.loc[i - 1, 'Total_apax'] + (1 - alpha) * y_expo.loc[i - 1, 'expo']

rms_expo = sqrt(mean_squared_error(y_expo.Total_apax, y_expo.expo))

print ("RMS value for exponential smoothing method ::::", rms_expo)
trace_trim_data = go.Scatter(x=y_expo['monthyear'], y=y_expo['Total_apax'], mode='lines', name='indate')
trace_out_expo = go.Scatter(x=y_expo['monthyear'], y=y_expo['expo'], mode='lines', name='exponeentialsmoothingforecast')

data_expo = [trace_trim_data, trace_out_expo]
layout = go.Layout(
    title="Exponential smoothing method :::" + str(rms_expo)
)
fig = go.Figure(data=data_expo,layout=layout)
plot(fig, filename="exponentialsmoothing.html")



#applying HOLT (double variable exponential smoothing)
y_holt = indata.copy()
fit1 = Holt(nm.asarray(y_holt['Total_apax'])).fit(smoothing_level=0.3, smoothing_slope=0.1)  # inc in smoothing level and keeping slope constant
for i in range(0, len(y_holt)):
    y_holt.loc[i, 'holt'] = fit1.fittedvalues[i]

rms_holt = sqrt(mean_squared_error(y_holt.Total_apax, y_holt.holt))
print ("RMS value for holt exponential smoothing method ::::", rms_holt)

trace_trim_data = go.Scatter(x=y_holt['monthyear'], y=y_holt['Total_apax'], mode='lines', name='indate')
trace_holt_expo = go.Scatter(x=y_holt['monthyear'], y=y_holt['holt'], mode='lines', name='holtexponentialsmoothingforecast')

data_holt_expo = [trace_trim_data, trace_holt_expo]

layout = go.Layout(
    title="Holt Exponential smoothing method :::" + str(rms_holt)
)
fig = go.Figure(data=data_holt_expo,layout=layout)
plot(fig, filename="holtexponentialsmoothing.html")


