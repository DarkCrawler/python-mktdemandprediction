# This file contains python code for running Holt - Winters (triple exponential smoothing for large markets
# Program computes rms value for both Holt Winters and arima models and recommends the best forecast from the two
# Author : Tushar Bisht

# Libraries
import pandas as pd
from plotly.offline import plot
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing
import numpy as nm
from pyramid.arima import auto_arima

# loading the combined file: <<change file path>>
data = pd.read_csv("<<PATH TO us_201401-201803.txt in data>>", sep='\t')

# selected market : selecting the market for which the forecast has to be run (this script can be further enhanced to fetch O&D passed by user)
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

#### Code logic to implement HOLT-Winters method with seasonality cycle as 12
y_hw = indata.copy()
fit2 = ExponentialSmoothing(nm.asarray(y_hw['Total_apax']), seasonal_periods=12, trend='add', seasonal='mul', ).fit()

# plotting holt winters prediction with actual data
y_hw_plot = pd.concat([indata, outdata])  # combining both in and out samnple to track prediction over the entire launch
y_hw_plot = y_hw_plot.reset_index()
y_hw_plot = y_hw_plot.drop("index", axis=1)
y_hw_plot['Holt_Winter'] = fit2.predict(start=0, end=len(y_hw_plot) - 1)  # predicting with the paramemters returned

# calculating rms value for halt winters method::
rms_holt_winters = sqrt(mean_squared_error(y_hw_plot.Total_apax, y_hw_plot.Holt_Winter))

trace_real = go.Scatter(x=y_hw_plot['monthyear'], y=y_hw_plot['Total_apax'], mode='lines', name='real')
trace_predict = go.Scatter(x=y_hw_plot['monthyear'], y=y_hw_plot['Holt_Winter'], mode='lines', name='predict')

data_plot = [trace_real, trace_predict]
layout = go.Layout(
    title="HoltWinter method ::: RMS :: " + str(rms_holt_winters)
)
fig = go.Figure(data=data_plot,layout=layout)
plot(fig, filename="holt_winter.html")

# Implementing auto:ARIMA for the same market
y_arima = indata.copy()

stepwise_model = auto_arima(y_arima['Total_apax'], start_p=1, start_q=1,
                            max_p=3, max_q=3, m=12,
                            start_P=0, seasonal=True,
                            D=1, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)

y_arima_fcst = outdata.copy()
y_arima_fcst = y_arima_fcst.reset_index()
y_arima_fcst = y_arima_fcst.drop(['index'], axis=1)
future_forecast = stepwise_model.predict(n_periods=15)  # since we have 15 months for the sample data we will be plotting the predicted value along with the real value

for i in range(0, len(y_arima_fcst)):
    y_arima_fcst.loc[i, 'arima'] = future_forecast[i]


# calculating rms value for halt winters method::
rms_arima = sqrt(mean_squared_error(y_arima_fcst.Total_apax, y_arima_fcst.arima))

# plotting arima fcst
trace_arima_real = go.Scatter(x=y_arima_fcst['monthyear'], y=y_arima_fcst['Total_apax'], mode='lines', name='real')
trace_arima_predict = go.Scatter(x=y_arima_fcst['monthyear'], y=y_arima_fcst['arima'], mode='lines', name='predict')
data_plot_arima = [trace_arima_real, trace_arima_predict]  # trace_arima
layout = go.Layout(
    title="Auto-Arima method ::: RMS :: " + str(rms_arima)
)
fig = go.Figure(data=data_plot_arima,layout=layout)
plot(fig, filename="arima_plot.html")



# the winning prediction method:::
print "rmse value for holt winters method:::", rms_holt_winters, "::: rmse value for auto - arima method :::", rms_arima
if rms_arima < rms_holt_winters:
    print "The market is better suited to be forecasted with Auto - Arima method, rmse value ::", rms_holt_winters
else:
    print "The market is better suited to be forecasted with HoltWinters method, rmse value ::", rms_holt_winters
