import pandas as pd
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse
import math

import os

os.chdir('D:\Kaggle Projects\Tabular Playground - Mar 2022')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

##################### Edit dataset #######################
## Split into date and time
train = train.drop(columns = ['row_id'])
train['x'] = train['x'].apply(lambda x: str(x))
train['y'] = train['y'].apply(lambda x: str(x))
train['time'] = train['time'].apply(lambda x: pd.to_datetime(x))
train = train.rename(columns= {'time':'period'})
train['date'] = train['period'].dt.strftime('%Y-%m-%d')
train['time'] = train['period'].dt.strftime('%H:%M:%S')
train['unit'] = train[['x','y','direction']].agg('-'.join, axis = 1)

##################### Advanced EDA  ######################
###### Visualising data ########
## Congestion Boxplot
fig, ax = plt.subplots(figsize = (100,70))
ax = sns.boxplot(x='date', y='congestion', data = train)
plt.xticks(rotation = 90)
plt.title('Boxplot of Congestion at date level')
plt.savefig('Boxplot of congestion at date level.pdf', bbox_inches = 'tight')
plt.show()
plt.close()
## Median seems very straight, probably the result of normalization of congestion

fig2, ax2 = plt.subplots(figsize = (100,70))
ax2 = sns.boxplot(x='time', y='congestion', data = train)
plt.xticks(rotation = 90)
plt.title('Boxplot of Congestion at time level')
plt.savefig('Boxplot of congestion at time level.pdf', bbox_inches = 'tight')
plt.show()
plt.close()
## Congestion begins to rapidly rise from 3.40am to 8.00am
## Probably the result of work traffic
## Congestion then steadily rises to peak around 4.00pm to 5.00pm

direction_list = list(train['direction'].drop_duplicates())

g = sns.catplot(
    data = train, x='time',y='congestion', col= 'direction', 
    kind = 'box', col_wrap = 4, aspect = 5)
(g.set_axis_labels('congestion')
 .set_xticklabels( rotation = 90))
plt.savefig('Boxplot of congestion at time and direction level.pdf', bbox_inches = 'tight')
plt.show()
plt.close()
## If want to show different subclasses, facet using catplot
## and use 'col' to subgroup

## Observation:
## NW and SE have a abnormally low and flat congestion, does not follow the trend showed in other directions
g1 = sns.catplot(
    data = train, x='time',y='congestion', col= 'unit', 
    kind = 'box', col_wrap = 4, aspect = 5)
(g1.set_axis_labels('congestion')
 .set_xticklabels( rotation = 90))
plt.savefig('Boxplot of congestion at time and unit level.pdf', bbox_inches = 'tight')
plt.show()
plt.close()

nw = train[train['direction']=='NW']
se = train[train['direction']=='SE']
ab_directions = train.loc[train['direction'].isin(['NW','SE']),:]
sb = train[train['direction']== 'SB']


g3 = sns.catplot(
    data = ab_directions, x='time',y='congestion', col= 'unit', 
    kind = 'box', col_wrap = 2, aspect = 5)
(g3.set_axis_labels('congestion')
 .set_xticklabels( rotation = 90))
plt.savefig('Boxplot of NW and SE congestion at time and unit level.pdf', bbox_inches = 'tight')
plt.show()
plt.close()

## It seems that NW and SE directions congestion level was
## very varied. Chart may be broken?

## so visualization isn't very helpful cus too much data to comprehend visually
## use manual tests

### Test for stationarity
## Augmented Dicket Fuller Test (ADH Test)
from statsmodels.tsa.stattools import adfuller

unit_list = list(set(train['unit']))

def adf_result(unit):
    utrain = train[train['unit'] == unit]
    adf = adfuller(utrain['congestion'], autolag = 't-stat')
    result = [unit,adf[0],adf[1],adf[4]['5%']]
    return result
# Unlike last time, my output cannot be in dataframe...

final_result = pd.DataFrame(map(adf_result,unit_list), columns = ['unit','t-statistic','p-value','5% crit'])
final_result['larger'] = np.where(abs(final_result['t-statistic']) < abs(final_result['5% crit']),'crit','t-stat')

nonstationary = final_result[(final_result['p-value']>= 0.05) &
                             (final_result['larger'] == 'crit')]
stationary = final_result[(final_result['p-value']< 0.05) &
                          (final_result['larger'] == 't-stat')]

## It looks like all of the road congestion follows a stationary pattern

## pd.DataFrme(data) joins as column, pd.DataFrame([data]) joins as row

### Test for seasonality
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# using 0-2-SB as example
example = train[train['unit']=='0-2-SB']

plot_acf(example['congestion'],lags = 10)
plt.savefig('Autocorrelation plot of train.pdf', bbox_inches = 'tight')
plot_pacf(example['congestion'],lags = 10, method = 'ols')
plt.savefig('Partial Autocorrelation plot of train.pdf', bbox_inches = 'tight')
plt.show()
plt.close()
 
### ARIMA Models
### ARIMA(1,0,1)
from statsmodels.tsa.arima.model import ARIMA

model101 = ARIMA(example['congestion'], order = (1,0,1)).fit()
print(model101.summary(), file = open('ARIMA Model 101.txt', 'w'))
# AR segment is statistically significant, but MA segment is not

model001 = ARIMA(example['congestion'], order = (0,0,1)).fit()
print(model001.summary(), file = open('ARIMA Model 001.txt', 'w'))
# MA is now statistically significant


model100 = ARIMA(example['congestion'], order = (1,0,0)).fit()
print(model100.summary(), file = open('ARIMA Model 100.txt', 'w'))
# AR is also statistically significant
# model001 seems to have higher AIC than model100, so continue towards model001

model002 = ARIMA(example['congestion'], order = (0,0,2)).fit()
print(model002.summary(), file = open('ARIMA Model 002.txt', 'w'))
# model001 has higher AIC than model002, so we predict with model001

################# Modelling stage ##########################
## using ARIMA(0,0,1)
def arima_predict(unit):
    unit_set = train[train['unit']==unit].reset_index(drop=True)
    model001 = ARIMA(unit_set['congestion'], order = (0,0,1)).fit()
    validation = unit_set[unit_set['date']>= '1991-08-01'].reset_index(drop = True)
    start_index = validation.index.min()+ unit_set.index.max()
    end_index = validation.index.max()+ unit_set.index.max()
    model001_pred = model001.predict(start = start_index,end = end_index).reset_index(drop = True)
## time series prediction need to use start and end index
    pred_result = pd.concat([validation[['period','unit','congestion']],model001_pred],axis = 1)
    return pred_result

arima_result001 = pd.concat(map(arima_predict, unit_list), axis = 0)

## using ARIMA(0,0,2)
def arima_predict(unit):
    unit_set = train[train['unit']==unit].reset_index(drop=True)
    model002 = ARIMA(unit_set['congestion'], order = (0,0,2)).fit()
    validation = unit_set[unit_set['date']>= '1991-08-01'].reset_index(drop = True)
    start_index = validation.index.min()+ unit_set.index.max()
    end_index = validation.index.max()+ unit_set.index.max()
    model002_pred = model002.predict(start = start_index,end = end_index).reset_index(drop = True)
## time series prediction need to use start and end index
    pred_result = pd.concat([validation[['period','unit','congestion']],model002_pred],axis = 1)
    return pred_result

arima_result002 = pd.concat(map(arima_predict, unit_list), axis = 0)

## using ARIMA(1,0,0)
def arima_predict(unit):
    unit_set = train[train['unit']==unit].reset_index(drop=True)
    model100 = ARIMA(unit_set['congestion'], order = (1,0,0)).fit()
    validation = unit_set[unit_set['date']>= '1991-08-01'].reset_index(drop = True)
    start_index = validation.index.min()+ unit_set.index.max()
    end_index = validation.index.max()+ unit_set.index.max()
    model100_pred = model100.predict(start = start_index,end = end_index).reset_index(drop = True)
## time series prediction need to use start and end index
    pred_result = pd.concat([validation[['period','unit','congestion']],model100_pred],axis = 1)
    return pred_result

arima_result100 = pd.concat(map(arima_predict, unit_list), axis = 0)

print('Model001 RMSE: ',math.sqrt(mse(arima_result001['congestion'],arima_result001['predicted_mean'])))
print('Model002 RMSE: ',math.sqrt(mse(arima_result002['congestion'],arima_result002['predicted_mean'])))
print('Model100 RMSE: ',math.sqrt(mse(arima_result100['congestion'],arima_result100['predicted_mean'])))
## Model001 has the best RMSE score.

#############Actual prediction###############
## Prepare test data
test['x'] = test['x'].apply(lambda x: str(x))
test['y'] = test['y'].apply(lambda x: str(x))
test['time'] = test['time'].apply(lambda x: pd.to_datetime(x))
test = test.rename(columns= {'time':'period'})
test['date'] = test['period'].dt.strftime('%Y-%m-%d')
test['time'] = test['period'].dt.strftime('%H:%M:%S')
test['unit'] = test[['x','y','direction']].agg('-'.join, axis = 1)

def arima_predict(unit):
    unit_set = train[train['unit']==unit].reset_index(drop=True)
    model001 = ARIMA(unit_set['congestion'], order = (0,0,1)).fit()
    test_set = test[test['unit']==unit].reset_index(drop=True)
    start_index = test_set.index.min() + unit_set.index.max()
    end_index = test_set.index.max() + unit_set.index.max()
    model001_pred = model001.predict(start = start_index,end = end_index).reset_index(drop=True)
## time series prediction need to use start and end index
## should also start from the end
    pred_result = pd.concat([test_set[['row_id','period','unit']],model001_pred],axis = 1)
    return pred_result

arima_result001 = pd.concat(map(arima_predict, unit_list), axis = 0)
final_test_result = arima_result001[['row_id','predicted_mean']].sort_values('row_id').rename(columns = {'predicted_mean':'congestion'})

### save output
final_test_result.to_csv('final_test_result.csv', index = False)
