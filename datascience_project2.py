#!/usr/bin/env python
# coding: utf-8

# In[267]:


# importing all necessary libraries.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import datetime
import scipy.stats
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


# In[127]:


# setting up the working directory.
os.chdir("C:\\Users\\SONY\\Documents")
os.getcwd()


# In[128]:


train=pd.read_csv("train_cab.csv")
test=pd.read_csv("test_cab.csv")


# In[129]:


train.head()


# In[130]:


train.shape


# In[131]:


test.head()


# In[132]:


# getting the details about the data sets.
train.info(), test.info()


# In[133]:


# Data cleaning.
train['passenger_count'].value_counts()


# In[10]:


test['passenger_count'].value_counts()


# In[134]:


train['fare_amount'].sort_values(ascending=False)


# In[135]:


train["fare_amount"] = pd.to_numeric(train["fare_amount"],errors = "coerce")


# In[136]:


train["fare_amount"].dtype


# In[137]:


train["fare_amount"].sort_values(ascending=False)


# In[138]:


sum(train["fare_amount"])is()


# In[139]:


sum(train['fare_amount']>453) # considring the fare amount above 453 as outlier dropping out the observation.


# In[140]:


sum(train['fare_amount']<=0) # at the same time fare amount cannot be 0 or negative


# In[141]:


# the passenger count maxiumum is 6 if considring an SUV, passengr count cannot be less than one.
sum(train['passenger_count']>6),sum(train['passenger_count']<1) # so filtering out those observation which satisfies the above condition.


# In[142]:


# Latitudes range from -90 to 90.Longitudes range from -180 to 180. Removing which does not satisfy these ranges
print('pickup_longitude above 180={}'.format(sum(train['pickup_longitude']>180)))
print('pickup_longitude below -180={}'.format(sum(train['pickup_longitude']<-180)))
print('pickup_latitude above 90={}'.format(sum(train['pickup_latitude']>90)))
print('pickup_latitude below -90={}'.format(sum(train['pickup_latitude']<-90)))
print('dropoff_longitude above 180={}'.format(sum(train['dropoff_longitude']>180)))
print('dropoff_longitude below -180={}'.format(sum(train['dropoff_longitude']<-180)))
print('dropoff_latitude below -90={}'.format(sum(train['dropoff_latitude']<-90)))
print('dropoff_latitude above 90={}'.format(sum(train['dropoff_latitude']>90)))


# In[143]:


# latitude and longitude cannot be comprised of zero value, so filtering up the values.
for i in ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']:
    print(i,'equal to 0={}'.format(sum(train[i]==0)))


# In[144]:


# Data cleaning. # by the above observation we can say that most of the data is invalid and we need to clean it.
train=train.drop(train[train['fare_amount']<1].index,axis=0)
train=train.drop(train[train['fare_amount']>453].index,axis=0)
train=train.drop(train[train['passenger_count']>6].index,axis=0)
train=train.drop(train[train['passenger_count']<1].index,axis=0)
train=train.drop(train[train['pickup_latitude']>90].index,axis=0)


# In[145]:


# checking the data after cleaning.
sum(train['fare_amount']<1),sum(train['passenger_count']>6),sum(train['passenger_count']<1),sum(train['fare_amount']>453),sum(train['pickup_latitude']>90)


# In[146]:


train.shape


# In[147]:


# FEATURE ENGINEERING 
# checking for missing values.
print(train.isnull().sum()),print(test.isnull().sum())


# In[148]:


# removing the NA observations.(as they are very less in count to impute.)# nearly 77 missing values.
train = train.drop(train[train['fare_amount'].isnull()].index, axis=0)
train = train.drop(train[train['passenger_count'].isnull()].index, axis=0)


# In[149]:


train.isnull().sum(),train.shape # now there are Zero missing values.


# In[150]:


train.describe()


# In[28]:


test.describe()


# In[151]:


train['passenger_count'].value_counts()


# In[152]:


train=train.drop(train[train['passenger_count']==1.3].index,axis=0) # passenger count cannot be 1.3 so dropping it out.


# In[153]:


test['passenger_count'].value_counts()


# In[154]:


#As we know that we have given pickup longitute and latitude values and same for drop. 
#So we need to calculate the distance Using the haversine formula and we will create a new variable called distance
from math import radians, cos, sin, asin, sqrt

def haversine(a):
    lon1=a[0]
    lat1=a[1]
    lon2=a[2]
    lat2=a[3]
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c =  2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km


# In[155]:


train['distance'] = train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[156]:


test['distance'] = test[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[157]:


train['distance'].sort_values(ascending=False)


# In[158]:


sum(train['distance']>150),sum(train['distance']<=0)


# In[159]:


train


# In[160]:


train=train.drop(train[train['distance']<=0].index,axis=0)
train=train.drop(train[train['distance']>150].index,axis=0)


# In[161]:


train['distance'].sort_values(ascending=False)


# In[162]:


sum(train['distance']>150),sum(train['distance']<=0)


# In[163]:


test['distance'].sort_values(ascending=False)


# In[164]:


sum(test['distance']<=0)


# In[165]:


test=test.drop(test[test['distance']==0].index,axis=0)


# In[166]:


train.describe()


# In[167]:


test.describe()


# In[168]:


train


# In[174]:


train.dtypes


# In[176]:


train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'],errors='coerce')


# In[177]:


test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])


# In[178]:


# seperating the Pickup_datetime column into separate field like year, month, day of the week, etc

train['year'] = train['pickup_datetime'].dt.year
train['Month'] = train['pickup_datetime'].dt.month
train['Date'] = train['pickup_datetime'].dt.day
train['Day'] = train['pickup_datetime'].dt.dayofweek
train['Hour'] = train['pickup_datetime'].dt.hour
train['Minute'] = train['pickup_datetime'].dt.minute
# lets do same for test dataset.         
test['year'] = test['pickup_datetime'].dt.year
test['Month'] = test['pickup_datetime'].dt.month
test['Date'] = test['pickup_datetime'].dt.day
test['Day'] = test['pickup_datetime'].dt.dayofweek
test['Hour'] = test['pickup_datetime'].dt.hour
test['Minute'] =test['pickup_datetime'].dt.minute


# In[180]:


train.info(),train.shape


# In[181]:


test.info(),test.shape


# In[184]:


train.head()


# In[186]:


# deleting the features.
dropfeatures = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude','Minute']
train = train.drop(dropfeatures, axis = 1)
drop_features = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude','Minute']
test = test.drop(drop_features, axis = 1)


# In[187]:


train.head()


# In[216]:


train.isnull().sum()


# In[215]:


train=train.dropna()


# In[189]:


train.shape


# In[190]:


test.head(), test.shape


# In[217]:


# converting the data in required data type.
train['passenger_count'] = train['passenger_count'].astype('int64')
train['year'] = train['year'].astype('int64')
train['Month'] = train['Month'].astype('int64')
train['Date'] = train['Date'].astype('int64')
train['Day'] = train['Day'].astype('int64')
train['Hour'] = train['Hour'].astype('int64')


# In[218]:


test['passenger_count'] = test['passenger_count'].astype('int64')
test['year'] = test['year'].astype('int64')
test['Month'] = test['Month'].astype('int64')
test['Date'] = test['Date'].astype('int64')
test['Day'] = test['Day'].astype('int64')
test['Hour'] = test['Hour'].astype('int64')


# In[219]:


train.dtypes


# In[220]:


test.dtypes


# In[222]:


# DATA VISUALIZATIONS.
plt.hist(train['passenger_count'],color='green')


# In[223]:


# passenger count for test data.
plt.hist(test['passenger_count'],color='green')


# In[224]:


# relationship between passenger count and fare amount.
plt.figure(figsize=(10,5))
plt.scatter(x="passenger_count",y="fare_amount", data=train,color='blue')
plt.xlabel('No. of passengers')
plt.ylabel('Fare_amount')
plt.show()


# In[225]:


# relationship between date and fare amount.
plt.figure(figsize=(15,6))
plt.scatter(x="Date",y="fare_amount", data=train,color='blue')
plt.xlabel('Date')
plt.ylabel('Fare_amount')
plt.show()


# In[227]:


# number of cabs with respect to hours..
plt.figure(figsize=(15,7))
train.groupby(train["Hour"])['Hour'].count().plot(kind="bar")
plt.show()


# In[228]:


# realationship between fare and hour
plt.figure(figsize=(10,5))
plt.scatter(x="Hour",y="fare_amount", data=train,color='blue')
plt.xlabel('Hour')
plt.ylabel('Fare_amount')
plt.show()


# In[229]:


# realationship between fare and day
plt.figure(figsize=(10,5))
plt.scatter(x="Day",y="fare_amount", data=train,color='blue')
plt.xlabel('Day')
plt.ylabel('Fare_amount')
plt.show()


# In[230]:


# realationship between fare and distance
plt.figure(figsize=(10,5))
plt.scatter(x="distance",y="fare_amount", data=train,color='blue')
plt.xlabel('distance')
plt.ylabel('fare')
plt.show()


# In[ ]:


# It is quite obvious that distance will effect the amount of fare


# In[ ]:


# features scaling


# In[231]:


# Normality check of training data is uniformly distributed or not
for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(train[i],bins='auto',color='green')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()


# In[232]:


#since skewness of target variable is high, apply log transform to reduce the skewness-
train['fare_amount'] = np.log1p(train['fare_amount'])
train['distance'] = np.log1p(train['distance'])


# In[233]:


# Normality Re-check to check data is uniformly distributed or not after log transformartion

for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(train[i],bins='auto',color='green')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()


# In[ ]:


# Here we can see bell shaped distribution. Hence our continous variables are now normally distributed, we will use not use any Feature Scalling technique. i.e, Normalization or Standarization for our training data


# In[234]:


# Normality check for test data is uniformly distributed or not-

sns.distplot(test['distance'],bins='auto',color='green')
plt.title("Distribution for Variable "+i)
plt.ylabel("Density")
plt.show()


# In[235]:


# since skewness of distance variable is high, apply log transform to reduce the skewness
test['distance'] = np.log1p(test['distance'])


# In[236]:


#rechecking the distribution for distance
sns.distplot(test['distance'],bins='auto',color='green')
plt.title("Distribution for Variable "+i)
plt.ylabel("Density")
plt.show()


# In[ ]:


# As we can see a bell shaped distribution. Hence our continous variables are now normally distributed, we will use not use any Feature Scalling technique. i.e, Normalization or Standarization for our test data


# In[ ]:


# Applying ML Algorithm


# In[237]:


# train test split for further modelling
X_train, X_test, y_train, y_test = train_test_split( train.iloc[:, train.columns != 'fare_amount'], 
                         train.iloc[:, 0], test_size = 0.20, random_state = 1)


# In[238]:


print(X_train.shape)
print(X_test.shape)


# In[241]:


# Building LR model on top of training dataset
fit_LR = LinearRegression().fit(X_train , y_train)


# In[242]:



#prediction on train data
pred_train_LR = fit_LR.predict(X_train)


# In[243]:


#prediction on test data
pred_test_LR = fit_LR.predict(X_test)


# In[246]:


# calculating RMSE for train data
RMSE_train_LR= np.sqrt(mean_squared_error(y_train, pred_train_LR))


# In[247]:


# calculating RMSE for test data
RMSE_test_LR = np.sqrt(mean_squared_error(y_test, pred_test_LR))


# In[248]:


print("Root Mean Squared Error For Training data = "+str(RMSE_train_LR))
print("Root Mean Squared Error For Test data = "+str(RMSE_test_LR))


# In[249]:


# calculate R^2 for train data
from sklearn.metrics import r2_score
r2_score(y_train, pred_train_LR)


# In[250]:


r2_score(y_test, pred_test_LR)


# In[ ]:





# In[ ]:


# DECISION TREE MODEL:


# In[251]:


fit_DT = DecisionTreeRegressor(max_depth = 2).fit(X_train,y_train)


# In[252]:


#prediction on train data
pred_train_DT = fit_DT.predict(X_train)

#prediction on test data
pred_test_DT = fit_DT.predict(X_test)


# In[253]:


# calculating RMSE for train data
RMSE_train_DT = np.sqrt(mean_squared_error(y_train, pred_train_DT))

# calculating RMSE for test data
RMSE_test_DT = np.sqrt(mean_squared_error(y_test, pred_test_DT))


# In[254]:


print("Root Mean Squared Error For Training data = "+str(RMSE_train_DT))
print("Root Mean Squared Error For Test data = "+str(RMSE_test_DT))


# In[255]:


# R^2 calculation for train data
r2_score(y_train, pred_train_DT)


# In[256]:


# R^2 calculation for test data
r2_score(y_test, pred_test_DT)


# In[ ]:





# In[ ]:


# RANDOM FOREST MODEL:


# In[257]:


fit_RF = RandomForestRegressor(n_estimators = 200).fit(X_train,y_train)


# In[258]:


# prediction on train data
pred_train_RF = fit_RF.predict(X_train)

# prediction on test data
pred_test_RF = fit_RF.predict(X_test)


# In[259]:


# calculating RMSE for train data
RMSE_train_RF = np.sqrt(mean_squared_error(y_train, pred_train_RF))

#calculating RMSE for test data
RMSE_test_RF = np.sqrt(mean_squared_error(y_test, pred_test_RF))


# In[260]:


print("Root Mean Squared Error For Training data = "+str(RMSE_train_RF))
print("Root Mean Squared Error For Test data = "+str(RMSE_test_RF))


# In[261]:


# calculate R^2 for train data
r2_score(y_train, pred_train_RF)


# In[262]:


# calculate R^2 for test data
r2_score(y_test, pred_test_RF)


# In[ ]:





# In[ ]:


# Prediction of fare from provided test dataset :


# In[ ]:


# We have already cleaned and processed our test dataset along with our training dataset. Hence we will be predicting using grid search CV for random forest model


# In[268]:


# Grid Search CV for random Forest model
regr = RandomForestRegressor(random_state = 0)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

# Create the grid
grid_search = {'n_estimators': n_estimator,
               'max_depth': depth}

# Grid Search Cross-Validation with 5 fold CV
gridcv_rf = GridSearchCV(regr, param_grid = grid_search, cv = 5)
gridcv_rf = gridcv_rf.fit(X_train,y_train)
view_best_params_GRF = gridcv_rf.best_params_

# Apply model on test data
predictions_GRF_test_Df = gridcv_rf.predict(test)


# In[269]:


predictions_GRF_test_Df


# In[270]:


test['Predicted_fare'] = predictions_GRF_test_Df


# In[271]:


test.head()

