# -*- coding: utf-8 -*-
"""Flight Fare Prediction Project.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

import pickle
# %matplotlib inline

# Source - https://www.kaggle.com/nikhilmittal/flight-fare-prediction-mh
train=pd.read_excel('Data_Train.xlsx')
sample = pd.read_excel('Sample_submission.xlsx')
test = pd.read_excel('Test_set.xlsx')

train.head()

test = pd.concat([test,sample],axis=1)

test.head()

train.shape,test.shape,train.shape[0]/(train.shape[0]+test.shape[0])*100

df= pd.concat([train,test])
df.shape

df.head()

"""### Feature Engineering"""

# Droping columns that does not seem practical to ask to a customer.
df.drop(labels=['Route','Arrival_Time','Additional_Info'],axis=1,inplace=True)

df.head()

df['Airline'].value_counts()

df['Source'].value_counts()

df['Destination'].value_counts()

df.isnull().sum()

print(df.shape)
df.dropna(inplace=True)
print(df.shape)

df.head()

df['Day']= df['Date_of_Journey'].str.split('/').str[0]
df['Month']= df['Date_of_Journey'].str.split('/').str[1]
df['Year']= df['Date_of_Journey'].str.split('/').str[2]

df.head()

df['Total_Stops']=df['Total_Stops'].str.replace('non-','0 ')

df.head()

df.info()

df['Stops'] = df['Total_Stops'].str.split().str[0]
df.head()

df['Departure_Hour'] = df['Dep_Time'].str.split(':').str[0]
df['Departure_Minute'] = df['Dep_Time'].str.split(':').str[1]

df.head()

#Converting the datatype o newly created features
df['Day'] = df['Day'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Year'] = df['Year'].astype(int)
df['Stops'] = df['Stops'].astype(int)
df['Departure_Hour'] = df['Departure_Hour'].astype(int)
df['Departure_Minute'] = df['Departure_Minute'].astype(int)

df.info()

print(df.Year.value_counts().index)
print("All data is of year 2019")

df.Airline.value_counts().index

source_dict = {y:x for x,y in enumerate(df.Source.value_counts().index.sort_values())}
source_dict

df.Destination.value_counts().index.sort_values()

destination_dict = {'Banglore':0,'Cochin':1,'Delhi':2,'Kolkata': 3,'Hyderabad':4,'New Delhi':5}

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Airline_Encoded']= le.fit_transform(df['Airline'].values)

df3 = df[['Airline']].copy()
df3['Encoded']=df['Airline_Encoded']
df3=df3.drop_duplicates('Airline').reset_index().iloc[:,1:]
d5=df3.Airline.values
d6=df3.Encoded.values
airline_dict = dict(zip(d5,d6))

print(airline_dict)

df['Source_Encoded']=df['Source'].map(source_dict)
df['Destination_Encoded']=df['Destination'].map(destination_dict)

df.head()

def convert_time(time_str):
    hour = int(time_str[:2])

    if 5 <= hour <= 11:
        return "Morning"
    elif 12 <= hour <= 16:
        return "Afternoon"
    elif 17 <= hour <= 20:
        return "Evening"
    else:
        return "Night"

df['Dep_Time']=df['Dep_Time'].apply(convert_time)

df.head()

"""Exploratory Data Analysis"""

import matplotlib.pyplot as plt
plt.style.use('classic')
plt.figure(figsize=(40,5))
plt.grid(visible=None, which='major', axis='both')
sns.countplot(x="Airline", data=df, hue=df['Airline'], palette="hls", legend=True)
plt.title('Flights Count of Different Airlines',fontsize=15)
plt.xlabel('Airline',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.show()

plt.figure(figsize=(40,10))
sns.boxplot(x=df['Airline'], y=df['Price'], hue=df['Airline'], palette='hls', legend=False)
plt.grid(visible=None, which='major', axis='both')
plt.title('Airlines Vs Price',fontsize=15)
plt.xlabel('Airline',fontsize=15)
plt.ylabel('Price',fontsize=15)
plt.show()

plt.figure(figsize=(40,10))
sns.boxplot(x=df['Airline'], y=df['Stops'], hue=df['Airline'], palette='hls', legend=False)
plt.grid(visible=None, which='major', axis='both')
plt.title('Airlines Vs Stops',fontsize=15)
plt.xlabel('Airline',fontsize=15)
plt.ylabel('Stops',fontsize=15)
plt.show()

plt.figure(figsize=(40,10))
sns.boxplot(x=df['Dep_Time'], y=df['Stops'], hue=df['Dep_Time'], palette='hls', legend=False)
plt.grid(visible=None, which='major', axis='both')
plt.title('Airlines Vs Departure Time',fontsize=15)
plt.xlabel('Airline',fontsize=15)
plt.ylabel('Departure Time',fontsize=15)
plt.show()

plt.figure(figsize=(10,5))
plt.grid(visible=None, which='major', axis='both')
sns.boxplot(x=df['Stops'], y=df['Price'], hue=df['Stops'], palette='hls', legend=False)
plt.title('Stops Vs Ticket Price',fontsize=15)
plt.xlabel('Stops',fontsize=15)
plt.ylabel('Price',fontsize=15)
plt.show()

plt.figure(figsize=(10,5))
plt.grid(visible=None, which='major', axis='both')
sns.boxplot(x=df['Dep_Time'], y=df['Price'], hue=df['Dep_Time'], palette='hls', legend=False)
plt.title('Departure Time Vs Ticket Price',fontsize=15)
plt.xlabel('Departure Time',fontsize=15)
plt.ylabel('Price',fontsize=15)
plt.show()

plt.figure(figsize=(24,10))

plt.subplot(1,2,1)
sns.boxplot(x='Source', y='Price', hue='Source', data=df, palette='hls', legend=False)
plt.title('Source City Vs Ticket Price', fontsize=20)
plt.xlabel('Source City', fontsize=15)
plt.ylabel('Price', fontsize=15)
plt.grid(visible=None, which='major', axis='both')

plt.subplot(1,2,2)
sns.boxplot(x='Destination', y='Price', hue='Destination', data=df, palette='hls', legend=False)
plt.title('Destination City Vs Ticket Price', fontsize=20)
plt.xlabel('Destination City', fontsize=15)
plt.ylabel('Price', fontsize=15)
plt.grid(visible=None, which='major', axis='both')

plt.show()

plt.style.use('dark_background')
plt.figure(figsize=(20,8))
sns.lineplot(data=df,x='Duration',y='Price',hue='Stops',palette='bright')
plt.title('Ticket Price Versus Flight Duration Based on Stops',fontsize=20)
plt.gca().set_xticklabels([])
plt.xlabel('Duration increases ----->',fontsize=15)
plt.ylabel('Price',fontsize=15)
plt.show()

df = df.drop(['Airline','Source','Destination','Date_of_Journey','Dep_Time','Duration','Total_Stops','Year'],axis=1)
df.head()

"""### Machine Learning"""

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

df.shape

df_train = df[0:10600]
df_test = df[10600:]

X = df_train.drop(['Price'],axis=1)
y = df_train.Price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = SelectFromModel(Lasso(alpha=0.005,random_state=0))
model.fit(X_train,y_train)

model.get_support(),model.get_params()

features_selected = X_train.columns[model.get_support()]

features_selected,X_train.shape,len(features_selected)

"""We see that year feature is not selected so we will eliminate Year feature from our dataset"""

X_train.head()

X_test.head()

"""### Feature Normalization"""

import scipy.stats as stat

for x in list(X_train.columns):
    X_train[x] = stat.yeojohnson(X_train[x])[0]

for y in list(X_test.columns):
    X_test[y] = stat.yeojohnson(X_test[y])[0]

X_train.head()

X_test.head()

"""### Linear Regression Model"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

lm = LinearRegression()

lm.fit(X_train,y_train)

predictions = lm.predict(X_test)

sns.kdeplot(x=predictions-y_test)

r2_score(y_true=y_test,y_pred=predictions)

"""As we can see that r2 score is fair if not the best and other algos can also be applied which wil eventually make r2_score significantly better than regression but since we were doing regression project we will limit to this algo this time and will explore other algos later."""

lm.coef_

plt.scatter(y_test,predictions)

y_test

predictions

lm.score(X_train,y_train)

"""Random Forest Regressor Model"""

from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor()
reg.fit(X_train,y_train)
tr_score=reg.score(X_train,y_train)
print("Train R^2 Score:", tr_score)
train_accuracy_percentage = tr_score * 100
formatted_accuracy = f"{train_accuracy_percentage:.2f}%"

print(f"Train Accuracy is: {formatted_accuracy}")

y_pred = reg.predict(X_test)

tt_score=r2_score(y_true=y_test,y_pred=y_pred)
print("Test R^2 Score:", tt_score)
test_accuracy_percentage = tt_score * 100
formatted_accuracy = f"{test_accuracy_percentage:.2f}%"

print(f"Test Accuracy is: {formatted_accuracy}")

"""Extra Trees Regressor Model"""

from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor()

etr.fit(X_train,y_train)
train_score=etr.score(X_train,y_train)
print("Train R^2 Score:", train_score)
train_accuracy_per = train_score * 100
formatted_acc = f"{train_accuracy_per:.2f}%"

print(f"Train Accuracy is: {formatted_acc}")

y_pred = etr.predict(X_test)

test_score=r2_score(y_true=y_test,y_pred=y_pred)
print("Test R^2 Score:", test_score)
test_accuracy_perc = test_score * 100
formatted_accr = f"{test_accuracy_perc:.2f}%"

print(f"Test Accuracy is: {formatted_accr}")

print("End of File")

