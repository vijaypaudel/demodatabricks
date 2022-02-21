# Databricks notebook source
pip install pandas

# COMMAND ----------

pip install numpy

# COMMAND ----------

pip install sklearn

# COMMAND ----------

pip install flask

# COMMAND ----------

import pickle

# COMMAND ----------

from flask import Flask,render_template,request,jsonify

# COMMAND ----------

pip install mlflow

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

df = pd.read_csv('/dbfs/FileStore/tables/test.csv')

# COMMAND ----------

df.head()

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

df.replace({'experience': {'one':1,'two':2, 'three':3,'four':4,'five':5, 'six':6,'seven':7,'eight':8, 'nine':9,
                           'ten':10,'eleven':11}}, inplace=True)

# COMMAND ----------

df.head(10)

# COMMAND ----------

x = df.drop(columns=['salary','index'])
y = pd.DataFrame(df['salary'])

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# COMMAND ----------

mlflow.sklearn.autolog()
 
# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
with mlflow.start_run():
  
  # Set the model parameters. 
  n_estimators = 190
  max_depth = 6
  max_features =7
  
  # Create and train model.
  rf = li = LinearRegression()
  model = li.fit(X_train,y_train)
  
  # Use the model to make predictions on the test dataset.
  pred_test = model.predict(X_test)

# COMMAND ----------




# COMMAND ----------

