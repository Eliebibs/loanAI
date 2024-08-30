import graphviz 
import xgboost
import hyperopt 
import mlflow 
import deepchecks
import neptune_xgboost
import sklearn
import pandas as pd
import numpy as np

"""
This file is used to prepare the data for the model.

There is a lot of data so keep the printing, analyzing, and visualization to a minimum. 
"""

def read_excel_data(path, sheet_names):
    loan_information = pd.read_excel(path, sheet_name=sheet_names[0])
    employment = pd.read_excel(path, sheet_name=sheet_names[1])
    personal_information = pd.read_excel(path, sheet_name=sheet_names[2])
    other_information = pd.read_excel(path, sheet_name=sheet_names[3])
    return loan_information, employment, personal_information, other_information

# Path to data
path = "Credit_Risk_Dataset.xlsx"

# Read data from file
sheet_names = ['loan_information', 'Employment', 'Personal_information', 'Other_information']

loan_information, employment, personal_information, other_information = read_excel_data(path, sheet_names)

"""
Following code merges the data from different sheets (data is stored on different pages) based on user_id
"""
# Merge data based on user_id
merged_df = pd.merge(loan_information, employment, left_on='User_id', right_on='User id')

#merge with personal information based on user_id
merged_df = pd.merge(merged_df, personal_information, left_on='User_id', right_on='User id')

#merge with other information based on user_id
merged_df = pd.merge(merged_df, other_information, left_on='User_id', right_on='User_id')

#set the merged data to the var df
df = merged_df

#assign frist 5 rows to head var
head = df.head()


