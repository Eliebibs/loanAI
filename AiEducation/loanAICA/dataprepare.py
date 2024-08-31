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

Keep in mind employmet type is misspelled like how it was just spelled


1. Read the data from the excel file and analyze it
2. Merge the data from different sheets based on user_id
3. Deal with missing values
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

# Describe the data and show the number of missing values per column
description = df.describe()
missing_values = df.isnull().sum()

"""
This section here is used to replace missing values in the columns with the value 'missing'
It is done for columns with categorical data where it makes "sense" to replace missing values with a "missing"
"""

replace_with = 'missing'
columns_to_replace = ['Social Profile', 'Is_verified', 'Married', 'Employmet type']

def replace_missing_values(df, columns, replacement):
    for column in columns:
        df[column].fillna(replacement, inplace=True)
    return df

#call the function to replace missing values
df = replace_missing_values(df, columns_to_replace, replace_with)

#reassign missing values after replacing missing values and print to ensure it worked (it did)
missing_values = df.isnull().sum()

#now deal with missing values for amount, assing 1 if missing and 0 if not
df['amount_missing'] = np.where(df['Amount'].isnull(), 1, 0)

#replace null with -1000 to differetiate it from the other values, reassign variables
replace_with = -1000
columns_to_replace = ['Amount']
#call the function
df = replace_missing_values(df, columns_to_replace, replace_with)

#reassign missing values and print to make sure it worked, it did
missing_values = df.isnull().sum()

"""
Replace missing values for tier of emplyment with Z since its an ordinal variable
"""

replace_with = 'Z'
columns_to_replace = ['Tier of Employment']

df = replace_missing_values(df, columns_to_replace, replace_with)


#drop missing values in industry and work experience since they only have 4 missing values (not significant)
df = df.dropna(subset=['Industry', 'Work Experience'])


#reassign missing values and print to make sure it worked, it did
missing_values = df.isnull().sum()
print(missing_values)





