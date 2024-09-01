import graphviz 
import xgboost
import hyperopt 
import mlflow 
import deepchecks
import neptune_xgboost
import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE

"""
This file is used to prepare the data for the model.

There is a lot of data so keep the printing, analyzing, and visualization to a minimum. 

Keep in mind employmet type is misspelled like how it was just spelled


1. Read the data from the excel file and analyze it
2. Merge the data from different sheets based on user_id
3. Deal with missing values
4. Drop columns that are not needed
5. Encode categorical data
6. Encode ordinal data
7. Convert boolean values to numbers
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


#reassign missing values and print to make sure it worked, it did
missing_values = df.isnull().sum()

columns_to_drop = ['Industry', 'User_id', 'User id_x', 'User id_y', 'Pincode', 'Role']

def drop_columns(df, columns):
    df.drop(columns=columns, inplace=True)
    return df

# Call the function to drop the specified columns
"""
Drop industry and role since there are too many categories
Drop user_id since we wont need that and data has already been merged across sheets
Drop pincode since it is hidden in data
"""
df = drop_columns(df, columns_to_drop)

#drop missing values in work experience since it only has 4 missing values (not significant) so remove its rows where its missing
df = df.dropna(subset=['Work Experience'])


"""
# Calculate the correlation matrix
float_columns = df.select_dtypes(include=['float', 'int']).columns
correlation_matrix = df[float_columns].corr()

# Generate the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')


# Save the heatmap to a file
plt.savefig('heatmap.png')

# Open the saved file using an image viewer
os.system('open heatmap.png')
"""

data = df

#categorical one hot encoding
categorical_columns = ['Gender', 'Home', 'Social Profile', 'Loan Category', 'Is_verified', 'Married', 'Employmet type']

#do one hot encoding with the get dummies from pandas
encoded_data = data = pd.get_dummies(data, columns=categorical_columns)


#ordinal encoding 
ordinal_features = ['Tier of Employment', 'Work Experience']

data = encoded_data

#create custom mapping for ordinal encoding
tier_employment_order = list(encoded_data['Tier of Employment'].unique())
tier_employment_order.sort()
work_experience_order = [0, '<1', '1-2', '2-3', '3-5', '5-10', '10+']

custom_mapping = [tier_employment_order, work_experience_order]

#perform ordinal encoding
ordinal_encoder = OrdinalEncoder(categories=custom_mapping)
data[ordinal_features] = ordinal_encoder.fit_transform(data[ordinal_features])

# Convert boolean values to numbers
boolean_columns = data.select_dtypes(include=bool).columns

for column in boolean_columns:
    data[column] = data[column].astype(int)


"""
Use oversampling to blanace the data since the target variable is imbalanced, there are many more values where
defaulter is 0 (no default) than 1 (default), so we use oversampling, which is a technique used to balance 
the data, to ensure the model is not biased towards the majority class (no default)
"""

target_column = 'Defaulter'

def oversample_data(data, target_column):
    # Separate the features and the target variable
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Apply SMOTE oversampling
    oversampler = SMOTE()
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Combine the resampled features and target variable into a new DataFrame
    resampled_data = pd.concat([X_resampled, y_resampled], axis=1)

    return resampled_data

# Call the oversample_data function
resampled_data = oversample_data(data, target_column)

data = resampled_data

# Print the number of rows where defaulter is 1
num_defaulters = len(data[data['Defaulter'] == 1])
print("Number of rows where defaulter is 1:", num_defaulters)

# Print the number of rows where defaulter is 0
num_non_defaulters = len(data[data['Defaulter'] == 0])
print("Number of rows where defaulter is 0:", num_non_defaulters)

# Print the total number of rows
total_rows = len(data)
print("Total number of rows:", total_rows)

