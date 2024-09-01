import graphviz 
import xgboost as xgb
import hyperopt as hp
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
from sklearn.model_selection import train_test_split
import neptune
from hyperopt.pyll import scope
from sklearn.metrics import mean_absolute_error
from hyperopt import fmin, tpe, hp, STATUS_OK
from neptune.integrations.xgboost import NeptuneCallback
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle


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
8. Balance the data using oversampling
9. Split the data into training and testing data
10. Train the model using XGBoost
11. Optimize the hyperparameters using hyperopt
12. Train the model with the best hyperparameters
13. Print the best hyperparameters
14. Print the classification report and confusion matrix
15. Save the model to a file
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
oversampling is done using the SMOTE technique and duplicates the minority class to balance the data
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

"""
After oversampling data to balance it, here is the stats
Number of rows where defaulter is 1: 130254
Number of rows where defaulter is 0: 130254
Total number of rows: 260508
"""

"""
Test training and testing data
test size is 20% of the data, 80% is used for training
the random state is set to 42 to ensure the same split is used each time
XGboost creates the validation set itself so we dont need to create a validation set
"""

#set x equal to all columns except the target column and y equal to the target column (defaulter)
X = data.drop(columns=[target_column])
y = data[target_column]

#split the data into training and testing data
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)


"""
Training the model with XGBoost
"""

#configure neptune
api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNTAwNmRjMi05NWVmLTQ0NDYtYTliMi1jN2IyM2YzODNmYTcifQ=="

run = neptune.init_run(project='loanAiCA', api_token=api_token)


neptune_callback = NeptuneCallback(run=run, log_tree=[0, 1, 2,3])


#using hyperopt to optimize the hyperparameters
#it sets a range for each hyperparameter and then tries to find the best combination of hyperparameters
search_space = {
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'max_depth': scope.int(hp.uniform('max_depth', 1, 100)),
    'min_child_weight': hp.loguniform('min_child_weight', -2, 3),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'gamma': hp.loguniform('gamma', -10, 10),
    'alpha': hp.loguniform('alpha', -10, 10),
    'lambda': hp.loguniform('lambda', -10, 10),
    'objective' : 'binary:logistic',
    'eval_metric': 'error',
    'seed': 123
}

#this function initiates the model with the params and trains it, it also logs models in neptune
def train_model_xgboost(params):
    start_time = time.time()

    #this creates the model and sets the params, num_boost_round is the number of boosting rounds, verbose_eval is set to false to not print the output, and the neptune callback is used to log the model
    model = xgb.XGBClassifier(params=params, num_boost_round=5000, verbose_eval=False, callbacks = [neptune_callback])

    run_time = time.time() - start_time

    #this trains the model
    model.fit(train_x, train_y)

    #this predicts the values, calculates the mean absolute error, and returns the loss
    predictions = model.predict(test_x)
    mae = mean_absolute_error(test_y, predictions)

    return {'status' : STATUS_OK, 'loss' : mae }

#this function is used to train the model with the best parameters, it is used to get the best parameters from the hyperopt search
#we are not using this function in the final model
def random_forest_classifier_grid_search(param_grid, x_train, y_train):

    #create randome forest classifier
    rf = RandomForestClassifier()

    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring = 'accuracy')
    grid_search.fit(x_train, y_train)

    return grid_search.best_params_


#finding best paremeters using the fmin function from hyperopt
#fn is the function to optimize
#space is the search space for the paraemters (previously defined)
#max_evals is the number of iterations to run (epochs), the more you do the better it is but it takes longer
#rstate is the random state
#trials is the trials object to store the results of the experiments

best_params = fmin(
    fn = train_model_xgboost, 
    space = search_space, 
    algo = tpe.suggest, 
    max_evals = 15, 
    rstate = 
    np.random.default_rng(123)
    #trials = spark_trials
    )

#access the best hyperparameters
best_hyperparams = {k : best_params[k] for k in best_params}

#Train the final model with XGBoost using the best hyperparameters
final_model = xgb.XGBClassifier(
    max_depth_best= int(best_hyperparams['max_depth']),
    learning_rate_best = best_hyperparams['learning_rate'],
    gamma_best = best_hyperparams['gamma'],
    subsample_best = best_hyperparams['subsample'],
    colsample_bytree_best = best_hyperparams['colsample_bytree'],
    random_state = 42,
    tree_method = 'hist', 
    enable_categorical = True # use GPU for faster training
)

final_model.fit(train_x, train_y) #train the final model

y_pred = final_model.predict(test_x) #predict the values

#following graphs are shown in neptune 

## Generate and log confusion matrix
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(final_model, test_x, test_y, ax=ax)
run['confusion_matrix'].upload(fig)
plt.close(fig)

# Generate and log ROC curve
fig, ax = plt.subplots()
RocCurveDisplay.from_estimator(final_model, test_x, test_y, ax=ax)
run['roc_curve'].upload(fig)
plt.close(fig)

# Generate and log Precision-Recall curve
fig, ax = plt.subplots()
PrecisionRecallDisplay.from_estimator(final_model, test_x, test_y, ax=ax)
run['precision_recall_curve'].upload(fig)
plt.close(fig)

run.stop()

# Print the best hyperparameters
print("Best Hyperparameters:")
for param, value in best_hyperparams.items():
    print(f"{param}: {value}")

#this prings out metrics to understand how well the model is performing
print("Classification Report: \n", classification_report(test_y, y_pred)) #print the classification report
print("Confusion Matrix: \n", confusion_matrix(test_y, y_pred)) #print the confusion matrix


"""
This saves the final model to my folder
# Define the file path
file_name = '/Users/eliebibliowicz/Desktop/AiEducation/loanAICA/bestFitModel.pkl'

# Save the best fit model to a file
with open(file_name, 'wb') as file:
    pickle.dump(final_model, file)

"""

"""
This logs the model to neptune
model = neptune.init_model(
    name="Prediction model",
    key="MOD", 
    project="eliebibliowicz/loanAiCA", 
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNTAwNmRjMi05NWVmLTQ0NDYtYTliMi1jN2IyM2YzODNmYTcifQ==", # your credentials
)

file_name = '/Users/eliebibliowicz/Desktop/AiEducation/loanAICA/bestFitModel.pkl'
model["model/bestFitModel.pkl"].upload(file_name)
"""
"""
used for grid search with random forest classifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
}
"""

#use grid search to find the best parameters for the random forest classifier (not used in the final model)
#this is antoher way to find the best parameters, not as efficient as hyperopt with xgboost
#best_params_forest = random_forest_classifier_grid_search(param_grid, train_x, train_y)


