import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import re
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--only", nargs="+", help="list of string values")

args = parser.parse_args()

only_models = []
if args.only:
    for value in args.only:
        only_models.append(value)

train_makes_tournament = False
train_seed = False
train_result = False

if len(only_models) == 0:
    train_makes_tournament = True
    train_seed = True
    train_result = True
else:
    for model in only_models:
        if model == 'makes_tournament':
            train_makes_tournament = True
        elif model == 'seed':
            train_seed = True
        elif model == 'result':
            train_result = True
        else:
            print(f'Unknown Argument: {model}')


df13_20 = pd.read_csv('data/cbb.csv')
df21 = pd.read_csv('data/cbb21.csv')
df23 = pd.read_csv('data/cbb23.csv')

# Several things required to clean the 2023 dataset
# Many cells have 2 values entered on seperate lines. Invariably, we only want the first
for i, row in df23.iterrows():
    for j, cell in row.items():
        if '\n' in cell:
            df23.iloc[i][j] = re.split('\n', cell)[0]

# Remove rows that are duplicates of the title row
rows_to_drop = []
for i, row in df23.iterrows():
    if row['2P%D'] == '2P%D':
        rows_to_drop.append(i)
        
df23 = df23.drop(rows_to_drop)

df23['W'] = df23.apply(lambda row: re.split('-', row['Rec'])[0], axis=1)

df23['YEAR'] = df23.apply(lambda row: 2023, axis=1)

old_new_col_label = {
    'Team': 'TEAM',
    'Conf': 'CONF',
    'AdjOE': 'ADJOE',
    'AdjDE': 'ADJDE',
    'Barthag': 'BARTHAG',
    'EFG%': 'EFG_O',
    'EFGD%': 'EFG_D',
    '2P%': '2P_O',
    '2P%D': '2P_D',
    '3P%': '3P_O',
    '3P%D': '3P_D',
    'Adj T.': 'ADJ_T'
}

df23 = df23.rename(columns=old_new_col_label)

df23 = df23.drop(columns=['Rk', 'Rec'])

# Need to add Year to 2021 Data
df21['YEAR'] = df21.apply(lambda row: 2021, axis=1)

master_df = pd.concat([df13_20, df23, df21], ignore_index=True, keys=list(df13_20.columns))

default_training_factors = [
    'Win_Pct',
    'ADJOE',
    'ADJDE',
    'BARTHAG',
    'EFG_O',
    'EFG_D',
    'TOR',
    'TORD',
    'ORB',
    'DRB',
    'FTR',
    'FTRD',
    '2P_O',
    '2P_D',
    '3P_O',
    '3P_D',
    'ADJ_T',
    'WAB'
]

def preprocess_historic_data(df, training_factors=default_training_factors):
    """
    Convert the df from the csv file to the df ready for learning.
    
    Should accomplish several things:
        - Filter to only use the relevant columns
        - Scale factors to prepare for KNN
        - Create any additional columns needed in DataFrame (ex. Win_Pct)
    """
    # Create any additional columns needed in DataFrame
    df['Win_Pct'] = df.apply(lambda row: float(row.W) / float(row.G), axis=1)

    # Scale factors to prepare for KNN
    scaler = MinMaxScaler()
    df[training_factors] = scaler.fit_transform(df[training_factors])
    
    # Filter to only use the relevant columns
    df = df[training_factors]
    
    return df

all_teams_scaled_and_vectorized_df = preprocess_historic_data(master_df)

historic_teams_query = master_df.query('YEAR != 2023')
historic_teams_scaled_and_vectorized_df = all_teams_scaled_and_vectorized_df.iloc[historic_teams_query.index.values]

X = historic_teams_scaled_and_vectorized_df.to_numpy()

if train_makes_tournament:
    # Use the SVM classifier with Grid Search Cross Validation for the makes_tournament_classifier
    makes_tournament_base = svm.SVC()
    params = {
        'C': [1, 5, 10],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'shrinking': [True, False],
        'decision_function_shape': ['ovo', 'ovr'],
        'gamma': ['scale', 'auto'],
        'coef0': [1, 5, 10],
        'probability': [True, False]
    }
    makes_tournament_classifier = GridSearchCV(makes_tournament_base, params)

    # Y Should be 1 if the team in X made the tournament, 0 otherwise
    historic_teams_query = master_df.query('YEAR != 2023')
    historic_teams_df = master_df.iloc[historic_teams_query.index.values]
    Y_df = historic_teams_df.apply(lambda row: 0 if pd.isna(row['SEED']) else 1, axis=1)
    Y_makes_tournament = Y_df.to_numpy()

    print('Training makes_tournament_classifier')
    makes_tournament_classifier.fit(X, Y_makes_tournament)
    joblib.dump(makes_tournament_classifier, 'assets/models/makes_tournament_classifier.joblib')


tournament_teams_query = master_df.query('POSTSEASON == POSTSEASON')
tournament_teams_df = master_df.iloc[tournament_teams_query.index.values]
X_tournament_teams = historic_teams_scaled_and_vectorized_df.loc[tournament_teams_query.index.values].to_numpy()

Y_seeds_df = tournament_teams_df['SEED']
Y_result_df = tournament_teams_df['POSTSEASON']

seed_classifier_base = KNeighborsClassifier()
result_classifier_base = KNeighborsClassifier()

knn_params = {
    'n_neighbors': list(range(3,16,2)),
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30, 40, 50],
    'p': [1, 2, 3]
}

seed_classifier = GridSearchCV(seed_classifier_base, knn_params)
result_classifier = GridSearchCV(result_classifier_base, knn_params)

if train_seed:
    print('Training seed_classifier')
    seed_classifier.fit(X_tournament_teams, Y_seeds_df.to_numpy())
    joblib.dump(seed_classifier, 'assets/models/seed_classifier.joblib')

if train_result:   
    print('Training result_classifier')
    result_classifier.fit(X_tournament_teams, Y_result_df.to_numpy())
    joblib.dump(result_classifier, 'assets/models/result_classifier.joblib')

joblib.dump(master_df, 'assets/dataframes/master_df.joblib')
joblib.dump(all_teams_scaled_and_vectorized_df, 'assets/dataframes/all_teams_scaled_and_vectorized_df.joblib')
joblib.dump(historic_teams_scaled_and_vectorized_df, 'assets/dataframes/historic_teams_scaled_and_vectorized_df.joblib')
