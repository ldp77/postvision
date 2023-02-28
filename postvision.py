import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import re
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

class PostVisionInstance():
    def __init__(self):
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

        self.master_df = pd.concat([df13_20, df23, df21], ignore_index=True, keys=list(df13_20.columns))


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

        # scaler = MinMaxScaler()
        # master_df[default_training_factors] = scaler.fit_transform(master_df[default_training_factors])
        # print(master_df.head())

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

        self.all_teams_scaled_and_vectorized_df = preprocess_historic_data(self.master_df)

        historic_teams_query = self.master_df.query('YEAR != 2023')
        self.historic_teams_scaled_and_vectorized_df = self.all_teams_scaled_and_vectorized_df.iloc[historic_teams_query.index.values]

        self.makes_tournament_classifier = svm.SVC()

        X = self.historic_teams_scaled_and_vectorized_df.to_numpy()

        # Y Should be 1 if the team in X made the tournament, 0 otherwise
        historic_teams_query = self.master_df.query('YEAR != 2023')
        historic_teams_df = self.master_df.iloc[historic_teams_query.index.values]
        Y_df = historic_teams_df.apply(lambda row: 0 if pd.isna(row['SEED']) else 1, axis=1)
        Y = Y_df.to_numpy()

        self.makes_tournament_classifier.fit(X, Y)

        tournament_teams_query = self.master_df.query('POSTSEASON == POSTSEASON')
        tournament_teams_df = self.master_df.iloc[tournament_teams_query.index.values]
        X_tournament_teams = self.historic_teams_scaled_and_vectorized_df.loc[tournament_teams_query.index.values].to_numpy()

        Y_seeds_df = tournament_teams_df['SEED']
        Y_result_df = tournament_teams_df['POSTSEASON']

        self.seed_classifier = KNeighborsClassifier()
        self.result_classifier = KNeighborsClassifier()

        self.seed_classifier.fit(X_tournament_teams, Y_seeds_df.to_numpy())
        self.result_classifier.fit(X_tournament_teams, Y_result_df.to_numpy())
    
    def get_team_as_vector(self, teamname, year=2023):
        team_query = self.master_df.query('TEAM == "{}" and YEAR == {}'.format(teamname, year))
        return self.all_teams_scaled_and_vectorized_df.iloc[team_query.index.values]

    def get_k_most_similar(self, teamname, k, year=2023):
        # Index not guaranteed to be [0, 1, 2, etc.] so need to keep track in a map
        index_distance_map = {}

        selected_current_team = self.get_team_as_vector(teamname, year)

        for i in self.historic_teams_scaled_and_vectorized_df.index.values:
            index_distance_map[i] = np.linalg.norm(selected_current_team.to_numpy() - self.historic_teams_scaled_and_vectorized_df.loc[i].to_numpy())

        most_similar = sorted(index_distance_map.items(), key=lambda x: x[1])

        return most_similar[0:k]

    def execute_query(self, teamname, k=5):
        most_similar = self.get_k_most_similar(teamname, k)
        most_similar_idx = [idx for idx, val in most_similar]
        most_similar_df = self.master_df.iloc[most_similar_idx]

        most_similar_list = [
            f"{int(row['YEAR'])} {row['TEAM']} ({row['W']}-{row['G'] - row['W']}): {row['SEED']} Seed, Finished {row['POSTSEASON']}" for i, row in most_similar_df.iterrows()
        ]

        selected_team = self.get_team_as_vector(teamname)

        makes_tournament_prediction = self.makes_tournament_classifier.predict(selected_team)[0]

        seed_prediction = None
        result_prediction = None

        if makes_tournament_prediction == 1:
            seed_prediction = self.seed_classifier.predict(self.get_team_as_vector(teamname))[0]
            result_prediction = self.result_classifier.predict(self.get_team_as_vector(teamname))[0]

        return {
            'most_similar': most_similar_list,
            'makes_tournament_prediction': str(makes_tournament_prediction),
            'seed_prediction': seed_prediction,
            'result_prediction': result_prediction
        }

# pvi = PostVisionInstance()
 
# current_teamname = 'Texas A&M'
# k = 5
# most_similar = pvi.get_k_most_similar(current_teamname, k)

# most_similar_idx = [idx for idx, val in most_similar]

# most_similar_df = pvi.master_df.iloc[most_similar_idx]

# for i, row in most_similar_df.iterrows():
#     print(f"{int(row['YEAR'])} {row['TEAM']} ({row['W']}-{row['G'] - row['W']}): {row['SEED']} Seed, Finished {row['POSTSEASON']}")