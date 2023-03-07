import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import re
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

class PostVisionInstance():
    def __init__(self):
        self.master_df = joblib.load('assets/dataframes/master_df.joblib')
        self.all_teams_scaled_and_vectorized_df = joblib.load('assets/dataframes/all_teams_scaled_and_vectorized_df.joblib')
        self.historic_teams_scaled_and_vectorized_df = joblib.load('assets/dataframes/historic_teams_scaled_and_vectorized_df.joblib')

        self.makes_tournament_classifier = joblib.load('assets/models/makes_tournament_classifier.joblib')
        self.seed_classifier = joblib.load('assets/models/seed_classifier.joblib')
        self.result_classifier = joblib.load('assets/models/result_classifier.joblib')
    
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
        try:
            most_similar = self.get_k_most_similar(teamname, k)
            most_similar_idx = [idx for idx, val in most_similar]
            most_similar_df = self.master_df.iloc[most_similar_idx]

            most_similar_list = [
                f"{int(row['YEAR'])} {row['TEAM']} ({row['W']}-{row['G'] - row['W']}): {row['SEED']} Seed, Finished {row['POSTSEASON']}" for i, row in most_similar_df.iterrows()
            ]

            selected_team = self.get_team_as_vector(teamname).values

            makes_tournament_prediction = self.makes_tournament_classifier.predict(selected_team)[0]

            seed_prediction = None
            result_prediction = None

            if makes_tournament_prediction == 1:
                seed_prediction = self.seed_classifier.predict(selected_team)[0]
                result_prediction = self.result_classifier.predict(selected_team)[0]

            return {
                'most_similar': most_similar_list,
                'makes_tournament_prediction': str(makes_tournament_prediction),
                'seed_prediction': seed_prediction,
                'result_prediction': result_prediction
            }
        except ValueError:
            return {
                'error': 'Given team name was not found'
            }