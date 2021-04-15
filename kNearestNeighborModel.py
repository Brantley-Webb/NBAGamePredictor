import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class kNearestNeighborModel:

    def test_train_dataframe_split(self, file_name):
        aggregated_game_stats_for_season = pd.read_csv(str(file_name))
        train_dataframe, test_dataframe = train_test_split(aggregated_game_stats_for_season, test_size=0.2, shuffle=True)
        return train_dataframe, test_dataframe
    def display_results(self, y_prediction, X_test):
        for game in range(len(y_prediction)):
            win_probability = round(y_prediction[game], 2)
            away_team = X_test.reset_index().drop(columns='index').loc[game, 'away_name']
            home_team = X_test.reset_index().drop(columns='index').loc[game, 'home_name']
            print(
                f'The away team: {away_team} have a probability of {win_probability} of beating the home team: {home_team}.')

    def display_accuracy(self, y_test, y_prediction):
        accuracy = accuracy_score(y_test, np.round(y_prediction))
        print("The accuracy of the model was: " + str(accuracy))

if __name__ == '__main__':
    our_file_name = "2019-NFL-Season-Dataset.csv"

    kNN = kNearestNeighborModel()
    our_train_dataframe, our_test_dataframe = kNN.test_train_dataframe_split(our_file_name)
