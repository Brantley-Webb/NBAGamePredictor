import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class SupportVectorMachine:

    def test_train_dataframe_split(self, file_name, subsample_size):
        aggregated_game_stats_for_season = pd.read_csv(str(file_name))
        aggregated_game_stats_for_season.sample(frac=subsample_size)
        train_dataframe, test_dataframe = train_test_split(aggregated_game_stats_for_season, test_size=0.2,
                                                           shuffle=True)
        return train_dataframe, test_dataframe

    def display_results(self, y_prediction, X_test):
        for game in range(len(y_prediction)):
            win_probability = round(y_prediction[game], 2)
            away_team = X_test.reset_index().drop(columns='index').loc[game, 'away_name']
            home_team = X_test.reset_index().drop(columns='index').loc[game, 'home_name']
            print(
                f'The away team: {away_team} have a probability of {win_probability} of beating the home team: {home_team}.')

    def display_accuracy(self, y_test, y_prediction, max_iterations, file):
        accuracy = accuracy_score(y_test, np.round(y_prediction))
        print("The accuracy of the model was: " + str(accuracy) + " with max iterations of " + str(max_iterations))
        file.write(str(accuracy) + ", " + str(max_iterations) + "\n")
        return accuracy


if __name__ == '__main__':
    our_file_name = "2019-NFL-Season-Dataset.csv"

    SVM = SupportVectorMachine()
    our_subsample_size = 1.0
    our_max_iterations = 1000
    our_increasing_factor = 1000
    file = open("accuracy_results.txt", "a+")
    file.truncate(0)
    while our_max_iterations <= 1000:
        our_train_dataframe, our_test_dataframe = SVM.test_train_dataframe_split(our_file_name, our_subsample_size)

        X_train = our_train_dataframe.drop(
            columns=['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
        y_train = our_train_dataframe[['result']]
        X_test = our_test_dataframe.drop(
            columns=['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
        y_test = our_test_dataframe[['result']]
        # scaler = StandardScaler()
        # scaler.fit(X_train)
        # scaler.transform(X_train)
        # scaler.fit(X_test)
        # scaler.transform(X_test)

        result = LinearSVC(max_iter=our_max_iterations)

        result.fit(X_train, np.ravel(y_train.values))
        y_pred = result.predict(X_test)

        accuracy = SVM.display_accuracy(y_test, y_pred, our_max_iterations, file)

        our_max_iterations = our_max_iterations + our_increasing_factor

    file.close()
