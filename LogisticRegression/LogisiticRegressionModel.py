import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:

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

    def display_accuracy(self, y_test, y_prediction, subsample_size, file):
        accuracy = accuracy_score(y_test, np.round(y_prediction))
        print("The accuracy of the model was: " + str(accuracy) + " with a subsample size of " + str(subsample_size))
        file.write(str(accuracy) + "\n")
        return accuracy


if __name__ == '__main__':
    our_file_name = "../NFLDataset/2019-NFL-Season-Dataset.csv"

    LRM = LogisticRegressionModel()
    our_subsample_size = 1.0
    # our_decreasing_factor = 0.05
    file = open("../LinearSVM/accuracy_results.txt", "a+")
    file.truncate(0)
    # while our_subsample_size > 0.0:
    # accuracy_list = list()
    for i in range(51):
        our_train_dataframe, our_test_dataframe = LRM.test_train_dataframe_split(our_file_name, our_subsample_size)

        X_train = our_train_dataframe.drop(
            columns=['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
        y_train = our_train_dataframe[['result']]
        X_test = our_test_dataframe.drop(
            columns=['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
        y_test = our_test_dataframe[['result']]

        result = LogisticRegression(penalty='l1', dual=False, tol=0.001, C=1.0, fit_intercept=True,
                                    intercept_scaling=1,
                                    class_weight='balanced', random_state=None, solver='liblinear', max_iter=1000,
                                    multi_class='ovr', verbose=0)

        result.fit(X_train, np.ravel(y_train.values))
        y_pred = result.predict_proba(X_test)
        y_pred = y_pred[:, 1]

        # LRM.display_results(y_pred, our_test_dataframe)
        accuracy = LRM.display_accuracy(y_test, y_pred, our_subsample_size, file)
        # accuracy_list.append(accuracy)

    # accuracy_average = sum(accuracy_list) / len(accuracy_list)
    # file.write(str(accuracy_average) + "\n")
    # our_subsample_size = our_subsample_size - our_decreasing_factor

    file.close()
