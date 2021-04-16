import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class kNearestNeighborModel:

    def test_train_dataframe_split(self, file_name):
        aggregated_game_stats_for_season = pd.read_csv(str(file_name))
        train_dataframe, test_dataframe = train_test_split(aggregated_game_stats_for_season, test_size=0.2, shuffle=True)
        return train_dataframe, test_dataframe

    def predict_result(self, win_prob):
        prob_whole = win_prob * 100
        randomly_generate = random.randrange(0, 101)
        print(randomly_generate)
        if prob_whole <= randomly_generate:
            return True
        else:
            return False

    def display_results(self, y_prediction, X_test):
        perc_prediction = []
        for game in range(len(y_prediction)):
            print("y prediction")
            print(y_prediction)
            win_probability = round(y_prediction[game], 2)
            away_team = X_test.reset_index().drop(columns='index').loc[game, 'away_name']
            home_team = X_test.reset_index().drop(columns='index').loc[game, 'home_name']
            print(
                f'The away team: {away_team} have a probability of {win_probability} of beating the home team: {home_team}.')
            predict = self.predict_result(win_probability)
            perc_prediction.append(predict)
            if predict:
                print("Prediction: " + away_team)
            else:
                print("Prediction: " + home_team)

    def display_accuracy(self, y_test, y_prediction):
        accuracy = accuracy_score(y_test, np.round(y_prediction))
        print("The accuracy of the model was: " + str(accuracy))

if __name__ == '__main__':
    our_file_name = "2019-NFL-Season-Dataset.csv"

    kNN = kNearestNeighborModel()
    our_train_dataframe, our_test_dataframe = kNN.test_train_dataframe_split(our_file_name)

    X_train = our_train_dataframe.drop(columns=['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
    y_train = our_train_dataframe[['result']]
    X_test = our_test_dataframe.drop(columns=['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
    y_test = our_test_dataframe[['result']]

    result = KNeighborsClassifier(16, weights='uniform', algorithm='auto',
                                  leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)

    result.fit(X_train, np.ravel(y_train.values))
    y_pred = result.predict_proba(X_test)
    y_pred = y_pred[:, 1]
    kNN.display_results(y_pred, our_test_dataframe)
    kNN.display_accuracy(y_test, y_pred)
