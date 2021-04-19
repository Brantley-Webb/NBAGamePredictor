import csv
import pandas as pd
import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt
import pydotplus

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz

class RandomForestModel:

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
        print("The accuracy of the model was: " + str(accuracy) + "\n")

    def get_accuracy(self, y_test, y_prediction):
        accuracy = accuracy_score(y_test,np.round(y_prediction))
        return accuracy

if __name__ == '__main__':
    file_name = "2019-NFL-Season-Dataset.csv"
    accuracy_list = []
    overall_accuracy = 0
    mean_accuracy = 0

    RandomForest = RandomForestModel()
    f = open("RandomForest.txt", "w")
    f.write("The Accuracy of all runs were: \n")

    for i in range(50):

        our_train_dataframe, our_test_dataframe = RandomForest.test_train_dataframe_split(file_name)

        X_train = our_train_dataframe.drop(columns=['Unnamed: 0', 'away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
        y_train = our_train_dataframe[['result']]
        X_test = our_test_dataframe.drop(columns=['Unnamed: 0', 'away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
        y_test = our_test_dataframe[['result']]



        result = RandomForestClassifier(n_estimators=1000, random_state=None, verbose=0, class_weight='balanced')

        result.fit(X_train, np.ravel(y_train.values))
        y_pred = result.predict_proba(X_test)
        y_pred = y_pred[:, 1]


        RandomForest.display_results(y_pred, our_test_dataframe)
        RandomForest.display_accuracy(y_test, y_pred)
        accuracy_list.append(RandomForest.get_accuracy(y_test, y_pred))
        stringAccuracy = str(RandomForest.get_accuracy(y_test, y_pred))
        f.write(stringAccuracy + "\n")

    for i in accuracy_list:
        overall_accuracy = overall_accuracy + i
    mean_accuracy = overall_accuracy / len(accuracy_list)
    print("The Accuracy of all runs were: " + str(accuracy_list))
    print("The Mean Accuracy of the Model was: " + str(mean_accuracy))

    f.write("The Mean Accuracy of the Model was: " + str(mean_accuracy))
    f.close()

    #Ploting the Feature Importance
    skplt.estimators.plot_feature_importances(result, max_num_features=22)
    plt.show()

    #Learning Curve
    skplt.estimators.plot_learning_curve(result, X_train, np.ravel(y_train))
    plt.show()

    #Ploting the Random Forest and Ploting a Decision Tree from it
    tree = result.estimators_[125]
    export_graphviz(tree, out_file='decision_tree_number_3.dot', feature_names=['win_perc_dif', 'first_downs_dif', 'fumbles_dif', 'interceptions_dif', 'net_pass_yards_dif', 'pass_attempts_dif', 'pass_completions_dif', 'pass_touchdowns_dif', 'pass_yards_dif', 'penalties_dif', 'points_dif', 'rush_attempts_dif', 'rush_touchdowns_dif', 'rush_yards_dif', 'time_of_possession_dif', 'times_sacked_dif', 'total_yards_dif', 'turnovers_dif', 'yards_from_penalties_dif', 'yards_lost_from_sacks_dif', 'fourth_down_perc_dif', 'third_down_perc_dif'], rounded=True, precision=1)
