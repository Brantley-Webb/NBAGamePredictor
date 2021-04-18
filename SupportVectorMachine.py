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

    def graph_svm(self, svm_classifier, training_samples, training_labels):
        plt.figure(figsize=(10, 5))
        for i, C in enumerate([1]):
            decision_function = svm_classifier.decision_function(training_samples)
            # we can also calculate the decision function manually
            # decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]
            # The support vectors are the samples that lie within the margin
            # boundaries, whose size is conventionally constrained to 1
            support_vector_indices = np.where(
                np.abs(decision_function) <= 1 + 1e-15)[0]
            support_vectors = list()
            for indice in support_vector_indices:
                support_vectors.append(training_samples.iloc[indice])
            # support_vectors = training_samples[support_vector_indices]

            plt.subplot(1, 2, i + 1)
            plt.scatter(training_samples[:, 0], training_samples[:, 1], c=training_labels, s=30, cmap=plt.cm.Paired)
            ax = plt.gca()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                                 np.linspace(ylim[0], ylim[1], 50))
            Z = svm_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                        linestyles=['--', '-', '--'])
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,
                        linewidth=1, facecolors='none', edgecolors='k')
            plt.title("C=" + str(C))
        plt.tight_layout()
        plt.show()


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
        SVM.graph_svm(result, X_train, y_train)

        our_max_iterations = our_max_iterations + our_increasing_factor

    file.close()
