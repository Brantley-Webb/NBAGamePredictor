import pandas as pd
import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class MachineLearningModelComparator:

    def test_train_dataframe_split(self, file_name, subsample_size, train_test_percentage):
        aggregated_game_stats_for_season = pd.read_csv(str(file_name))
        aggregated_game_stats_for_season.sample(frac=subsample_size)
        train_dataframe, test_dataframe = train_test_split(aggregated_game_stats_for_season, test_size=train_test_percentage,
                                                           shuffle=True)
        return train_dataframe, test_dataframe


if __name__ == '__main__':
    our_file_name = "../NFLDataset/2019-NFL-Season-Dataset.csv"

    MLComparator = MachineLearningModelComparator()
    our_subsample_size = 1.0
    our_max_iterations = 1000
    our_test_split = 0.20
    our_c = 1.0

    our_train_dataframe, our_test_dataframe = MLComparator.test_train_dataframe_split(our_file_name, our_subsample_size, our_test_split)

    X_train = our_train_dataframe.drop(
        columns=['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
    y_train = our_train_dataframe[['result']]
    X_test = our_test_dataframe.drop(
        columns=['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
    y_test = our_test_dataframe[['result']]

    scaler_and_svm = make_pipeline(StandardScaler(), LinearSVC(max_iter=our_max_iterations, C=our_c))

    lr = LogisticRegression(penalty='l1', dual=False, tol=0.001, C=1.0, fit_intercept=True,
                                intercept_scaling=1,
                                class_weight='balanced', random_state=None, solver='liblinear', max_iter=1000,
                                multi_class='ovr', verbose=0)

    knn = KNeighborsClassifier(15, weights='uniform', algorithm='auto',
                                  leaf_size=30, p=1, metric='manhattan', metric_params=None, n_jobs=None)

    rf = RandomForestClassifier(n_estimators=1000, random_state=None, verbose=0, class_weight='balanced')

    svm_proba = scaler_and_svm.fit(X_train, np.ravel(y_train.values)).decision_function(X_test)
    lr_proba = lr.fit(X_train, np.ravel(y_train.values)).predict_proba(X_test)
    knn_proba = knn.fit(X_train, np.ravel(y_train.values)).predict_proba(X_test)
    rf_proba = rf.fit(X_train, np.ravel(y_train.values)).predict_proba(X_test)

    probas_list = [svm_proba, lr_proba, knn_proba, rf_proba]
    classifier_names = ['Linear SVM', 'Logistic Regression', 'kNN', 'Random Forest']
    skplt.metrics.plot_calibration_curve(np.ravel(y_test), probas_list, classifier_names)
    plt.show()

