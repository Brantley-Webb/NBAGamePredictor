# NFLGamePredictor
## Overview
This is group 14's final project. For this project we 
wanted to see if machine learning could be utilized to predict the winner 
of NFL games. Since sports prediction is such a factor driven field, 
machine learning models can have a tough time predicting the outcomes.

## Data
The data was pulled from the [sports reference python package](https://pypi.org/project/sportsreference/#documentation). This is an API that scraps data
from the [sports reference website](https://www.sports-reference.com/). We specifically looked at the 2019 NFL season.
Our code for scraping this data is in the ``NFLDataset`` folder. This contains the python code, titled ``NFLSeasonDatasetCreator.py``
that specifically pulls the data and does some data cleanup that is helpful for our machine learning models. The dataset that was used for all
experiments is also in this folder and titled ``2019-NFL-Season-Dataset.csv``.

## Environment
We utilized [PyCharm](https://www.jetbrains.com/pycharm/) to run and debug our code. We also used their tools to help
set up the virtual environments that were needed.

## The Models
The models were used from [scikit-learns python package](https://scikit-learn.org/stable/index.html). We also used a
plotting python package called [scikit-plot](https://scikit-plot.readthedocs.io/en/stable/index.html) to help plot some of our graphs.
Other graphs were done by pulling information from our python files into txt files and moving that data onto excel.

### Logistic Regression
The logistic regression code is located in the ``LogisticRegression`` folder, which
contains the ``LogisticRegressionModel.py`` python file. It imports the 
csv file found in the ``NFLDataset`` folder, performs a train/test split on it, and 
performs any of the experiments on the model. This model was the baseline for our project.

### K Nearest Neighbor
The kNN code is located in the ``kNN`` folder, which
contains the ``kNearestNeighborModel.py`` python file. It imports the 
csv file found in the ``NFLDataset`` folder, performs a train/test split on it, and 
performs any of the experiments on the model. It also contains a txt file which contains results
from some of our experiment results.

### Random Forest
The Random Forest code is located in the ``RandomForest`` folder, which
contains the ``RandomForestModel.py`` python file. It imports the 
csv file found in the ``NFLDataset`` folder, performs a train/test split on it, and 
performs any of the experiments on the model. It also contains files which show a decision tree
of the Random Forest. This was used for visualization purposes. It also contains a txt file which contains results
from some of our experiment results.

### Linear SVM
The Linear SVM code is located in the ``LinearSVM`` folder, which
contains the ``SupportVectorMachine.py`` python file. It imports the 
csv file found in the ``NFLDataset`` folder, performs a train/test split on it, and 
performs any of the experiments on the model.
It also contains txt files which contains results
from some of our experiment results.
