import json
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class RandomForestModel:
    def __init__(self, config_path, data_paths, feature_name, X_train, y_train, X_dev, y_dev, test=False,
                 baseline=False):
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        with open(data_paths, 'r') as data_file:
            self.data_paths = json.load(data_file)

        self.model_name = 'RF'
        self.feature_name = feature_name
        self.X_train = X_train
        self.X_dev = X_dev
        self.y_train = y_train
        self.y_dev = y_dev
        self.parameters = self.config.get('parameters', {})
        self.model = None
        self.baseline = baseline
        self.test = test

    def run(self):
        if not self.parameters:
            print("\nHyperparameters are not tuned yet.")
            self.tune()

        self.model = RandomForestClassifier(**self.parameters, verbose=1, n_jobs=-1, class_weight="balanced")
        self.train()

    def train(self):
        print("\nTraining a Random Forest Classifier...")
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        print("\nEvaluating the Random Forest Classifier...")
        y_pred_train = self.model.predict(self.X_train)
        y_pred_dev = self.model.predict(self.X_dev)

        print("\nAccuracy on training set: %.3f" % metrics.accuracy_score(self.y_train, y_pred_train))
        print("\nAccuracy on development set: %.3f" % metrics.accuracy_score(self.y_dev, y_pred_dev))

    def tune(self):
        parameters = {
            "n_estimators": [100, 200, 300],
            "max_features": ['auto', 'sqrt'],
            "max_depth": [10, 20, 30],
            "criterion": ["gini", "entropy"]
        }
        print("\nRunning Grid Search for Random Forest classifier...")
        clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs=-1, verbose=3, scoring='recall_macro')
        clf.fit(self.X_train, self.y_train)

        print("\nBest hyperparameters:", clf.best_params_)
        self.parameters.update(clf.best_params_)