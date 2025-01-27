import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import data_loader
class Model:
    def __init__(self, target_type, datafile = "./data.csv"):
        self.target_type = target_type
        if target_type == "cooler":
            self.target = 5
            self.pca = 2
            self.lr = 0.001
            self.layers = (50,)
        elif target_type == "bar":
            self.target = 2
            self.pca = 150
            self.lr = 0.01
            self.layers = (100, 200)
        elif target_type == "pump":
            self.target = 3
            self.pca = 0
            self.lr = 0.01
            self.layers = (50,)
        elif target_type == "valve":
            self.target = 4
            self.pca = 150
            self.lr = 0.001
            self.layers = (100, 200)
        elif target_type == "stable":
            self.target = 1
            self.pca = 50
            self.lr = 0.001
            self.layers = (50,)
        else:
            self.target_type = "cooler"
            self.target = 5
            self.pca = 2
            self.lr = 0.001
            self.layers = (50,)
        self.datafile = datafile
        self.classifier = None
        self.accuracy = 0
        self.scaler = None
        self.scaler2 = None
        self.pca_transformer = None
    def predict(self, x, x_converted = False):
        if not x_converted:
            x_features = data_loader.loadObservation(x)
        else:
            x_features = x
        x_features = x_features.reshape(1,-1)
        x_features = self.scaler.transform(x_features)
        if self.pca > 0:
            x_features = self.pca_transformer.transform(x_features)
            x_features = self.scaler2.transform(x_features)
        Y_pred = self.classifier.predict(x_features)
        return Y_pred
    def fit(self):
        data = pd.read_csv(self.datafile)
        y = data.copy().iloc[:, data.shape[1]-self.target]
        x = data.copy().iloc[:, 0:data.shape[1]-5]
        x = x.values
        y = y.values
        n = 10
        fold = StratifiedKFold(n_splits = n)
        folds = fold.split(x, y)
        sets = []
        for i, (train_index, test_index) in enumerate(folds):
            X_train_new = x[train_index]
            X_test_new = x[test_index]
            Y_train_new = y[train_index]
            Y_test_new = y[test_index]
            scaler = StandardScaler()
            scaler.fit(X_train_new)
            X_train_new = scaler.transform(X_train_new)
            X_test_new = scaler.transform(X_test_new)
            if (self.pca > 0):
                pca = PCA(n_components = self.pca)
                X_train_new = pca.fit_transform(X_train_new)
                X_test_new = pca.transform(X_test_new)
                scaler2 = StandardScaler()
                X_train_new = scaler2.fit_transform(X_train_new)
                X_test_new = scaler2.transform(X_test_new)
            else:
                scaler2 = None
                pca = None
            set_ = {"X_train":X_train_new.copy(), "Y_train": Y_train_new.copy(), "X_test":X_test_new.copy(), "Y_test": Y_test_new.copy(), "scaler1" : scaler, "scaler2": scaler2, "PCA": pca}
            sets.append(set_)

        models = [MLPClassifier(hidden_layer_sizes = self.layers, max_iter = 200, learning_rate_init=self.lr, activation="identity", solver="lbfgs") for _ in range(n)]
        accuracies = []
        for i in range(n):
            X_train = sets[i]["X_train"]
            X_test = sets[i]["X_test"]
            Y_train = sets[i]["Y_train"]
            Y_test = sets[i]["Y_test"]
            m = models[i]
            m.fit(X_train, Y_train)
            Y_pred = m.predict(X_test)
            accuracy = accuracy_score(Y_test, Y_pred) * 100
            accuracies.append(accuracy)
        mean_acc = sum(accuracies) / n
        best = 0
        for i in range(n):
            if abs(accuracies[i] - mean_acc) < abs(accuracies[best] - mean_acc):
                best = i
        self.accuracy = accuracies[best]
        self.classifier = models[best]
        self.scaler = sets[best]["scaler1"]
        self.scaler2 = sets[best]["scaler2"]
        self.pca_transformer = sets[best]["PCA"]



