from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

final_result = "name;acc;f1\n"
data = pd.read_csv("./data.csv")
y = data.copy().iloc[:,data.shape[1]-5]
x = data.copy().iloc[:, 0:data.shape[1]-5]
n = 10
fold = StratifiedKFold(n_splits = n)
folds = fold.split(x, y)
sets = []
for i, (train_index, test_index) in enumerate(folds):
	X_train_new = x.iloc[train_index]
	X_test_new = x.iloc[test_index]
	Y_train_new = y.iloc[train_index]
	Y_test_new = y.iloc[test_index]

	scaler = StandardScaler()
	scaler.fit(X_train_new)
	X_train_new = scaler.transform(X_train_new)
	X_test_new = scaler.transform(X_test_new)
	set_ = {"X_train":X_train_new.copy(), "Y_train": Y_train_new.copy(), "X_test":X_test_new.copy(), "Y_test": Y_test_new.copy()}
	sets.append(set_)

models = [DecisionTreeClassifier(), KNeighborsClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), GaussianNB(), SVC(), MLPClassifier()]
names = ["DT", "KNN", "RF", "GBC", "GNB", "SVM", "MLP"]
#models = [DecisionTreeClassifier(), GaussianNB()]
#names = ["DT", "GNB"]
final_result = final_result + "NO PCA;;\n"
for m in range(len(models)):
	accuracies = 0
	f1s = 0
	for i in range(len(sets)):
		X_train = sets[i]["X_train"]
		X_test = sets[i]["X_test"]
		Y_train = sets[i]["Y_train"]
		Y_test = sets[i]["Y_test"]
		res = ""
		models[m].fit(X_train, Y_train)
		Y_pred = models[m].predict(X_test)
		accuracy = accuracy_score(Y_pred, Y_test) * 100
		f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
		accuracies = accuracies + accuracy
		f1s = f1s + f1
	res = res + f"{names[m]};{accuracies/10:.2f};{f1s/10:.2f}; \n"
	final_result = final_result + res
final_result = final_result + "PCA(N);;\n"
for m in range(len(models)):
	accuracies = 0
	f1s = 0
	for i in range(len(sets)):
		X_train = sets[i]["X_train"]
		X_test = sets[i]["X_test"]
		Y_train = sets[i]["Y_train"]
		Y_test = sets[i]["Y_test"]

		pca = PCA()
		X_train = pca.fit_transform(X_train)
		X_test = pca.transform(X_test)
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)
		res = ""
		models[m].fit(X_train, Y_train)
		Y_pred = models[m].predict(X_test)
		accuracy = accuracy_score(Y_pred, Y_test) * 100
		f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
		accuracies = accuracies + accuracy
		f1s = f1s + f1
	res = res + f"{names[m]};{accuracies/10:.2f};{f1s/10:.2f}; \n"
	final_result = final_result + res
final_result = final_result + "PCA(300);;\n"
for m in range(len(models)):
	accuracies = 0
	f1s = 0
	for i in range(len(sets)):
		X_train = sets[i]["X_train"]
		X_test = sets[i]["X_test"]
		Y_train = sets[i]["Y_train"]
		Y_test = sets[i]["Y_test"]
		pca = PCA(n_components = 300)
		X_train = pca.fit_transform(X_train)
		X_test = pca.transform(X_test)
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)
		res = ""
		models[m].fit(X_train, Y_train)
		Y_pred = models[m].predict(X_test)
		accuracy = accuracy_score(Y_pred, Y_test) * 100
		f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
		accuracies = accuracies + accuracy
		f1s = f1s + f1
	res = res + f"{names[m]};{accuracies/10:.2f};{f1s/10:.2f}; \n"
	final_result = final_result + res
final_result = final_result + "PCA(150);;\n"
for m in range(len(models)):
	accuracies = 0
	f1s = 0
	for i in range(len(sets)):
		X_train = sets[i]["X_train"]
		X_test = sets[i]["X_test"]
		Y_train = sets[i]["Y_train"]
		Y_test = sets[i]["Y_test"]
		pca = PCA(n_components = 150)
		X_train = pca.fit_transform(X_train)
		X_test = pca.transform(X_test)
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)
		res = ""
		models[m].fit(X_train, Y_train)
		Y_pred = models[m].predict(X_test)
		accuracy = accuracy_score(Y_pred, Y_test) * 100
		f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
		accuracies = accuracies + accuracy
		f1s = f1s + f1
	res = res + f"{names[m]};{accuracies/10:.2f};{f1s/10:.2f}; \n"
	final_result = final_result + res
final_result = final_result + "PCA(50);;\n"
for m in range(len(models)):
	accuracies = 0
	f1s = 0
	for i in range(len(sets)):
		X_train = sets[i]["X_train"]
		X_test = sets[i]["X_test"]
		Y_train = sets[i]["Y_train"]
		Y_test = sets[i]["Y_test"]
		pca = PCA(n_components = 50)
		X_train = pca.fit_transform(X_train)
		X_test = pca.transform(X_test)
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)
		res = ""
		models[m].fit(X_train, Y_train)
		Y_pred = models[m].predict(X_test)
		accuracy = accuracy_score(Y_pred, Y_test) * 100
		f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
		accuracies = accuracies + accuracy
		f1s = f1s + f1
	res = res + f"{names[m]};{accuracies/10:.2f};{f1s/10:.2f}; \n"
	final_result = final_result + res
final_result = final_result + "PCA(17);;\n"
for m in range(len(models)):
	accuracies = 0
	f1s = 0
	for i in range(len(sets)):
		X_train = sets[i]["X_train"]
		X_test = sets[i]["X_test"]
		Y_train = sets[i]["Y_train"]
		Y_test = sets[i]["Y_test"]
		pca = PCA(n_components = 17)
		X_train = pca.fit_transform(X_train)
		X_test = pca.transform(X_test)
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)
		res = ""
		models[m].fit(X_train, Y_train)
		Y_pred = models[m].predict(X_test)
		accuracy = accuracy_score(Y_pred, Y_test) * 100
		f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
		accuracies = accuracies + accuracy
		f1s = f1s + f1
	res = res + f"{names[m]};{accuracies/10:.2f};{f1s/10:.2f}; \n"
	final_result = final_result + res
final_result = final_result + "PCA(2);;\n"
for m in range(len(models)):
	accuracies = 0
	f1s = 0
	for i in range(len(sets)):
		X_train = sets[i]["X_train"]
		X_test = sets[i]["X_test"]
		Y_train = sets[i]["Y_train"]
		Y_test = sets[i]["Y_test"]
		pca = PCA(n_components = 2)
		X_train = pca.fit_transform(X_train)
		X_test = pca.transform(X_test)
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)
		res = ""
		models[m].fit(X_train, Y_train)
		Y_pred = models[m].predict(X_test)
		accuracy = accuracy_score(Y_pred, Y_test) * 100
		f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
		accuracies = accuracies + accuracy
		f1s = f1s + f1
	res = res + f"{names[m]};{accuracies/10:.2f};{f1s/10:.2f}; \n"
	final_result = final_result + res
result_file = open("result.csv", "w")
result_file.write(final_result)
result_file.close()

