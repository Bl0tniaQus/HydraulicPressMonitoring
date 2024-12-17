import AutoEncoder
import torch
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
from timeit import default_timer as timer
import pandas as pd

n_epochs = 30
lr = 0.1

final_result = ""
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


	set_ = {"X_train":X_train_new.copy(), "Y_train": Y_train_new.copy(), "X_test":X_test_new.copy(), "Y_test": Y_test_new.copy(), "AE" : AutoEncoder.AE_train(X_train_new, n_epochs,lr)}
	sets.append(set_)

models = [DecisionTreeClassifier(), KNeighborsClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), GaussianNB(), SVC(), MLPClassifier()]
names = ["DT", "KNN", "RF", "GBC", "GNB", "SVM", "MLP"]
final_result = final_result + "***AutoEncoder***\n"
for m in range(len(models)):
	accuracies = 0
	f1s = 0
	for i in range(len(sets)):
		X_train = sets[i]["X_train"]
		X_test = sets[i]["X_test"]
		Y_train = sets[i]["Y_train"]
		Y_test = sets[i]["Y_test"]
		AE = sets[i]["AE"]

		X_train_tensor = torch.from_numpy(X_train)
		X_train_tensor = X_train_tensor.to(torch.float32)
		X_train_encoded = AE.encode(X_train_tensor)
		X_test_tensor = torch.from_numpy(X_test)
		X_test_tensor = X_test_tensor.to(torch.float32)
		X_test_encoded = AE.encode(X_test_tensor)
		res = ""
		models[m].fit(X_train_encoded.detach().numpy(), Y_train)
		Y_pred = models[m].predict(X_test_encoded.detach().numpy())
		accuracy = accuracy_score(Y_pred, Y_test) * 100
		f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
		accuracies = accuracies + accuracy
		f1s = f1s + f1
	res = res + f"{names[m]} - Acc: {accuracies/10:.2f}, F1: {accuracies/10:.2f}; \n"
	final_result = final_result + res
result_file = open("result_AE.txt", "w")
result_file.write(final_result)
result_file.close()

final_result = ""
data = pd.read_csv("./data.csv")
y = data.copy().iloc[:,data.shape[1]-5]
x = data.copy().iloc[:, 0:data.shape[1]-5]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

AE = AutoEncoder.AE_train(X_train, n_epochs, lr)

final_result = final_result + "***AutoEncoder***\n"
for m in range(len(models)):
	res = ""
	X_train_tensor = torch.from_numpy(X_train)
	X_train_tensor = X_train_tensor.to(torch.float32)
	X_train_encoded = AE.encode(X_train_tensor)
	models[m].fit(X_train_encoded.detach().numpy(), Y_train)
	test_start = timer()
	for i in range(X_test.shape[0]):
		x_obs = X_test.iloc[[i]]
		x_obs = scaler.transform(x_obs)
		X_obs_tensor = torch.from_numpy(x_obs)
		X_obs_tensor = X_obs_tensor.to(torch.float32)
		X_obs_encoded = AE.encode(X_obs_tensor)
		Y_pred = models[m].predict(X_obs_encoded.detach().numpy())
	test_end = timer()
	t = (test_end - test_start) / X_test.shape[0]
	res = res + f"{names[m]} - time: {t:.5f}; \n"
	final_result = final_result + res
result_file = open("result_AE_t.txt", "w")
result_file.write(final_result)
result_file.close()
