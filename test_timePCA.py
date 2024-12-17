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
import warnings
warnings.filterwarnings('ignore')
final_result = "name;time\n"
data = pd.read_csv("./data.csv")
y = data.copy().iloc[:,data.shape[1]-5]
x = data.copy().iloc[:, 0:data.shape[1]-5]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
models = [DecisionTreeClassifier(), KNeighborsClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), GaussianNB(), SVC(), MLPClassifier()]
names = ["DT", "KNN", "RF", "GBC", "GNB", "SVM", "MLP"]
#models = [DecisionTreeClassifier(), GaussianNB()]
#names = ["DT", "GNB"]
final_result = final_result + "NO PCA;;\n"
for m in range(len(models)):
	res = ""
	models[m].fit(X_train, Y_train)
	test_start = timer()
	for i in range(X_test.shape[0]):
		x_obs = X_test.iloc[[i]]
		x_obs = scaler.transform(x_obs)
		Y_pred = models[m].predict(x_obs)
	test_end = timer()
	t = (test_end - test_start) / X_test.shape[0]
	res = res + f"{names[m]};{t:.5f}\n"
	final_result = final_result + res
final_result = final_result + "PCA (N);;\n"
for m in range(len(models)):
	res = ""
	pca = PCA()
	X_PCA = X_train.copy()
	X_PCA = pca.fit_transform(X_PCA)
	models[m].fit(X_PCA, Y_train)
	test_start = timer()
	for i in range(X_test.shape[0]):
		x_obs = X_test.iloc[[i]]
		x_obs = scaler.transform(x_obs)
		x_obs = pca.transform(x_obs)
		Y_pred = models[m].predict(x_obs)
	test_end = timer()
	t = (test_end - test_start) / X_test.shape[0]
	res = res + f"{names[m]};{t:.5f}\n"
	final_result = final_result + res
final_result = final_result + "PCA (300);;\n"
for m in range(len(models)):
	res = ""
	pca = PCA(n_components = 300)
	X_PCA = X_train.copy()
	X_PCA = pca.fit_transform(X_PCA)
	models[m].fit(X_PCA, Y_train)
	test_start = timer()
	for i in range(X_test.shape[0]):
		x_obs = X_test.iloc[[i]]
		x_obs = scaler.transform(x_obs)
		x_obs = pca.transform(x_obs)
		Y_pred = models[m].predict(x_obs)
	test_end = timer()
	t = (test_end - test_start) / X_test.shape[0]
	res = res + f"{names[m]};{t:.5f}\n"
	final_result = final_result + res
final_result = final_result + "PCA (150);;\n"
for m in range(len(models)):
	res = ""
	pca = PCA(n_components = 150)
	X_PCA = X_train.copy()
	X_PCA = pca.fit_transform(X_PCA)
	models[m].fit(X_PCA, Y_train)
	test_start = timer()
	for i in range(X_test.shape[0]):
		x_obs = X_test.iloc[[i]]
		x_obs = scaler.transform(x_obs)
		x_obs = pca.transform(x_obs)
		Y_pred = models[m].predict(x_obs)
	test_end = timer()
	t = (test_end - test_start) / X_test.shape[0]
	res = res + f"{names[m]};{t:.5f}\n"
	final_result = final_result + res
final_result = final_result + "PCA (50);;\n"
for m in range(len(models)):
	res = ""
	pca = PCA(n_components = 50)
	X_PCA = X_train.copy()
	X_PCA = pca.fit_transform(X_PCA)
	models[m].fit(X_PCA, Y_train)
	test_start = timer()
	for i in range(X_test.shape[0]):
		x_obs = X_test.iloc[[i]]
		x_obs = scaler.transform(x_obs)
		x_obs = pca.transform(x_obs)
		Y_pred = models[m].predict(x_obs)
	test_end = timer()
	t = (test_end - test_start) / X_test.shape[0]
	res = res + f"{names[m]};{t:.5f}\n"
	final_result = final_result + res
final_result = final_result + "PCA (17);;\n"
for m in range(len(models)):
	res = ""
	pca = PCA(n_components = 17)
	X_PCA = X_train.copy()
	X_PCA = pca.fit_transform(X_PCA)
	models[m].fit(X_PCA, Y_train)
	test_start = timer()
	for i in range(X_test.shape[0]):
		x_obs = X_test.iloc[[i]]
		x_obs = scaler.transform(x_obs)
		x_obs = pca.transform(x_obs)
		Y_pred = models[m].predict(x_obs)
	test_end = timer()
	t = (test_end - test_start) / X_test.shape[0]
	res = res + f"{names[m]};{t:.5f}\n"
	final_result = final_result + res
final_result = final_result + "PCA (2);;\n"
for m in range(len(models)):
	res = ""
	pca = PCA(n_components = 2)
	X_PCA = X_train.copy()
	X_PCA = pca.fit_transform(X_PCA)
	models[m].fit(X_PCA, Y_train)
	test_start = timer()
	for i in range(X_test.shape[0]):
		x_obs = X_test.iloc[[i]]
		x_obs = scaler.transform(x_obs)
		x_obs = pca.transform(x_obs)
		Y_pred = models[m].predict(x_obs)
	test_end = timer()
	t = (test_end - test_start) / X_test.shape[0]
	res = res + f"{names[m]};{t:.5f}\n"
	final_result = final_result + res
result_file = open("result_t.csv", "w")
result_file.write(final_result)
result_file.close()

