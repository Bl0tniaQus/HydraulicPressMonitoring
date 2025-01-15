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
print("started")
start = timer()
def GridSearch(sets, hl, a, s, lr, e):
	accuracies = 0
	f1s = 0
	for i in range(len(sets)):
		X_train = sets[i]["X_train"]
		X_test = sets[i]["X_test"]
		Y_train = sets[i]["Y_train"]
		Y_test = sets[i]["Y_test"]
		res = ""
		model = MLPClassifier(hidden_layer_sizes = hl, activation = a, solver = s, learning_rate_init=lr, max_iter = e)
		model.fit(X_train, Y_train)
		Y_pred = model.predict(X_test)
		accuracy = accuracy_score(Y_test, Y_pred) * 100
		f1 = f1_score(Y_test, Y_pred, average="weighted") * 100
		accuracies = accuracies + accuracy
		f1s = f1s + f1
	res = f"{hl};{a};{s};{lr};{e};{accuracies/10:.2f};{f1s/10:.2f}\n"
	return res



data = pd.read_csv("./data.csv")
y = data.copy().iloc[:,data.shape[1]-4]
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
	pca = PCA(n_components = 150)
	X_train_new = pca.fit_transform(X_train_new)
	X_test_new = pca.transform(X_test_new)
	scaler2 = StandardScaler()
	X_train_new = scaler.fit_transform(X_train_new)
	X_test_new = scaler.transform(X_test_new)
	set_ = {"X_train":X_train_new.copy(), "Y_train": Y_train_new.copy(), "X_test":X_test_new.copy(), "Y_test": Y_test_new.copy()}
	sets.append(set_)
print("sets completed")
hidden_layer_sizes = [(100,), (50,), (100, 200,)]
activations = ['identity']
solvers = ['lbfgs']
learning_rates = [0.001, 0.01]
max_epochs = [200, 300]

n_models = len(hidden_layer_sizes) * len(activations) * len(solvers) * len(learning_rates) * len(max_epochs)
i = 1
final_result = "layers;activation;solver;lr;epochs;acc;f1"
print("training")
for hl in hidden_layer_sizes:
	for a in activations:
		for s in solvers:
			for lr in learning_rates:
				for e in max_epochs:
					try:
						res = GridSearch(sets, hl, a, s ,lr, e)
					except:
						print("error occured")
						res = ""
					final_result = final_result + res
					print(f"finished model {i}/{n_models}")
					i = i+1

print("finished")
result_file = open("result_valve_cv.csv", "w")
result_file.write(final_result)
result_file.close()
end = timer()
print(f"time: {end - start}")

