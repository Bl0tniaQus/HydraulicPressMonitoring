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
from sklearn.model_selection import StratifiedKFold, train_test_split
from timeit import default_timer as timer
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
print("started")
start = timer()
def GridSearch(data, hl, a, s, lr, e):
	accuracies = 0
	f1s = 0
	X_train = data[0]
	X_test = data[1]
	Y_train = data[2]
	Y_test = data[3]
	res = ""
	model = MLPClassifier(hidden_layer_sizes = hl, activation = a, solver = s, learning_rate_init=lr, max_iter = e)
	model.fit(X_train, Y_train)
	Y_pred = model.predict(X_test)
	accuracy = accuracy_score(Y_pred, Y_test) * 100
	f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
	accuracies = accuracies + accuracy
	f1s = f1s + f1
	res = f"{hl};{a};{s};{lr};{e};{accuracies/10:.2f};{f1s/10:.2f}\n"
	return res



data = pd.read_csv("./data.csv")
y = data.copy().iloc[:,data.shape[1]-3]
x = data.copy().iloc[:, 0:data.shape[1]-5]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
n = 5
sets = []
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
data = [X_train, X_test, Y_train, Y_test]
hidden_layer_sizes = [(100,), (50,), (50, 100,), (100, 100,), (50, 100, 200,), (200,), (100,200,), (50, 100, 50,), (100, 200, 100,), (100, 200, 300, 200, 100,)]
activations = ['identity', 'logistic', 'tanh', 'relu']
solvers = ['lbfgs', 'adam', 'sgd']
learning_rates = [0.001, 0.01, 0.1, 0.2]
max_epochs = [200, 300, 500, 1000]
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
						res = GridSearch(data, hl, a, s ,lr, e)
					except:
						print("error occured")
						res = ""
					final_result = final_result + res
					print(f"finished model {i}/{n_models}")
					i = i+1

print("finished")
result_file = open("result_pump.csv", "w")
result_file.write(final_result)
result_file.close()
end = timer()
print(f"time: {end - start}")

