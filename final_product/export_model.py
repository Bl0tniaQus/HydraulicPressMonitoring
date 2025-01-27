from model import Model
import pickle
import data_loader
model = Model("cooler")
model.fit()
# ~ data_loader.loadAll("../raw_data")
print(model.accuracy)
obs = data_loader.readDictFromFiles("../raw_data", 0)
y = model.predict(obs)
print(y)
