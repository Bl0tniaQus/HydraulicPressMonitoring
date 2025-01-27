import model
import pickle
import data_loader

model_file = open("model_cooler", "rb")
model = pickle.load(model_file)
model_file.close()

obs = data_loader.readDictFromFiles("../raw_data", 711)
Y_pred = model.predict(obs)
print(Y_pred)

