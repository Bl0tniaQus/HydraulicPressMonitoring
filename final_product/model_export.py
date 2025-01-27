from model import Model
import pickle
model = Model("cooler")
model.fit()
model_file = open("model_cooler", "wb")
pickle.dump(model, model_file)
model_file.close()

