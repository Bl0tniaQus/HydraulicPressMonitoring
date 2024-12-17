from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import pandas as pd

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(498, 249),
            torch.nn.ReLU(),
            torch.nn.Linear(249, 83),
            torch.nn.ReLU(),
            torch.nn.Linear(83, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 17)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(17, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 83),
            torch.nn.ReLU(),
            torch.nn.Linear(83, 249),
            torch.nn.ReLU(),
            torch.nn.Linear(249, 498),
            torch.nn.Sigmoid()
        )
    def decode(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

def AE_train(n_epochs, lr):
    epochs = n_epochs
    outputs = []
    losses = []
    data = pd.read_csv("./data.csv")
    y = data.copy().iloc[:,data.shape[1]-5]
    x = data.copy().iloc[:, 0:data.shape[1]-5]
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1)
    model = AE()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = lr,weight_decay = 1e-8)
    for epoch in range(n_epochs):
        for i in range(X_train.shape[0]):
            obs = X_train.iloc[i:]
            print(obs)
            obs_t = torch.from_numpy(obs.values)
            print(obs_t)
            output = model.decode(obs_t)
            loss = loss_function(output, obs_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss)
        outputs.append((epoch, i, output))
AE_train(1,0.1)

