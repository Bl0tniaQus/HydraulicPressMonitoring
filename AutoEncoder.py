from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
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
            torch.nn.Linear(25, 17),
            torch.nn.ReLU()
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
    def forward(self, x):
        return self.decode(x)
    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

def AE_train(x, n_epochs, lr):
    epochs = n_epochs
    outputs = []
    losses = []
    scaler = StandardScaler()
    model = AE()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adadelta(model.parameters())
    for epoch in range(n_epochs):
        for i in range(x.shape[0]):
            obs = x[i]
            obs_t = torch.from_numpy(obs)
            obs_t = obs_t.to(torch.float32)
            enc = model.encode(obs_t)
            output = model(obs_t)
            loss = loss_function(output, obs_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss)
    return model

