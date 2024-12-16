from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

class CSVData(Dataset):
    def __init__(self):
        self.data = pd.read_csv("data.csv")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            return self.data.iloc[idx, 1:]


class AE(torch.nn.Module):
    def __init__(self):
        super.init()
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

