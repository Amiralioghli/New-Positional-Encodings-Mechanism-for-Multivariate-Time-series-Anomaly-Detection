import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("mps")

class DataLoaderCreator:
    def __init__(self, train_csv_path, test_csv_path, window_size, batch_size=32, test_size=0.15 , device = device):
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.device = device

    def create_sequences(self, inputs, labels):
        input_seq = []
        label_seq = []
        for i in range(labels.shape[0] - (self.window_size - 1)):
            input_seq.append(inputs.iloc[i:i + self.window_size].values)
            label_seq.append(labels.iloc[i + (self.window_size - 1)])
        input_seq = np.array(input_seq)
        label_seq = np.array(label_seq)
        return torch.tensor(input_seq, dtype=torch.float32 , device=self.device), torch.tensor(label_seq, dtype=torch.long , device=self.device)

    def load_data(self):
        train_data = pd.read_csv(self.train_csv_path)
        test_data = pd.read_csv(self.test_csv_path)

        train_input = train_data.iloc[:, :-1]
        train_label = train_data.iloc[:, -1:].astype(dtype=int)
        x_train, x_val, y_train, y_val = train_test_split(train_input, train_label, test_size=self.test_size)
        x_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1:].astype(dtype=int)

        x_train, y_train = self.create_sequences(x_train, y_train)
        x_val, y_val = self.create_sequences(x_val, y_val)
        x_test, y_test = self.create_sequences(x_test, y_test)

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        return train_loader, val_loader, test_loader


