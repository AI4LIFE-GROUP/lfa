import torch
import torch.nn as nn

from torch.utils.data import Dataset

import numpy as np
import math

import os

#####create model classes

class LinearRegressionNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        out = self.linear(x)
        return out
    


class LogisticRegressionNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out
    


#3 hidden layers, 8 nodes per layer, relu + sigmoid activation
class FFNNClassification(nn.Module):
    def __init__(self, input_dim, n_nodes_per_layer=8):
        super().__init__()
        self.n_nodes_per_layer = n_nodes_per_layer
        
        self.linear1 = nn.Linear(input_dim, self.n_nodes_per_layer)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(self.n_nodes_per_layer, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):       
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        out = self.sigmoid(out)
        
        return out



#5 hidden layers, 8 nodes per layer, relu + sigmoid activation
class FFNNClassificationB(nn.Module):
    def __init__(self, input_dim, n_nodes_per_layer=8):
        super().__init__()
        self.n_nodes_per_layer = n_nodes_per_layer
        
        self.linear1 = nn.Linear(input_dim, self.n_nodes_per_layer)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(self.n_nodes_per_layer, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):       
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        out = self.relu3(out)
        out = self.linear4(out)
        out = self.relu4(out)
        out = self.linear5(out)
        out = self.sigmoid(out)
        
        return out
    


#8 hidden layers, 8 nodes per layer, relu + sigmoid activation
#5 hidden layers, 8 nodes per layer, relu + sigmoid activation
class FFNNClassificationC(nn.Module):
    def __init__(self, input_dim, n_nodes_per_layer=8):
        super().__init__()
        self.n_nodes_per_layer = n_nodes_per_layer
        
        self.linear1 = nn.Linear(input_dim, self.n_nodes_per_layer)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.relu5 = nn.ReLU()
        self.linear6 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.relu6 = nn.ReLU()
        self.linear7 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.relu7 = nn.ReLU()
        self.linear8 = nn.Linear(self.n_nodes_per_layer, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):       
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        out = self.relu3(out)
        out = self.linear4(out)
        out = self.relu4(out)
        out = self.linear5(out)
        out = self.relu5(out)
        out = self.linear6(out)
        out = self.relu6(out)
        out = self.linear7(out)
        out = self.relu7(out)
        out = self.linear8(out)
        out = self.sigmoid(out)
        
        return out



#3 hidden layers, each 8 nodes, tanh activation
class FFNNRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.n_nodes_per_layer = 8
        self.linear1 = nn.Linear(input_dim, self.n_nodes_per_layer)
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.tanh2 = nn.Tanh()
        self.linear3 = nn.Linear(self.n_nodes_per_layer, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.tanh1(out)
        out = self.linear2(out)
        out = self.tanh2(out)
        out = self.linear3(out)
        
        return out



#5 hidden layers, each 8 nodes, tanh activation
class FFNNRegressionB(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.n_nodes_per_layer = 8
        self.linear1 = nn.Linear(input_dim, self.n_nodes_per_layer)
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.tanh2 = nn.Tanh()
        self.linear3 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.tanh3 = nn.Tanh()
        self.linear4 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.tanh4 = nn.Tanh()
        self.linear5 = nn.Linear(self.n_nodes_per_layer, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.tanh1(out)
        out = self.linear2(out)
        out = self.tanh2(out)
        out = self.linear3(out)
        out = self.tanh3(out)
        out = self.linear4(out)
        out = self.tanh4(out)
        out = self.linear5(out)
        
        return out



#5 hidden layers, each 8 nodes, tanh activation
class FFNNRegressionC(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.n_nodes_per_layer = 8
        self.linear1 = nn.Linear(input_dim, self.n_nodes_per_layer)
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.tanh2 = nn.Tanh()
        self.linear3 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.tanh3 = nn.Tanh()
        self.linear4 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.tanh4 = nn.Tanh()
        self.linear5 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.tanh5 = nn.Tanh()
        self.linear6 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.tanh6 = nn.Tanh()
        self.linear7 = nn.Linear(self.n_nodes_per_layer, self.n_nodes_per_layer)
        self.tanh7 = nn.Tanh()
        self.linear8 = nn.Linear(self.n_nodes_per_layer, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.tanh1(out)
        out = self.linear2(out)
        out = self.tanh2(out)
        out = self.linear3(out)
        out = self.tanh3(out)
        out = self.linear4(out)
        out = self.tanh4(out)
        out = self.linear5(out)
        out = self.tanh5(out)
        out = self.linear6(out)
        out = self.tanh6(out)
        out = self.linear7(out)
        out = self.tanh7(out)
        out = self.linear8(out)
        
        return out


    
#####create Dataset

class DatasetFromCSV(Dataset):
    def __init__(self, filepath, target_idx):
        self.data = np.loadtxt(filepath, delimiter=',', dtype=np.float64, skiprows=1)
        self.X = torch.from_numpy(np.delete(self.data, target_idx, axis=1)).float() #[num_samples, num_features]
        self.y = torch.from_numpy(self.data[:, [target_idx]]).float() #[num_samples, 1]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.data.shape[0]
    
    