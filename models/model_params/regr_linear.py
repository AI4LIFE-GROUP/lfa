import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

from functions_model_training import LinearRegressionNN, DatasetFromCSV


#datasets
test_size = 0.20
train_str = int((1-test_size)*100)
test_str = int(test_size*100)
dataset_name = 'who'
task = 'regr'
train_ds = DatasetFromCSV(filepath=f'./data/clean/{dataset_name}-clean-train{train_str}-normalized.csv', target_idx=-1)
test_ds = DatasetFromCSV(filepath=f'./data/clean/{dataset_name}-clean-test{test_str}-normalized.csv', target_idx=-1)

#dataloaders
batch_size=64
train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True)

#model
model = LinearRegressionNN(input_dim=train_ds.X.shape[1])

#training parameters
n_epochs = 100
criterion = nn.MSELoss()
learning_rate = 0.2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
seed = 12345

#model description to add to name of checkpoint file
ckpt_descrip=f'linear_{dataset_name}'

