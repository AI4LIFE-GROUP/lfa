import numpy as np
import torch

import sys
sys.path.append("../code")
from models.functions_model_training import LinearRegressionNN, FFNNRegression, FFNNRegressionB, FFNNRegressionC, LogisticRegressionNN, FFNNClassification, FFNNClassificationB, FFNNClassificationC


def load_data_and_model(data_name, model_name, task):
	#load data points for which to provide explanations
	test_size = 0.20
	test_str = int(test_size*100)
	filepath = f'./data/clean/{data_name}-clean-test{test_str}-normalized-subset.csv' #100 randomly-selected test set points
	data = np.loadtxt(filepath, delimiter=',', dtype=np.float64, skiprows=1)
	data = torch.from_numpy(data).float()
	X = data[:, 0:-1]
	y = data[:, -1]

	#load model checkpoint
	models_dict = {
		'who_linear': LinearRegressionNN(input_dim=X.shape[1]), #instantiate models
		'who_ffnnA': FFNNRegression(input_dim=X.shape[1]),
		'who_ffnnB': FFNNRegressionB(input_dim=X.shape[1]),
		'who_ffnnC': FFNNRegressionC(input_dim=X.shape[1]),
		'heloc_logistic': LogisticRegressionNN(input_dim=X.shape[1]),
		'heloc_ffnnA': FFNNClassification(input_dim=X.shape[1]),
		'heloc_ffnnB': FFNNClassificationB(input_dim=X.shape[1]),
		'heloc_ffnnC': FFNNClassificationC(input_dim=X.shape[1]),
	}

	model_f = models_dict[f'{data_name}_{model_name}']     
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if device=='cuda':
		checkpoint = torch.load(f'./models/model_checkpoints/{task}_{model_name}_{data_name}_ckpt.pth')
	if device=='cpu':
		checkpoint = torch.load(f'./models/model_checkpoints/{task}_{model_name}_{data_name}_ckpt.pth', map_location=torch.device('cpu'))
	model_f.load_state_dict(checkpoint['model'])

	return X, y, model_f


