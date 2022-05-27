import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from sklearn.metrics import accuracy_score

from art.classifiers import PyTorchClassifier

def flattern(X):
	return X.reshape(X.shape[0], -1)

def toOneHot(y):
	return np.concatenate((y.reshape(-1, 1)==0, y.reshape(-1, 1)==1, y.reshape(-1, 1)==2), axis=-1).astype(int)

def oneHotToZeroOne(y):
	return np.argmax(y, axis=1)

class Net(nn.Module):
	def __init__(self, input_size, layers_size=100, output_size=3, num_layers=3, dropout=None):
		super(Net, self).__init__()
		self.linears = nn.ModuleList([nn.Linear(input_size, layers_size)])
		self.linears.extend([nn.Linear(layers_size, layers_size) for i in range(1, num_layers-1)])
		self.linears.append(nn.Linear(layers_size, output_size))
		if dropout is not None:
			self.dropouts= nn.ModuleList([nn.Dropout(p=dropout)])
			self.dropouts.extend([nn.Dropout(p=dropout) for i in range(1, num_layers-1)])
			self.dropouts.append(nn.Dropout(p=dropout))
		self.dropout = dropout
							
	def forward(self, x):
		if self.dropout is None:
			for i in range(len(self.linears)-1):
				x = F.relu(self.linears[i](x))
			x = self.linears[-1](x)
		else:
			for i in range(len(self.linears)-1):
				x = self.dropouts[i](F.relu(self.linears[i](x)))
			x = self.dropouts[-1](self.linears[-1](x))
		return x

class torchNNThreeClass:
	def __init__(self, input_size, layers_size=100, output_size=3, num_layers=3, dropout=None, weight_decay=0):
		model = Net(input_size=input_size, layers_size=layers_size, output_size=output_size, num_layers=num_layers, dropout=dropout)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)

		self.classifier = PyTorchClassifier(
			model=model,
			loss=criterion,
			optimizer=optimizer,
			input_shape=input_size,
			nb_classes=3,
		)

		self.input_size=input_size
		self.layers_size=layers_size
		self.output_size=output_size
		self.num_layers=num_layers


	def fit(self, X, y):
		assert y.min() == 0, "Please use 0, 1 label"
		self.classifier.fit(X, toOneHot(y), batch_size=X.shape[0], nb_epochs=100)

	def predict(self, X):
		y_pred = np.argmax(self.classifier.predict(X), axis=1)
		return y_pred

	def score(self, X, y):
		assert y.min() == 0, "Please use 0, 1 label"
		y_pred = np.argmax(self.classifier.predict(X), axis=1)
		return accuracy_score(y, y_pred)

	def decision_function(self, X):
		return self.classifier.predict(X)

	def save(self, DATA_DIR, name):
		self.classifier.save(name+'_nn', DATA_DIR)

	def load(self, DATA_DIR, name):
		model = Net(input_size=self.input_size, layers_size=self.layers_size, output_size=self.output_size, num_layers=self.num_layers)
		model_state = torch.load(DATA_DIR+name+'_nn.model')
		model.load_state_dict(model_state)

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.parameters(), lr=0.01)
		optimizer_state = torch.load(DATA_DIR+name+'_nn.optimizer')
		optimizer.load_state_dict(optimizer_state)

		self.classifier = PyTorchClassifier(
			model=model,
			loss=criterion,
			optimizer=optimizer,
			input_shape=self.input_size,
			nb_classes=3,
		)

