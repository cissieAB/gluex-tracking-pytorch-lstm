
"""
- Load the data from "train.csv" and wrap it into the batched training and validation PyTorch dataloaders.
- Define the NN structure.

mailto: xmei@jlab.org, 10/05/2022
"""

import numpy as np
import pandas

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

DATA_DIM = 6
DATA_WIDTH = 150
BATCH_SIZE = 1256
EPOCHS = 100
HIDDEN_DIM = 128

MODEL_STATE_DICT_PATH = 'model_state_dict.pt'

# wget https://halldweb.jlab.org/talks/ML_lunch/Sep2019/MLchallenge2_training.csv
# mv MLchallenge2_training.csv train_data.csv
train_csv = pandas.read_csv("train_data.csv")
train_csv.dropna(inplace=True)
# train_x = np.array(train_csv)[0:100, :]  # start from small dataset first
train_x = np.array(train_csv)  # of dim (len, 150)

train_data = np.reshape(train_x, (train_x.shape[0], DATA_WIDTH // DATA_DIM, DATA_DIM))

# normalization
for i in range(train_data.shape[2]):
    minimum = train_data[:, :, i].min()
    maximum = train_data[:, :, i].max()
    train_data[:, :, i] = (train_data[:, :, i] - minimum) / (maximum - minimum)

# transfer data samples into sequences
train_x = []
train_y = []
for i in range(17):
    xs = train_data[:, i:7 + i, :]
    ys = train_data[:, 7 + i:8 + i, :]
    train_x.extend(xs)
    train_y.extend(ys)

train_x = np.array(train_x)
train_y = np.array(train_y)
# print(train_x.shape, train_y.shape)

# Split the whole set into the training and validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)
train_y = np.squeeze(train_y)
val_y = np.squeeze(val_y)

# Wrap the data into batched dataloader
tensor_x, tensor_y = torch.Tensor(train_x), torch.Tensor(train_y)
tensor_val_x, tensor_val_y = torch.Tensor(val_x), torch.Tensor(val_y)

training_set = TensorDataset(tensor_x, tensor_y)
validation_set = TensorDataset(tensor_val_x, tensor_val_y)

training_dataloader = DataLoader(training_set, batch_size=BATCH_SIZE)
validation_dataloader = DataLoader(validation_set, batch_size=BATCH_SIZE)


class LSTMNetwork(nn.Module):
    """
    Define the LSTM network structure.
    """

    def __init__(self, data_dim, hidden_dim):
        super(LSTMNetwork, self).__init__()

        self.hidden_size = hidden_dim
        self.data_dim = data_dim

        # Pytorch LSTM ref: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # By default, PyTorch LSTM's return_sequences=True while for Keras, return_sequences=False,
        # The difference and conversion between them is given \
        # at https://stackoverflow.com/questions/62204109/return-sequences-false-equivalent-in-pytorch-lstm
        self.layer_lstm1 = nn.LSTM(self.data_dim, self.hidden_size, batch_first=True)
        self.layer_lstm2 = nn.LSTM(self.hidden_size, 64, batch_first=True)
        self.layer_lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.layer_output = nn.Linear(32, self.data_dim)

    def forward(self, x):
        # Three LSTM layers, the h/c tensors are by default initialized as zeros
        out, _ = self.layer_lstm1(x)
        out, _ = self.layer_lstm2(out)
        out, _ = self.layer_lstm3(out)
        out = out[:, -1, :]  # mapping return_sequences=False for layer lstm3, note batch_first=True

        # Linear output layer
        out = self.layer_output(out)

        return out
