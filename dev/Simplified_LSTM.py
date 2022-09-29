import numpy as np
import pandas
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

import logging
logging.basicConfig(
    filename='lstm-full.log',
    filemode='w',
    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Tested on farm A100 GPU but failed. It says requiring higher version of libtorch.
# torch.backends.cuda.matmul.allow_tf32 = True  # for Ampere architecture

BATCH_SIZE = 1256
DATA_DIM = 6
NN_HIDDEN_DIM = 128

train_csv = pandas.read_csv("train_data.csv")
train_csv.dropna(inplace=True)
train_x = np.array(train_csv)[0:100, :]  # start from small dataset first
# train_x = np.array(train_csv)

train_data = np.reshape(train_x, (train_x.shape[0], 25, 6))

for i in range(train_data.shape[2]):
    minimum = train_data[:, :, i].min()
    maximum = train_data[:, :, i].max()
    train_data[:, :, i] = (train_data[:, :, i] - minimum) / (maximum-minimum)

train_x = []
train_y = []
for i in range(17):
    xs = train_data[:, i:7+i, :]
    ys = train_data[:, 7+i:8+i, :]
    train_x.extend(xs)
    train_y.extend(ys)

train_x = np.array(train_x)
train_y = np.array(train_y)
# print(train_x.shape, train_y.shape)

# Split the training and validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)
train_y = np.squeeze(train_y)
val_y = np.squeeze(val_y)


tensor_x, tensor_y = torch.Tensor(train_x), torch.Tensor(train_y)
tensor_val_x, tensor_val_y = torch.Tensor(val_x), torch.Tensor(val_y)

training_set = TensorDataset(tensor_x,tensor_y)
validation_set = TensorDataset(tensor_val_x, tensor_val_y)

training_dataloader = DataLoader(training_set, batch_size=BATCH_SIZE)
validation_dataloader = DataLoader(validation_set, batch_size=BATCH_SIZE)


class LSTMNetwork(nn.Module):
    """
    Define the LSTM network struture.
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


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = LSTMNetwork(
    data_dim=DATA_DIM,
    hidden_dim=NN_HIDDEN_DIM
).to(device)
# print(model)

loss_fn = nn.L1Loss()
metric_fn = nn.MSELoss()

# Simplified optimizer
# TODO: add adaptive lr later
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_id, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()  # reset grad to 0
        loss.backward()
        optimizer.step()  # update the model parameters

        if batch_id % 100 == 0:
            loss, error, current = loss.item(), metric_fn(pred, y).item(), batch_id * len(X)
            logging.info(f"batch_id: {batch_id:>6d}, loss: {loss:>7f}, mse: {error:>8f}  [{current:>7d}/{size:>7d}]")

    loss, mse = loss.item(), metric_fn(pred, y)
    return loss, mse


def validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()

    val_loss /= num_batches
    val_mse = metric_fn(pred, y).item()
    return val_loss, val_mse


epochs = 100
his_loss, his_mse, his_val_loss, his_val_mse = [], [], [], []

for t in range(epochs):
    loss, mse = train(training_dataloader, model, loss_fn, optimizer)
    val_loss, val_mse = validate(validation_dataloader, model, loss_fn)
    print(f"Epoch {t + 1:>4d}: loss={loss:>7f}, mse={mse:>8f}, val_loss={val_loss:>7f}, val_mse={val_mse:>7f}")
    logging.info(f"Epoch {t + 1:>4d}: loss={loss:>7f}, mse={mse:>8f}, val_loss={val_loss:>7f}, val_mse={val_mse:>7f}")

    his_loss.append(loss)
    his_mse.append(mse)
    his_val_loss.append(val_loss)
    his_val_mse.append(val_mse)


"""
Some visualization
"""
fig, ax = plt.subplots()
plt.plot(his_loss, label="loss")
plt.plot(his_val_loss, label="val_loss")
plt.yscale("log")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('L1 loss (log scale)')
plt.margins(0.05)
plt.subplots_adjust(left=0.2)
plt.title('Losses of the training process')
plt.savefig('./training-loss.png')


"""
Evaluation with the whole evaluation dataset
"""
preds = model.forward(tensor_val_x)
# print(f"preds.shape={preds.shape}", f"tensor_val_y.shape={tensor_val_y.shape}")
diff = (tensor_val_y - preds).detach().numpy()  # convert to a numpy array

plt.rcParams.update({'font.size': 15})
labels = ['x', 'y', 'z', 'px', 'py', 'pz']
fig, ax = plt.subplots(2, 3, figsize=(25, 6))
fig.suptitle('Model evaluation with the full validation dataset')
for i in range(6):
    row_id, col_id = i // 3, i % 3
    # TODO: change y-axis to percentage
    ax[row_id, col_id].hist(diff[:, i], weights=np.ones(len(diff[:, i])) / len(diff[:, i]), label=labels[i])
    ax[row_id, col_id].legend()
fig.text(0.5, 0.04, 'Difference between predictions and truth', ha='center')
fig.text(0.04, 0.5, 'Percentage (%)', va='center', rotation='vertical')
plt.savefig('./evaluation-error-percentage.png')


# TODO: save the model params and tmp data
