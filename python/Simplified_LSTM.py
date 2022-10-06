"""
Mimic the Keras LSTM code at
https://github.com/nathanwbrei/phasm/blob/main/python/2022.05.29_GlueX_tracking_v0.1.ipynb

Observed differences:
- The training dataset is larger than that in the referred demonstrated jupyter notebook.
- Looks like the referred Keras notebook runs faster than this code :(. Larger dataset might be a reason, but there
 should be more.

Some notes:
- Tested on a single TitanRTX or a single T4.
  1 TitanRTX takes ~50 minutes to complete the training process.
- Tested on farm A100 GPU but failed. It says requiring higher version of libtorch.

mailto: xmei@jlab.org, 10/05/2022
"""

import logging
import os

import torch
from matplotlib import pyplot as plt
from torch import nn

from utils import training_dataloader, validation_dataloader, \
    DATA_DIM, HIDDEN_DIM, EPOCHS, BATCH_SIZE, \
    MODEL_STATE_DICT_PATH, MODEL_TORCH_SCRIPT_PATH, LSTMNetwork

# torch.backends.cuda.matmul.allow_tf32 = True  # for Ampere architecture

"""
Logging configuration
"""
SLURM_JID = os.getenv('SLURM_JOB_ID') if os.getenv('SLURM_JOB_ID') else 'current'
logging.basicConfig(
    filename=f'training-full_{SLURM_JID}.log',
    filemode='w',
    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

NN_CLIP_VALUE = 1.0


class Net:

    def __init__(self, data_dim=DATA_DIM, hidden_dim=HIDDEN_DIM, epochs=EPOCHS):
        """
        Configurations of the training process
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training on {self.device} device")

        self.model = LSTMNetwork(data_dim, hidden_dim).to(self.device)
        # print(self.model)

        self.loss_fn = nn.L1Loss()
        self.metric_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.85,
            patience=5,
            threshold=1e-5,
            min_lr=1e-6,
            verbose=True
        )

        self.epochs = epochs

    def step(self, dataloader):
        num_batches = len(dataloader)
        size = len(dataloader.dataset)
        pred, loss_sum, mse_sum = 0.0, 0.0, 0.0

        for batch_id, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)
            cur_loss, cur_metric = self.loss_fn(pred, y), self.metric_fn(pred, y)
            loss_sum += cur_loss.item()
            mse_sum += cur_metric.item()

            # Back propagation
            self.optimizer.zero_grad()  # reset grad to 0
            cur_loss.backward()
            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=NN_CLIP_VALUE)
            self.optimizer.step()  # update the model parameters

            if batch_id % 100 == 0:  # logging
                current = batch_id * BATCH_SIZE
                logging.info(f"batch_id: {batch_id:>5d}, cur_loss: {cur_loss}, cur_mse: {cur_metric}  [{current:>8d}/{size:>8d}]")

        loss_sum = loss_sum / num_batches
        mse_sum = mse_sum / num_batches
        # print("step", loss_sum, mse_sum)
        return loss_sum, mse_sum  # return float, not tensor

    def validate(self, dataloader):
        num_batches = len(dataloader)
        val_loss, val_mse = 0.0, 0.0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                val_loss += self.loss_fn(pred, y).item()
                val_mse += self.metric_fn(pred, y).item()

        val_loss /= num_batches
        val_mse /= num_batches
        # print("validate", val_loss, val_mse)
        return val_loss, val_mse

    def train(self, train_dataloader, val_dataloader):
        his = {'loss': [], 'mse': [], 'val_loss': [], 'val_mse': []}  # lives on CPU
        logging.info("Training started.\n")

        for it in range(self.epochs):
            loss, mse = self.step(train_dataloader)
            val_loss, val_mse = self.validate(val_dataloader)

            print(f"Epoch {it + 1:>4d}: loss={loss}, mse={mse}, val_loss={val_loss}, val_mse={val_mse}, "
                  f"lr={self.optimizer.param_groups[0]['lr']}")
            logging.info(
                f"Epoch {it + 1:>4d}: loss={loss}, mse={mse}, val_loss={val_loss}, val_mse={val_mse}, lr={self.optimizer.param_groups[0]['lr']}\n")

            self.scheduler.step(val_loss)  # adjust lr based on val_loss

            his['loss'].append(loss)
            his['mse'].append(mse)
            his['val_loss'].append(val_loss)
            his['val_mse'].append(val_mse)

        logging.info("Training stopped. \n\n")
        return his


print(f"Torch version {torch.__version__}")
net = Net()
his = net.train(training_dataloader, validation_dataloader)

"""
Visualization of the training process
"""
fig, ax = plt.subplots()
plt.plot(his['loss'], label="loss")
plt.plot(his['val_loss'], label="val_loss")
plt.yscale("log")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('L1 loss (log scale)')
plt.margins(0.05)
plt.subplots_adjust(left=0.2)
plt.title('The L1 loss along the training process')
plt.savefig(f"./training-loss_{SLURM_JID}.png")

"""
Save the model params

Comment out because it looks like the model params are included in the TorchScript.
"""
# print("\n#########################################")
# print(net.model, "\n")
# print("Model's state_dict:")
# for param_tensor in net.model.state_dict():
#     print("\t", param_tensor, "\t", net.model.state_dict()[param_tensor].size())
# torch.save(net.model.state_dict(), MODEL_STATE_DICT_PATH)
# print(f"Save model.state_dict() to {MODEL_STATE_DICT_PATH}\n")

"""
Save the model into TorchScript
"""
print("\n#########################################")
model_scripted = torch.jit.script(net.model)  # Export to TorchScript
print("TorchScript: \n", model_scripted.code, "\n", model_scripted.graph)
model_scripted.save(MODEL_TORCH_SCRIPT_PATH)  # Save the TorchScript
print(f"Save the model's TorchScript to {MODEL_TORCH_SCRIPT_PATH}\n")
