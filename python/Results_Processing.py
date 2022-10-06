"""
Evaluation with the whole evaluation dataset

Reference: https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""

import numpy as np
import os
import logging
import torch
from matplotlib import pyplot as plt

from utils import validation_dataloader, DATA_DIM, BATCH_SIZE, MODEL_TORCH_SCRIPT_PATH

SLURM_JID = os.getenv('SLURM_JOB_ID') if os.getenv('SLURM_JOB_ID') else 'current'
logging.basicConfig(
    filename=f'inference-full_{SLURM_JID}.log',
    filemode='w',
    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

"""
Load model via model.state_dict

No need now since we are loading TorchScript now.
"""
# model = LSTMNetwork(DATA_DIM, HIDDEN_DIM)
# model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH))

"""
Load model via TorchScript
"""
model = torch.jit.load(MODEL_TORCH_SCRIPT_PATH)


model.eval()  # necessary

# The below two numpy arrays live on CPU
pred_all = np.zeros((len(validation_dataloader.dataset), DATA_DIM))
y_all = np.zeros((len(validation_dataloader.dataset), DATA_DIM))

print("\n###########################################")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

loss_fn = torch.nn.L1Loss()
loss_sum = 0.0

print(f"Start inference on {device}")
logging.info(f"Start inference on {device}")
with torch.no_grad():
    for batch_id, (X, y) in enumerate(validation_dataloader):
        pred = model(X.to(device)).cpu()
        cur_idx, cur_loss = batch_id * BATCH_SIZE, loss_fn(pred, y).item()
        pred_all[cur_idx:(cur_idx + len(X)), :] = pred.detach().numpy()  # move to cpu and convert to a numpy array
        y_all[cur_idx:(cur_idx + len(X)), :] = y
        # print(f"  batch_id={batch_id:>4d}, l1_loss={cur_loss}")
        logging.info(f"batch_id={batch_id:>4d}, l1_loss={cur_loss}")
        loss_sum += cur_loss

print("Inference completed.\n")
loss_sum /= len(validation_dataloader)
diff = y_all - pred_all
print(f"diff.shape={diff.shape}, l1_loss={loss_sum}")

# print(diff)

"""
Visualization
"""
plt.rcParams.update({'font.size': 15})
labels = ['x', 'y', 'z', 'px', 'py', 'pz']
fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
fig.suptitle('Model evaluation with the full validation dataset')
for i in range(6):
    row_id, col_id = i // 3, i % 3
    ax[row_id, col_id].hist(
        diff[:, i],
        weights=100 * np.ones(len(diff[:, i])) / len(diff[:, i]),
        label=labels[i]
    )
    ax[row_id, col_id].legend()
fig.text(0.5, 0.04, 'Difference between predictions and truth', ha='center')
fig.text(0.04, 0.5, 'Percentage counts (%)', va='center', rotation='vertical')
plt.subplots_adjust(left=0.1)
plt.savefig(f"./evaluation-error-percentage_{SLURM_JID}.png")

"""
Interesting error if we do not batch the validation dataset.

On TitanRTX
# CUDA out of memory. Tried to allocate 11.04 GiB (GPU 0; 23.65 GiB total capacity; 18.20 GiB already allocated;
# 4.46 GiB free; 18.21 GiB reserved in total by PyTorch)
# If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.
# See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
# srun: error: sciml1903: task 0: Exited with exit code 1

On T4
# RuntimeError: CUDA out of memory. 
# Tried to allocate 14.51 GiB (GPU 0; 14.76 GiB total capacity; 3.37 GiB already allocated; 
# 10.49 GiB free; 3.37 GiB reserved in total by PyTorch) 
# If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation. 
# See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
"""
