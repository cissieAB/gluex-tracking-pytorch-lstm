# An LSTM model implemented with PyTorch

This repo is to reproduce the GlueX tracking algorithm with PyTorch, which originally implemented with
 TensorFlow Keras [here](https://github.com/nathanwbrei/phasm/blob/main/python/2022.05.29_GlueX_tracking_v0.1.ipynb).
 It is aimed to a future integration with [phasm](https://github.com/nathanwbrei/phasm).


To the best of my knowledge, mimic everything in the original Keras notebook, including the same:
- Shuffled training dataset;
- Batch size, epochs, NN size, loss function, optimizer, clip value;
- Learning rate scheduler.


### Python code structure
- [utils.py](python/utils.py): define the NN structure and wrap the datasets into batched PyTorch dataloaders.
- [LSTM_training.py](python/LSTM_training.py): train the NN with the whole training dataset with $epochs=100$.
 Save the trained model into a TorchScript.
- [validation_processing.py](python/validation_processing.py): load the model from the TorchScript.
 Validate the model with a validation dataset of (661644, 6).


### Results

Table: results after 100 training epochs

| Exp              | `loss` | `mse`      | `val_loss` | `val_mse`  | `lr`       |     Time | Training `X` size |
|:-----------------|:-------|:-----------|:-----------|:-----------|:-----------|---------:|------------------:|
| Keras TitanRTX*2 | 0.0015 | 6.8281e-06 | 0.0018     | 7.2508e-06 | 3.7715e-05 | ~20 mins |   (1910698, 7, 6) |
| PyTorch TitanRTX | 0.0011 | 3.1423e-06 | 0.0010     | 2.9384e-06 | 6.1413e-05 | ~60 mins |   (2646573, 7, 6) |
| PyTorch T4       | 0.0016 | 1.9300e-05 | 0.0016     | 1.9548e-05 | 5.2200e-05 | ~80 mins |   (2646573, 7, 6) |


The code is tested on a single `ifarm` TitanRTX/T4 GPU. Results are available at:
- [./res/training-loss](./res/training-loss): images of the losses along the training process.
- [./res/job-logs](./res/job-logs): the detailed job logs. An example of
 how losses are changed along the epochs, batches and time is [here](./res/job-logs/training-full_66284590_TitanRTX.log).
- [./res/evaluation](./res/evaluation): images of the evaluation results. [This](./res/evaluation/cmp.md) is
 a comparison between the evaluation errors with Epochs=1 and Epochs=100.

### NN definition
Table: the LSTM network definition, where batch_size=1256 and seq_len=7.

| Layer   | Input size                 | Output size                  | Param # |
|:--------|:---------------------------|:-----------------------------|--------:|
| LSTM_1  | (batch_size, seq_len, 6)   | (batch_size, seq_len, 128)   |   69120 |
| LSTM_2  | (batch_size, seq_len, 128) | (batch_size, seq_len, 64)    |   49408 |
| LSTM_3  | (batch_size, seq_len, 64)  | (batch_size, 32)             |   12416 |
| Linear  | (batch_size, 32)           | (batch_size, 6)              |     198 |

The parameter counts of the layers are taken from the original Keras `model.summary()`.

### Dataset
Download the training dataset as below.

```commandline
wget https://halldweb.jlab.org/talks/ML_lunch/Sep2019/MLchallenge2_training.csv
mv MLchallenge2_training.csv train_data.csv
```
Compared to the dataset at the time of executing the Keras notebook,
 the new dataset is about 40% larger (2646573 v.s. 1910698).

After sequencing, the dimension of the whole training dataset (as on 10/20/2022) is (2646573, 7, 6), with
 each epoch containing ~2108 batches. We train 100 epochs in total.


### TODOs
- [ ] Use C++ to load the TorchScript of the model trained by Python. Test the inference accuracy.
- [ ] A100 PyTorch error. Find the reason why the Python code cannot run on A100 (while cpp can).
```
/home/xmei/.local/lib/python3.6/site-packages/torch/cuda/__init__.py:143: UserWarning: 
NVIDIA A100 80GB PCIe with CUDA capability sm_80 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70.
If you want to use the NVIDIA A100 80GB PCIe GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
```

### References
- Keras APIs: https://keras.io/api/
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- PyTorch tutorials
  - [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
  - [From a LSTM cell to a Multilayer LSTM Network with PyTorch](https://towardsdatascience.com/from-a-lstm-cell-to-a-multilayer-lstm-network-with-pytorch-2899eb5696f3)
  - [Loading a TorchScript model in C++](https://pytorch.org/tutorials/advanced/cpp_export.html)

---
Last updated on 10/20/2022 by xmei@jlab.org

