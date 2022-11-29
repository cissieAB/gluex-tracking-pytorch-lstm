# An LSTM model implemented with PyTorch

This repo is to reproduce the GlueX tracking algorithm with PyTorch, which originally implemented with
 TensorFlow Keras [here](https://github.com/nathanwbrei/phasm/blob/main/python/2022.05.29_GlueX_tracking_v0.1.ipynb).
 It is aimed to a future integration with [phasm](https://github.com/nathanwbrei/phasm).


To the best of my knowledge, mimic everything in the original Keras notebook, including the same:
- Shuffled training dataset;
- Batch size, epochs, NN size, loss function, optimizer, clip value;
- Learning rate scheduler.


## Python code structure
- [utils.py](python/utils.py): define the NN structure and wrap the datasets into batched PyTorch dataloaders.
- [LSTM_training.py](python/LSTM_training.py): train the NN with the whole training dataset with 100 epochs.
 Save the trained model into a TorchScript.
- [validation_processing.py](python/validation_processing.py): load the model from the TorchScript.
 Validate the model with a validation dataset of (661644, 6).
- [submit-training-job.slurm](python/submit-training-job.slurm): a one-step slurm script to run jobs on farm.

## Configurations

### Conda PyTorch environment
Based on my own experience, the bare-metal python3.9+pip3+cudnn8.6 installation would always fail on A100
because of mismatch cudnn/torch versions. This is solved by installing the latest pytorch (as of Nov-28-2022)
via conda virtual environments as guided [here](https://pytorch.org/get-started/locally/).
A [conda environment file](environment.yml) is provided to show my environment configurations.

```bash
# install pytorch via conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
# create conda env from yml file
conda env create -f environment.yml
```

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
 the new dataset is about 38.5% larger (2646573 v.s. 1910698).

After sequencing, the dimension of the whole training dataset (as on 10/20/2022) is (2646573, 7, 6), with
 each epoch containing ~2108 batches. We train 100 epochs in total.


## Results

Table: results after 100 training epochs

| Exp              | `loss` | `mse`      | `val_loss` | `val_mse`  | `lr`       |     Time | Training `X` size |
|:-----------------|:-------|:-----------|:-----------|:-----------|:-----------|---------:|------------------:|
| Keras+TitanRTX*2 | 0.0015 | 6.8281e-06 | 0.0018     | 7.2508e-06 | 3.7715e-05 | ~20 mins |   (1910698, 7, 6) |
| PyTorch+TitanRTX | 0.0015 | 2.0858e-05 | 0.0015     | 2.0803e-05 | 3.7715e-05 | ~55 mins |   (2646573, 7, 6) |
| PyTorch+T4       | 0.0012 | 8.0547e-06 | 0.0012     | 7.6509e-06 | 4.4371e-05 | ~65 mins |   (2646573, 7, 6) |
| PyTorch+A100     | 0.0010 | 2.6062e-06 | 0.0010     | 2.5379e-06 | 5.2201e-05 | ~45 mins |   (2646573, 7, 6) |


The code is tested on a single `ifarm` TitanRTX/T4/A100 GPU. Results are available at:
- [./res/training-loss](./res/training-loss): images of the losses along the training process.
- [./res/job-logs](./res/job-logs): the detailed job logs. An example of
 how losses are changed along the epochs, batches and time is [here](./res/job-logs/training-full_51367970_A100.log).
- [./res/evaluation](./res/evaluation): images of the evaluation results. [This](./res/evaluation/cmp.md) is
 a comparison between the evaluation errors with Epochs=1 and Epochs=100.


## References
- Keras APIs: https://keras.io/api/
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- PyTorch tutorials
  - [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
  - [From a LSTM cell to a Multilayer LSTM Network with PyTorch](https://towardsdatascience.com/from-a-lstm-cell-to-a-multilayer-lstm-network-with-pytorch-2899eb5696f3)
  - [Loading a TorchScript model in C++](https://pytorch.org/tutorials/advanced/cpp_export.html)

---
Last updated on 12/01/2022 by xmei@jlab.org

