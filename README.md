# An LSTM model implemented with PyTorch

This repo is to reproduce the GlueX tracking algorithm, which originally implemented with
 TensorFlow Keras [here](https://github.com/nathanwbrei/phasm/blob/main/python/2022.05.29_GlueX_tracking_v0.1.ipynb),
 with PyTorch, for a future integration with [phasm](https://github.com/nathanwbrei/phasm).

## LSTM Model
[The PyTorch code](python/Simplified_LSTM.py) is using the same LSTM network structure as listed below,
 where batch_size=1256 and seq_len=7.
The parameter counts of the layers are taken from the original Keras `model.summary()`.

| Layer   | Input size                 | Output size                  | Param # |
|:--------|:---------------------------|:-----------------------------|--------:|
| LSTM_1  | (batch_size, seq_len, 6)   | (batch_size, seq_len, 128)   |   69120 | 
| LSTM_2  | (batch_size, seq_len, 128) | (batch_size, seq_len, 64)    |   49408 |  
| LSTM_3  | (batch_size, seq_len, 64)  | (batch_size, 32)             |   12416 |
| Linear  | (batch_size, 32)           | (batch_size, 6)              |     198 |



## Dataset
Download the training dataset as below.

```commandline
wget https://halldweb.jlab.org/talks/ML_lunch/Sep2019/MLchallenge2_training.csv
mv MLchallenge2_training.csv train_data.csv
```
As the dataset has changed after the Keras code's implementation, a 100%
 matched result is impossible.

After sequencing, the dimension of the whole training dataset (as on 09/30/2022) is (2646573, 7, 6), with
 each epoch containing ~2108 batches. We train 100 epochs in total.

## Results

[//]: # (Figure: the training loss on a T4 GPU)

[//]: # (![training-loss-T4]&#40;./res/training-loss/training-loss_65506053_T4.png&#41;)

Table: the results after 100 training epochs

| Exp              | `loss`   | `mse`      | `val_loss` | `val_mse`  | `lr`        |     Time | Training `X` size |
|:-----------------|:---------|:-----------|:-----------|:-----------|:------------|---------:|------------------:|
| Keras TitanRTX*2 | 0.0015   | 6.8281e-06 | 0.0018     | 7.2508e-06 | 3.7715e-05  | ~20 mins |   (1910698, 7, 6) |
| PyTorch TitanRTX | 0.000957 | 0.000007   | 0.000988   | 0.000002   | 5.22006e-05 | ~50 mins |   (2646573, 7, 6) |
| PyTorch T4       | 0.001056 | 1.7273e-06 | 0.000971   | 2.3378e-06 | 5.22006e-05 | ~60 mins |   (2646573, 7, 6) |


The code is tested on a single `ifarm` TitanRTX/T4 GPU. Results are available at:
- [./res/training-loss](./res/training-loss): images of the losses along the training process.
- [./res/job-log](./res/job-log): the detailed job logs. An example of
 how losses are changed along the epochs, batches and time is [here](./res/job-log/train-full_65238781_TitanRTX.log).
- [./res/evaluation](./res/evaluation): images of the evaluation results. [This](./res/evaluation/cmp.md) is
 a comparison between the evaluation errors with Epochs=1 and Epochs=100.



## References
- Keras APIs: https://keras.io/api/
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- PyTorch Tutorials
  - [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
  - [From a LSTM cell to a Multilayer LSTM Network with PyTorch](https://towardsdatascience.com/from-a-lstm-cell-to-a-multilayer-lstm-network-with-pytorch-2899eb5696f3)

---
Last updated on 10/06/2022 by xmei@jlab.org

