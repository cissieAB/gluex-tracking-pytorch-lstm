# An LSTM model implemented with PyTorch

This repo is to reproduce the GlueX tracking algorithm, which originally implemented with
 TensorFlow Keras [here](https://github.com/nathanwbrei/phasm/blob/main/python/2022.05.29_GlueX_tracking_v0.1.ipynb),
 with PyTorch, for a future integration with [phasm](https://github.com/nathanwbrei/phasm).

[The PyTorch code](python/Simplified_LSTM.py) is using the same LSTM network structure as listed below,
 where batch_size=1256 and seq_len=7.
The parameter counts of the layers are taken from the original Keras `model.summary()`.

| Layer   | Input size                 | Output size                  | Param # |
|:--------|:---------------------------|:-----------------------------|--------:|
| LSTM_1  | (batch_size, seq_len, 6)   | (batch_size, seq_len, 128)   |   69120 | 
| LSTM_2  | (batch_size, seq_len, 128) | (batch_size, seq_len, 64)    |   49408 |  
| LSTM_3  | (batch_size, seq_len, 64)  | (batch_size, 32)             |   12416 |
| Linear  | (batch_size, 32)           | (batch_size, 6)              |     198 |



### Dataset
Download the dataset at the below address.

```commandline
wget https://halldweb.jlab.org/talks/ML_lunch/Sep2019/MLchallenge2_training.csv
mv MLchallenge2_training.csv train_data.csv

wget https://halldweb.jlab.org/talks/ML_lunch/Sep2019/MLchallenge2_testing_inputs.csv
mv MLchallenge2_testing_inputs.csv test_data.csv
```
As the dataset has changed after the Keras code's implementation, a 100%
 matched result is impossible.

After sequencing, the dimension of the whole training dataset is (2646573, 7, 6), with
 each epoch containing ~2108 batches. We train 100 epochs in total.

### Early results
The code is tested with an `ifarm` RTX GPU. Some early results are available.
- [./res/training-loss](./res/training-loss): images of the losses along the training process.
- [./res/job-log](./res/job-log): the detailed job logs. An example of
 how losses are changed along the epochs, batches and time is [here](./res/job-log/lstm-full_65238781.log).
- [./res/evaluation](./res/evaluation): images of the evaluation results.

### TODO
The cuda out-of-memory error at the final evaluation stage, where we
 try to feed the GPU with the whole validation dataset. Error msg is
 as below. This might be solved by batching the validation set while
 copying back the data from GPU to CPU on the fly.

```python
preds = model.forward(tensor_val_x.to(device))

# CUDA out of memory. Tried to allocate 11.04 GiB (GPU 0; 23.65 GiB total capacity; 18.20 GiB already allocated;
# 4.46 GiB free; 18.21 GiB reserved in total by PyTorch)
# If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.
# See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
# srun: error: sciml1903: task 0: Exited with exit code 1
```

---
### References
- Keras APIs: https://keras.io/api/
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- [From a LSTM cell to a Multilayer LSTM Network with PyTorch](https://towardsdatascience.com/from-a-lstm-cell-to-a-multilayer-lstm-network-with-pytorch-2899eb5696f3)


---
Last updated on 09/30/2022 by xmei@jlab.org

