# WCO model

## Requirements
* Ranger optimizer:
    * https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
* pytorch
* torch_multi_head_attention
* scikit-learn

## How to train the model
Since we propose a two-stage model, we train Xgboost model first and use the pre-trained model to generate cross feature for second embedding model
Note:the processed_data.pickle file contains raw data and preprocessed data by Sundong's jupyter notebook

1. Run generate_loader.py
This will train and evaluate Xgboost model and XGB+LR model
Aslo, the scipt will dump a pickle file for embedding model input
2. Run train.py
you can tune the hyper parameters by adding args after train.py
e.g. python3 train.py --epoch 10 --l2 1e-6 etc.
**Important:** make sure you create a folder named "results" to store the result
```
--epoch: number of epochs
--l2: l2 regularization 
--dim: dimension for hidden layer
--lr: learning rate
--head_num: number of heads for self-attention
--fusion: fusion type for last layer (concat or attention)
--act: activation function (Relu or Mish)
--device: The device name for training, if train with cpu please use:"cpu" 
--output: save the performance output in a csv file
```