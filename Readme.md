# DATE: Dual Attentive Tree-aware Embedding for CustomsFrauds Detection

This is our implementation for the paper:

> DATE: Dual Attentive Tree-aware Embedding for CustomsFrauds Detection. Sundong Kim∗, Yu-Che Tsai∗, Karandeep Singh, Yeonsoo Choi, Etim Ibok, Cheng-Te Li, Meeyoung Cha. 

Submitting to KDD'2020

## Requirements
* Ranger optimizer:
    * https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
* pytorch==1.0.0
* torch_multi_head_attention
* scikit-learn==0.21.0
* numpy==1.16.4
* pandas==0.25.3 

## How to train the model
Our proposed DATE is a two-stage model, we train XGBoost model first and use the pre-trained model to generate cross feature for second embedding model.

1. Run preprocess_data.py 
This script would run the preprocessing for raw data from customs and dump a preprocessed file.
2. Run generate_loader.py
This will train and evaluate XGBoost model and XGB+LR model
Also, the script will dump a pickle file for embedding model input
3. Run train.py
you can tune the hyper parameters by adding args after train.py
e.g. python3 train.py --epoch 10 --l2 1e-6 etc.
**Important:** make sure you create a folder named "results" to store the result
```
--epoch: number of epochs
--l2: l2 regularization 
--dim: dimension for hidden layer
--use_self: Use leaf-wise self attention or not 
--alpha: The adaptive weight to balance the scale and importance for regression loss
--lr: learning rate
--head_num: number of heads for self-attention
--act: activation function (Relu or Mish)
--device: The device name for training, if train with cpu please use:"cpu" 
--output: save the performance output in a csv file
```
