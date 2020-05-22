# DATE: Dual Attentive Tree-aware Embedding for Customs Frauds Detection
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This is our implementation for the paper:

\*Sundong Kim, \*Yu-Che Tsai, Karandeep Singh, Yeonsoo Choi, Etim Ibok, Cheng-Te Li, and Meeyoung Cha. DATE: Dual Attentive Tree-aware Embedding for Customs Fraud Detection. Proceedings of ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD). Virtual Event, USA, August 23-27, 2020. (\*: equally-contributed first authors) (To Appear)

## Requirements
* Ranger optimizer:
    * https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
* pytorch==1.0.0
* torch_multi_head_attention
* scikit-learn==0.21.0
* numpy==1.16.4
* pandas==0.25.3 

## Model Architecture
![](https://i.imgur.com/0BmFe7K.jpg)


## How to Train the Model
Our proposed DATE is a two-stage model, we train XGBoost model first and use the pre-trained model to generate cross feature for second embedding model.

1. Run preprocess_data.py 
This script would run the preprocessing for raw data from customs and dump a preprocessed file.
2. Run generate_loader.py
This will train and evaluate Xgboost model and XGB+LR model.
Also, the scipt will dump a pickle file for embedding model input.
3. Run train.py
You can tune the hyper parameters by adding args after train.py.
e.g. python3 train.py --epoch 10 --l2 1e-6 etc.
__Important:__ make sure you create a folder named "results" to store the result.
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

## Main Results
![](https://i.imgur.com/20EwrQQ.png)

## Other Experiments & Codes 
Code for auxiliary experiments are uploaded in the `experiments/` directory.
* revcls: Section 5.1, date_cls and date_rev results 
* training-length: Section 5.4, effects on training length
* corrupted-data: Section 6, way to leverage existing data
* hyperparameter-analysis: Section 7.1-2, hyperparameter analysis
* loss-weight: Section 7.3, date_cls and date_rev by controlling alpha




## Synthetic Data
For your understanding, synthetic import declarations are uploaded in the `data/` directory.
To use this repository, users are expected to preprocess their own import declarations into the similar format.
Note that the variable name used in the code can be slightly different from the synthetic data.

|   sgd.id | sgd.date        |   importer.id | ... |   total.taxes | illicit   | revenue   |
|-------|------------------|----------|-----|------------------|------------------|----------|
|    SGD1 | 13-01-02 |    IMP56598 | ... | 468 | 0    | 0   |
|    SGD2 | 13-01-02 |   IMP109304 | ... | 94337 | 0  | 0    |
|    SGD3 | 13-01-02 |  IMP383355  | ... | 10826 | 0    | 0   |
|    ...  | ...      |  ... | ... |   ...  | ...  | ...   | 
|  SGD31 | 13-01-02 | IMP239001 | ... | 2707 | 1 | 1302 |
|    ...  | ...      |  ... | ... |   ...  | ...  | ...    | 
|   SGD100000 | 17-12-31  |  IMP821186 | ... |  2713 | 0    | 0 |
