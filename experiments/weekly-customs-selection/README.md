## Apply DATE model for Weekly Customs Selection (or Daily)
Written by: Sundong Kim, Institute for Basic Science

#### Preparation
* First, locate the training data under the directory './data/'
* Second, locate the weekly dataset under the directory './weekly-data/', the weekly dataset name should follow 'week*_ano.csv'

#### Prediction
* To make a prediction, run pilot.sh ($ bash pilot.sh). By running this bash script, XGBoost classifier, XGBoost + Logistic regression, and two variants of our DATE model predict the test data.
* The prediction result will be saved under the './weekly-data/' with the file name of 'week*_ano_result'.

#### Parameter tuning
* You can adjust some parameters in the script, you can refer the arguments of preprocess_data.py, generate_loader.py, and train.py, which are listed just below the __main__ part. (e.g., You can change the learning rate of the DATE model by adding --lr 0.01)
* In our setting, we consider training period from 2016-01-01 ~ 2017-10-01 and 2017-10-01 ~ 2017-12-31 as a validation period.

#### Results 
Considering the total number of imports are over two thousands, Top-10 high fraudulent transactions, and bottom-10 low fraudulent transactions selected by DATE and other models are quite consistent each other.

[See this figure](./DATE-customs-selection.png)