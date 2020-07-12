import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle

from collections import defaultdict
from itertools import islice, combinations
from datetime import datetime as dt
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('./data/synthetic-imports-declarations.csv', encoding = "ISO-8859-1")
df = df.dropna(subset=["illicit"])
df = df.sort_values("sgd.date")
print("Finish loading data...")

def merge_attributes(df: pd.DataFrame, *args: str) -> None:
    """
    dtype df: dataframe
    dtype *args: strings (attribute names that want to be combined)
    """
    iterables = [df[arg].astype(str) for arg in args]
    columnName = '&'.join([*args]) 
    fs = [''.join([v for v in var]) for var in zip(*iterables)]
    df.loc[:, columnName] = fs
    
    
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    dtype df: dataframe
    rtype df: dataframe
    """
    df = df.dropna(subset=['fob.value', 'total.taxes']) # Remove 170 rows which does not have FOB, CIF value.
    df.loc[:, 'Unitprice'] = df['cif.value']/df['quantity']
    df.loc[:, 'WUnitprice'] = df['cif.value']/df['gross.weight']
    df.loc[:, 'TaxRatio'] = df['total.taxes'] / df['cif.value']
    df.loc[:, 'TaxUnitquantity'] = df['total.taxes'] / df['quantity']
    df.loc[:, 'FOBCIFRatio'] = df['fob.value']/df['cif.value']
    df.loc[:, 'HS6'] = df['tariff.code'].apply(lambda x: int(x // 10000))
    df.loc[:, 'HS4'] = df['HS6'].apply(lambda x: int(x // 100))
    df.loc[:, 'HS2'] = df['HS4'].apply(lambda x: int(x // 100))
    # Factor some thing
    df.loc[:, 'HS6.Origin'] = [str(i)+'&'+j for i, j in zip(df['HS6'], df['country'])]

    
# #     Made a general function "merge_attributes" for supporting any combination

# #     Generated all possible combinations, But the final AUC is smaller than just adding three combinations active below.
#     candFeaturesCombine = ['office.id','importer.id','country','HS6','declarant.id']
#     for subset in combinations(candFeaturesCombine, 2):
#         merge_attributes(df, *subset)
    
#     for subset in combinations(candFeaturesCombine, 3):
#         merge_attributes(df, *subset)
        
    merge_attributes(df, 'office.id','importer.id')
    merge_attributes(df, 'office.id','HS6')
    merge_attributes(df, 'office.id','country')
    
    df['sgd.date'] = df['sgd.date'].apply(lambda x: dt.strptime(x, '%y-%m-%d'))
    df.loc[:, 'SGD.DayofYear'] = df['sgd.date'].dt.dayofyear
    df.loc[:, 'SGD.WeekofYear'] = df['sgd.date'].dt.weekofyear
    df.loc[:, 'SGD.MonthofYear'] = df['sgd.date'].dt.month
    return df


def find_risk_profile(df: pd.DataFrame, feature: str, topk_ratio: float, adj: float, option: str) -> list or dict:
    """
    dtype feature: str
    dtype topk_ratio: float (range: 0-1)
    dtype adj: float (to modify the mean)
    dtype option: str ('topk', 'ratio')
    rtype: list(option='topk') or dict(option='ratio')
    
    The option topk is usually better than the ratio because of overfitting.
    """

    # Top-k suspicious item flagging
    if option == 'topk':
        total_cnt = df.groupby([feature])['illicit']
        nrisky_profile = int(topk_ratio*len(total_cnt))+1
        # prob_illicit = total_cnt.mean()  # Simple mean
        adj_prob_illicit = total_cnt.sum() / (total_cnt.count()+adj)  # Smoothed mean
        return list(adj_prob_illicit.sort_values(ascending=False).head(nrisky_profile).index)
    
    # Illicit-ratio encoding (Mean target encoding)
    # Refer: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-munging/target-encoding.html
    # Refer: https://towardsdatascience.com/why-you-should-try-mean-encoding-17057262cd0
    elif option == 'ratio':
        # For target encoding, we just use 70% of train data to avoid overfitting (otherwise, test AUC drops significantly)
        total_cnt = df.sample(frac=0.7).groupby([feature])['illicit']
        nrisky_profile = int(topk_ratio*len(total_cnt))+1
        # prob_illicit = total_cnt.mean()  # Simple mean
        adj_prob_illicit = total_cnt.sum() / (total_cnt.count()+adj)  # Smoothed mean
        return adj_prob_illicit.to_dict()
    
    
def tag_risky_profiles(df: pd.DataFrame, profile: str, profiles: list or dict, option: str) -> pd.DataFrame:
    """
    dtype df: dataframe
    dtype profile: str
    dtype profiles: list(option='topk') or dictionary(option='ratio')
    dtype option: str ('topk', 'ratio')
    rtype: dataframe
    
    The option topk is usually better than the ratio because of overfitting.
    """
    # Top-k suspicious item flagging
    if option == 'topk':
        d = defaultdict(int)
        for id in profiles:
            d[id] = 1
    #     print(list(islice(d.items(), 10)))  # For debugging
        df.loc[:, 'RiskH.'+profile] = df[profile].apply(lambda x: d[x])
    
    # Illicit-ratio encoding
    elif option == 'ratio':
        overall_ratio_train = np.mean(train.illicit) # When scripting, saving it as a class variable is clearer.
        df.loc[:, 'RiskH.'+profile] = df[profile].apply(lambda x: profiles.get(x), overall_ratio_train)
    return df


# Dataset settings
# data_length = df.shape[0]
# train_ratio = 0.6
# valid_ratio = 0.8
# train_length = int(data_length*train_ratio)
# valid_length = int(data_length*valid_ratio)

# split train/valid/test set
# train = df.iloc[:train_length,:]
# valid = df.iloc[train_length:valid_length,:]
# test = df.iloc[valid_length:,:]
train = df[df["sgd.date"] < "13-10-01"]
valid = df[(df["sgd.date"] >= "13-10-01") & (df["sgd.date"] < "13-11-01")]
test = df[df["sgd.date"] >= "13-11-01"]

# save label data
train_reg_label = train['revenue'].values
valid_reg_label = valid['revenue'].values
test_reg_label = test['revenue'].values
train_cls_label = train["illicit"].values
valid_cls_label = valid["illicit"].values
test_cls_label = test["illicit"].values

# Run preprocessing
train = preprocess(train)
valid = preprocess(valid)
test = preprocess(test)

# save labels
train_reg_label = train['revenue'].values
valid_reg_label = valid['revenue'].values
test_reg_label = test['revenue'].values
train_cls_label = train["illicit"].values
valid_cls_label = valid["illicit"].values
test_cls_label = test["illicit"].values

# Add a few more risky profiles
risk_profiles = {}
profile_candidates = ['importer.id', 'declarant.id', 'HS6.Origin', 'tariff.code', 'quantity', 'HS6', 'HS4', 'HS2', 'office.id'] + [col for col in train.columns if '&' in col]

for profile in profile_candidates:
    option = 'topk'
    risk_profiles[profile] = find_risk_profile(train, profile, 0.1, 10, option=option)
    train = tag_risky_profiles(train, profile, risk_profiles[profile], option=option)
    valid = tag_risky_profiles(valid, profile, risk_profiles[profile], option=option)
    test = tag_risky_profiles(test, profile, risk_profiles[profile], option=option)

# Features to use in a classifier
column_to_use = ['fob.value', 'cif.value', 'total.taxes', 'gross.weight', 'quantity', 'Unitprice', 'WUnitprice', 'TaxRatio', 'FOBCIFRatio', 'TaxUnitquantity', 'tariff.code', 'HS6', 'HS4', 'HS2', 'SGD.DayofYear', 'SGD.WeekofYear', 'SGD.MonthofYear'] + [col for col in train.columns if 'RiskH' in col] 
X_train = train[column_to_use].values
X_valid = valid[column_to_use].values
X_test = test[column_to_use].values
print("Data size:")
print(train.shape, valid.shape,test.shape)

# impute nan
X_train = np.nan_to_num(X_train, 0)
X_valid = np.nan_to_num(X_valid, 0)
X_test = np.nan_to_num(X_test, 0)

# store all data in a dictionary
all_data = {"raw":{"train":train,"valid":valid,"test":test},
 "xgboost_data":{"train_x":X_train,"train_y":train_cls_label,\
                 "valid_x":X_valid,"valid_y":valid_cls_label,\
                 "test_x":X_test,"test_y":test_cls_label},
 "revenue":{"train":train_reg_label,"valid":valid_reg_label,"test":test_reg_label}}

# make sure the data size are correct
print("Checking data size...")
print(X_train.shape[0], train_cls_label.shape, train_reg_label.shape)
print(X_valid.shape[0], valid_cls_label.shape, valid_reg_label.shape)
print(X_test.shape[0], test_cls_label.shape, test_reg_label.shape)

from collections import Counter
print("Checking label distribution")
cnt = Counter(train_cls_label)
print("Training:",cnt[1]/cnt[0])
cnt = Counter(valid_cls_label)
print("Validation:",cnt[1]/cnt[0])
cnt = Counter(test_cls_label)
print("Testing:",cnt[1]/cnt[0])

# pickle a variable to a file
file = open('./processed_data.pickle', 'wb')
pickle.dump(all_data, file)
file.close()

