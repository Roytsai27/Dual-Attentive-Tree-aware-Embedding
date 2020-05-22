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

df = pd.read_csv('../../../../../Sharedfolder/Ndata.merged.anonymized.single.relabeled.csv', encoding = "ISO-8859-1")
df = df.dropna(subset=["illicit"])
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
    df = df.dropna(subset=['FOB.VALUE', 'TOTAL.TAXES']) # Remove 170 rows which does not have FOB, CIF value.
    df.loc[:, 'Unitprice'] = df['CIF.VALUE']/df['QUANTITY']
    df.loc[:, 'WUnitprice'] = df['CIF.VALUE']/df['GROSS.WEIGHT']
    df.loc[:, 'TaxRatio'] = df['TOTAL.TAXES'] / df['CIF.VALUE']
    df.loc[:, 'TaxUnitquantity'] = df['TOTAL.TAXES'] / df['QUANTITY']
    df.loc[:, 'FOBCIFRatio'] = df['FOB.VALUE']/df['CIF.VALUE']
    df.loc[:, 'HS6'] = df['TARIFF.CODE'].apply(lambda x: int(x // 10000))
    df.loc[:, 'HS4'] = df['HS6'].apply(lambda x: int(x // 100))
    df.loc[:, 'HS2'] = df['HS4'].apply(lambda x: int(x // 100))
    # Factor some thing
    df.loc[:, 'HS6.Origin'] = [str(i)+'&'+j for i, j in zip(df['HS6'], df['ISO3'])]

    
# #     Made a general function "merge_attributes" for supporting any combination

# #     Generated all possible combinations, But the final AUC is smaller than just adding three combinations active below.
#     candFeaturesCombine = ['OFFICE','IMPORTER.TIN','ISO3','HS6','DECLARANT.CODE']
#     for subset in combinations(candFeaturesCombine, 2):
#         merge_attributes(df, *subset)
    
#     for subset in combinations(candFeaturesCombine, 3):
#         merge_attributes(df, *subset)
        
    merge_attributes(df, 'OFFICE','IMPORTER.TIN')
    merge_attributes(df, 'OFFICE','HS6')
    merge_attributes(df, 'OFFICE','ISO3')
    
    
    # Day of Year of SGD.DATE
    tmp2 = {}
    for date in set(df['SGD.DATE']):
        tmp2[date] = dt.strptime(date, '%y-%m-%d')
    tmp_day = {}
    tmp_week = {}
    tmp_month = {}
    yearStart = dt(tmp2[date].date().year, 1, 1)
    for item in tmp2:
        tmp_day[item] = (tmp2[item] - yearStart).days
        tmp_week[item] = int(tmp_day[item] / 7)
        tmp_month[item] = int(tmp_day[item] / 30)
        
    df.loc[:, 'SGD.DayofYear'] = df['SGD.DATE'].apply(lambda x: tmp_day[x])
    df.loc[:, 'SGD.WeekofYear'] = df['SGD.DATE'].apply(lambda x: tmp_week[x])
    df.loc[:, 'SGD.MonthofYear'] = df['SGD.DATE'].apply(lambda x: tmp_month[x])
    
    
    # RECEIPT-SGD time  # To-Do: We should consider where there aren't any receipt date.
    tmp = {}
    for date in set(df['SGD.DATE']).union(set(df['RECEIPT.DATE'])):
        tmp[date] = dt.strptime(date, '%y-%m-%d')
    df.loc[:, 'RECEIPT.DATE-SGD.DATE'] = df['RECEIPT.DATE'].apply(lambda x: tmp[x]) - df['SGD.DATE'].apply(lambda x: tmp[x])
    df.loc[:, 'RECEIPT.DATE-SGD.DATE'] = df['RECEIPT.DATE-SGD.DATE'].apply(lambda x: x.days)
    
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




dates = ['16-09-01']
for start_date in dates[::-1]:
    train = df[(df['SGD.DATE'] >= start_date) & (df['SGD.DATE'] < '16-12-01')]
    valid = df[(df['SGD.DATE'] >= '16-12-01') & (df['SGD.DATE'] < '17-01-01')]
    test = df[(df['SGD.DATE'] >= '17-01-01')]

    # save label data
    train_reg_label = train['RAISED_TAX_AMOUNT'].values
    valid_reg_label = valid['RAISED_TAX_AMOUNT'].values
    test_reg_label = test['RAISED_TAX_AMOUNT'].values
    train_cls_label = train["illicit"].values
    valid_cls_label = valid["illicit"].values
    test_cls_label = test["illicit"].values

    # Run preprocessing
    train = preprocess(train)
    valid = preprocess(valid)
    test = preprocess(test)
    print("Data size:")
    print(train.shape, valid.shape,test.shape)

    # save labels
    train_reg_label = train['RAISED_TAX_AMOUNT'].values
    valid_reg_label = valid['RAISED_TAX_AMOUNT'].values
    test_reg_label = test['RAISED_TAX_AMOUNT'].values
    train_cls_label = train["illicit"].values
    valid_cls_label = valid["illicit"].values
    test_cls_label = test["illicit"].values

    # Add a few more risky profiles
    risk_profiles = {}
    profile_candidates = ['IMPORTER.TIN', 'DECLARANT.CODE', 'HS6.Origin', 'TARIFF.DESCRIPTION', 'TARIFF.CODE', 'QUANTITY', 'HS6', 'HS4', 'HS2', 'OFFICE'] + [col for col in train.columns if '&' in col]

    for profile in profile_candidates:
        option = 'topk'
        risk_profiles[profile] = find_risk_profile(train, profile, 0.1, 10, option=option)
        train = tag_risky_profiles(train, profile, risk_profiles[profile], option=option)
        valid = tag_risky_profiles(valid, profile, risk_profiles[profile], option=option)
        test = tag_risky_profiles(test, profile, risk_profiles[profile], option=option)

    # Features to use in a classifier
    column_to_use = ['FOB.VALUE', 'CIF.VALUE', 'TOTAL.TAXES', 'GROSS.WEIGHT', 'QUANTITY', 'EXCHANGERATE', 'Unitprice', 'WUnitprice', 'EXCHANGERATE', 'TaxRatio', 'FOBCIFRatio', 'TaxUnitquantity', 'TARIFF.CODE', 'HS6', 'HS4', 'HS2', 'SGD.DayofYear', 'SGD.WeekofYear', 'SGD.MonthofYear'] + [col for col in train.columns if 'RiskH' in col] 
    X_train = train[column_to_use].values
    X_valid = valid[column_to_use].values
    X_test = test[column_to_use].values

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
    print("Checking labled distribution")
    cnt = Counter(train_cls_label)
    print("Training:",cnt[1]/cnt[0])
    cnt = Counter(valid_cls_label)
    print("Validation:",cnt[1]/cnt[0])
    cnt = Counter(test_cls_label)
    print("Testing:",cnt[1]/cnt[0])

    # pickle a variable to a file
    file_name = f'./data/processed_data_{start_date}.pickle'
    file = open(file_name, 'wb')
    pickle.dump(all_data, file)
    file.close()

