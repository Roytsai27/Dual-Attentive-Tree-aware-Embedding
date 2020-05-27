import numpy as np
import pandas as pd
import pickle
import random
import pickle

def noiseby10percent(df, transaction_rate, noise_proportion):
    indices = np.random.choice(len(df), int(len(df)*transaction_rate), replace=False)
    before = np.sum(df.loc[indices, 'CIF.VALUE'])
    df.loc[indices, 'CIF.VALUE'] = df.loc[indices, 'CIF.VALUE'] * noise_proportion
    df.loc[indices, 'FOB.VALUE'] = df.loc[indices, 'FOB.VALUE'] * noise_proportion
    df.loc[indices, 'TOTAL.TAXES'] = df.loc[indices, 'TOTAL.TAXES'] * noise_proportion
    after = np.sum(df.loc[indices, 'CIF.VALUE'])
    print(before, after)
    return df

def adjust_illicit_val_deterministic(df, roll_back_constant):
    indices = list(df[df['illicit'] == 1].index)
    before = np.sum(df.loc[indices, 'CIF.VALUE'])
    df.loc[indices, 'CIF.VALUE'] = df.loc[indices, 'CIF.VALUE'] * roll_back_constant
    df.loc[indices, 'FOB.VALUE'] = df.loc[indices, 'FOB.VALUE'] * roll_back_constant
    df.loc[indices, 'TOTAL.TAXES'] = df.loc[indices, 'TOTAL.TAXES'] * roll_back_constant
    after = np.sum(df.loc[indices, 'CIF.VALUE'])
    print(before, after)
    return df

def adjust_illicit_val_stochastic(df, roll_back_constant):
    indices = list(df[df['illicit'] == 1].index)
    before = np.sum(df.loc[indices, 'CIF.VALUE'])
    roll_back_proportion = np.random.normal(loc=roll_back_constant, scale=0.1, size=len(df.loc[indices, 'CIF.VALUE']))
    df.loc[indices, 'CIF.VALUE'] = df.loc[indices, 'CIF.VALUE'] * roll_back_proportion
    df.loc[indices, 'FOB.VALUE'] = df.loc[indices, 'FOB.VALUE'] * roll_back_proportion
    df.loc[indices, 'TOTAL.TAXES'] = df.loc[indices, 'TOTAL.TAXES'] * roll_back_proportion
    after = np.sum(df.loc[indices, 'CIF.VALUE'])
    print(before, after)
    return df

if __name__ == '__main__': 

    for method in ['rollback-none', 'rollback-stochastic', 'rollback-deterministic']:

        df = pd.read_csv('../../../../../Sharedfolder/Ndata.merged.anonymized.single.relabeled.csv', encoding = "ISO-8859-1")
        df = df.dropna(subset=["illicit"])

        stats = df.groupby(['illicit'])['CIF.VALUE'].apply(np.mean)
        roll_back_constant = stats[0] / stats[1]

        if method == 'rollback-stochastic':
            df = adjust_illicit_val_stochastic(df, roll_back_constant)
            print(method, df.groupby(['illicit'])['CIF.VALUE'].apply(np.mean))
        if method == 'rollback-deterministic':
            df = adjust_illicit_val_deterministic(df, roll_back_constant)
            print(method, df.groupby(['illicit'])['CIF.VALUE'].apply(np.mean))
        if method == 'rollback-none':
            df = df

        file_path = f'./data/Ndata.merged.anonymized.single.relabeled.{method}.p'
        df.to_pickle(file_path)

        df = pd.read_pickle(file_path)
        print(method, df.groupby(['illicit'])['CIF.VALUE'].apply(np.mean))