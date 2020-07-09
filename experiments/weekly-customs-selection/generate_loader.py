import numpy as np 
import pandas as pd 
import pickle
import copy
import os 
from xgboost import XGBClassifier, XGBRegressor
from utils import find_best_threshold,process_leaf_idx,stratify_sample, writeResult
from sklearn.metrics import f1_score,roc_auc_score
import torch
import torch.utils.data as Data
import warnings
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import  OneHotEncoder
warnings.filterwarnings("ignore")


 # Xgboost+LR model 



if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--week',
                        type=int,
                        default=2,
                        help="week number: e.g., --week 2")
    parser.add_argument('--date',
                        type=str,
                        default='16-01-01',
                        help="training staring date")
    
    args = parser.parse_args()
    input_path = './data/Nigeria_pilot_weekly/week'+str(args.week)+'_ano.csv'
    output_path = './data/Nigeria_pilot_weekly/week'+str(args.week)+'_ano_result.csv'


    # load preprocessed data
    # 
    dates = [args.date]
    for start_date in dates[::-1]:
        file_name = f'./data/processed_data_{start_date}.pickle'
        with open(file_name,"rb") as f :
            processed_data = pickle.load(f)
        print(processed_data.keys())
        print("Finish loading data...")

        # train/test data 
        train = processed_data["raw"]["train"]
        valid = processed_data["raw"]["valid"]
        test = processed_data["raw"]["test"]

        # Revenue data for regression target 
        revenue_train, revenue_valid,revenue_test = processed_data["revenue"]["train"],\
                                                    processed_data["revenue"]["valid"],\
                                                    processed_data["revenue"]["test"]

        # normalize revenue by f(x) = log(x+1)/max(xi)
        norm_revenue_train, norm_revenue_valid, norm_revenue_test = np.log(revenue_train+1), np.log(revenue_valid+1), np.log(revenue_test+1) 
        global_max = max(norm_revenue_train) 
        
        norm_revenue_train = norm_revenue_train/global_max
        norm_revenue_valid = norm_revenue_valid/global_max
        norm_revenue_test = norm_revenue_test/global_max

        # Xgboost data 
        xgb_trainx = processed_data["xgboost_data"]["train_x"]
        xgb_trainy = processed_data["xgboost_data"]["train_y"]
        xgb_validx = processed_data["xgboost_data"]["valid_x"]
        xgb_validy = processed_data["xgboost_data"]["valid_y"]
        xgb_testx = processed_data["xgboost_data"]["test_x"]
        xgb_testy = processed_data["xgboost_data"]["test_y"]
        
#         xgb_trainy_rev = processed_data["revenue"]["train"]
#         xgb_validy_rev = processed_data["revenue"]["valid"]
#         xgb_testy_rev = processed_data["revenue"]["test"]

        # build xgboost classifier
        print("Training xgboost classifier...")
        xgb_clf = XGBClassifier(n_estimators=100, max_depth=4)
        xgb_clf.fit(xgb_trainx,xgb_trainy)
        
        # Predict test set by xgboost classifer
        print("------Xgboost model prediction result------")
        test_pred_XGB = xgb_clf.predict_proba(xgb_testx)[:,1]
        
        print('XGB_cls result', test_pred_XGB[:10])
        writeResult(test_pred_XGB, 'XGB', input_path, output_path)

#         # build xgboost regressor
#         print("Training xgboost regressor...")
#         xgb_reg = XGBRegressor(n_estimators=100, max_depth=4)
#         xgb_reg.fit(xgb_trainx,np.log(xgb_trainy_rev+1))
        
#         # Predict test set by xgboost regressor
#         print("------Xgboost model prediction result------")
#         test_pred_XGB_reg = np.exp(xgb_reg.predict(xgb_testx))-1
        
#         print('XGB_reg result', test_pred_XGB_reg[:1000])
#         writeResult(test_pred_XGB_reg, 'XGB_rev', input_path, output_path)
        

        # get leaf index from xgboost model 
        X_train_leaves = xgb_clf.apply(xgb_trainx)
        X_valid_leaves = xgb_clf.apply(xgb_validx)
        X_test_leaves = xgb_clf.apply(xgb_testx)
        train_rows = X_train_leaves.shape[0]

        # one-hot encoding for leaf index
        xgbenc = OneHotEncoder(categories="auto")
        lr_trainx = xgbenc.fit_transform(X_train_leaves)
        lr_validx = xgbenc.transform(X_valid_leaves)
        lr_testx = xgbenc.transform(X_test_leaves)

        # model 
        print("Training Logistic regression model...")
        lr = LogisticRegression()
        lr.fit(lr_trainx, xgb_trainy)
        test_pred_XGB_LR = lr.predict_proba(lr_testx)[:,1]
        
        print('XGB+LR_cls result', test_pred_XGB_LR[:10])
        writeResult(test_pred_XGB_LR, 'XGB+LR', input_path, output_path)

        # user & item information 
        train_raw_importers = train['IMPORTER.TIN'].values
        train_raw_items = train['TARIFF.CODE'].values
        valid_raw_importers = valid['IMPORTER.TIN'].values
        valid_raw_items = valid['TARIFF.CODE'].values
        test_raw_importers = test['IMPORTER.TIN']
        test_raw_items = test['TARIFF.CODE']

        # we need padding for unseen user or item 
        importer_set = set(train_raw_importers)
        item_set = set(train_raw_items)

        # Remember to +1 for zero padding 
        importer_mapping = {v:i+1 for i,v in enumerate(importer_set)} 
        hs6_mapping = {v:i+1 for i,v in enumerate(item_set)}
        importer_size = len(importer_mapping) + 1
        item_size = len(hs6_mapping) + 1
        train_importers = [importer_mapping[x] for x in train_raw_importers]
        train_items = [hs6_mapping[x] for x in train_raw_items]

        # for test data, we use padding_idx=0 for unseen data
        valid_importers = [importer_mapping.get(x,0) for x in valid_raw_importers]
        valid_items = [hs6_mapping.get(x,0) for x in valid_raw_items]
        test_importers = [importer_mapping.get(x,0) for x in test_raw_importers] # use dic.get(key,deafault) to handle unseen
        test_items = [hs6_mapping.get(x,0) for x in test_raw_items]

        # Preprocess
        train_rows = train.shape[0]
        valid_rows = valid.shape[0] + train_rows
        X_leaves = np.concatenate((X_train_leaves, X_valid_leaves, X_test_leaves), axis=0) # make sure the dimensionality
        transformed_leaves, leaf_num = process_leaf_idx(X_leaves)
        train_leaves, valid_leaves, test_leaves = transformed_leaves[:train_rows],\
                                                  transformed_leaves[train_rows:valid_rows],\
                                                  transformed_leaves[valid_rows:]

        # Convert to torch type
        train_leaves = torch.tensor(train_leaves).long()
        train_user = torch.tensor(train_importers).long()
        train_item = torch.tensor(train_items).long()

        valid_leaves = torch.tensor(valid_leaves).long()
        valid_user = torch.tensor(valid_importers).long()
        valid_item = torch.tensor(valid_items).long()

        test_leaves = torch.tensor(test_leaves).long()
        test_user = torch.tensor(test_importers).long()
        test_item = torch.tensor(test_items).long()

        # cls data
        train_label_cls = torch.tensor(xgb_trainy).float()
        valid_label_cls = torch.tensor(xgb_validy).float()
        test_label_cls = torch.tensor(xgb_testy).float()

        # revenue data 
        train_label_reg = torch.tensor(norm_revenue_train).float()
        valid_label_reg = torch.tensor(norm_revenue_valid).float()
        test_label_reg = torch.tensor(norm_revenue_test).float()

        # create dataloader 

        train_dataset = Data.TensorDataset(train_leaves,train_user,train_item,train_label_cls,train_label_reg)
        valid_dataset = Data.TensorDataset(valid_leaves,valid_user,valid_item,valid_label_cls,valid_label_reg)
        test_dataset = Data.TensorDataset(test_leaves,test_user,test_item,test_label_cls,test_label_reg)



        data4embedding = {"train_dataset":train_dataset,"valid_dataset":valid_dataset,"test_dataset":test_dataset,\
                          "leaf_num":leaf_num,"importer_num":importer_size,"item_size":item_size}

        # save data
        file_to_save = f'./data/torch_data_{start_date}.pickle'
        with open(file_to_save, 'wb') as f:
            pickle.dump(data4embedding, f, protocol=pickle.HIGHEST_PROTOCOL)