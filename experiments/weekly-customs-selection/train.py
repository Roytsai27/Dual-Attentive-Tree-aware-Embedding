import argparse
import os
import pickle
import warnings
from collections import defaultdict
import random, string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from model.AttTreeEmbedding import Attention, AttentionalTreeEmbeddig
from ranger import Ranger
from utils import torch_threshold, fgsm_attack, metrics, writeResult

warnings.filterwarnings("ignore")


def train(args):
    # get configs
    epochs = args.epoch
    dim = args.dim
    lr = args.lr
    weight_decay = args.l2
    head_num = args.head_num
    aggregate="sum"
    device = args.device
    act = args.act
    fusion = args.fusion
    beta = args.beta
    model = AttentionalTreeEmbeddig(leaf_num,importer_size,item_size,\
                                    dim,head_num,\
                                    fusion_type=fusion,act=act,device=device,
                                    ).to(device)

    # initialize parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # optimizer & loss 
    optimizer = Ranger(model.parameters(), weight_decay=weight_decay,lr=lr)
    cls_loss_func = nn.BCELoss()
    reg_loss_func = nn.MSELoss()

    # save best model
    global_best_score = 0
    model_state = None

    # early stop settings 
    stop_rounds = 3
    no_improvement = 0
    current_score = None 

    for epoch in range(epochs):
        for step, (batch_feature,batch_user,batch_item,batch_cls,batch_reg) in enumerate(train_loader):
            model.train() # prep to train model
            batch_feature,batch_user,batch_item,batch_cls,batch_reg =  \
            batch_feature.to(device), batch_user.to(device), batch_item.to(device),\
             batch_cls.to(device), batch_reg.to(device)
            batch_cls,batch_reg = batch_cls.view(-1,1), batch_reg.view(-1,1)

            # model output
            classification_output, regression_output, hidden_vector = model(batch_feature,batch_user,batch_item)

            # FGM attack
            adv_vector = fgsm_attack(model,cls_loss_func,hidden_vector,batch_cls,0.01)
            adv_output = model.pred_from_hidden(adv_vector) 

            # calculate loss
            adv_loss = beta * cls_loss_func(adv_output,batch_cls)
            cls_loss = cls_loss_func(classification_output,batch_cls)
            revenue_loss = 10 * reg_loss_func(regression_output, batch_reg)
            loss = cls_loss + revenue_loss + adv_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step+1) % 1000 ==0:  
                print("CLS loss:%.4f, REG loss:%.4f, ADV loss:%.4f, Loss:%.4f"\
                %(cls_loss.item(),revenue_loss.item(),adv_loss.item(),loss.item()))
                
        # evaluate 
        model.eval()
        print("Validate at epoch %s"%(epoch+1))
        y_prob, y_rev, val_loss = model.eval_on_batch(valid_loader)
        y_pred_tensor = torch.tensor(y_prob).float().to(device)
        best_threshold, val_score, roc = torch_threshold(y_prob,xgb_validy)
        overall_f1, auc, precisions, recalls, f1s, revenues = metrics(y_prob,xgb_validy,revenue_valid)
        select_best = np.mean(f1s)
        print("Over-all F1:%.4f, AUC:%.4f, F1-top:%.4f" % (overall_f1, auc, select_best) )

        print("Evaluate at epoch %s"%(epoch+1))
        y_prob, y_rev, val_loss = model.eval_on_batch(test_loader)
        y_pred_tensor = torch.tensor(y_prob).float().to(device)

        # save best model 
        if select_best > global_best_score:
            global_best_score = select_best
            torch.save(model,model_path)
        
         # early stopping 
        if current_score == None:
            current_score = select_best
            continue
        if select_best < current_score:
            current_score = select_best
            no_improvement += 1
        if no_improvement >= stop_rounds:
            print("Early stopping...")
            break 
        if select_best > current_score:
            no_improvement = 0
            current_score = None

def evaluate(save_model, exp_id):
    print()
    print("--------Evaluating DATE model---------")
    # create best model
    best_model = torch.load(model_path)
    best_model.eval()
    y_prob, y_rev, test_loss = best_model.eval_on_batch(test_loader)

    return y_prob, y_rev

    


if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',
                        type=int,
                        default=5,
                        help="Number of epochs")
    parser.add_argument('--dim',
                        type=int,
                        default=16,
                        help="Hidden layer dimension")
    parser.add_argument('--lr',
                        type=float,
                        default=0.005,
                        help="learning rate")
    parser.add_argument('--l2',
                        type=float,
                        default=0.01,
                        help="l2 reg")
    parser.add_argument('--beta',
                        type=float,
                        default=0.00,
                        help="Adversarial loss weight")
    parser.add_argument('--head_num',
                        type=int,
                        default=4,
                        help="Number of heads for self attention")
    parser.add_argument('--fusion',
                        type=str,
                        choices=["concat","attention"],
                        default="concat",
                        help="Fusion method for user/item/leaf embedding")
    parser.add_argument('--act',
                        type=str,
                        choices=["mish","relu"],
                        default="relu",
                        help="Activation function")
    parser.add_argument('--device',
                        type=str,
                        choices=["cuda:0","cuda:1","cuda:2","cuda:3","cuda:4","cuda:5","cpu"],
                        default="cuda:1",
                        help="device name for training")
    parser.add_argument('--output',
                        type=str,
                        default="clsrev.csv",
                        help="Name of output file")
    parser.add_argument('--save',
                        type=int,
                        default=0,
                        help="save model or not")
    parser.add_argument('--date',
                        type=str,
                        default='16-01-01',
                        help="training staring date")
    parser.add_argument('--week',
                        type=int,
                        default=2,
                        help="week number: e.g., --week 2")
    
    args = parser.parse_args()
    epochs = args.epoch
    starting_date = args.date                    
    dim = args.dim
    lr = args.lr
    weight_decay = args.l2
    head_num = args.head_num
    save_model = args.save
    act = args.act
    fusion = args.fusion
    beta = args.beta
    print(args)
    input_path = './data/Nigeria_pilot_weekly/week'+str(args.week)+'_ano.csv'
    output_path = './data/Nigeria_pilot_weekly/week'+str(args.week)+'_ano_result.csv'
    
    
    # load torch dataset 
    data_path = f'./data/torch_data_{starting_date}.pickle'
    print(data_path)
    with open(data_path,"rb") as f:
        data = pickle.load(f)

    # get torch dataset 
    train_dataset = data["train_dataset"]
    valid_dataset = data["valid_dataset"]
    test_dataset = data["test_dataset"]

    # create dataloader
    batch_size = 128
    train_loader = Data.DataLoader(
        dataset=train_dataset,     
        batch_size=batch_size,      
        shuffle=True,               
    )
    valid_loader = Data.DataLoader(
        dataset=valid_dataset,     
        batch_size=batch_size,      
        shuffle=False,               
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,     
        batch_size=batch_size,      
        shuffle=False,               
    )

    # parameters for model 
    leaf_num = data["leaf_num"]
    importer_size = data["importer_num"]
    item_size = data["item_size"]

    # global variables
    xgb_validy = valid_loader.dataset.tensors[-2].detach().numpy()
    xgb_testy = test_loader.dataset.tensors[-2].detach().numpy()
    revenue_valid = valid_loader.dataset.tensors[-1].detach().numpy()
    revenue_test = test_loader.dataset.tensors[-1].detach().numpy()

    # model information
    
    exp_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    model_path = f'./saved_models/DATE_{starting_date}_{exp_id}.pkl'
    train(args)
    y_pred_prob, y_pred_rev = evaluate(save_model, exp_id)
    
    
## Try to retrieve RAISED_AMT_TAX, but due to value imbalance, does not look very nice 
#     prep_file_name = f'./data/processed_data_{starting_date}.pickle'
#     with open(prep_file_name,"rb") as f2 :
#         processed_data = pickle.load(f2)
#     revenue_train = processed_data["revenue"]["train"]
#     y_pred_rev *= np.log(max(revenue_train)+1) 
#     y_pred_rev = np.exp(y_pred_rev)-1

    print('DATE_CLS', y_pred_prob[:10])
    print('DATE_REV', y_pred_rev[:10])
    
    
        
        
    writeResult(y_pred_prob, 'DATE_CLS', input_path, output_path)
    writeResult(y_pred_rev, 'DATE_REV', input_path, output_path)
    
    
    
