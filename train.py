import argparse
import os
import pickle
import warnings
import time 
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from model.AttTreeEmbedding import Attention, DATE
from ranger import Ranger
from utils import torch_threshold, fgsm_attack, metrics

warnings.filterwarnings("ignore")

# load torch dataset 
with open("./torch_data.pickle","rb") as f:
    data = pickle.load(f)

# get torch dataset 
train_dataset = data["train_dataset"]
valid_dataset = data["valid_dataset"]
test_dataset = data["test_dataset"]

# create dataloader
batch_size = 256
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
curr_time = str(time.time())
model_name = "DATE"
model_path = "./saved_models/%s%s.pkl" % (model_name,curr_time)

def train(args):
    # get configs
    epochs = args.epoch
    dim = args.dim
    lr = args.lr
    weight_decay = args.l2
    head_num = args.head_num
    device = args.device
    act = args.act
    fusion = args.fusion
    beta = args.beta
    alpha = args.alpha
    use_self = args.use_self
    agg = args.agg
    model = DATE(leaf_num,importer_size,item_size,\
                                    dim,head_num,\
                                    fusion_type=fusion,act=act,device=device,\
                                    use_self=use_self,agg_type=agg,
                                    ).to(device)
    model = nn.DataParallel(model,device_ids=[0,1])

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

            # FGSM attack
            adv_vector = fgsm_attack(model,cls_loss_func,hidden_vector,batch_cls,0.01)
            adv_output = model.module.pred_from_hidden(adv_vector) 

            # calculate loss
            adv_loss_func = nn.BCELoss(weight=batch_cls) 
            adv_loss = beta * adv_loss_func(adv_output,batch_cls) 
            cls_loss = cls_loss_func(classification_output,batch_cls)
            revenue_loss = alpha * reg_loss_func(regression_output, batch_reg)
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
        y_prob, val_loss = model.module.eval_on_batch(valid_loader)
        y_pred_tensor = torch.tensor(y_prob).float().to(device)
        best_threshold, val_score, roc = torch_threshold(y_prob,xgb_validy)
        overall_f1, auc, precisions, recalls, f1s, revenues = metrics(y_prob,xgb_validy,revenue_valid)
        select_best = np.mean(f1s)
        print("Over-all F1:%.4f, AUC:%.4f, F1-top:%.4f" % (overall_f1, auc, select_best) )

        print("Evaluate at epoch %s"%(epoch+1))
        y_prob, val_loss = model.module.eval_on_batch(test_loader)
        y_pred_tensor = torch.tensor(y_prob).float().to(device)
        overall_f1, auc, precisions, recalls, f1s, revenues = metrics(y_prob,xgb_testy,revenue_test,best_thresh=best_threshold)
        print("Over-all F1:%.4f, AUC:%.4f, F1-top:%.4f" %(overall_f1, auc, np.mean(f1s)) )

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

def evaluate(save_model):
    print()
    print("--------Evaluating DATE model---------")
    # create best model
    best_model = torch.load(model_path)
    best_model.eval()

    # get threshold
    y_prob, val_loss = best_model.module.eval_on_batch(valid_loader)
    best_threshold, val_score, roc = torch_threshold(y_prob,xgb_validy)

    # predict test 
    y_prob, val_loss = best_model.module.eval_on_batch(test_loader)
    overall_f1, auc, precisions, recalls, f1s, revenues = metrics(y_prob,xgb_testy,revenue_test,best_threshold)
    best_score = f1s[0]
    os.system("rm %s"%model_path)
    if save_model:
        scroed_name = "./saved_models/%s_%.4f.pkl" % (model_name,overall_f1)
        torch.save(best_model,scroed_name)
    
    return overall_f1, auc, precisions, recalls, f1s, revenues


if __name__ == '__main__':
    # Parse argument
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', 
                        type=str, 
                        default="DATE", 
                        help="Name of model",
                        )
    parser.add_argument('--epoch', 
                        type=int, 
                        default=5, 
                        help="Number of epochs",
                        )
    parser.add_argument('--dim', 
                        type=int, 
                        default=16, 
                        help="Hidden layer dimension",
                        )
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.005, 
                        help="learning rate",
                        )
    parser.add_argument('--l2',
                        type=float,
                        default=0.01,
                        help="l2 reg",
                        )
    parser.add_argument('--alpha',
                        type=float,
                        default=10,
                        help="Regression loss weight",
                        )
    parser.add_argument('--beta', type=float, default=0.00, help="Adversarial loss weight")
    parser.add_argument('--head_num', type=int, default=4, help="Number of heads for self attention")
    parser.add_argument('--use_self', type=int, default=1, help="Wheter to use self attention")
    parser.add_argument('--fusion', type=str, choices=["concat","attention"], default="concat", help="Fusion method for final embedding")
    parser.add_argument('--agg', type=str, choices=["sum","max","mean"], default="sum", help="Aggreate type for leaf embedding")
    parser.add_argument('--act', type=str, choices=["mish","relu"], default="relu", help="Activation function")
    parser.add_argument('--device', type=str, choices=["cuda:0","cuda:1","cpu"], default="cuda:0", help="device name for training")
    parser.add_argument('--output', type=str, default="full.csv", help="Name of output file")
    parser.add_argument('--save', type=int, default=1, help="save model or not")

    # args
    args = parser.parse_args()
    epochs = args.epoch
    dim = args.dim
    lr = args.lr
    weight_decay = args.l2
    head_num = args.head_num
    save_model = args.save
    act = args.act
    fusion = args.fusion
    alpha = args.alpha
    beta = args.beta
    use_self = args.use_self
    agg = args.agg
    print(args)
    train(args)
    overall_f1, auc, precisions, recalls, f1s, revenues = evaluate(save_model)

    # save result
    
    
    output_file =  "./results/" + args.output
    print("Saving result...",output_file)
    with open(output_file, 'a') as ff:
        # print(args,file=ff)
        print()
        print("""Metrics:\nf1:%.4f auc:%.4f\nPr@1:%.4f Pr@2:%.4f Pr@5:%.4f Pr@10:%.4f\nRe@1:%.4f Re@2:%.4f Re@5:%.4f Re@10:%.4f\nRev@1:%.4f Rev@2:%.4f Rev@5:%.4f Rev@10:%.4f""" \
              % (overall_f1, auc,\
                 precisions[0],precisions[1],precisions[2],precisions[3],\
                 recalls[0],recalls[1],recalls[2],recalls[3],\
                 revenues[0],revenues[1],revenues[2],revenues[3]
                 ),
                 ) 
        output_metric = [dim,overall_f1,auc] + precisions + recalls + revenues
        output_metric = list(map(str,output_metric))
        print(" ".join(output_metric),file=ff)
        
        # print("Model:%s epoch:%d dim:%d lr:%f l2:%f beta:%f heads:%d fusion:%s activation:%s"
        #       % (model_name, epochs, dim, lr, weight_decay, beta, head_num,fusion,act),file=ff) 
        # print("""Metrics:\nf1:%.4f auc:%.4f\nPr@1:%.4f Pr@2:%.4f Pr@5:%.4f Pr@10:%.4f\nRe@1:%.4f Re@2:%.4f Re@5:%.4f Re@10:%.4f\nRev@1:%.4f Rev@2:%.4f Rev@5:%.4f Rev@10:%.4f"""  \
        #       % (overall_f1, auc,\
        #          precisions[0],precisions[1],precisions[2],precisions[3],\
        #          recalls[0],recalls[1],recalls[2],recalls[3],\
        #          revenues[0],revenues[1],revenues[2],revenues[3]
        #          ),
        #          file=ff)      
