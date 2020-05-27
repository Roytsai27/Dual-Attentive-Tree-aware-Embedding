# load packages
import pickle
import numpy as np
import torch
import torch.utils.data as Data
import pandas as pd 
import matplotlib.pyplot as plt 
import sys 
sys.path.append('..')
from model.AttTreeEmbedding import Attention, FusionAttention, DATE

# load preprocessed dataset
with open("../processed_data.pickle","rb") as f:
    processed = pickle.load(f)
test = processed["raw"]["test"]
test = test.reset_index(drop=True) # raw data(test set)

# load trained DATE model
model = torch.load("../saved_models/DATE_0.2146.pkl").module

# load torch dataset 
with open("../torch_data.pickle","rb") as f:
    data = pickle.load(f)
test_dataset = data["test_dataset"]

# create dataloader
batch_size = 128 # the data size per batch
test_loader = Data.DataLoader( # Note that you could iterate dataloader with for loop to obtain batches
    dataset=test_dataset,     
    batch_size=batch_size,      
    shuffle=False,               
)

attention_weight = [] # create a list to store attention weights for each transaction
predicted_illicit_prob = []
predicted_rev = []
device = model.device # the device id of model and data should be consistant

# iterate over test data
for batch_feature,batch_user,batch_item,batch_cls,batch_reg in test_loader:
    model.eval() # this line is to set the model for evaluation stage by stoping some parameters(e.g. dropout)
    
    # convert the data into same device as our model 
    batch_feature,batch_user,batch_item,batch_cls,batch_reg =  \
    batch_feature.to(device), batch_user.to(device), batch_item.to(device),\
     batch_cls.to(device), batch_reg.to(device)
    batch_cls,batch_reg = batch_cls.view(-1,1), batch_reg.view(-1,1)
    
    print(batch_feature.shape, batch_cls.shape, batch_reg.shape)

    # model output
    classification_output, regression_output, _ = model(batch_feature,batch_user,batch_item) # get the prediction output
    
    # convert torch.tensor to numpy array
    classification_output = classification_output.detach().cpu().numpy()
    regression_output = regression_output.detach().cpu().numpy()
    att_w = model.attention_w.detach().cpu().numpy()
    att_w = att_w.reshape(-1,100)
    
    attention_weight.extend(att_w.tolist())
    predicted_illicit_prob.extend(classification_output.ravel().tolist())
    predicted_rev.extend(regression_output.ravel().tolist())

# conver python list to numpy array
attention_weight = np.array(attention_weight)
predicted_illicit_prob = np.array(predicted_illicit_prob)
predicted_rev = np.array(predicted_rev)

plt.bar(range(100),attention_weight.mean(axis=0))
plt.xlabel("Leaf node")
plt.ylabel("Weights")
plt.title("The averaged attentive weights among all transactions")
plt.show()


number_sampled = 1
illicit_sample = test[test.illicit==1].sample(number_sampled,random_state=1) # sample from illicit samples
transaction_id = illicit_sample.index[0] # get the index of sampled data
print("Index of sampled data:",transaction_id)

print("Checking the inspection information of sampled data")
print("-"*52)
for text in illicit_sample['INSPECTION.INFORMATION'].values:
    print(text.strip())
    print("-"*52)
    
# load leaf_index dictionary to project lead-id into XGBoost tree structure
with open("../leaf_index.pickle","rb") as f:
    leaf_index = pickle.load(f)
    
# first, take a look at the prediction probability of the transaction
print("Prediction probability:%.4f" % predicted_illicit_prob[transaction_id])
top_k = 5 # number of cross feature to observe (select top k)
test_leaves = test_dataset.tensors[0].numpy() # the leaf index for all transactions
tree_id = np.argsort(attention_weight[transaction_id])[-top_k:] # obtain which trees have the hightest weights
print("Trees with the highest attention weights:",tree_id)
leaf_id = [test_leaves[transaction_id][i] for i in tree_id] # obtain the leaf-id for the trees
print("The corresponding index of the leaf node:",leaf_id)

# transform the leaf index to the original XGBoost tree structure
# Variable leaf_index is a dictionary with {leaf-id: {tree-id,node-id}}
xgb_cross_feature = [leaf_index[l] for l in leaf_id] 
print("The node id in the original XGBoost model:",xgb_cross_feature)

# read the contents of the file
with open('../xgb_model.txt', 'r') as f:
    txt_model = f.read()

# Note: To maintain the clearness, only print the result for the first tree. Please print the whole text data when analyzing by yourself
print(txt_model[:1015]) 