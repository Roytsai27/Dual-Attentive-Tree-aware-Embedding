import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
import numpy as np 
from torch_multi_head_attention import MultiHeadAttention

class Mish(nn.Module):
    def __init__(self):
        super(Mish,self).__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))


class FusionAttention(nn.Module):
    def __init__(self,dim):
        super(FusionAttention, self).__init__()
        self.attention_matrix = nn.Linear(dim, dim)
        self.project_weight = nn.Linear(dim,1)
    def forward(self, inputs):
        query_project = self.attention_matrix(inputs) # (b,t,d) -> (b,t,d2)
        query_project = F.leaky_relu(query_project)
        project_value = self.project_weight(query_project) # (b,t,h) -> (b,t,1)
        attention_weight = torch.softmax(project_value, dim=1) # Normalize and calculate weights (b,t,1)
        attention_vec = inputs * attention_weight
        attention_vec = torch.sum(attention_vec,dim=1)
        return attention_vec, attention_weight


class Attention(nn.Module):
    def __init__(self,dim,hidden,aggregate="sum"):
        super(Attention, self).__init__()
        self.attention_matrix = nn.Linear(dim, hidden)
        self.project_weight = nn.Linear(hidden*2,hidden)
        self.h = nn.Parameter(torch.rand(hidden,1))
        self.agg_type = aggregate
    def forward(self, query,key): # assume key==value
        dim = query.size(-1)
        batch,time_step = key.size(0) ,key.size(1)
        
        # concate input query and key 
        query = query.view(batch,1,dim)
        query = query.expand(batch,time_step,-1)
        cat_vector = torch.cat((query,key),dim=-1)
        
        # project to single value
        project_vector = self.project_weight(cat_vector) 
        project_vector = torch.relu(project_vector)
        attention_alpha = torch.matmul(project_vector,self.h)
        attention_weight = torch.softmax(attention_alpha, dim=1) # Normalize and calculate weights (b,t,1)
        attention_vec = key * attention_weight
        
        # aggregate leaves
        if self.agg_type == "max":
            attention_vec,_ = torch.max(attention_vec,dim=1)
        elif self.agg_type =="mean":
            attention_vec = torch.mean(attention_vec,dim=1)
        elif self.agg_type =="sum":
            attention_vec = torch.sum(attention_vec,dim=1)
        return attention_vec, attention_weight


class DATE(nn.Module):
    def __init__(self,max_leaf,importer_size,item_size,dim,head_num=4,fusion_type="concat",act="relu",device="cpu",use_self=True,agg_type="sum"):
        super(DATE, self).__init__()
        self.d = dim
        self.device = device
        if act == "relu":
            self.act = nn.LeakyReLU()
        elif act == "mish":
            self.act = Mish() 
        self.fusion_type = fusion_type
        self.use_self = use_self

        # embedding layers 
        self.leaf_embedding = nn.Embedding(max_leaf,dim)
        self.user_embedding = nn.Embedding(importer_size,dim,padding_idx=0)
        self.user_embedding.weight.data[0] = torch.zeros(dim)
        self.item_embedding = nn.Embedding(item_size,dim,padding_idx=0)
        self.item_embedding.weight.data[0] = torch.zeros(dim)

        # attention layer
        self.attention_bolck = Attention(dim,dim,agg_type).to(device)
        self.self_att = MultiHeadAttention(dim,head_num).to(device)
        self.fusion_att = FusionAttention(dim)

        # Hidden & output layer
        self.layer_norm = nn.LayerNorm((100,dim))
        self.fussionlayer = nn.Linear(dim*3,dim)
        self.hidden = nn.Linear(dim,dim)
        self.output_cls_layer = nn.Linear(dim,1)
        self.output_reg_layer = nn.Linear(dim,1)
    
    def forward(self,feature,uid,item_id):
        leaf_vectors = self.leaf_embedding(feature)
        if self.use_self:
            leaf_vectors = self.self_att(leaf_vectors,leaf_vectors,leaf_vectors)
            leaf_vectors = self.layer_norm(leaf_vectors)
        importer_vector = self.user_embedding(uid)
        item_vector = self.item_embedding(item_id)
        query_vector = importer_vector * item_vector
        set_vector, self.attention_w = self.attention_bolck(query_vector,leaf_vectors)
        
        # concat the user, item and tree vectors into a fusion embedding
        if self.fusion_type == "concat":
            fusion = torch.cat((importer_vector, item_vector, set_vector), dim=-1)
            fusion = self.act(self.fussionlayer(fusion))
        elif self.fusion_type == "attention":
            importer_vector, item_vector, set_vector = importer_vector.view(-1,1,self.d), item_vector.view(-1,1,self.d), set_vector.view(-1,1,self.d)
            fusion = torch.cat((importer_vector, item_vector, set_vector), dim=1)
            fusion,_ = self.fusion_att(fusion)
        else:
            raise "Fusion type error"
        hidden = self.hidden(fusion)
        hidden = self.act(hidden)

        # multi-task output 
        classification_output = torch.sigmoid(self.output_cls_layer(hidden))
        regression_output = torch.relu(self.output_reg_layer(hidden))
        return classification_output, regression_output, hidden

    def pred_from_hidden(self,hidden):
        classification_output = torch.sigmoid(self.output_cls_layer(hidden))
        return classification_output 

    def eval_on_batch(self,test_loader): # predict test data using batch 
        final_output = []
        cls_loss = []
        reg_loss = []
        for batch in test_loader:
            batch_feature, batch_user, batch_item, batch_cls, batch_reg = batch
            batch_feature,batch_user,batch_item,batch_cls,batch_reg =  \
            batch_feature.to(self.device), batch_user.to(self.device),\
            batch_item.to(self.device), batch_cls.to(self.device), batch_reg.to(self.device)
            batch_cls,batch_reg = batch_cls.view(-1,1), batch_reg.view(-1,1)
            y_pred_prob, y_pred_rev,_ = self.forward(batch_feature,batch_user,batch_item)

            # compute classification loss
            cls_losses = nn.BCELoss()(y_pred_prob,batch_cls)
            cls_loss.append(cls_losses.item())

            # compute regression loss 
            reg_losses = nn.MSELoss()(y_pred_rev, batch_reg)
            reg_loss.append(reg_losses.item())

            # store predicted probability 
            y_pred = y_pred_prob.detach().cpu().numpy().tolist()
            final_output.extend(y_pred)

        print("CLS loss: %.4f, REG loss: %.4f"% (np.mean(cls_loss), np.mean(reg_loss)) )
        return np.array(final_output).ravel(), np.mean(cls_loss)+ np.mean(reg_loss)
