##############
#1LigandAtom
#3Protein
#5 Pocket

import time
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

#import os
#print(os.getcwd())


from Config135 import (
    ROOT,
    PK_PATH,
    PK_FEATURE_SIZE,
    max_pkt_len,
    TRAIN_SET_LIST,
    BATCH_SIZE,
    PT_FEATURE_SIZE,
    max_seq_len,
    #max_smi_len,
    PP_PATH,
    CHECKPOINT_PATH,
    CHECKPOINT_PATH1,
    TOTAL_EPOCH,
    LL_FEATURE_PATH,
    LL_LENGTH,
    #ANGLE_FEATURE_PATH,
    #AngLENGTH
    )
"""from dataloader3_1 import CustomDataset
#from binmodel import Model
from binmodelS2_2 import Model"""

#from configCapla import ROOT, data_path, Dis_path, TRAIN_SET_LIST, PT_FEATURE_SIZE, max_smi_len, max_seq_len, BATCH_SIZE,TOTAL_EPOCH, CHECKPOINT_PATH,CHECKPOINT_PATH1
from Dataloader135 import CustomDataset
from model135 import Model

#from utils import LogCoshLoss
#from MSEloss import Ang_loss
#from myloss import Ang_loss

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


#from model import *

#from sklearn.model_selection import train_test_split

Val_losses = []
Train_losses=[]
metrics=[]
def train1():
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device_name}")
    device = torch.device(device_name)
    model = Model().to(device)
    print(f"Model instance created and loaded in {device_name}")

    print("Starting training process")
    
   

    #X_train, X_val, Y_train, Y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    
    
    """dataset=CustomDataset(
       pid_path=TRAIN_SET_LIST,
       label_path=TRAIN_LABEL_LIST,
    )"""
    
    dataset=CustomDataset(
     
        pid_path=TRAIN_SET_LIST )
    """dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,)"""
    
    
    """dataloader = DataLoader(
        dataset=CustomDataset(
            pid_path=TRAIN_SET_LIST,
            label_path=TRAIN_LABEL_LIST,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )"""
    
    
    
    print("length dataloader",len(dataset))
    train_indices, val_indices=split_indices(len(dataset),val_pct=0.2)
    print(" train_indices, val_indices", len(train_indices), len(val_indices))
    
    train_sampler=SubsetRandomSampler(train_indices)
    
    train_dataloader = DataLoader(
         dataset,
         batch_size=BATCH_SIZE,
         sampler=train_sampler)
    
    valid_sampler=SubsetRandomSampler(val_indices)
    valid_dataloader = DataLoader(
         dataset,
         batch_size=BATCH_SIZE,
         sampler=valid_sampler)
    
    print(" valid_dataloader", valid_dataloader)
    
    """for x,y in  valid_dataloader:
        abc=x
        a1=len(x)
        abc1=y
        print(x,y)
        #print(x.view(x.size(0)))
        #print(x.size(0))
        break"""
    
    
    learning_rate = 0.001
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    #optimizer = optim.AdamW(model.parameters())
    #loss_function = LogCoshLoss()
    loss_function = nn.L1Loss()
    #loss_function = Ang_loss()

    best_epoch = -1
    best_loss = 100000000
    

   
    
    for n_epoch in range(1, TOTAL_EPOCH + 1):
        print(f"Starting epoch {n_epoch} / {TOTAL_EPOCH}")
       
        start_time = time.time()
        
        trainloss=train_one_epoch(model,train_dataloader, optimizer,  loss_function ,device)
        valloss=validate_one_epoch(model, valid_dataloader,loss_function,device)
        Train_losses.append( trainloss)
        Val_losses.append(valloss)
        
        print(f"Epoch time: {time.time() - start_time}")
        
        
        if len(Val_losses) > 1 and valloss < min(Val_losses[:-1]):
            # save trained model
            torch.save(model.state_dict(), CHECKPOINT_PATH / f"Angle01_{n_epoch}_{round(valloss, 4)}.pt")
            #torch.save(model.state_dict(), CHECKPOINT_PATH /'%5.3f.pt' % valloss)
        
            
        
       
        print('Epoch [{}/{}] Val_loss:{:.4f}'.format(n_epoch, TOTAL_EPOCH + 1, valloss ))
        
    np.savetxt(CHECKPOINT_PATH1 / "trainloss.lst" , np.array(Train_losses),fmt='%f')
    np.savetxt(CHECKPOINT_PATH1 / "valloss.lst", np.array(Val_losses),fmt='%f')
        
  
            
    #loss_plot(Val_losses,Train_losses)
    
   


def split_indices(n,val_pct):
    n_val=int(val_pct*n)
    idxs=np.random.permutation(n)
    
    return idxs[n_val:],idxs[:n_val]
#****************************
def train_one_step(model,x,y, device, optimizer, loss_function):
    optimizer.zero_grad()
    for i in range(len(x)):
        x[i] = x[i].to(device)
    
        
    y = y.to(device)
    
    
    output = model(x)
    loss = loss_function(output.view(-1), y.view(-1))
    
    loss.backward()
    optimizer.step()
    return loss #return batch loss

def train_one_epoch(model,train_dataloader, optimizer, loss_function ,device):
    model.train()
    train_loss=0
    for x,y in train_dataloader:
        #print(len(x))
        #break;
        loss=train_one_step(model,x,y, device, optimizer, loss_function)# batch loss
        #scheduler.step()
        train_loss += loss.item() / len(y)# avegrage of loss of each batch 
    print( " train_loss", train_loss)
    return train_loss #total of loss for one epoch
    

def validate_one_step(model,x,y,device, loss_function):
    for i in range(len(x)):
        x[i] = x[i].to(device) 
    y = y.to(device)
    
    output = model(x)
    loss = loss_function(output.view(-1), y.view(-1))
   
    return loss

    
def validate_one_epoch(model, valid_dataloader,loss_function,device):
    model.eval()
    valid_loss=0
    
    for x,y in valid_dataloader:
        with torch.no_grad():
            loss=validate_one_step(model, x,y,device, loss_function)
        valid_loss+=loss.item()/len(y)
        return valid_loss

def loss_plot(Val_losses,Train_losses):
    plt.plot(range(1, TOTAL_EPOCH + 1), Val_losses,label='Validation Loss')
    plt.plot(range(1, TOTAL_EPOCH + 1),   Train_losses,label='Training Loss')
    plt.title(' Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    train1()