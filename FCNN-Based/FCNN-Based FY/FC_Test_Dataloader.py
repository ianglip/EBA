from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

#from config import LL_LENGTH, LENGTH, P_FDIM, LL_FEATURE_PATH, ANGLE_FEATURE_PATH, PP_FEATURE_PATH
#from configCapla import ROOT, data_path, Dis_path, TRAIN_SET_LIST,PT_FEATURE_SIZE, max_smi_len, max_seq_len,BATCH_SIZE,TEST_SET_LIST, CHECKPOINT_PATH1 

from FC_Test_Config import (
    TRAIN_SET_LIST,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CHECKPOINT_PATH1,
    TOTAL_EPOCH,)


#**********************************************



class CustomDataset(Dataset):
     
    def __init__(self, pid_path: Path ):
        
        all_pids: list = np.loadtxt(fname=str(pid_path.absolute()), dtype='str').tolist()
        
        print("pid len",len(all_pids))
        self.y_labels=[]
        self.prediction12=[]
        self.prediction24=[]
        self.prediction234=[]
        self.prediction135=[]
        self.prediction345=[]
        self.prediction1234=[]
        self.prediction1345=[]
        self.prediction45=[]
        
        """for i, pid in enumerate(all_pids):
            print(pid)
            """
        with open( CHECKPOINT_PATH1 / "Test2016_290label.lst", 'r') as file:
            for line in file:
                line.strip()
                self.y_labels.append(line)
        
        with open( CHECKPOINT_PATH1 / "PredictedModel1.lst", 'r') as file:
            for line in file:
                line.strip()
                self.prediction12.append(line)
        
        with open( CHECKPOINT_PATH1 / "PredictedModel2.lst", 'r') as file:
            for line in file:
                line.strip()
                self.prediction24.append(line)
        
        with open( CHECKPOINT_PATH1 / "PredictedModel3.lst", 'r') as file:
            for line in file:
                line.strip()
                self.prediction234.append(line)
                
        with open( CHECKPOINT_PATH1 / "PredictedModel4.lst", 'r') as file:
            for line in file:
                line.strip()
                self.prediction135.append(line)
                
        with open(CHECKPOINT_PATH1 / "PredictedModel5.lst", 'r') as file:
             for line in file:
                 line.strip()
                 self.prediction345.append(line)
                
        with open( CHECKPOINT_PATH1 / "PredictedModel6.lst", 'r') as file:
            for line in file:
                line.strip()
                self.prediction1234.append(line)
                
        with open( CHECKPOINT_PATH1 / "PredictedModel7.lst", 'r') as file:
            for line in file:
                line.strip()
                self.prediction1345.append(line) 
        
        with open( CHECKPOINT_PATH1 / "PredictedModel10.lst", 'r') as file:
            for line in file:
                line.strip()
                self.prediction45.append(line)   
           
      
        print(len(self.y_labels))
        print(len(self.prediction12))
        print(len(self.prediction24))
        print(len(self.prediction234))
        print(len(self.prediction135))
        print(len(self.prediction345))
        print(len(self.prediction1234))
        print(len(self.prediction1345))
        print(len(self.prediction45))
       
        
        
        
                    
    def __getitem__(self, idx):
        p12  = np.float32(self.prediction12[idx])
        p24  = np.float32(self.prediction24[idx])
        p234  = np.float32(self.prediction234[idx])
        p135  = np.float32(self.prediction135[idx])
        p345  = np.float32(self.prediction345[idx])
        p1234  = np.float32(self.prediction1234[idx])
        p1345  = np.float32(self.prediction1345[idx])
        p45  = np.float32(self.prediction45[idx])
        y_label= np.float32(self.y_labels[idx])
        return ( p12,p24,p234,p135,p345,p1234,p1345,p45), y_label
    
    """def __getitem__(self):
        p12  = np.float32(self.prediction12)
        p24  = np.float32(self.prediction24)
        p234  = np.float32(self.prediction234)
        p135  = np.float32(self.prediction135)
        p345  = np.float32(self.prediction345)
        p1234  = np.float32(self.prediction1234)
        p1345  = np.float32(self.prediction1345)
        y_label= np.float32(self.y_labels)
        return ( p12,p24,p234,p135,p345,p1234,p1345), y_label"""

    def __len__(self):
        return len(self.y_labels)