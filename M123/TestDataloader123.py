


##############
#1 Ligand Atom
#2 Angles
#3 Protein



import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


from M123.Testconfig123 import (

    ROOT,
    
    TEST_SET_LIST,
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
    ANGLE_FEATURE_PATH,
    AngLENGTH,
   
    )





LL_FEATURE=17

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}

CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)


#pid_path=TRAIN_SET_LIST
#label_path=TRAIN_LABEL_LIST

def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=np.int32)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch] - 1

    return X

            
def map_to_bins( angle, bins_ranges, step, max_val):
#def map_to_bins( angle, bins_ranges, step, max_val):
     L = len(angle)
     B = np.full(L, 0)
     for i in range(L):
         if angle[i] > max_val:
             B[i] = len(bins_ranges)+1
         else:
             for indx_bin, bin_i in enumerate(bins_ranges):
                 if angle[i] > bin_i and angle[i] <= bin_i+step:
                     B[i] = indx_bin+1
     return B       
            
class CustomDataset123(Dataset):
    
    
    #def __init__(self, pid_path: Path, label_path: Path):
    #def __init__(self, pid_path: Path, label_path: Path):
    #def __init__(self, pid_path: Path, max_seq_len,  PT_FEATURE_SIZE ):
    def __init__(self, pid_path: Path ):
        #print("Loading data")
        
       
        all_pids = np.loadtxt(fname=str(TEST_SET_LIST), dtype='str').tolist()
        #all_pids: list = np.loadtxt(fname=str(pid_path.absolute()), dtype='str').tolist()
        #all_pids: list = np.loadtxt(fname=str(pid_path.absolute()), dtype='str').tolist()
        #y_all: list = np.loadtxt(fname=str(label_path.absolute()), dtype='float').tolist()
        
        #*************************************************
        self.y_labels = []
        
        #self.ll_data = np.zeros((len(all_pids), max_smi_len))  # ll_info['smile_features'].shape[0]
        self.ll_data = np.zeros((len(all_pids), LL_LENGTH,LL_FEATURE)) 
        self.pp_data = np.zeros((len(all_pids), max_seq_len, PT_FEATURE_SIZE))
        self.angle_data = np.zeros((len(all_pids), AngLENGTH))
        #self.angle_data = np.zeros((len(all_pids), LENGTH, 40))
        #self.Bin_Dist_data = np.zeros((len(all_pids), DistLENGTH))
        self.y_labels = []
        self.angleBinAll = []
        self.LenangleBin = []
     
        
       
        
        affinity = {}
        affinity_df = pd.read_csv(ROOT / "affinity_data.txt", delimiter='\t')
        for _, row in affinity_df.iterrows():
            affinity[row[0]] = row[1]
        #self.affinity = affinity
        affinity = affinity
        
        
        
        for i, pid in enumerate(all_pids):
            print(pid)
            #self.y_labels.append(affinity[pid])
            self.y_labels.append(affinity[pid])
            
            #ligand_smi=label_smiles(self.smi[pid], max_smi_len)
            #self.ll_data[i]= ligand_smi
            
            with open(f"{LL_FEATURE_PATH.absolute()}/{pid}.pkl", "rb") as dif:
                ll_info = pickle.load(dif)
            #self.ll_data[i] = ll_info['smile_features'][:LL_LENGTH]
            if ll_info.shape[0] > LL_LENGTH :
                self.ll_data[i, :, :] = ll_info[ :LL_LENGTH, :]
            else:
                self.ll_data[i, :ll_info.shape[0], :] = ll_info[:,:]
            
            #self.affinity[pid]
            
            
            _seq_tensor = pd.read_csv(f"{PP_PATH.absolute()}/{pid}.csv" , index_col=0)
            if "idx" in _seq_tensor.columns:_seq_tensor = _seq_tensor.drop(['idx'], axis=1).values[:max_seq_len]
            else:_seq_tensor = _seq_tensor.values[:max_seq_len] 
            seq_tensor = np.zeros((max_seq_len, PT_FEATURE_SIZE))
            seq_tensor[:len(_seq_tensor)] = _seq_tensor
            self.pp_data[i] =  seq_tensor 
            
            
            with open(f"{ANGLE_FEATURE_PATH.absolute()}/{pid}_angle_info.pkl", "rb") as dif:
                self.angle_info = pickle.load(dif)   #pl_angle Dim 40 is ok or pl_angle_1D
                #self.LenangleBin.append(len(self.angle_info['BinAngle1D']))
                if self.angle_info['BinAngle1D'].shape[0] > AngLENGTH:
                    self.angle_data[i, :] = self.angle_info['BinAngle1D'][:AngLENGTH]
                else:
                    self.angle_data[i, :self.angle_info['BinAngle1D'].shape[0]] = self.angle_info['BinAngle1D']
            
           
                    
       
           
        np.savetxt( CHECKPOINT_PATH1 / "Test2016_290label.lst",  self.y_labels, delimiter='\t', fmt='%f')
        #np.savetxt( CHECKPOINT_PATH1 / "lengthDistan.lst",  self.LenDistBin, delimiter='\t', fmt='%f')
        #*************************************************


        
            
        
        
        
    
   

    def __getitem__(self, idx):
        pp = np.float32(self.pp_data[idx, :])
        #al = np.float32(self.Bin_angle_data[idx, :])
        al = np.int32(self.angle_data[idx, :])
        ll = np.float32(self.ll_data[idx, :])
        return (pp, al, ll), self.y_labels[idx]

    def __len__(self):
        return len(self.y_labels)
    
    
    


angle2=my_instance.Bin_angle_data
ll=my_instance.ll_data
pp=my_instance.pp_data"""

 
