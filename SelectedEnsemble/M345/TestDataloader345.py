


##############

#3 Protein
#4 Ligand SMILES
#5 Pocket

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

#from config import LL_LENGTH, LENGTH, P_FDIM, LL_FEATURE_PATH, ANGLE_FEATURE_PATH, PP_FEATURE_PATH
#from configCapla import ROOT, data_path, Dis_path, TRAIN_SET_LIST,PT_FEATURE_SIZE, max_smi_len, max_seq_len,BATCH_SIZE,TEST_SET_LIST, CHECKPOINT_PATH1 

from M345.Testconfig345_1 import (
#from Testconfig345_1 import (
    ROOT,
    PK_PATH,
    PK_FEATURE_SIZE,
    max_pkt_len,
    #TRAIN_SET_LIST,
    TEST_SET_LIST,
    BATCH_SIZE,
    PT_FEATURE_SIZE,
    max_seq_len,
    max_smi_len,
    PP_PATH,
    CHECKPOINT_PATH,
    CHECKPOINT_PATH1,
    TOTAL_EPOCH,
    #LL_FEATURE_PATH,
    #LL_LENGTH,
    #ANGLE_FEATURE_PATH,
    #AngLENGTH,
    SMI_PATH
    
    )

"""LL_LENGTH = 150
LENGTH = 1000
P_FDIM = 58


#BATCH_SIZE = 16
#TOTAL_EPOCH = 200


ROOT = Path("/home/mac/Research2023/1DCNN-Angle/JuliaFeature")
DATASET_ROOT = ROOT / "Features"
LL_FEATURE_PATH = DATASET_ROOT / "LL_features"
ANGLE_FEATURE_PATH = DATASET_ROOT / "PLangle"
PP_FEATURE_PATH = DATASET_ROOT / "PP_featuresNew"

TRAIN_SET_LIST = DATASET_ROOT /"TrainPDBID.lst" #"train_set.lst"
TRAIN_LABEL_LIST =DATASET_ROOT / "Trainlabel.lst"

TEST_SET_LIST = ROOT / "core2016.lst"
TEST_LABEL_LIST = ROOT / ""  """

"""ROOT = Path("/home/mac/Griffith Research/CAPlaData/TrainAndValidation/AlltrainingValidation")
data_path=Path("/home/mac/Griffith Research/CAPlaData/TrainAndValidation")
Angle_path= Path("/home/mac/Research2023/1DCNN-Angle/JuliaFeature/Features/PLangle")
TRAIN_SET_LIST=data_path/"TrainPDBID.lst" """

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}


"""CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64, ":": 65, "~": 66}"""


CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)

LL_FEATURE=17
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
            
class CustomDataset345(Dataset):
    
    
    #def __init__(self, pid_path: Path, label_path: Path):
    #def __init__(self, pid_path: Path, label_path: Path):
    def __init__(self, pid_path: Path ):
        #print("Loading data")
        
       
        all_pids = np.loadtxt(fname=str(TEST_SET_LIST), dtype='str').tolist()
        #all_pids: list = np.loadtxt(fname=str(pid_path.absolute()), dtype='str').tolist()
        
        
        #all_pids: list = np.loadtxt(fname=str(pid_path.absolute()), dtype='str').tolist()
        #y_all: list = np.loadtxt(fname=str(label_path.absolute()), dtype='float').tolist()
        
        #*************************************************
        self.max_pkt_len =63
        
        self.ll_data1 = np.zeros((len(all_pids), max_smi_len))  # ll_info['smile_features'].shape[0]
        #self.ll_data2 = np.zeros((len(all_pids), LL_LENGTH,LL_FEATURE)) 
        #self.angle_data = np.zeros((len(all_pids), AngLENGTH))
        self.pk_data = np.zeros((len(all_pids), max_pkt_len, PK_FEATURE_SIZE))
        self.pp_data = np.zeros((len(all_pids), max_seq_len, PT_FEATURE_SIZE))
        
        self.y_labels = []
       
       
     
        
        #ligands_df = pd.read_csv( ROOT / "smi.csv")
        #ligands_df = pd.read_csv( ROOT / "training_Validation_smi.txt", delimiter='\t')
        ligands_df = pd.read_csv( SMI_PATH, delimiter='\t')
        ligands = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
        self.smi = ligands 
        #self.max_smi_len = max_smi_len
        #smi = ligands
        
        
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
            
            ligand_smi=label_smiles(self.smi[pid], max_smi_len)
            self.ll_data1[i]= ligand_smi 
            
       
                    
            _pkt_tensor = pd.read_csv(f"{PK_PATH.absolute()}/{pid}.csv" , index_col=0)
            #_pkt_tensor = pd.read_csv(ROOT / PK_PATH.absolute() , index_col=0)
            if "idx" in _pkt_tensor.columns: _pkt_tensor = _pkt_tensor.drop(['idx'], axis=1).values[:self.max_pkt_len] 
            else:_pkt_tensor = _pkt_tensor.values[:self.max_pkt_len] 
            
            pkt_tensor = np.zeros((self.max_pkt_len, PK_FEATURE_SIZE))
            pkt_tensor[:len(_pkt_tensor)] = _pkt_tensor
            self.pk_data[i] =  pkt_tensor        
                    
            _seq_tensor = pd.read_csv(f"{PP_PATH.absolute()}/{pid}.csv" , index_col=0)
            #_seq_tensor = pd.read_csv(ROOT / PP_PATH.absolute() , index_col=0)
            if "idx" in _seq_tensor.columns:_seq_tensor = _seq_tensor.drop(['idx'], axis=1).values[:max_seq_len]
            else:_seq_tensor = _seq_tensor.values[:max_seq_len] 
            seq_tensor = np.zeros((max_seq_len, PT_FEATURE_SIZE))
            seq_tensor[:len(_seq_tensor)] = _seq_tensor
            self.pp_data[i] =  seq_tensor 
            

            
            
            
           
        np.savetxt( CHECKPOINT_PATH1 / "Test2016_290label.lst",  self.y_labels, delimiter='\t', fmt='%f')
        #np.savetxt( CHECKPOINT_PATH1 / "lengthDistan.lst",  self.LenDistBin, delimiter='\t', fmt='%f')
        #*************************************************


        
            
        
        
        
    
   

    def __getitem__(self, idx):
      
        
        
        ll1 = np.int32(self.ll_data1[idx, :])
        #ll2 = np.float32(self.ll_data2[idx, :])
        #al = np.int32(self.angle_data[idx, :])
        pk = np.float32(self.pk_data[idx, :])
        pp = np.float32(self.pp_data[idx, :])
        return ( ll1,pk,pp), self.y_labels[idx]

    def __len__(self):
        return len(self.y_labels)
    
    
    
"""from torch.utils.data import DataLoader
dataloader = DataLoader(
    dataset=CustomDataset(pid_path=TRAIN_SET_LIST, max_seq_len=1000, max_smi_len=150, PT_FEATURE_SIZE=40),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

my_instance = CustomDataset(TRAIN_SET_LIST,max_seq_len=1000, max_smi_len=150, PT_FEATURE_SIZE=40)


angle2=my_instance.Bin_angle_data
ll=my_instance.ll_data
pp=my_instance.pp_data"""

 
