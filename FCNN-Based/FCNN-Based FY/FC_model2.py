import numpy as np
import torch
from torch import nn
import torch.nn.functional as F




class Squeeze(nn.Module):  # Dimention Module
    @staticmethod
    def forward(input_data: torch.Tensor):
        return input_data.squeeze()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cat_dropout = nn.Dropout(0.1)#0.3 0.2
        # FNN
        self.classifier = nn.Sequential(
            nn.Linear(8,32),
            nn.PReLU(),
            nn.Linear(32, 16),
            nn.PReLU(),
            nn.Linear(16, 8),
            nn.PReLU(),
            nn.Linear(8, 4),
            nn.PReLU(),
            nn.Linear(4, 1)
        )
       
        
    
    def forward(self, data): #batch of data
        p12,p24,p234,p135,p345,p1234,p1345,p45= data
        
        p12 = p12.to(torch.float32)
        p24 = p24.to(torch.float32)
        p234 = p234.to(torch.float32)
        p135 = p135.to(torch.float32)
        p345 = p345.to(torch.float32)
        p1234 = p1234.to(torch.float32)
        p1345 = p1345.to(torch.float32)
        p45 = p45.to(torch.float32)
        
        p12 = p12.unsqueeze(1)
        p24 = p24.unsqueeze(1)
        p234 = p234.unsqueeze(1)
        p135 = p135.unsqueeze(1)
        p345 = p345.unsqueeze(1)
        p1234 = p1234.unsqueeze(1)
        p1345 = p1345.unsqueeze(1)
        p45 = p45.unsqueeze(1)
        
        #print("p12",p12.shape)
        concat = torch.cat([ p12,p24,p234,p135,p345,p1234,p1345,p45], dim=1)
        #print("concat",concat.shape)
        #concat = self.cat_dropout(concat)

        output = self.classifier(concat)
        
        return output
