
from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader


from Testconfig345_1 import (
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

from TestDataloader345 import CustomDataset345
from model345_1 import Model345


import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt


def forward_pass(model, x, device):
    model.eval()
    for i in range(len(x)):
        x[i] = x[i].to(device)
    return model(x)


def read_file_names(directory_path):
    try:
        # List all files in the directory
        files = os.listdir(directory_path)
        
        # Filter out directories, if needed
        files = [file for file in files if os.path.isfile(os.path.join(directory_path, file))]
        
        return files
    except Exception as e:
        print(f"Error: {e}")
        return None


R=[]
def test():
    
    file_names = read_file_names(CHECKPOINT_PATH )
    
    for i in range(0, len(file_names)):
        weight_file_path = CHECKPOINT_PATH /  file_names[i] #"Angle01_375_0.0053.pt" #"Angle01_353_0.0051.pt" 
        #print(file_names[i])                           #   "Angle01_397_0.0044.pt" #"Angle01_258_0.0052.pt"
        #weight_file_path = CHECKPOINT_PATH / "Angle01_321_0.0054.pt"
        #weight_file_path ='/home/mac/Research2023/1DCNN-Angle/JuliaFeature/models/Angle01_290_0.0067.pt'
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device_name}")
        device = torch.device(device_name)
        model = Model345().to(device)
        #model.load_state_dict(torch.load(weight_file_path, map_location=device))
        model.load_state_dict(torch.load(weight_file_path, map_location=torch.device('cpu')))
        print(f"Model weights loaded in {device_name}")

   
         
        dataloader = DataLoader(
            dataset=CustomDataset345(
               pid_path = TEST_SET_LIST),
            batch_size=1
        )
    
        print("Test started")
        #with open("/home/julia/Research/AffinityPred/Results/YPred_core2016.lst", "w") as file:
        #with open("/home/mac/Research2023/1DCNN-Angle/JuliaFeature/Predicted2016.lst", "w") as file:
        #with open(CHECKPOINT_PATH1 / "BinPredicted2016.lst", "w") as file:
        with open(CHECKPOINT_PATH1 / "Predicted2016.lst", "w") as file:
            with torch.no_grad():
                for x, y in dataloader:
                    prediction = forward_pass(model, x, device)
                    prediction = prediction.cpu().detach().numpy()
                    for value in prediction[0]:        	
                        #print(value)
                        rounded_value = round(float(value), 2)  
                        file.write(f"{rounded_value}\n")
        print("Test done")
        
#****************************************************  
        predicted=[]
        
        try:
            with open(CHECKPOINT_PATH1 / "Predicted2016.lst", 'r') as file:
            #with open('/home/mac/Research2023/JNewAngle/Result/Predicted2016.lst', 'r') as file:
                for line in file:
                    # Process each line as needed
                    line.strip()
                    predicted.append(line)
                      # Prints each line after stripping newline characters
        except FileNotFoundError:
            print(f"Predicted File not found:")
        except Exception as e:
            print(f"An error occurred: {e}")
                
        #print(predicted)


        Actual=[]
        try:
            #with open(CHECKPOINT_PATH1 / "Y_core2016.lst", 'r') as file:
            with open(CHECKPOINT_PATH1 / "Test2016_290label.lst", 'r') as file:
            #with open('/home/mac/Research2023/JNewAngle/Result/Y_core2016.lst', 'r') as file:
                for line in file:
                    # Process each line as needed
                    line.strip()
                    Actual.append(line)
                      # Prints each line after stripping newline characters
        except FileNotFoundError:
            print(f"Test label File not found:")
        except Exception as e:
            print(f"An error occurred: {e}")
                
        #print(Actual)



        actual_score =np.array(Actual, dtype='float32') 
        predicted_score = np.array(predicted, dtype='float32').round(2)




        #r = stats.pearsonr(test_y1, y_pred2)[0]
        ###################################


        #concordance index
        
        #########################################


        #print("Actual",actual_score)
        #print("\nPredicted",predicted_score)
        print("\nlength",len(actual_score), len(predicted_score))
        ## Pearson correlation coefficient (R)
        r = stats.pearsonr(actual_score, predicted_score)[0]
        R.append(np.around( r, 3))
       
        #print(actual_score) 
        #print(predicted_score) 

        ## Mean square error(MSE), Root mean squared error (RMSE)
        mse = mean_squared_error(actual_score, predicted_score)
        rmse = sqrt(mse)

         

        ## Mean absolute error (MAE)
        mae = mean_absolute_error(actual_score, predicted_score)

         

        ## standard deviation (SD) from https://github.com/bioinfocqupt/Sfcnn/blob/main/scripts/evaluation.ipynb
        sd = np.sqrt(sum((np.array(actual_score)-np.array(predicted_score))**2)/(len(actual_score)-1))

        CI=c_index(actual_score, predicted_score) 

        print("Pearson correlation coefficient = ", np.around( r, 3))
        print("Root mean squared error = ", np.around( rmse, 3))
        print("Mean absolute error = ", np.around( mae, 3))
        print("Mean square error = ", np.around( mse, 3))
        print("standard deviation = ", np.around( sd, 3))
        print("Concordance Index  = " ,np.around( CI, 3))
        print("Model Weight  =", file_names[i])

        #with open('/home/jiffriya/Dataset2016/MSfCNN/TrainingData/Figure2013/EMetrics2013.txt', 'a') as f:
        #with open('/home/mac/Dataset/myprogram/Newscfnn/AngleGenerate/EMetrics2013.txt', 'a') as f:
        with open(CHECKPOINT_PATH1 / "Evaluate.txt", 'a') as f:
            f.write("Pearson correlation coefficient = %f "%(np.around( r, 3)))
            f.write("\nRoot mean squared error = %f" %np.around( rmse, 3))
            f.write("\nMean absolute error = %f" %np.around( mae, 3))
            f.write("\nMean square error = %f" %np.around( mse, 3))
            f.write("\nstandard deviation = %f" % np.around( sd, 3))
            f.write("\nConcordance Index  = %f" % np.around( CI, 3))
            f.write("\nModel Weight  = %s\n\n" %file_names[i])
            #f.write("\nTest Loss%f " % loss1)
            #f.write("Test MAE = %f " % mae1)
    
    #print(R)
    R_max=max(R)
    R_max_index = R.index(max(R))
    print("Max R", max(R))
    print("Max R index", R_max_index)
    print("Weight ", file_names[R_max_index])
    with open(CHECKPOINT_PATH1 / "Evaluate.txt", 'a') as f:
        f.write("Max R = %f "%(np.around( R_max, 3)))
        f.write("\nMax R index= %d "%R_max_index)
        f.write("\nWeight Name= %s "%file_names[R_max_index])
        
    
      
    
    
def c_index(y_true, y_pred):
    
    #y_true=test_y1 
   
    summ = 0
    pair = 0

    for i in range(1, len(y_true)):
        for j in range(0, i):
            pair += 1
            if y_true[i] > y_true[j]:
                summ += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            elif y_true[i] < y_true[j]:
                summ += 1 * (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            else:
                pair -= 1
    if pair != 0:
        result= summ / pair
    else:
        result=0
    return result


if __name__ == "__main__":
    test()

