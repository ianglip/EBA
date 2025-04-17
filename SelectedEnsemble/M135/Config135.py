from pathlib import Path

BATCH_SIZE =32#64#32#4#8#16
TOTAL_EPOCH = 200
#AngLENGTH=1800
#max_smi_len=150
max_pkt_len=63
PK_FEATURE_SIZE=40
LL_LENGTH = 50
#AngLENGTH=1800
PT_FEATURE_SIZE=40
max_seq_len=1000


ROOT = Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData")
PP_PATH = ROOT / "AlltrainingValidation"
LL_FEATURE_PATH = ROOT / "Ligand_Atom_Feature"
PK_PATH = ROOT / "PocketTraingValidation"
#ANGLE_FEATURE_PATH =Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData/Angle_November/Angle_CNOS")


CHECKPOINT_PATH = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/M135/models")

CHECKPOINT_PATH1 = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/M135/Result")
#CHECKPOINT_PATH_2 = Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/code/CV5/models2")

TRAIN_SET_LIST= ROOT / "train_set.lst" 


############# CASF 2016 290 #################  01

"""ROOT = Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData")
PP_PATH = ROOT/ "Testset"/ "Test2016_290"/"global" 
#SMI_PATH= Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData/Testset/Test2016_290_smi.txt")
LL_FEATURE_PATH = ROOT / "Ligand_Atom_Feature"
#ANGLE_FEATURE_PATH = ROOT / "Angle_November"/ "Angle_CNOS"
PK_PATH= ROOT/ "Testset"/ "Test2016_290"/ "pocket"

CHECKPOINT_PATH = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/M135/models")

CHECKPOINT_PATH1 = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/M135/Result")

TEST_SET_LIST = ROOT/"Testset"/ "PID_Test2016_290.lst"  """