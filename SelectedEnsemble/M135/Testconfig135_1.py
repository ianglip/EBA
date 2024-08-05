#############
#1LigandAtom
#3Protein
#5 Pocket

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

##############Training PDBBind2020#######################

ROOT = Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData/PDBBind2020Train")
PP_PATH = ROOT / "Protein"
LL_FEATURE_PATH = ROOT / "Ligand_Atom"
PK_PATH = ROOT / "Pocket"
#SMI_PATH=ROOT / "training_Validation_smi.txt"
#ANGLE_FEATURE_PATH = ROOT / "Angle"

CHECKPOINT_PATH = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/Oldmodels/M135/models2")

CHECKPOINT_PATH1 = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/Oldmodels/M135/Result")

#TEST_SET_LIST=ROOT / "training_PID.lst" 
TEST_SET_LIST=ROOT / "validation_PID.lst" 



############# CASF 2016 290 #################  01

"""ROOT = Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData")
PP_PATH = ROOT/ "Testset"/ "Test2016_290"/"global" 
#SMI_PATH= Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData/Testset/Test2016_290_smi.txt")
LL_FEATURE_PATH = ROOT / "Ligand_Atom_Feature"
#ANGLE_FEATURE_PATH = ROOT / "Angle_November"/ "Angle_CNOS"
PK_PATH= ROOT/ "Testset"/ "Test2016_290"/ "pocket"

CHECKPOINT_PATH = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/Oldmodels/M135/models2")

CHECKPOINT_PATH1 = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/Oldmodels/M135/Result")

TEST_SET_LIST = ROOT/"Testset"/ "PID_Test2016_290.lst" """

############# CASF 2013 195 #################  02


"""ROOT = Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData")
PP_PATH = ROOT/ "Testset"/ "Test2013_195"/"global" 
#SMI_PATH= Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData/Testset/Test2013_195_smi.txt")
LL_FEATURE_PATH = ROOT / "Ligand_Atom_Feature"
#ANGLE_FEATURE_PATH = ROOT / "Angle_November"/ "Angle_CNOS"
PK_PATH= ROOT/ "Testset"/ "Test2013_195"/ "pocket"

CHECKPOINT_PATH = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/Oldmodels/M135/models2")

CHECKPOINT_PATH1 = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/Oldmodels/M135/Result")

TEST_SET_LIST = ROOT/"Testset"/ "PID_Test2013_195.lst" """


##############################################


#############  CSAR-HiQ_51 #################  03

"""ROOT = Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData")
PP_PATH = ROOT/ "Testset"/ "CSAR-HiQ_51"/"global" 
#SMI_PATH= Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData/Testset/CSAR-HiQ_51_smi.txt")
LL_FEATURE_PATH =Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData/Testset/HIQ_Atom_AngleFeature/Feature/LigandAtomFeature/CSAR-HIQ")
#ANGLE_FEATURE_PATH = ROOT /"Testset"/"HIQ_Atom_AngleFeature"/"Feature"/"AngleFeatureCNOS"/"CSAR-HIQ_AngleCNOS"
PK_PATH= ROOT/ "Testset"/ "CSAR-HiQ_51"/ "pocket"

CHECKPOINT_PATH = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/Oldmodels/M135/models2")

CHECKPOINT_PATH1 = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/Oldmodels/M135/Result")

TEST_SET_LIST = ROOT/"Testset"/ "PID_CSAR-HiQ_51.lst" """


################################################################

#############  CSAR-HiQ_36 #################  04
"""ROOT = Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData")
PP_PATH = ROOT/ "Testset"/ "CSAR-HiQ_36"/"global" 
#SMI_PATH= Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData/Testset/CSAR-HiQ_36_smi.txt")
LL_FEATURE_PATH =Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData/Testset/HIQ_Atom_AngleFeature/Feature/LigandAtomFeature/CSAR-HIQ")
#ANGLE_FEATURE_PATH = ROOT /"Testset"/"HIQ_Atom_AngleFeature"/"Feature"/"AngleFeatureCNOS"/"CSAR-HIQ_AngleCNOS"
PK_PATH= ROOT/ "Testset"/ "CSAR-HiQ_36"/ "pocket"

CHECKPOINT_PATH = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/Oldmodels/M135/models2")

CHECKPOINT_PATH1 = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/Oldmodels/M135/Result")

TEST_SET_LIST = ROOT/"Testset"/ "PID_CSAR-HiQ_36.lst" """
############### PDBBind 2020 ##############################
"""ROOT = Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData/PDBBind2020TestSet")

PP_PATH= Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData/PDBBind2020TestSet/protein")
PK_PATH=Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData/PDBBind2020TestSet/Pocket")
#SMI_PATH =Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData/PDBBind2020TestSet/ligand_SMILES/test.txt")
LL_FEATURE_PATH =Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData/PDBBind2020TestSet/Ligand_Atom")
#ANGLE_FEATURE_PATH =Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData/PDBBind2020TestSet/Angle")

CHECKPOINT_PATH = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/Oldmodels/M135/models2")

CHECKPOINT_PATH1 = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/Oldmodels/M135/Result")



TEST_SET_LIST = ROOT/"PDBBind2020TestPID.lst" """

############### DUD ##############################
"""ROOT = Path("/export/home/s5274138/CombinedDTIMay2023/1DCNNAngle/DiheadTrain/Capla_Angle/CaplaData/akt1/pocketRes")

PP_PATH= ROOT/"protein.csv"
PK_PATH=ROOT/"pocket.csv"
#SMI_PATH =ROOT/"DUD_smi.txt"
LL_FEATURE_PATH =ROOT/"Ligand_Atom"
#ANGLE_FEATURE_PATH =ROOT/"Angle"

CHECKPOINT_PATH = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/Oldmodels/M135/models2")

CHECKPOINT_PATH1 = Path("/export/home/s5274138/CaplaEnsemple/NewTrain/Oldmodels/M135/Result")

#TEST_SET_LIST = ROOT/"decoys_ids.lst"
TEST_SET_LIST = ROOT/"actives_ids.lst" """
