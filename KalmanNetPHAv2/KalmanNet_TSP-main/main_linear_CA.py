import torch
from datetime import datetime
import pandas as pd
from Simulations.Linear_sysmdl import SystemModel
import Simulations.config as config
import Simulations.utils as utils
from Simulations.Linear_CA.parameters import F_gen,F_CV,H_identity,H_onlyPos,\
   Q_gen,Q_CV,R_3,R_2,R_onlyPos,\
   m,m_cv

from Filters.KalmanFilter_test import KFTest
import numpy as np
from KNet.KalmanNet_nn import KalmanNetNN

from Pipelines.Pipeline_EKF import Pipeline_EKF as Pipeline

from Plot import Plot_extended as Plot

################
### Get Time ###
################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

print("load data 2904")
[train_input2904, train_target2904, cv_input, cv_target, test_input, test_target] = torch.load('INPUT/29042012/cal_Net_x.pt', map_location=device)
print("load data 2201 and test on 2201:")
[train_input2201, train_target2201, cv_input, cv_target, test_input, test_target] = torch.load('INPUT/22012012/cal_Net_x.pt', map_location=device)

train_input = torch.cat([torch.tensor(train_input2201), torch.tensor(train_input2904)], dim=0).to(device)
train_target = torch.cat([torch.tensor(train_target2201), torch.tensor(train_target2904)], dim=0).to(device)

print("Pipeline Start")
####################################
### Generative Parameters For CA ###
####################################
args = config.general_settings()
### Dataset parameters
total_samples = train_target.size(dim = 0)                     # added
args.N_E = round(total_samples * 0.85)             # 1000
args.N_CV = round(total_samples * 0.15)            # 100
args.N_T = total_samples - args.N_E - args.N_CV    # 200
offset = 0 ### Init condition of dataset           # what for?
args.randomInit_train = False                       # True
args.randomInit_cv = False                          # True
args.randomInit_test = False                        # True

args.T = args.N_T                                  # Độ dài 1 sequence     # 100
args.T_test = args.N_T                             # Độ dài tập test       # 100
### training parameters
KnownRandInit_train = True # if true: use known random init for training, else: model is agnostic to random init
KnownRandInit_cv = True
KnownRandInit_test = True
                                                   # What these 3 for?
args.use_cuda = False # use GPU or not
args.n_steps = 1000                # 4000
args.n_batch = 1                                   # 10
args.lr = 1e-4
args.wd = 1e-4

if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

if(args.randomInit_train or args.randomInit_cv or args.randomInit_test):
   std_gen = 1
else:
   std_gen = 0

if(KnownRandInit_train or KnownRandInit_cv or KnownRandInit_test): # nếu k random thì mặc định là 0?
   std_feed = 0
else:
   std_feed = 1

# m1x_0 = torch.zeros(m)                       # Initial State, lỗi trong EKF
# m1x_0_cv = torch.zeros(m_cv)                 # Initial State for CV
m1x_0_cv = train_target[0].squeeze()
m2x_0 = std_feed * std_feed * torch.eye(m)   # Initial Covariance for feeding to filters and KNet
m2x_0_gen = std_gen * std_gen * torch.eye(m) # Initial Covariance for generating dataset
m2x_0_cv = std_feed * std_feed * torch.eye(m_cv) # Initial Covariance for CV
# m2x_0_cv = torch.eye(m_cv)
#############################
###  Dataset Generation   ###
#############################
### PVA or P
Loss_On_AllState = False      # if false: only calculate loss on position
Train_Loss_On_AllState = False# if false: only calculate training loss on position
CV_model = True               # if true: use CV model, else: use CA model

####################
### System Model ###
####################
# Generation model (CA)    # mình k dùng nên comment
# sys_model_gen = SystemModel(F_gen, Q_gen, H_onlyPos, R_onlyPos, args.T, args.T_test)
# sys_model_gen.InitSequence(m1x_0, m2x_0_gen)# x0 and P0

# Feed model (to KF, KalmanNet) 
if CV_model:
   H_onlyPos = torch.tensor([[1, 0]]).float()
   sys_model = SystemModel(F_CV, Q_CV, H_onlyPos, R_onlyPos, args.T, args.T_test)
   sys_model.InitSequence(m1x_0_cv, m2x_0_cv)# x0 and P0
else: # có như không, để cho vui
   sys_model = SystemModel(F_gen, Q_gen, H_onlyPos, R_onlyPos, args.T, args.T_test)
   sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0

print("Start Data Gen")       # mình k cần
# utils.DataGen(args, sys_model_gen, DatafolderName+DatafileName)
# print("Load Original Data")
# [train_input, train_target, cv_input, cv_target, test_input, test_target,train_init,cv_init,test_init] = torch.load(DatafolderName+DatafileName, map_location=device)
if CV_model:# set state as (p,v) instead of (p,v,a)            # thêm require grad để chạy KNET
   train_target = train_target[:,0:m_cv,:].requires_grad_()
   # train_init = train_init[:,0:m_cv]          # ori
   train_init = train_target[:, 0:m_cv].requires_grad_()
   cv_target = cv_target[:,0:m_cv,:].requires_grad_()
   # cv_init = cv_init[:,0:m_cv]                # ori
   cv_init = cv_target[:, 0:m_cv].requires_grad_()
   test_target = test_target[:,0:m_cv,:].requires_grad_()
   # test_init = test_init[:,0:m_cv]            # ori
   test_init = test_target[:, 0:m_cv].requires_grad_()   # test_init = [250,2,1] mà

print("Data Shape")
print("testset state x size:",test_target.size())
print("testset observation y size:",test_input.size())
print("trainset state x size:",train_target.size())
print("trainset observation y size:",train_input.size())
print("cvset state x size:",cv_target.size())
print("cvset observation y size:",cv_input.size())

print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
##############################
### Evaluate Kalman Filter ###
##############################
# print("Evaluate Kalman Filter")     # lỗi dimension
# if args.randomInit_test and KnownRandInit_test:
#    [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model, test_input, test_target, allStates=Loss_On_AllState, randomInit = True, test_init=test_init)
# else:
#    [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model, test_input, test_target, allStates=Loss_On_AllState, test_init=test_init)

##########################
### Evaluate KalmanNet ###
##########################
# Build Neural Network
path_results = 'KNet/2days/'
model_name = '2daysmodelteston2201x.pt'

KNet_model = KalmanNetNN()
KNet_model.NNBuild(sys_model, args)
print("Number of trainable parameters for KNet pass 1:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
## Train Neural Network
KNet_Pipeline = Pipeline(strTime, path_results, model_name)
KNet_Pipeline.setssModel(sys_model)
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setTrainingParams(args)
if (KnownRandInit_train):
   print("Train KNet with Known Random Initial State")
   print("Train Loss on All States (if false, loss on position only):", Train_Loss_On_AllState)
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, MaskOnState=not Train_Loss_On_AllState, randomInit = True, cv_init=cv_init,train_init=train_init)
else:
   print("Train KNet with Unknown Initial State")
   print("Train Loss on All States (if false, loss on position only):", Train_Loss_On_AllState)
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, MaskOnState=not Train_Loss_On_AllState)

torch.save(KNet_model, path_results + model_name)

if (KnownRandInit_test): 
   print("Test KNet with Known Random Initial State")
   ## Test Neural Network
   print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,KNet_out,RunTime] = KNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,MaskOnState=not Loss_On_AllState,randomInit=True,test_init=test_init)
else: 
   print("Test KNet with Unknown Initial State")
   ## Test Neural Network
   print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,KNet_out,RunTime] = KNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,MaskOnState=not Loss_On_AllState)

      
print("Save")
KNet_out = KNet_out.detach().numpy()
KNet_out = np.squeeze(KNet_out, axis=-1)
df_KNet = pd.DataFrame(KNet_out)
df_KNet.to_csv("OUTPUT/KNET/2201/2daysKNEToutx.csv", index=False)
#