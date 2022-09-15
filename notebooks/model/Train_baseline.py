#!/usr/bin/env python

#SBATCH --job-name=BSL

#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hzhao@teco.edu

#SBATCH --export=ALL

#SBATCH --time=48:00:00

#SBATCH --partition=sdil
#SBATCH --gres=gpu:1


import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("/pfs/data5/home/kit/tm/px3192/I2S0W2C2_CFC/")
sys.path.append('/pfs/data5/home/kit/tm/px3192/I2S0W2C2_CFC/notebooks/model/')

from experiment import Exp

from dataloaders import data_set,data_dict
import torch
import yaml
import os

scaling = float(sys.argv[2])
data = sys.argv[1]

if scaling == 1.0:
    scaling = int(1)
# 参数设置

# 训练参数  除了路径 其他不要变

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

args = dotdict()   
# TODO change the path as relative path
args.to_save_path     = "../../../ISWC2022LearnableFilter/Run_logs"              
args.freq_save_path   = "../../../ISWC2022LearnableFilter/Freq_data"
args.window_save_path = "../../../ISWC2022LearnableFilter/Sliding_window"
args.root_path        = "../../../datasets"


args.drop_transition  = False
args.datanorm_type    = "standardization" # None ,"standardization", "minmax"


args.batch_size       = 256                                                    
args.shuffle          = True
args.drop_last        = False
args.train_vali_quote = 0.90                                           


# training setting 
args.train_epochs            = 150

args.learning_rate           = 0.001  
args.learning_rate_patience  = 5
args.learning_rate_factor    = 0.1


args.early_stop_patience     = 15

args.use_gpu                 = True if torch.cuda.is_available() else False
args.gpu                     = 0
args.use_multi_gpu           = False

args.optimizer               = "Adam"
args.criterion               = "CrossEntropy"

## 数据参数



args.data_name                        =  data

args.wavelet_filtering                = False
args.wavelet_filtering_regularization = False
args.wavelet_filtering_finetuning     = False


args.difference       = False 
args.filtering        = False
args.magnitude        = False
args.weighted_sampler = False




args.pos_select       = None
args.sensor_select    = None


args.representation_type = "time"
args.exp_mode            = "LOCV"
if args.data_name      ==  "skodar":
    args.exp_mode            = "SOCV"

config_file = open('../../configs/data.yaml', mode='r')
data_config = yaml.load(config_file, Loader=yaml.FullLoader)
config = data_config[args.data_name]

args.root_path       = os.path.join(args.root_path,config["filename"])
args.sampling_freq   = config["sampling_freq"]
args.num_classes     =  config["num_classes"]
window_seconds       = config["window_seconds"]
args.windowsize      =   int(window_seconds * args.sampling_freq) 
args.input_length    =  args.windowsize
# input information
args.c_in            = config["num_channels"]

if args.wavelet_filtering :
    
    if args.windowsize%2==1:
        N_ds = int(torch.log2(torch.tensor(args.windowsize-1)).floor()) - 2
    else:
        N_ds = int(torch.log2(torch.tensor(args.windowsize)).floor()) - 2

    args.f_in            =  args.number_wavelet_filtering*N_ds+1
else:
    args.f_in            =  1

## 模型参数

args.filter_scaling_factor = scaling
args.model_type            = "deepconvlstm"

# 实验

for seed in [1,2,3,4,5]:
    args.seed = seed
    exp = Exp(args)
    exp.train()



