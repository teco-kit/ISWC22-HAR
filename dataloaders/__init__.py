from torch.utils.data import Dataset
import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
#from dataloaders.utils import PrepareWavelets,FiltersExtention
# ----------------- har -----------------
"""
from .dataloader_uci_har import UCI_HAR_DATA
from .dataloader_daphnet_har import Daphnet_HAR_DATA
from .dataloader_pamap2_har import PAMAP2_HAR_DATA
from .dataloader_opportunity_har import Opportunity_HAR_DATA
from .dataloader_wisdm_har import WISDM_HAR_DATA
from .dataloader_dsads_har import DSADS_HAR_DATA
from .dataloader_sho_har import SHO_HAR_DATA
from .dataloader_complexHA_har import ComplexHA_HAR_DATA
from .dataloader_mhealth_har import Mhealth_HAR_DATA
from .dataloader_motion_sense_har import MotionSense_HAR_DATA
from .dataloader_usc_had_har import USC_HAD_HAR_DATA
from .dataloader_skoda_r_har import SkodaR_HAR_DATA
from .dataloader_skoda_l_har import SkodaL_HAR_DATA
from .dataloader_single_chest_har import Single_Chest_HAR_DATA
from .dataloader_HuGaDBV2_har import HuGaDBV2_HAR_DATA
from .dataloader_utd_mhad_w_har import UTD_MHAD_W_HAR_DATA
from .dataloader_utd_mhad_t_har import UTD_MHAD_T_HAR_DATA
from .dataloader_synthetic_har import SYNTHETIC_HAR_DATA
from .dataloader_unimib_har import UNIMIB_HAR_DATA
from .dataloader_hhar_har import HHAR_HAR_DATA
from .dataloader_ward_har import WARD_HAR_DATA
data_dict = {"ucihar" : UCI_HAR_DATA,
             "daphnet" : Daphnet_HAR_DATA,
             "pamap2" : PAMAP2_HAR_DATA,
             "oppo"  : Opportunity_HAR_DATA,
             "wisdm" : WISDM_HAR_DATA,
             "dsads" : DSADS_HAR_DATA,
             "sho" : SHO_HAR_DATA,
             "complexha" : ComplexHA_HAR_DATA,
             "mhealth" : Mhealth_HAR_DATA,
             "motionsense" : MotionSense_HAR_DATA,
             "uschad" : USC_HAD_HAR_DATA,
             "skodar" : SkodaR_HAR_DATA,
             "skodal" : SkodaL_HAR_DATA,
             "singlechest" : Single_Chest_HAR_DATA,
             "hugadbv2" : HuGaDBV2_HAR_DATA,
             "utdmhadw" : UTD_MHAD_W_HAR_DATA,
             "utdmhadt" : UTD_MHAD_T_HAR_DATA,
             "synthetic_1" : SYNTHETIC_HAR_DATA,
             "synthetic_2" : SYNTHETIC_HAR_DATA,
             "synthetic_3" : SYNTHETIC_HAR_DATA,
             "synthetic_4" : SYNTHETIC_HAR_DATA,
             "unimib"   : UNIMIB_HAR_DATA,
             "hhar"     : HHAR_HAR_DATA,
             "ward"     : WARD_HAR_DATA}

"""
from .dataloader_HAPT_har import HAPT_HAR_DATA
from .dataloader_EAR_har  import EAR_HAR_DATA
from .dataloader_RW_har import REAL_WORLD_HAR_DATA
from .dataloader_OPPO_har import Opportunity_HAR_DATA
from .dataloader_PAMAP_har import PAMAP2_HAR_DATA
from .dataloader_SKODAR_har import SkodaR_HAR_DATA
from .dataloader_DSADS_har import DSADS_HAR_DATA
from .dataloader_DG_har import Daphnet_HAR_DATA
from .dataloader_USCHAD_har import USC_HAD_HAR_DATA
from .dataloader_WISDM_har import WISDM_HAR_DATA
data_dict = {"hapt"  : HAPT_HAR_DATA,
             "ear"   : EAR_HAR_DATA,
             "oppo"  : Opportunity_HAR_DATA,
             "rw"    : REAL_WORLD_HAR_DATA,
             "pamap2": PAMAP2_HAR_DATA,
             "skodar": SkodaR_HAR_DATA,
             "dsads" : DSADS_HAR_DATA,
             "dg"    : Daphnet_HAR_DATA,
             "uschad": USC_HAD_HAR_DATA,
             "wisdm" : WISDM_HAR_DATA}

class data_set(Dataset):
    def __init__(self, args, dataset, flag):
        """
        args : a dict , In addition to the parameters for building the model, the parameters for reading the data are also in here
        dataset : It should be implmented dataset object, it contarins train_x, train_y, vali_x,vali_y,test_x,test_y
        flag : (str) "train","test","vali"
        """
        self.args = args
        self.flag = flag
        self.load_all = args.load_all
        self.data_x = dataset.normalized_data_x
        self.data_y = dataset.data_y
        if flag in ["train","vali"] or self.args.exp_mode in ["SOCV","FOCV"]:
            #print("get train_slidingwindows ")
            self.slidingwindows = dataset.train_slidingwindows
        else:
            #print("get test_slidingwindows ")
            self.slidingwindows = dataset.test_slidingwindows
        self.act_weights = dataset.act_weights

        if self.args.representation_type in ["freq","time_freq"]:
            if flag in  ["train","vali"]:
                self.freq_path      = dataset.train_freq_path
                self.freq_file_name = dataset.train_freq_file_name
                if self.load_all :
                    self.data_freq   = dataset.data_freq
            else:
                self.freq_path      = dataset.test_freq_path
                self.freq_file_name = dataset.test_freq_file_name
                self.load_all = False

        if self.flag == "train":
            # load train
            self.window_index =  dataset.train_window_index
            print("Train data number : ", len(self.window_index))


        elif self.flag == "vali":
            # load vali

            self.window_index =  dataset.vali_window_index
            print("Validation data number : ",  len(self.window_index))  


        else:
            # load test
            self.window_index = dataset.test_window_index
            print("Test data number : ", len(self.window_index))  
            
            
        all_labels = list(np.unique(dataset.data_y))
        to_drop = list(dataset.drop_activities)
        label = [item for item in all_labels if item not in to_drop]
        self.nb_classes = len(label)
        assert self.nb_classes==len(dataset.no_drop_activites)

        classes = dataset.no_drop_activites
        self.class_transform = {x: i for i, x in enumerate(classes)}
        self.class_back_transform = {i: x for i, x in enumerate(classes)}
        self.input_length = self.slidingwindows[0][2]-self.slidingwindows[0][1]
        self.channel_in = self.data_x.shape[1]-2


        #if self.args.wavelet_filtering:
        #    SelectedWavelet = PrepareWavelets(K=self.args.number_wavelet_filtering, length=self.args.windowsize)
        #    self.ScaledFilter = FiltersExtention(SelectedWavelet)
        #    if self.args.windowsize%2==1:
        #        self.Filter_ReplicationPad1d = torch.nn.ReplicationPad1d(int((self.args.windowsize-1)/2))
        #    else:
        #        self.Filter_ReplicationPad1d = torch.nn.ReplicationPad1d(int(self.args.windowsize/2))

        if self.flag == "train":
            print("The number of classes is : ", self.nb_classes)
            print("The input_length  is : ", self.input_length)
            print("The channel_in is : ", self.channel_in)


    def __getitem__(self, index):
        #print(index)
        index = self.window_index[index]
        start_index = self.slidingwindows[index][1]
        end_index = self.slidingwindows[index][2]

        if self.args.representation_type == "time":

            if self.args.sample_wise ==True:
                sample_x = np.array(self.data_x.iloc[start_index:end_index, 1:-1].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x))))
            else:
                sample_x = self.data_x.iloc[start_index:end_index, 1:-1].values

            sample_y = self.class_transform[self.data_y.iloc[start_index:end_index].mode().loc[0]]
            #print(sample_x.shape)


            sample_x = np.expand_dims(sample_x,0)
            #print(sample_x.shape)
            return sample_x, sample_y,sample_y

        elif self.args.representation_type == "freq":

            if self.load_all:
                    sample_x = self.data_freq[self.freq_file_name[index]]
            else:
                with open(os.path.join(self.freq_path,"{}.pickle".format(self.freq_file_name[index])), 'rb') as handle:
                    sample_x = pickle.load(handle)

            sample_y = self.class_transform[self.data_y.iloc[start_index:end_index].mode().loc[0]]

            return sample_x, sample_y,sample_y

        else:

            if self.args.sample_wise ==True:
                sample_ts_x = np.array(self.data_x.iloc[start_index:end_index, 1:-1].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x))))
            else:
                sample_ts_x = self.data_x.iloc[start_index:end_index, 1:-1].values


            if self.load_all:
                    sample_fq_x = self.data_freq[self.freq_file_name[index]]
            else:
                with open(os.path.join(self.freq_path,"{}.pickle".format(self.freq_file_name[index])), 'rb') as handle:
                    sample_fq_x = pickle.load(handle)

            sample_y = self.class_transform[self.data_y.iloc[start_index:end_index].mode().loc[0]]


            return sample_ts_x, sample_fq_x , sample_y

    def __len__(self):
        return len(self.window_index)

