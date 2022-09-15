import pandas as pd
import numpy as np
import os
import scipy.io as sio

from dataloaders.dataloader_base import BASE_DATA

# ========================================       SkodaR_HAR_DATA               =============================
class SkodaR_HAR_DATA(BASE_DATA):

    """
    Activity recognition dataset - Skoda Mini Checkpoint
    Brief Description of the Dataset:
    ---------------------------------

    Sensors
    This dataset contains 10 classes, recorded with a 2x10 USB sensors placed on the left and right upper and lower arm.

    Sensor sample rate is approximately 98Hz.
    The locations of the sensors on the arms is documented in the figure.

    right_classall_clean.mat and left_classall_clean.mat: matlab .mat files with original datafor right and left arm sensors

    label value:
        32 null class
        48 write on notepad
        49 open hood
        50 close hood
        51 check gaps on the front door
        52 open left front door
        53 close left front door
        54 close both left door
        55 check trunk gaps
        56 open and close trunk
        57 check steering wheel
    """

    def __init__(self, args):

        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （ UCI 数据的采样频率？？）
            wavelet : Methods of wavelet transformation

        """

        # Column 1: label
        # Column 2+s*7: sensor id
        # Column 2+s*7+1: X acceleration calibrated
        # Column 2+s*7+2: Y acceleration calibrated
        # Column 2+s*7+3: Z acceleration calibrated
        # Column 2+s*7+4: X acceleration raw
        # Column 2+s*7+5: Y acceleration raw
        # Column 2+s*7+6: Z acceleration raw

        self.used_cols = [0]+[2 + s * 7 for s in range(10)] + [3 + s *7 for s in range(10)] + [4 + s *7 for s in range(10)]
        self.used_cols.sort()

        # there are total 30 sensors 
        col_names = ["acc_x","acc_y", "acc_z"]
        self.col_names    =  ["activity_id"] + [j  for k in [[item+"_"+str(i) for item in col_names] for i in range(1,11)] for j in k ]

        # These two variables represent whether all sensors can be filtered according to position and sensor type
        # pos_filter ------- >  filter according to position
        # sensor_filter ----->  filter according to the sensor type
        self.pos_filter         = None
        self.sensor_filter      = None

        # selected_cols will be updated according to user settings. User have to set -- args.pos_select, args.sensor_select---
        self.selected_cols      = None
        # Filtering channels according to the Position
        self.selected_cols      = self.Sensor_filter_acoording_to_pos_and_type(args.pos_select, self.pos_filter, self.col_names[1:], "position")
        # Filtering channels according to the Sensor Type
        if self.selected_cols is None:
            self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.col_names[1:], "Sensor Type")
        else:
            self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.selected_cols, "Sensor Type")



        self.label_map = [(32, "null class"),
                          (48, "write on notepad"),
                          (49, "open hood"),
                          (50, "close hood"),
                          (51, "check gaps on the front door"),
                          (52, "open left front door"),
                          (53, "close left front door"),
                          (54, "close both left door"),
                          (55, "check trunk gaps"),
                          (56, "open and close trunk"),
                          (57, "check steering wheel")]

        self.drop_activities = [32]

        self.train_keys   = []  # no use 
        self.vali_keys    = []  # no use 
        self.test_keys    = []  # no use 

        self.exp_mode     = args.exp_mode
        self.split_tag = "sub"

        self.LOCV_keys = []
        self.all_keys = [1]
        self.sub_ids_of_each_sub = {}

        self.file_encoding = {}  # no use 

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(SkodaR_HAR_DATA, self).__init__(args)

    def load_all_the_data(self, root_path):
        print(" ----------------------- load all the data -------------------")

        data_dict = sio.loadmat(file_name=os.path.join(root_path,"right_classall_clean.mat"), squeeze_me=True)
        df_all = data_dict[list(data_dict.keys())[3]]

        df_all = df_all[:, self.used_cols]
        df_all = pd.DataFrame(df_all,columns=self.col_names)

        df_all["sub_id"] = 1
        df_all["sub"] = 1

        # Downsampling!
        df_all.reset_index(drop=True,inplace=True)
        index_list = list(np.arange(0,df_all.shape[0],3))
        df_all = df_all.iloc[index_list]


        self.sub_ids_of_each_sub[1] = [1]
        # label transformation
        df_all["activity_id"]=df_all["activity_id"].map(self.labelToId)

        df_all = df_all.set_index('sub_id')

        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        if self.selected_cols:
            df_all = df_all[self.selected_cols+["sub"]+["activity_id"]]
        else:
            df_all = df_all[self.col_names[1:]+["sub"]+["activity_id"]]



        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 

        return data_x, data_y