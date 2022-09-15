import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA

# ========================================       WISDM_HAR_DATA             =============================
class WISDM_HAR_DATA(BASE_DATA):

    """

    https://www.cis.fordham.edu/wisdm/dataset.php
    Wireless Sensor Data Mining (WISDM) Lab

    BASIC INFO ABOUT THE DATA:
    ---------------------------------

    Sampling rate:  20Hz (1 sample every 50ms)

    raw.txt follows this format: [user],[activity],[timestamp],[x-acceleration],[y-accel],[z-accel];

    Fields: *user  nominal, 1..36

    activity nominal, { Walking Jogging Sitting Standing Upstairs Downstairs }
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


        # There is only one file which includs the collected data from 33 users
        # delete the second column it is the timestamp
        self.used_cols    = [0,1,3,4,5]
        self.col_names    =  ['sub','activity_id','timestamp', 'acc_x', 'acc_y', 'acc_z']


        # pos_filter ------- >  filter according to position
        # sensor_filter ----->  filter according to the sensor type
        self.pos_filter         = None
        self.sensor_filter      = None

        # selected_cols will be updated according to user settings. User have to set -- args.pos_select, args.sensor_select---
        self.selected_cols      = None
        # Filtering channels according to the Position
        self.selected_cols      = self.Sensor_filter_acoording_to_pos_and_type(args.pos_select, self.pos_filter, self.col_names, "position")
        # Filtering channels according to the Sensor Type
        if self.selected_cols is None:
            self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.col_names, "Sensor Type")
        else:
            self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.selected_cols, "Sensor Type")




        self.label_map = [(0, 'Walking'), 
                          (1, 'Jogging'),
                          (2, 'Sitting'),
                          (3, 'Standing'), 
                          (4, 'Upstairs'),
                          (5, 'Downstairs')]

        self.drop_activities = []
        # TODO This should be referenced by other paper
        # TODO , here the keys for each set will be updated in the readtheload function

        self.train_keys   = [1,2,3,4,5,  7,8,9,10,11,  13,14,15,16,17,  19,20,21,22,23,  25,26,27,28,29,  31,32,34,35,36]
        self.vali_keys    = []
        self.test_keys    = [6,12,18,24,30,33]

        self.exp_mode     = args.exp_mode

        self.split_tag = "sub"

        self.LOCV_keys = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20],[21,22,23,24],[25,26,27],[28,29,30],[31,32,33],[34,35,36]]
        self.all_keys = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
        self.sub_ids_of_each_sub = {}

        self.file_encoding = {} # no use

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(WISDM_HAR_DATA, self).__init__(args)


    def load_all_the_data(self, root_path):
        print(" ----------------------- load all the data -------------------")
        df_all = pd.read_csv(os.path.join(root_path,"WISDM_ar_v1.1_raw.txt"),header=None,names=self.col_names)
        df_all["acc_z"]=df_all["acc_z"].replace('\;','',regex=True).astype(float) #清洗掉z-axis中的符号
        df_all =df_all.iloc[:,self.used_cols]
        df_all.dropna(inplace=True)

        df_all['act_block'] = ( (df_all['sub'].shift(1) != df_all['sub'])).astype(int).cumsum()
        sub_id_list = []
        for index in df_all.act_block.unique():
            temp_df = df_all[df_all["act_block"]==index]
            sub = temp_df["sub"].unique()[0]
            sub_id = "{}_{}".format(sub,index)
            sub_id_list.extend([sub_id]*temp_df.shape[0])

            if sub not in self.sub_ids_of_each_sub.keys():
                self.sub_ids_of_each_sub[sub] = []
            self.sub_ids_of_each_sub[sub].append(sub_id)    

        df_all["sub_id"] =     sub_id_list
        del df_all["act_block"]

        label_mapping = {'Walking':0, 
                         'Jogging':1,
                          'Sitting':2,
                          'Standing':3, 
                          'Upstairs':4,
                          'Downstairs':5}

        df_all["activity_id"] = df_all["activity_id"].map(label_mapping)
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        df_all = df_all.set_index('sub_id')

        if self.selected_cols:
            df_all = df_all[self.selected_cols+["sub"]+["activity_id"]]
        else:
            df_all = df_all[["acc_x","acc_y","acc_z"]+["sub"]+["activity_id"]]


        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]
        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 
        return data_x, data_y
