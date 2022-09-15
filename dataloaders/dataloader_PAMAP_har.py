import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA
# ================================= PAMAP2 HAR DATASET ============================================
class PAMAP2_HAR_DATA(BASE_DATA):
    """
    PAMAP2_Dataset: Physical Activity Monitoring

    BASIC INFO ABOUT THE DATA:
    ---------------------------------
    sampling frequency: 100Hz

    position of the sensors:
      1 IMU over the wrist on the dominant arm
      1 IMU on the chest
      1 IMU on the dominant side's ankle


    9 subjects participated in the data collection:
      mainly employees or students at DFKI
      1 female, 8 males
      aged 27.22 ± 3.31 years

    Each of the data-files contains 54 columns per row, the columns contain the following data:
      1 timestamp (s)
      2 activityID (see II.2. for the mapping to the activities)
      3 heart rate (bpm)
      4-20 IMU hand
      21-37 IMU chest
      38-54 IMU ankle

    The IMU sensory data contains the following columns:
      1 temperature (°C)  !!!!! DROP
      2-4 3D-acceleration data (ms-2), scale: ±16g, resolution: 13-bit
      5-7 3D-acceleration data (ms-2), scale: ±6g, resolution: 13-bit*
      8-10 3D-gyroscope data (rad/s)
      11-13 3D-magnetometer data (μT)
      14-17 orientation (invalid in this data collection) !!!!!!!!!!!DROP
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

        # !!!!!! Depending on the setting of each data set!!!!!!
        # the 0th column is time step 
        self.used_cols    = [1,# this is "label"
                             # TODO check the settings of other paper 
                             # the second column is heart rate (bpm) --> ignore?
                             # each IMU sensory has 17 channals , 3-19,20-36,38-53
                             # the first temp ignores
                             # the last four channel according to the readme are invalide
                             # 4, 5, 6,   7, 8, 9,   10, 11, 12,   13, 14, 15,        # IMU Hand
                             # 21, 22, 23,   24, 25, 26,   27, 28, 29,   30, 31, 32,  # IMU Chest
                             # 38, 39, 40,    41, 42, 43,    44, 45, 46,    47, 48, 49   # IMU ankle
                             4, 5, 6,   10, 11, 12,      # IMU Hand
                             21, 22, 23,  27, 28, 29,    # IMU Chest
                             38, 39, 40,   44, 45, 46,     # IMU ankle
                            ]
        # form the columns name , [label, 12*[hand], 12*[chest], 12*[ankle]]

        self.col_names = ['activity_id',
                          'acc_x_hand', 'acc_y_hand', 'acc_z_hand',
                          'gyro_x_hand', 'gyro_y_hand', 'gyro_z_hand',
                          'acc_x_chest', 'acc_y_chest', 'acc_z_chest',
                          'gyro_x_chest', 'gyro_y_chest', 'gyro_z_chest',
                          'acc_x_ankle', 'acc_y_ankle', 'acc_z_ankle',
                          'gyro_x_ankle', 'gyro_y_ankle',  'gyro_z_ankle']


        # These two variables represent whether all sensors can be filtered according to position and sensor type
        # pos_filter ------- >  filter according to position
        # sensor_filter ----->  filter according to the sensor type
        self.pos_filter         = ["hand", "chest", "ankle"]
        self.sensor_filter      = ["acc", "gyro"]


        # selected_cols will be updated according to user settings. User have to set -- args.pos_select, args.sensor_select---
        self.selected_cols      = None
        # Filtering channels according to the Position
        self.selected_cols      = self.Sensor_filter_acoording_to_pos_and_type(args.pos_select, self.pos_filter, self.col_names[1:], "position")
        # Filtering channels according to the Sensor Type
        if self.selected_cols is None:
            self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.col_names[1:], "Sensor Type")
        else:
            self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.selected_cols, "Sensor Type")



        self.label_map = [ 
            (0, 'other'),
            (1, 'lying'),
            (2, 'sitting'),
            (3, 'standing'),
            (4, 'walking'),
            (5, 'running'),
            (6, 'cycling'),
            (7, 'nordic walking'),

            (12, 'ascending stairs'),
            (13, 'descending stairs'),
            (16, 'vacuum cleaning'),
            (17, 'ironing'),

            (24, 'rope jumping')
        ]
        # As can be seen from the PerformedActivitiesSummary.pdf, some activities are not performed
        # TODO this should be chosen by reading related work
        # self.drop_activities = [0,9,10,11,18,19,20] #TODO check!!!!
        self.drop_activities = [0]

        # 'subject101.dat', 'subject102.dat', 'subject103.dat',  'subject104.dat', 
        # 'subject105.dat', 'subject107.dat', 'subject108.dat', 'subject109.dat'
        self.train_keys   = [1,2,3,4,5,7,8,9]
        self.vali_keys    = []
        # 'subject106.dat'
        self.test_keys    = [6]

        self.exp_mode     = args.exp_mode

        self.split_tag    = "sub"
 

        self.LOCV_keys = [[1],[2],[3],[4],[5],[6],[7],[8],[9]]
        self.all_keys = [1,2,3,4,5,6,7,8,9]
        self.sub_ids_of_each_sub = {}

        self.file_encoding = {'subject101.dat':1, 'subject102.dat':2, 'subject103.dat':3, 
                              'subject104.dat':4, 'subject105.dat':5, 'subject106.dat':6,
                              'subject107.dat':7, 'subject108.dat':8, 'subject109.dat':9} 

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(PAMAP2_HAR_DATA, self).__init__(args)

    def load_all_the_data(self, root_path):
        print(" ----------------------- load all the data -------------------")
        file_list = os.listdir(root_path)
        
        df_dict = {}
        for file in file_list:
            sub_data = pd.read_table(os.path.join(root_path,file), header=None, sep='\s+')
            sub_data =sub_data.iloc[:,self.used_cols]
            sub_data.columns = self.col_names

            # if missing values, imputation
            sub_data = sub_data.interpolate(method='linear', limit_direction='both')
            sub = int(self.file_encoding[file])
            sub_data['sub_id'] =sub
            sub_data["sub"] = sub

            if sub not in self.sub_ids_of_each_sub.keys():
                self.sub_ids_of_each_sub[sub] = []
            self.sub_ids_of_each_sub[sub].append(sub)
            df_dict[self.file_encoding[file]] = sub_data   


        # all data
        df_all = pd.concat(df_dict)
        # Downsampling!
        df_all.reset_index(drop=True,inplace=True)
        index_list = list(np.arange(0,df_all.shape[0],3))
        df_all = df_all.iloc[index_list]

        df_all = df_all.set_index('sub_id')


        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)


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
