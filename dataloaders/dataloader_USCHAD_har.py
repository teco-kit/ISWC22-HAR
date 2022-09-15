import pandas as pd
import numpy as np
import os

import scipy.io as sio
from dataloaders.dataloader_base import BASE_DATA

# ================================= USC_HAD_HAR_DATA ============================================
class USC_HAD_HAR_DATA(BASE_DATA):
    """

    **********************************************
    Section 1: Device Configuration


    2. Sampling rate: 100Hz
    3. Accelerometer range: +-6g
    4. Gyroscope range: +-500dps


    **********************************************
    Section 2: Data Format
    Each activity trial is stored in an .mat file.

    The naming convention of each .mat file is defined as:
    a"m"t"n".mat, where
    "a" stands for activity
    "m" stands for activity number
    "t" stands for trial
    "n" stands for trial number

    Each .mat file contains 13 fields:
    1. title: USC Human Motion Database
    2. version: it is version 1.0 for this first round data collection
    3. date
    4. subject number
    5. age
    6. height
    7. weight
    8. activity name
    9. activity number
    10. trial number
    11. sensor_location
    12. sensor_orientation
    13. sensor_readings

    For sensor_readings field, it consists of 6 readings:
    From left to right:
    1. acc_x, w/ unit g (gravity)
    2. acc_y, w/ unit g
    3. acc_z, w/ unit g
    4. gyro_x, w/ unit dps (degrees per second)
    5. gyro_y, w/ unit dps
    6. gyro_z, w/ unit dps

    **********************************************
    Section 3: Activities
    1. Walking Forward
    2. Walking Left
    3. Walking Right
    4. Walking Upstairs
    5. Walking Downstairs
    6. Running Forward
    7. Jumping Up
    8. Sitting
    9. Standing
    10. Sleeping
    11. Elevator Up
    12. Elevator Down

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
        # because this dataset only has 6 columns, the label is saved in the file name, so this used cols will not be used
        self.used_cols    = [0,1,2,3,4,5]
        # This dataset only has this 6 channels
        self.col_names = [ 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z' ]


        # These two variables represent whether all sensors can be filtered according to position and sensor type
        # pos_filter ------- >  filter according to position
        # sensor_filter ----->  filter according to the sensor type
        self.pos_filter         = None
        self.sensor_filter      = ["acc","gyro"]

        # selected_cols will be updated according to user settings. User have to set -- args.pos_select, args.sensor_select---
        self.selected_cols      = None
        # Filtering channels according to the Position
        self.selected_cols      = self.Sensor_filter_acoording_to_pos_and_type(args.pos_select, self.pos_filter, self.col_names, "position")
        # Filtering channels according to the Sensor Type
        if self.selected_cols is None:
            self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.col_names, "Sensor Type")
        else:
            self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.selected_cols, "Sensor Type")


        # The original labels are from 1 to 12
        self.label_map = [(1, "Walking Forward"),
                          (2, "Walking Left"),
                          (3, "Walking Right"),
                          (4, "Walking Upstairs"),
                          (5, "Walking Downstairs"),
                          (6, "Running Forward"),
                          (7, "Jumping Up"),
                          (8, "Sitting"),
                          (9, "Standing"),
                          (10, "Sleeping"),
                          (11, "Elevator Up"),
                          (12, "Elevator Down")]

        # As can be seen from the readme
        self.drop_activities = []



        self.train_keys   = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ]
        self.vali_keys    = [  ]
        self.test_keys    = [ 13, 14 ]

        self.exp_mode     = args.exp_mode

        self.split_tag = "sub"

        self.LOCV_keys = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14]]
        self.all_keys = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        self.sub_ids_of_each_sub = {}

        self.file_encoding = {}  # no use 
	
        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(USC_HAD_HAR_DATA, self).__init__(args)

    def load_all_the_data(self, root_path):
        print(" ----------------------- load all the data -------------------")

        activities = range(1, 13)

        df_dict = {}
        for sub in range(1, 15):

            for activity in activities:

                for trial in range(1, 6):

                    sub_data = sio.loadmat("%s/Subject%d%sa%dt%d.mat" % (root_path, sub, os.sep, activity, trial))
                    sub_data = pd.DataFrame(np.array(sub_data['sensor_readings']))

                    sub_data =sub_data.iloc[:,self.used_cols]
                    sub_data.columns = self.col_names
					
                    sub_id = "{}_{}_{}".format(sub,activity,trial)
                    sub_data["sub_id"] = sub_id
                    sub_data["sub"] = sub
                    sub_data["activity_id"] = activity

                    df_dict[sub_id] = sub_data   

                    if sub not in self.sub_ids_of_each_sub.keys():
                        self.sub_ids_of_each_sub[sub] = []
                    self.sub_ids_of_each_sub[sub].append(sub_id)


        df_all = pd.concat(df_dict)
        df_all = df_all.set_index('sub_id')


        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        if self.selected_cols:
            df_all = df_all[self.selected_cols+["sub"]+["activity_id"]]
        else:
            df_all = df_all[self.col_names+["sub"]+["activity_id"]]

        # Label Transformation
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 
        return data_x, data_y
