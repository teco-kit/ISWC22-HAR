import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA

# ========================================       DSADS_HAR_DATA               =============================
class DSADS_HAR_DATA(BASE_DATA):
    """
    https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities

    Daily and Sports Activities Data Set

    Brief Description of the Dataset:
    ---------------------------------
    Each of the 19 activities is performed by eight subjects (4 female, 4 male, between the ages 20 and 30) for 5 minutes.
    Total signal duration is 5 minutes for each activity of each subject.
    The subjects are asked to perform the activities in their own style and were not restricted on how the activities should be performed. 
    For this reason, there are inter-subject variations in the speeds and amplitudes of some activities.
	
    The activities are performed at the Bilkent University Sports Hall, in the Electrical and Electronics Engineering Building, and in a flat outdoor area on campus. 
    Sensor units are calibrated to acquire data at 25 Hz sampling frequency. 
    The 5-min signals are divided into 5-sec segments so that 480(=60x8) signal segments are obtained for each activity.
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

        # there are total 3 sensors :ACC Gyro Mag
        # amounted in 5 places "T", "RA", "LA", "RL", "LL"
        # In total 45 Channels
		
        self.used_cols = used_cols = [0,  1,  2,  3,  4,  5,  6,  7,  8,
                                      9, 10, 11, 12, 13, 14, 15,  16, 17,
                                      18, 19, 20, 21, 22, 23, 24, 25, 26,
                                      27, 28, 29, 30, 31, 32, 33, 34, 35,
                                      36, 37, 38, 39, 40, 41, 42, 42, 44]

        col_list       =  ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z","mag_x","mag_y","mag_z"]
        pos_list       = ["T", "RA", "LA", "RL", "LL"]
        self.col_names = [item for sublist in [[col+"_"+pos for col in col_list] for pos in pos_list] for item in sublist]


        # These two variables represent whether all sensors can be filtered according to position and sensor type
        # pos_filter ------- >  filter according to position
        # sensor_filter ----->  filter according to the sensor type
        self.pos_filter         = ["T", "RA", "LA", "RL", "LL"]
        self.sensor_filter      = ["acc","gyro","mag"]


        # selected_cols will be updated according to user settings. User have to set -- args.pos_select, args.sensor_select---
        self.selected_cols      = None
        # Filtering channels according to the Position
        self.selected_cols      = self.Sensor_filter_acoording_to_pos_and_type(args.pos_select, self.pos_filter, self.col_names, "position")
        # Filtering channels according to the Sensor Type
        if self.selected_cols is None:
            self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.col_names, "Sensor Type")
        else:
            self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.selected_cols, "Sensor Type")

        self.label_map = [(0, '01'), # sitting (A1),
                          (1, "02"), # standing (A2),
                          (2, "03"), # lying on back and on right side (A3 and A4),
                          (3, "04"), # lying on back and on right side (A3 and A4),
                          (4, "05"), # ascending and descending stairs (A5 and A6),
                          (5, "06"), # ascending and descending stairs (A5 and A6),
                          (6, "07"), # standing in an elevator still (A7)
                          (7, "08"), # and moving around in an elevator (A8),
                          (8, "09"), # walking in a parking lot (A9),
                          (9, "10"), # walking on a treadmill with a speed of 4 km/h (in flat and 15 deg inclined positions) (A10 and A11),
                          (10, "11"), # walking on a treadmill with a speed of 4 km/h (in flat and 15 deg inclined positions) (A10 and A11),
                          (11, "12"), # running on a treadmill with a speed of 8 km/h (A12),
                          (12, "13"), # exercising on a stepper (A13),
                          (13, "14"), # exercising on a cross trainer (A14),
                          (14, "15"), # cycling on an exercise bike in horizontal and vertical positions (A15 and A16),
                          (15, "16"), # cycling on an exercise bike in horizontal and vertical positions (A15 and A16),
                          (16, "17"), # rowing (A17),
                          (17, "18"), # jumping (A18),
                          (18, "19")] #  playing basketball (A19).

        self.drop_activities = []

        # TODO , here the keys for each set will be updated in the readtheload function
        self.train_keys   = [1,2,3,4,5,6,7]
        self.vali_keys    = []
        self.test_keys    = [8]


        self.LOCV_keys = [[1],[2],[3],[4],[5],[6],[7],[8]]
        self.all_keys = [1,2,3,4,5,6,7,8]
        self.sub_ids_of_each_sub = {}

        self.exp_mode     = args.exp_mode
        self.split_tag = "sub"

        self.file_encoding = {}  # no use 
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(DSADS_HAR_DATA, self).__init__(args)

    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -------------------")

        df_dict = {}

        for action in os.listdir(root_path):
            action_name = action[1:]

            for user in os.listdir(os.path.join(root_path,action)):
                user_name = user[1:]

                for seg in os.listdir(os.path.join(root_path,action,user)):
                    seg_name = seg[1:3]

                    sub_data = pd.read_csv(os.path.join(root_path,action,user,seg),header=None)
                    sub_data =sub_data.iloc[:,self.used_cols]
                    sub_data.columns = self.col_names

                    sub_id = "{}_{}_{}".format(user_name,seg_name,action_name)
                    sub_data["sub_id"] = sub_id
                    sub_data["sub"] = int(user_name)
                    sub_data["activity_id"] = action_name

                    sub = int(user_name)
                    if sub not in self.sub_ids_of_each_sub.keys():
                        self.sub_ids_of_each_sub[sub] = []
                    self.sub_ids_of_each_sub[sub].append(sub_id)

                    df_dict[sub_id] = sub_data

        df_all = pd.concat(df_dict)
        df_all = df_all.set_index('sub_id')


        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        if self.selected_cols:
            df_all = df_all[self.selected_cols+["sub"]+["activity_id"]]
        else:
            df_all = df_all[self.col_names+["sub"]+["activity_id"]]

        label_mapping = {item[1]:item[0] for item in self.label_map}
        # because the activity label in the df is not encoded, thet are  "01","02",...,"19"
        # first, map them in to nummeric number
        df_all["activity_id"] = df_all["activity_id"].map(label_mapping)
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 

        return data_x, data_y


