import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA

# ========================================    Daphnet_HAR_DATA        =============================
class Daphnet_HAR_DATA(BASE_DATA):

    """
    BASIC INFO ABOUT THE DATA:
    ---------------------------------
    The dataset comprises 3 wearable wireless acceleration sensors (see [10] for sensor details) recording 3D acceleration at 64 Hz. 
    The sensors are placed at the ankle (shank), on the thigh just above the knee, and on the hip.

    0: not part of the experiment. For instance the sensors are installed on the user or the user is performing activities unrelated to the experimental protocol, such as debriefing
    1: experiment, no freeze (can be any of stand, walk, turn)
    2: freeze
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

        # In this documents in doc/documentation.html, all columns definition coulde be found   (or in the column_names)

        # Time of sample in millisecond
        # Ankle (shank) acceleration - horizontal forward acceleration [mg]
        # Ankle (shank) acceleration - vertical [mg]
        # Ankle (shank) acceleration - horizontal lateral [mg]
        # Upper leg (thigh) acceleration - horizontal forward acceleration [mg]
        # Upper leg (thigh) acceleration - vertical [mg]
        # Upper leg (thigh) acceleration - horizontal lateral [mg]
        # Trunk acceleration - horizontal forward acceleration [mg]
        # Trunk acceleration - vertical [mg]
        # Trunk acceleration - horizontal lateral [mg]
        # Annotations (see Annotations section)

        # the first cols no use
        self.used_cols    =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.col_names    =  ["acc_x_ankle", "acc_y_ankle", "acc_z_ankle",
                              "acc_x_leg",   "acc_y_leg",   "acc_z_leg",
                              "acc_x_trunk", "acc_y_trunk","acc_z_trunk",
                              "activity_id"]


        # These two variables represent whether all sensors can be filtered according to position and sensor type
        # pos_filter ------- >  filter according to position
        # sensor_filter ----->  filter according to the sensor type
        self.pos_filter         = ["ankle","leg","trunk"]
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


        self.label_map = [
            (0, 'Other'),
            (1, 'No freeze'),
            (2, 'Freeze')
        ]

        self.drop_activities = [0]

        self.train_keys   = ["1_1", "1_2", # 'S01R01.txt', 'S01R02.txt',
                             "2_2", #'S02R02.txt', 
                             "3_1", "3_2", # 'S03R01.txt', 'S03R02.txt',
                             "3_3", # 'S03R03.txt', 
                             "4_1", # 'S04R01.txt',
                             "5_1", #'S05R01.txt' ]
                             "5_2",  #'S05R02.txt'
                             "6_1", "6_2", # 'S06R01.txt', 'S06R02.txt',
                             "7_1", "7_2", # 'S07R01.txt', 'S07R02.txt',
                             "8_1", #'S08R01.txt', 
                             "9_1", # 'S09R01.txt', 
                             "10_1" ] #'S10R01.txt' 

        self.vali_keys    =[]

        self.test_keys    = ["2_1", # 'S02R01.txt',
                             "2_2"] #'S02R02.txt', 


        self.exp_mode     = args.exp_mode
        if self.exp_mode == "LOCV":
            self.split_tag = "sub"
        else:
            self.split_tag = "sub_id"

        self.LOCV_keys = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
        self.all_keys = [1,2,3,4,5,6,7,8,9,10]
        self.sub_ids_of_each_sub = {}


        self.file_encoding = {'S01R01.txt':"1_1", 'S01R02.txt':"1_2",
                              'S02R01.txt':"2_1", 'S02R02.txt':"2_2",
                              'S03R01.txt':"3_1", 'S03R02.txt':"3_2", 'S03R03.txt':"3_3",
                              'S04R01.txt':"4_1",
                              'S05R01.txt':"5_1", 'S05R02.txt':"5_2",
                              'S06R01.txt':"6_1", 'S06R02.txt':"6_2",
                              'S07R01.txt':"7_1", 'S07R02.txt':"7_2",
                              'S08R01.txt':"8_1",
                              'S09R01.txt':"9_1",
                              'S10R01.txt':"10_1"}
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(Daphnet_HAR_DATA, self).__init__(args)

    def load_all_the_data(self, root_path):
        print(" ----------------------- load all the data -------------------")
        file_list = os.listdir(root_path)

        assert len(file_list) == 17
        df_dict = {}

        for file in file_list:
            sub_data = pd.read_table(os.path.join(root_path,file),header=None, delim_whitespace=True)
            sub_data =sub_data.iloc[:,self.used_cols]
            sub_data.columns = self.col_names

            # This data set dose not need label transformation
            sub = int(self.file_encoding[file].split("_")[0])
            sub_data['sub_id'] = self.file_encoding[file]
            sub_data["sub"] = sub
            if sub not in self.sub_ids_of_each_sub.keys():
                self.sub_ids_of_each_sub[sub] = []
            self.sub_ids_of_each_sub[sub].append(self.file_encoding[file])

            df_dict[self.file_encoding[file]] = sub_data


        df_all = pd.concat(df_dict)
        df_all = df_all.set_index('sub_id')


        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        if self.selected_cols:
            df_all = df_all[self.selected_cols+["sub"]+["activity_id"]]
        else:
            df_all = df_all[self.col_names[:-1]+["sub"]+["activity_id"]]


        # it is not necessary here
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 
            
        return data_x, data_y