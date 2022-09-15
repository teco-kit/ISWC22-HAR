import pandas as pd
import numpy as np
import os
import re

from dataloaders.dataloader_base import BASE_DATA

# ========================================    HAPT_HAR_DATA        =============================
class HAPT_HAR_DATA(BASE_DATA):

    """

    """

    def __init__(self, args):

        """


        """



        # If used_cols is [], it means that all columns will be used
        self.used_cols          = []
        # if used_cols is not none, the length of col_names should be same as the used_cols
        self.col_names          = ['acc_x_1', 'acc_y_1', 'acc_z_1','gyro_x_2', 'gyro_y_2', 'gyro_z_2']

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


        self.label_map = [
            (0, "Null_Class"),
            (1, "WALKING"),   
            (2, "WALKING_UPSTAIRS" ),
            (3, "WALKING_DOWNSTAIRS"),
            (4, "SITTING"),
            (5, "STANDING"),
            (6, "LAYING"),
            (7, "STAND_TO_SIT"),
            (8, "SIT_TO_STAND"),
            (9, "SIT_TO_LIE"),
            (10, "LIE_TO_SIT"),
            (11, "STAND_TO_LIE"),
            (12, "LIE_TO_STAND")
        ]

        self.drop_activities    = [0]

        self.train_keys         = [1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        self.vali_keys          = []
        self.test_keys          = []


        self.exp_mode           = args.exp_mode
        self.split_tag          = "sub"


        self.LOCV_keys = [[1,  2,  3],  [4,  5,  6],  [7,  8,  9],  
                          [10, 11, 12], [13, 14, 15], [16, 17, 18], 
                          [19, 20, 21], [22, 23, 24], [25, 26, 27], 
                          [28, 29, 30]]

        self.all_keys = [1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                         21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

        self.sub_ids_of_each_sub = {}


        self.file_encoding = {}  # no use
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(HAPT_HAR_DATA, self).__init__(args)

    def load_all_the_data(self, root_path):
        print(" ----------------------- load all the data -------------------")
        labels = np.loadtxt(os.path.join(root_path, 'labels.txt'), delimiter=' ')

        # filter out the acc/gyro files
        acc_data  = [f for f in os.listdir(root_path) if 'acc' in f]
        gyro_data = [f for f in os.listdir(root_path) if 'gyro' in f]

        df_dict = {}

        # this dataset has 30 subjects
        for sbj in range(30):
            if sbj < 9:
                acc_sbj_files = [f for f in acc_data if 'user0' + str(sbj + 1) in f]
                gyro_sbj_files = [f for f in gyro_data if 'user0' + str(sbj + 1) in f]
            else:
                acc_sbj_files = [f for f in acc_data if 'user' + str(sbj + 1) in f]
                gyro_sbj_files = [f for f in gyro_data if 'user' + str(sbj + 1) in f]


            for acc_sbj_file in acc_sbj_files:

                acc_tmp_data = np.loadtxt(os.path.join(root_path, acc_sbj_file), delimiter=' ')
                # get the sub and exp
                sub_str = re.sub('[^0-9]', '', acc_sbj_file.split('_')[2])
                exp_str = re.sub('[^0-9]', '', acc_sbj_file.split('_')[1])

                sub_int = int(sub_str)
                exp_int = int(exp_str)

                sub_id = "{}_{}".format(sub_int,exp_int)

                gyro_tmp_data = np.loadtxt(os.path.join(root_path, 'gyro_exp' + exp_str + '_user' + sub_str + '.txt'), delimiter=' ')

                sub_data = pd.DataFrame(np.concatenate((acc_tmp_data, gyro_tmp_data), axis=1))

                # Column 1: experiment number ID, 
                # Column 2: user number ID, 
                # Column 3: activity number ID 
                # Column 4: Label start point (in number of signal log samples (recorded at 50Hz))
                # Column 5: Label end point (in number of signal log samples)
                sub_labels = labels[(labels[:, 0] == exp_int) & (labels[:, 1] == sub_int)]

                sub_data.columns = self.col_names

                sub_data["sub"] = sub_int
                sub_data["sub_id"] = sub_id
                # lables
                sub_data["activity_id"] = 0
                for label_triplet in sub_labels:
                    sub_data.iloc[int(label_triplet[3]):int(label_triplet[4] + 1), -1] = label_triplet[2]

                if sub_int not in self.sub_ids_of_each_sub.keys():
                    self.sub_ids_of_each_sub[sub_int] = []
                self.sub_ids_of_each_sub[sub_int].append(sub_id)
                df_dict[sub_id] = sub_data

        df_all = pd.concat(df_dict)
        df_all = df_all.set_index('sub_id')

        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        if self.selected_cols:
            df_all = df_all[self.selected_cols+["sub"]+["activity_id"]]
        else:
            df_all = df_all[self.col_names+["sub"]+["activity_id"]]

        # it is not necessary here
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)





        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 

        return data_x, data_y