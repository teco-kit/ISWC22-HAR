import pandas as pd
import numpy as np
import os
import re

from dataloaders.dataloader_base import BASE_DATA



SI_SUFFIX = '_SI'
def merge_to_dataframe(df_esense, df_earconnect, col):

    df_esense = df_esense.reset_index(drop=True)

    df_earconnect     = df_earconnect[['timestamp', col]].reset_index(drop=True)
    df_earconnect     = df_earconnect.dropna(subset=[col]).reset_index(drop=True)

    min_timestamp = df_esense.iloc[0,2]
    max_timestamp = df_esense.iloc[-1,2]
    
    timestamps = np.array(df_esense["timestamp"])
    index_dict = {}

    for i in range(df_earconnect.shape[0]):
        time = df_earconnect.iloc[i,0]
        index = np.where(timestamps > time)[0]
        if len(index)> 0 :
            index = index[0]
            if index == 0:
                index_dict[index] = i
            else:
                timestamp_1 = timestamps[index-1]
                timestamp_2 = timestamps[index]
                assert timestamp_2-time>=0
                assert time-timestamp_1>=0
                if timestamp_2-time>time-timestamp_1:
                    index = index-1
                    if index in index_dict.keys():
                        time_exist = df_earconnect.iloc[index_dict[index],0]
                        if np.abs(time_exist-timestamp_1)>=np.abs(time-timestamp_1):
                            index_dict[index] = i
                    else:
                        index_dict[index] = i
                else:
                    if index in index_dict.keys():
                        time_exist = df_earconnect.iloc[index_dict[index],0]
                        if np.abs(time_exist-timestamp_2)>=np.abs(time-timestamp_2):
                            index_dict[index] = i   
                    else:
                        index_dict[index] = i
        else:
            break



    values = [np.nan]*df_esense.shape[0]
    for key in index_dict.keys():
        values[key] = df_earconnect.iloc[index_dict[key],1]

    return values

def _timestampToSI(df):
    df['timestamp' + SI_SUFFIX] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000
    return df

def _trim(df):
    time_sec_label = 'timestamp' + SI_SUFFIX
    assert(time_sec_label in df.columns)
    secs_to_trim = 5

    df = df[df[time_sec_label] > secs_to_trim]
    df = df[df[time_sec_label] < df[time_sec_label].iloc[-1] - secs_to_trim]  
    # 抛开前后5秒得
    return df


# ========================================    EAR_HAR_DATA        =============================
class EAR_HAR_DATA(BASE_DATA):

    """

    """

    def __init__(self, args):

        """


        """




        self.used_cols          = []
        self.col_names          = ['acc_x', 'acc_y', 'acc_z', 
                                   'gyro_x', 'gyro_y', 'gyro_z', 
                                   'heart_rate', 'body_temp']

        self.pos_filter         = None
        self.sensor_filter      = ["acc","gyro"]
        self.selected_cols      = None

        if args.pos_select is not None:
            if self.pos_filter is None:
                raise Exception('This dataset cannot be selected by sensor positions!')
            else:
                col_names = []
                for col in self.col_names:
                    selected = False
                    for pos in args.pos_select:
                        assert pos in self.pos_filter
                        if pos in col:
                            selected = True
                    if selected:
                        col_names.append(col)
                self.selected_cols = col_names

        if args.sensor_select is not None:
            if self.sensor_filter is None:
                raise Exception('This dataset cannot be selected by sensor types!')
            else:
                col_names = []
                if self.selected_cols is not None:
                    cols = self.selected_cols
                else:
                    cols = self.col_names
                for col in cols:

                    selected = False
                    for type in args.sensor_select:

                        assert type in self.sensor_filter
                        if type in col:
                            selected = True
                    if selected:
                        col_names.append(col)
                self.selected_cols = col_names


        self.label_map = [(0, 'Sitzen'), 
                          (1, 'Stehen'),
                          (2, 'Liegen'),
                          (3, 'Joggen'), 
                          (4, 'Gehen'),
                          (5, 'Essen'),
                          (6, 'Staubsaugen'), 
                          (7, 'Zähne putzen'),
                          (8, 'Treppenlaufen'),
                          (9, 'Fensterputzen'), 
                          (10, 'Spülmaschine ausräumen'),
                          (11, 'Hände waschen'),
                          (12, 'Yoga mit Video'), 
                          (13, 'Home Workout'),
                          (14, 'Obst oder Gemüse schneiden'),
                          (15, 'Lesen'), 
                          (16, 'Computerspielen'),
                          (17, 'Tippen am Computer'), 
                          (18, "Video anschauen im Sitzen"),
                          (19, "Geschirrspülen")
                         ]
        self.drop_activities    = []

        self.train_keys         = [1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                   12, 13, 14, 15, 16, 17, 18, 19, 20]
        self.vali_keys          = []
        self.test_keys          = []


        self.exp_mode           = args.exp_mode
        self.split_tag          = "sub"


        self.LOCV_keys = [[1,  2,  3], [ 4,  5,  6],  [7,  8,  9],  
                          [10, 12], [13, 14, 15], [16, 17, 18], 
                          [19, 20]]

        self.all_keys = [1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                         12, 13, 14, 15, 16, 17, 18, 19, 20]

        self.sub_ids_of_each_sub = {}


        self.file_encoding = {}  # no use
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(EAR_HAR_DATA, self).__init__(args)

    def load_all_the_data(self, root_path):
        print(" ----------------------- load all the data -------------------")
        df_dict = {}
        sub_list = os.listdir(root_path)

        for sub,sub_name in enumerate(sub_list):

            file_list = os.listdir(os.path.join(root_path,sub_name))
            file_list = [i for i in file_list if "csv" in i]

            for file_index, file in enumerate(file_list):
                file_activity_name = file.split("-")[0]
                assert file_activity_name in ["Sitzen","Stehen","Liegen", "Joggen", "Gehen", "Essen", 
                                              "Staubsaugen", "Zähne putzen", "Treppenlaufen", "Fensterputzen", 
                                              "Spülmaschine ausräumen", "Geschirrspülen", "Hände waschen", 
                                              "Yoga mit Video", "Home Workout", "Obst oder Gemüse schneiden", 
                                              "Lesen", "Computerspielen", "Tippen am Computer", "Video anschauen im Sitzen"]

                df = pd.read_csv(os.path.join(root_path, sub_name, file))


                del df["button"]
                del df["oxygen_saturation"]
                del df["pulse_rate"]


                df_earconnect = df[df["device_name"]=="earconnect"]
                df_esense     = df[df["device_name"]!="earconnect"]

                if df_esense.shape[0]==0:
                    continue


                df_esense = df_esense[['device_name', 'device_address', 'timestamp','acc_x', 'acc_y', 'acc_z',  'gyro_x', 'gyro_y', 'gyro_z']]        

                if df_earconnect.shape[0] > 10:
                    df_earconnect           = df_earconnect[['device_name', 'device_address', 'timestamp','heart_rate', 'body_temp']]
                    df_esense["heart_rate"] = merge_to_dataframe(df_esense, df_earconnect, "heart_rate")
                    df_esense["body_temp"]  = merge_to_dataframe(df_esense, df_earconnect, "body_temp")
                else:
                    df_esense["heart_rate"] = 90.1
                    df_esense["body_temp"]  = 36.5

                df_esense = _timestampToSI(df_esense)
                df_esense = _trim(df_esense)    
                df_esense.interpolate(method='linear', limit_direction='both',inplace=True)


                sub_data = df_esense[['acc_x', 'acc_y', 'acc_z',
                                       'gyro_x', 'gyro_y', 'gyro_z', 
                                       'heart_rate', 'body_temp']].copy()
                del df_esense



                sub_id               = "{}_{}".format(sub+1, file_index)
                sub_data["sub"]     = sub+1
                sub_data["sub_id"]  = sub_id

                sub_data["activity_id"] = file_activity_name

                if sub+1 not in self.sub_ids_of_each_sub.keys():
                        self.sub_ids_of_each_sub[sub+1] = []
                self.sub_ids_of_each_sub[sub+1].append(sub_id)

                df_dict[sub_id] = sub_data




        df_all = pd.concat(df_dict)
        df_all = df_all.set_index('sub_id')


        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        if self.selected_cols:
            df_all = df_all[self.selected_cols+["sub"]+["activity_id"]]
        else:
            df_all = df_all[self.col_names+["sub"]+["activity_id"]]


        label_mapping = {item[1]:item[0] for item in self.label_map}
        df_all["activity_id"] = df_all["activity_id"].map(label_mapping)

        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        df_all = df_all[self.col_names+["sub"]+["activity_id"]]

        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()
        # it is not necessary here
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 

        return data_x, data_y