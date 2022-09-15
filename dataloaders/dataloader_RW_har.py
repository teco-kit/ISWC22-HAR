import re
import numpy as np
import pandas as pd
from glob import glob
import os
import zipfile
from io import BytesIO


from dataloaders.dataloader_base import BASE_DATA

# ========================================    REAL_WORLD_HAR_DATA        =============================
class REAL_WORLD_HAR_DATA(BASE_DATA):

    """


    """

    def __init__(self, args):

        """


        """

        self.used_cols = [ ]
        self.col_names    =   ['acc_x_chest',     'acc_y_chest',     'acc_z_chest', 
                               'acc_x_forearm',   'acc_y_forearm',   'acc_z_forearm',
                               'acc_x_head',      'acc_y_head',      'acc_z_head', 
                               'acc_x_shin',      'acc_y_shin',      'acc_z_shin', 
                               'acc_x_thigh',     'acc_y_thigh',     'acc_z_thigh', 
                               'acc_x_upperarm',  'acc_y_upperarm',  'acc_z_upperarm',
                               'acc_x_waist',     'acc_y_waist',     'acc_z_waist']

        # These two variables represent whether all sensors can be filtered according to position and sensor type
        # pos_filter ------- >  filter according to position
        # sensor_filter ----->  filter according to the sensor type
        self.pos_filter         = ["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"]
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
                          (0, "climbingdown"),
                          (1, "climbingup"),   
                          (2, "jumping" ),
                          (3, "lying"),
                          (4, "running"),
                          (5, "sitting"),
                          (6, "standing"),
                          (7, "walking")
                          ]

        self.drop_activities = []


        self.train_keys   = [1,  2,  3,  4,   6,  7,  8,  9,  11, 12, 13, 14]
        self.vali_keys    = []
        self.test_keys    = [5,10,15]

        self.exp_mode     = args.exp_mode
        self.split_tag = "sub"

        self.LOCV_keys = [[1],  [2],  [3],  [4],  [5],  [6],  [7],  [8],  [9], [10], [11], [12], [13], [14], [15]]
        self.all_keys = [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15]
        self.sub_ids_of_each_sub = {}

        self.file_encoding = {} # no use

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(REAL_WORLD_HAR_DATA, self).__init__(args)


    def check_rwhar_zip(self, path):
        # verify that the path is to the zip containing csv and not another zip of csv

        if any(".zip" in filename for filename in zipfile.ZipFile(path, "r").namelist()):
            # There are multiple zips in some cases
            with zipfile.ZipFile(path, "r") as temp:
                path = BytesIO(temp.read(
                    max(temp.namelist())))  # max chosen so the exact same acc and gyr files are selected each time (repeatability)
        return path

    def rwhar_load_csv(self, path):
        # Loads up the csv at given path, returns a dictionary of data at each location

        path = self.check_rwhar_zip(path)
        tables_dict = {}
        with zipfile.ZipFile(path, "r") as Zip:
            zip_files = Zip.namelist()

            for csv in zip_files:
                if "csv" in csv:
                    loc = csv[csv.rfind("_") + 1:csv.rfind(".")]

                    sensor = csv[:3]
                    prefix = sensor.lower() + "_"
                    table = pd.read_csv(Zip.open(csv))
                    table.rename(columns={"attr_x": prefix + "x"+ "_"+loc,
                                          "attr_y": prefix + "y"+ "_"+loc,
                                          "attr_z": prefix + "z"+ "_"+loc,
                                          "attr_time": "timestamp",
                                          }, inplace=True)
                    table.drop(columns="id", inplace=True)
                    tables_dict[loc] = table

        return tables_dict

    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -------------------")
        subject_dir = os.listdir(root_path)
        df_dict = {}

        for sub in subject_dir:
            if "proband" not in sub:
                continue

            sub_int = int(sub[7:]) # proband is 7 letters long so subject num is number following that

            for trial,activity in  enumerate([ "climbingdown","climbingup","jumping","lying","running","sitting","standing","walking"]): 

                activity_name = "_" + activity + "_csv.zip"
                path_acc = root_path + '/' + sub + "/acc" + activity_name 
				
                sub_dic = self.rwhar_load_csv(path_acc)

                resampled_sub_dic={}
                for key in sub_dic.keys():
                    temp = sub_dic[key].copy()
                    temp["timestamp"] = pd.to_datetime(temp["timestamp"], unit='ms')
                    temp =temp.set_index("timestamp")
                    temp = temp.resample('20ms').mean().interpolate(method='linear')
                    resampled_sub_dic[key] =temp

                sub_data = pd.DataFrame()
                for key in resampled_sub_dic.keys():
                    sub_data = pd.merge(sub_data,resampled_sub_dic[key],  left_index=True, right_index=True,how="outer")

                sub_data["activity_id"] = activity
                sub_data["sub"] = sub_int
                sub_id = "{}_{}".format(sub_int,trial)
                sub_data["sub_id"] = sub_id

                if sub_int not in self.sub_ids_of_each_sub.keys():
                    self.sub_ids_of_each_sub[sub_int] = []
                self.sub_ids_of_each_sub[sub_int].append(sub_id)
                df_dict[sub_id] = sub_data

        df_all = pd.concat(df_dict)
        df_all = df_all.dropna()
        df_all = df_all.set_index('sub_id')

        label_mapping = {item[1]:item[0] for item in self.label_map}

        df_all["activity_id"] = df_all["activity_id"].map(label_mapping)
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        if self.selected_cols:
            df_all = df_all[self.selected_cols+["sub"]+["activity_id"]]
        else:
            df_all = df_all[self.col_names+["sub"]+["activity_id"]]


        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]
        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 
        return data_x, data_y
