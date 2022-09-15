import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA
# TODO the cols ! name
# ========================================      Opportunity_HAR_DATA         =============================
class Opportunity_HAR_DATA(BASE_DATA):
    """
    OPPORTUNITY Dataset for Human Activity Recognition from Wearable, Object, and Ambient Sensors
	
    Brief Description of the Dataset:
    ---------------------------------
    Each .dat file contains a matrix of data in text format. 
    Each line contains the sensor data sampled at a given time (sample rate: 30Hz). 
    For more detail . please reffer to the docomentation.html
    """
    def __init__(self, args):

        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （sample rate: 30Hz)）
            wavelet : Methods of wavelet transformation

        """

        # In this documents in doc/documentation.html, all columns definition coulde be found   (or in the column_names)
        # the sensors between 134 and 248 are amounted on devices, so they will not to be considered
        # Specifically, the following columns were used for the challenge: 
        # =============================================================
        # 1-37, 38-46, 51-59, 64-72, 77-85, 90-98, 103-134, 244, 250.
        # 0 milisconds
        all_columns = ["timestamp",
                       # Position : RKN
                       "acc_x_RKN", "acc_y_RKN", "acc_z_RKN", 
                       # Position : HIP			   
                       "acc_x_HIP", "acc_y_HIP", "acc_z_HIP",
                       # Position : LUA^	
                       "acc_x_LUA^", "acc_y_LUA^", "acc_z_LUA^",
                       # Position : RUA-
                       "acc_x_RUA-", "acc_y_RUA-", "acc_z_RUA-",
                       # Position : LH
                       "acc_x_LH", "acc_y_LH", "acc_z_LH",
                       # Position : BACK
                       "acc_x_BACK", "acc_y_BACK", "acc_z_BACK",
                       # Position : RKN-
                       "acc_x_RKN-", "acc_y_RKN-", "acc_z_RKN-", 
                       # Position : RWR
                       "acc_x_RWR", "acc_y_RWR", "acc_z_RWR",
                       # Position : RUA^
                       "acc_x_RUA^", "acc_y_RUA^", "acc_z_RUA^",
                       # Position : LUA-
                       "acc_x_LUA-", "acc_y_LUA-", "acc_z_LUA-", 
                       # Position : LWR
                       "acc_x_LWR", "acc_y_LWR", "acc_z_LWR",
                       # Position : RH
                       "acc_x_RH", "acc_y_RH", "acc_z_RH",
                       # Position : IBack
                       "acc_x_IBack", "acc_y_IBack", "acc_z_IBack", "gyro_x_IBack", "gyro_y_IBack", "gyro_z_IBack","magnetic_x_IBack", "magnetic_y_IBack", "magnetic_z_IBack",
                       "Quaternion_1_IBack", "Quaternion_2_IBack", "Quaternion_3_IBack", "Quaternion_4_IBack", 
                       # Position : IRUA
                       "acc_x_IRUA", "acc_y_IRUA", "acc_z_IRUA", "gyro_x_IRUA", "gyro_y_IRUA", "gyro_z_IRUA","magnetic_x_IRUA", "magnetic_y_IRUA", "magnetic_z_IRUA",
                       "Quaternion_1_IRUA", "Quaternion_2_IRUA", "Quaternion_3_IRUA", "Quaternion_4_IRUA", 
                       # Position : IRLA
                       "acc_x_IRLA", "acc_y_IRLA", "acc_z_IRLA", "gyro_x_IRLA", "gyro_y_IRLA", "gyro_z_IRLA","magnetic_x_IRLA", "magnetic_y_IRLA", "magnetic_z_IRLA",
                       "Quaternion_1_IRLA", "Quaternion_2_IRLA", "Quaternion_3_IRLA", "Quaternion_4_IRLA", 
                       # Position : ILUA
                       "acc_x_ILUA", "acc_y_ILUA", "acc_z_ILUA", "gyro_x_ILUA", "gyro_y_ILUA", "gyro_z_ILUA","magnetic_x_ILUA", "magnetic_y_ILUA", "magnetic_z_ILUA",
                       "Quaternion_1_ILUA", "Quaternion_2_ILUA", "Quaternion_3_ILUA", "Quaternion_4_ILUA", 
                       # Position : ILLA
                       "acc_x_ILLA", "acc_y_ILLA", "acc_z_ILLA", "gyro_x_ILLA", "gyro_y_ILLA", "gyro_z_ILLA","magnetic_x_ILLA", "magnetic_y_ILLA", "magnetic_z_ILLA",
                       "Quaternion_1_ILLA", "Quaternion_2_ILLA", "Quaternion_3_ILLA", "Quaternion_4_ILLA", 
                       # Position : LSHOE
                       "EU_x_LSHOE",  "EU_y_LSHOE", "EU_z_LSHOE", "acc_x_Nav_LSHOE", "acc_y_Nav_LSHOE", "acc_z_Nav_LSHOE", "acc_x_body_LSHOE", "acc_y_body_LSHOE", "acc_z_body_LSHOE",
                       "AngVel_x_body_LSHOE","AngVel_y_body_LSHOE","AngVel_z_body_LSHOE","AngVel_x_Nav_LSHOE","AngVel_y_Nav_LSHOE","AngVel_z_Nav_LSHOE","Compass_none_LSHOE",
                       # Position : RSHOE
                       "EU_x_RSHOE",  "EU_y_RSHOE", "EU_z_RSHOE", "acc_x_Nav_RSHOE", "acc_y_Nav_RSHOE", "acc_z_Nav_RSHOE", "acc_x_body_RSHOE", "acc_y_body_RSHOE", "acc_z_body_RSHOE",
                       "AngVel_x_body_RSHOE","AngVel_y_body_RSHOE","AngVel_z_body_RSHOE","AngVel_x_Nav_RSHOE","AngVel_y_Nav_RSHOE","AngVel_z_Nav_RSHOE","Compass_none_RSHOE",
                       # Position : Cup  + 。。。。。
                       "drop", "drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop",
                       "drop", "drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop",
                       "drop", "drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop",
                       "drop", "drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop",
                       "drop", "drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop","drop",
                       "drop", "drop","drop","drop","drop","drop","drop","drop","drop",
                       # Label
                       "Locomotion","HL_Activity","LL_Left_Arm","LL_Left_Arm_Object","LL_Right_Arm","ML_Both_Arms","activity_id"]

        self.used_cols = [#1,  2,   3, # Accelerometer RKN^ 
                          #4,  5,   6, # Accelerometer HIP
                          #7,  8,   9, # Accelerometer LUA^ 
                          #10, 11,  12, # Accelerometer RUA_
                          #13, 14,  15, # Accelerometer LH
                          #16, 17,  18, # Accelerometer BACK
                          #19, 20,  21, # Accelerometer RKN_ 
                          #22, 23,  24, # Accelerometer RWR
                          #25, 26,  27, # Accelerometer RUA^
                          #28, 29,  30, # Accelerometer LUA_ 
                          #31, 32,  33, # Accelerometer LWR
                          #34, 35,  36, # Accelerometer RH
                          37, 38,  39, 40, 41, 42, 43, 44, 45, # InertialMeasurementUnit BACK
                          50, 51,  52, 53, 54, 55, 56, 57, 58, # InertialMeasurementUnit RUA
                          63, 64,  65, 66, 67, 68, 69, 70, 71, # InertialMeasurementUnit RLA 
                          76, 77,  78, 79, 80, 81, 82, 83, 84, # InertialMeasurementUnit LUA
                          89, 90,  91, 92, 93, 94, 95, 96, 97,  # InertialMeasurementUnit LLA 45
                          102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, # InertialMeasurementUnit L-SHOE
                          118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, # InertialMeasurementUnit R-SHOE 32
                          249  # Label
                         ]


        self.col_names    =  []
        for index in self.used_cols:
            self.col_names.append(all_columns[index])
        # These two variables represent whether all sensors can be filtered according to position and sensor type
        # pos_filter ------- >  filter according to position
        # sensor_filter ----->  filter according to the sensor type
        self.pos_filter         = None
        self.sensor_filter      = ["acc","gyro","magnetic","EU","AngVel"]


        # selected_cols will be updated according to user settings. User have to set -- args.pos_select, args.sensor_select---
        self.selected_cols      = None
        # Filtering channels according to the Position
        self.selected_cols      = self.Sensor_filter_acoording_to_pos_and_type(args.pos_select, self.pos_filter, self.col_names, "position")
        # Filtering channels according to the Sensor Type
        if self.selected_cols is None:
            self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.col_names, "Sensor Type")
        else:
            self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.selected_cols, "Sensor Type")

        self.label_map = [(0,      'Other'),
                          (406516, 'Open Door 1'),
                          (406517, 'Open Door 2'),
                          (404516, 'Close Door 1'),
                          (404517, 'Close Door 2'),
                          (406520, 'Open Fridge'),
                          (404520, 'Close Fridge'),
                          (406505, 'Open Dishwasher'),
                          (404505, 'Close Dishwasher'),
                          (406519, 'Open Drawer 1'),
                          (404519, 'Close Drawer 1'),
                          (406511, 'Open Drawer 2'),
                          (404511, 'Close Drawer 2'),
                          (406508, 'Open Drawer 3'),
                          (404508, 'Close Drawer 3'),
                          (408512, 'Clean Table'),
                          (407521, 'Drink from Cup'),
                          (405506, 'Toggle Switch')]


        self.drop_activities = []

        self.train_keys   = [11,12,13,14,15,16,
                             21,22,23,      26,
                             31,32,33,      36,
                             41,42,43,44,45,46]
        # 'S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat',  'S1-ADL5.dat', 'S1-Drill.dat', # subject 1
        # 'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat',                                'S2-Drill.dat', # subject 2
        # 'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat',                                'S3-Drill.dat'  # subject 3
        # 'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat'   'S4-ADL5.dat', 'S4-Drill.dat'] # subject 4
        self.vali_keys    = [ ]
        # 'S2-ADL4.dat', 'S2-ADL5.dat','S3-ADL4.dat', 'S3-ADL5.dat'
        self.test_keys    = [24,25,34,35]

        self.exp_mode     = args.exp_mode
        if self.exp_mode == "LOCV":
            self.split_tag = "sub"
        else:
            self.split_tag = "sub_id"


        self.LOCV_keys = [[1],[2],[3],[4]]
        self.all_keys = [1,2,3,4]
        self.sub_ids_of_each_sub = {}

        self.file_encoding = {'S1-ADL1.dat':11, 'S1-ADL2.dat':12, 'S1-ADL3.dat':13, 'S1-ADL4.dat':14, 'S1-ADL5.dat':15, 'S1-Drill.dat':16,
                              'S2-ADL1.dat':21, 'S2-ADL2.dat':22, 'S2-ADL3.dat':23, 'S2-ADL4.dat':24, 'S2-ADL5.dat':25, 'S2-Drill.dat':26,
                              'S3-ADL1.dat':31, 'S3-ADL2.dat':32, 'S3-ADL3.dat':33, 'S3-ADL4.dat':34, 'S3-ADL5.dat':35, 'S3-Drill.dat':36,
                              'S4-ADL1.dat':41, 'S4-ADL2.dat':42, 'S4-ADL3.dat':43, 'S4-ADL4.dat':44, 'S4-ADL5.dat':45, 'S4-Drill.dat':46}

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(Opportunity_HAR_DATA, self).__init__(args)



    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -------------------")
	 
        file_list = os.listdir(root_path)
        file_list = [file for file in file_list if file[-3:]=="dat"] # in total , it should be 24

        assert len(file_list) == 24

        df_dict = {}

        for file in file_list:
            sub_data = pd.read_table(os.path.join(root_path,file), header=None, sep='\s+')
            sub_data =sub_data.iloc[:,self.used_cols]
            sub_data.columns = self.col_names

            # TODO check missing labels? 
            sub_data = sub_data.interpolate(method='linear', limit_direction='both')

            sub = int(file[1])
            sub_data['sub_id'] = self.file_encoding[file]
            sub_data["sub"] = sub


            if sub not in self.sub_ids_of_each_sub.keys():
                self.sub_ids_of_each_sub[sub] = []
            self.sub_ids_of_each_sub[sub].append(self.file_encoding[file])

            df_dict[self.file_encoding[file]] = sub_data
            
        # all data
        df_all = pd.concat(df_dict)
        df_all = df_all.set_index('sub_id')

        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        if self.selected_cols:
            df_all = df_all[self.selected_cols+["sub"]+["activity_id"]]
        else:
            df_all = df_all[self.col_names[:-1]+["sub"]+["activity_id"]]

        # label transformation
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)


        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 

        return data_x, data_y

