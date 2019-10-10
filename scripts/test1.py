import glob
from data_processing import (make_dict_from_csv, prepare_dataset, plot_beat_dyn, add_dyn_values_for_markings_in_features_info, 
                             modify_continuous_features, modify_categorical_features, plot_dyn_with_markings_values_boxplots)
import numpy as np
from collections import namedtuple
import tkinter
import matplotlib
matplotlib.use("macOSX")
import matplotlib.pyplot as plt
import pandas as pd
import copy

files_beat = glob.glob('../beat_time/*.csv')
files_dyn = glob.glob('../beat_dyn/*.csv')
files_mark = glob.glob('../markings/*.csv')
files_mark_dyn = glob.glob('../markings_dyn/*.csv')

Mazurka_info = prepare_dataset(files_beat, files_dyn, files_mark, files_mark_dyn)

plot_dyn_with_markings_values_boxplots(Mazurka_info['M06-1'], 1, 3)

import ipdb; ipdb.set_trace()
# Marckings features for training model 
features_info = make_dict_from_csv('../marking_INFO_dyn.csv')
features_info_with_dyn_values = add_dyn_values_for_markings_in_features_info(Mazurka_info, features_info)


features_info_continuous = modify_continuous_features(features_info_with_dyn_values, ['Dist_PR', 'Dist_N'])
categories_mapping, final_features_info = modify_categorical_features(features_info_continuous, ['PR_M', 'N_M', 'Annot_PR', 'Annot_N', 'Annot_M', 'M'])


# datapoints_for_pid9048 = copy.copy(final_features_info)
# datapoints_for_pid9058 = copy.copy(final_features_info)
# datapoints_for_pid9059 = copy.copy(final_features_info)
# datapoints_for_pid9104 = copy.copy(final_features_info)
# datapoints_for_pid9063 = copy.copy(final_features_info)
# datapoints_for_pid9072 = copy.copy(final_features_info)
# datapoints_for_pid9192 = copy.copy(final_features_info)
# datapoints_for_pid9118 = copy.copy(final_features_info)


# def get_dyn_values_for_pianist(feats, pID):
#     for ind, de in enumerate(feats['EN']):
#         for du in de:
#             print(pID, du[0])
#             if pID in du[0]:
#                 feats['EN'][ind] = du[1]
#             else: 
#                 feats['EN'][ind] = 'None'
#     return feats

# feats_pid9048 = get_dyn_values_for_pianist(datapoints_for_pid9048, 'pid9048')
# import ipdb; ipdb.set_trace()




