import glob
from data_processing import (make_dict_from_csv, prepare_dataset, plot_beat_dyn, plot_dyn_change_curves,
                             add_dyn_values_for_markings_in_features_info, modify_continuous_features, 
                             modify_categorical_features, get_clusters, plot_dyn_with_markings_values_boxplots, get_names_from_mazurka)
import numpy as np
from collections import namedtuple
import tkinter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import copy

files_beat = glob.glob('../beat_time/*.csv')
files_dyn = glob.glob('../beat_dyn/*.csv')
files_mark = glob.glob('../markings/*.csv')
files_mark_dyn = glob.glob('../markings_dyn/*.csv')

Mazurka_info = prepare_dataset(files_beat, files_dyn, files_mark, files_mark_dyn)
# plot_dyn_with_markings_values_boxplots(Mazurka_info['M06-1'], 1, 3)
get_clusters(Mazurka_info['M63-3'])
import ipdb; ipdb.set_trace()
plot_dyn_with_markings_values_boxplots(Mazurka_info['M63-3'], ['Rubinstein(1966)', 'Magaloff'])


plot_beat_dyn(Mazurka_info['M63-3'][1:10])
# plot_dyn_change_curves(Mazurka_info['M63-3'][:10])



import ipdb; ipdb.set_trace()
# # Markings features for training model 
# features_info = make_dict_from_csv('../marking_INFO_dyn.csv')
# features_info_with_dyn_values = add_dyn_values_for_markings_in_features_info(Mazurka_info, features_info)


# features_info_continuous = modify_continuous_features(features_info_with_dyn_values, ['Dist_PR', 'Dist_N'])
# categories_mapping, final_features_info = modify_categorical_features(features_info_continuous, ['PR_M', 'N_M', 'Annot_PR', 'Annot_N', 'Annot_M', 'M'])



