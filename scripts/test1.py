import glob
from data_processing import make_dict_from_csv, merge_dicts, prepare_dataset, plot_beat_dyn
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

files_beat = glob.glob('../beat_time/*.csv')
files_dyn = glob.glob('../beat_dyn/*.csv')
files_mark = glob.glob('../markings/*.csv')

Mazurka_info = prepare_dataset(files_beat, files_dyn, files_mark)
plot_beat_dyn(Mazurka_info['M06-3'][4:6])