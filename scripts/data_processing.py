import numpy as np
import pandas as pd
import glob
import os
from collections import namedtuple
import matplotlib.pyplot as plt

Mazurka_ID = ['M06-1', 
              'M06-2', 
              'M06-3', 
              'M06-4', 
              'M07-1', 
              'M07-2', 
              'M07-3', 
              'M17-1', 
              'M17-2', 
              'M17-3', 
              'M17-4', 
              'M24-1', 
              'M24-2', 
              'M24-3', 
              'M24-4', 
              'M30-1', 
              'M30-2', 
              'M30-3', 
              'M30-4', 
              'M33-1', 
              'M33-2', 
              'M33-3', 
              'M33-4', 
              'M41-1', 
              'M41-2', 
              'M41-3', 
              'M41-4', 
              'M50-1', 
              'M50-2', 
              'M50-3', 
              'M56-1', 
              'M56-2', 
              'M56-3', 
              'M59-1', 
              'M59-2', 
              'M59-3', 
              'M63-1', 
              'M63-2', 
              'M63-3', 
              'M67-1', 
              'M67-2', 
              'M67-3', 
              'M67-4', 
              'M68-1', 
              'M68-2', 
              'M68-3']

######### File parsing #########

def add_measure_number_and_beat_number_on_header(file):
    df = pd.read_csv(file)
    new_columns = df.columns.values

    new_columns[0] = 'measure_number'
    new_columns[1] = 'beat_number'
    df.columns = new_columns
    df.to_csv(file)
    return

def make_dict_from_csv(file):
    df = pd.read_csv(file)
    return df.to_dict()

######## Data processing ########

def merge_dicts(dict1, dict2):
    return {**dict1, **dict2}

def norm_by_max(values):
    return [x / max(values) for x in values]


def prepare_dataset(files_beat, files_dyn, files_mark):
    """
    Gets files from MazurkaBL folder
    returns dictionary: key: Mazurka ID,
                        value: list of namedtuple objects
                        Pianist: 'id beat dyn markings'
    """

    Mazurka_info = {}

    Pianist = namedtuple('Pianist', 'id beat dyn markings')

    files_ID = []
    for M_ID in Mazurka_ID:
        files_ID.append([(M_ID, fb, fd, fm) for fb in files_beat if M_ID in fb for fd in files_dyn if M_ID in fd for fm in files_mark if M_ID in fm])

    for [(M_ID, fb, fd, mark)] in files_ID:
        tuple_beats = [tuple(x) for x in make_dict_from_csv(fb).items()]
        tuple_dyns = [tuple(x) for x in make_dict_from_csv(fd).items()]
        tuple_all = []
        for tub in tuple_beats:
            for tud in tuple_dyns:
                if tub[0] == tud[0]:
                    tuple_all.append(Pianist(id = tub[0], beat = [*tub[1].values()], dyn = norm_by_max([*tud[1].values()]), markings = make_dict_from_csv(mark)))
        Mazurka_info[M_ID] = tuple_all
    return Mazurka_info

def get_markings_dyn_values(Mazurka_info)

    return

####### Vector processing #######




######## Plotting tools #########
    
def plot_beat_dyn(M_info_pianist):
    plt.figure(figsize=(18, 16), dpi= 80)
    plt.subplot(211)
    for pianist in M_info_pianist:
        plt.plot(range(len(pianist.beat) -1 ), norm_by_max(np.diff(pianist.beat)))
        plt.title('Inter-beat-intervals in Mazurka recording', fontsize=14)
        plt.xlabel('Score beats', fontsize=14)
        plt.xticks([v[0] for v in [list(mark.values()) for mark in list(pianist.markings.values())]], 
                   [m.split('.')[0] for m in [*pianist.markings.keys()]], rotation='vertical', fontsize=14) 
        plt.ylabel('IBIs (normalised)', fontsize=14)
        plt.tight_layout()

    plt.subplot(212)
    for pianist in M_info_pianist:
        plt.plot(range(len(pianist.dyn)), pianist.dyn)
        plt.title('Dynamics per score beat in Mazurka recording', fontsize=14)
        plt.xlabel('Score beats', fontsize=14)
        plt.xticks([v[0] for v in [list(mark.values()) for mark in list(pianist.markings.values())]], 
                   [m.split('.')[0] for m in [*pianist.markings.keys()]], rotation='vertical', fontsize=14) 
        plt.ylabel('Dynamics in smoothed sones (normalised)', fontsize=14)
        plt.tight_layout()
    plt.show()

