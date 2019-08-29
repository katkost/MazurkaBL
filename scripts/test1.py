import glob
from data_processing import make_dict_from_csv, merge_dicts
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

files_beat = glob.glob('../beat_time/*.csv')
files_dyn = glob.glob('../beat_dyn/*.csv')
files_mark = glob.glob('../markings/*.csv')

def prepare_dataset(files_beat, files_dyn, markings):
    """
    Gets files from MazurkaBL folder
    returns dictionary: key: Mazurka ID,
                        value: list of namedtuple objects
                        Pianist: 'id beat dyn markings'
    """
    
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
                    tuple_all.append(Pianist(id = tub[0], beat = tub[1], dyn = tud[1], markings = make_dict_from_csv(mark)))
        Mazurka_info[M_ID] = tuple_all

    return Mazurka_info


def plot_beat_dyn(M_info):
    plt.figure()
    plt.subplot(211)
    for pianist in M_info:
        plt.plot([*pianist.beat.keys()], np.diff([*pianist.beat.values()]))
    
    plt.subplot(2,1,2)
    for pianist in M_info:
        plt.plot([*pianist.dyn.keys()], np.diff([*pianist.dyn.values()]))
    plt.show()
