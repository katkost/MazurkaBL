import numpy as np
import pandas as pd
import glob
import os
from collections import namedtuple, Counter
import tkinter
import matplotlib
matplotlib.use("Agg")
from mpld3 import plugins
from mpld3.utils import get_id
import matplotlib.pyplot as plt
from functools import partial

from sklearn.cluster import KMeans

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

df = pd.read_csv('../pianistID_name.csv')
data_map = df.to_dict('split')['data']

# Name_from_ID = get_name_ID_map('../pianistID_name.csv')

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
    return df.to_dict('list')

def make_dict_from_csv_without_header(file):
    df = pd.read_csv(file, header=None, usecols=[1,2], names=['location', 'num_cp'])
    return df.to_dict('list')

def make_dict_from_csv_without_header2(file):
    df = pd.read_csv(file, header=None, usecols=[0,1], names=['time_frame_in_sec', 'sone_value'])
    return df.to_dict('list')

def get_sones(folder, mazurka_id):
    return sorted(glob.glob(os.path.join(folder, mazurka_id + '/*.csv')))

def read_txt_for_cps(file):
    df = pd.read_csv(file, sep=" ", header=None)
    data = df.to_dict('list')
    return [list(map(int, a[1:-1].split(','))) for a in data[0]]

######## Data processing ########

def get_name_from_ID_map(ID, data_map):
    name = ''
    for d in data_map:
        if d[1] in ID:
            return d[0]
    return name

def get_names_from_mazurka(M_info):
    names = []
    for m_info in M_info:
        names.append(get_name_from_ID_map(m_info.id, data_map))
    return names

def norm_by_max(values):
    return [np.round(x / max(values), 3) for x in values]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_seconds_for_change_points(beats, change_points):
    return [beats[cp-1] for cp in change_points] 

def get_change_point_positions_in_sone_data(sones_locs, seconds_for_cp):
    return list(map(partial(find_nearest, sones_locs), seconds_for_cp))

def remove_redundant_keys(dictionary, keys_to_remove):
    """
    Removes redundant information from the dictionaries
    """
    for k in keys_to_remove:
        del dictionary[k]
    return dictionary


def prepare_dataset(files_beat, files_dyn, files_mark, files_mark_dyn):
    """
    Gets files from MazurkaBL folder
    returns dictionary: key: Mazurka ID,
                        value: list of namedtuple objects
                        Pianist: 'id beat dyn markings markings_dyn dyns_in_markings_dyn dyn_change'
    """

    Mazurka_info = {}

    Pianist = namedtuple('Pianist', 'id beat dyn markings markings_dyn dyns_in_markings_dyn dyn_change')

    files_ID = []
    print ('Retrieving information from the .csv files...')
    for M_ID in Mazurka_ID:
        files_ID.append([(M_ID, fb, fd, fm, fmd) for fb in files_beat if M_ID in fb 
                                                 for fd in files_dyn if M_ID in fd 
                                                 for fm in files_mark if M_ID in fm 
                                                 for fmd in files_mark_dyn if M_ID in fmd])

    for [(M_ID, fb, fd, mark, mark_dyn)] in files_ID:
        beats = make_dict_from_csv(fb)
        dyns = make_dict_from_csv(fd)

        beats = remove_redundant_keys(beats, ['Unnamed: 0', 'measure_number', 'beat_number'])
        dyns = remove_redundant_keys(dyns, ['Unnamed: 0', 'measure_number', 'beat_number'])
        
        tuple_all = []
        for id1, vals_beat in beats.items():
            for id2, vals_dyn in dyns.items():
                if id1 == id2:
                    # round beat and dyn values
                    vals_beat = [np.round(v, 3) for v in vals_beat]
                    vals_dyn = [np.round(v, 3) for v in vals_dyn]

                    # store info about dyns in markings
                    markings_dyn_positions = [int(v[0]) for v in make_dict_from_csv(mark_dyn).values()]
                    pianist_dyn_mark_values = list(map(partial(get_marking_dyn_value, vals_dyn), markings_dyn_positions))      

                    tuple_all.append((Pianist(id = id1, 
                                              beat = vals_beat, 
                                              dyn = norm_by_max(vals_dyn), 
                                              markings = make_dict_from_csv(mark), 
                                              markings_dyn = make_dict_from_csv(mark_dyn),
                                              dyns_in_markings_dyn = pianist_dyn_mark_values,
                                              dyn_change = list(map(map_discrete_variation_value, values_pairs(pianist_dyn_mark_values)))
                                              )))
        Mazurka_info[M_ID] = tuple_all
    print ("Done!")
    return Mazurka_info

def prepare_dataset_change_points(folder_sones, files_cp_per_recording_in_mazurka, files_total_cp, files_mark):
    """
    Gets files from MazurkaBL folder
    returns dictionary: key: Mazurka ID,
                        value: list of namedtuple objects
                        Pianist: 'id sones cp dyn cp_total_in_mazurka'
    """

    Mazurka_info = {}

    Pianist = namedtuple('Pianist', 'id sones cp cp_total_in_mazurka markings')

    tuple_all = []
    print ('Retrieving information from the change points data files...')
    # exclude 'M63-2' for the change-point data
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
                'M63-3', 
                'M67-1', 
                'M67-2', 
                'M67-3', 
                'M67-4', 
                'M68-1', 
                'M68-2', 
                'M68-3']
    for M_ID in Mazurka_ID:
        tuple_all = []
        # Get total change points in Mazurka
        for cp_total_file in files_total_cp:
            if M_ID in cp_total_file:
                cp_total_in_m = make_dict_from_csv_without_header(cp_total_file)

        # Get markings in Mazurka
        for marking_file in files_mark:
            if M_ID in marking_file:
                markings = make_dict_from_csv(marking_file)

        # Get change-point lists per recording
        for cp_file in files_cp_per_recording_in_mazurka:
            if M_ID in cp_file:
                rec_cps_in_mazurka = read_txt_for_cps(cp_file)

        # Get sone values per recording
        sone_files = get_sones(folder_sones, M_ID)

        # We assume the lists in files_cp_per_recording_in_mazurka are in alphabetical order
        for ind, sf in enumerate(sone_files):
            sones = make_dict_from_csv_without_header2(sf)
            tuple_all.append(Pianist(id = sf.split('/')[3].split('Ntot')[0], 
                                    sones = sones, 
                                    cp = rec_cps_in_mazurka[ind], 
                                    cp_total_in_mazurka = cp_total_in_m, 
                                    markings = markings))

        Mazurka_info[M_ID] = tuple_all
    print ("Done!")
    return Mazurka_info


####### clustering task #######

def values_pairs(list):
    return [(first, second) for first, second in zip(list, list[1:])]

def map_discrete_variation_value(values):
    # Pairwise comparison of sequential data points
    # by creating four equally spaced discretization points.
    v1 = values[0]
    v2 = values[1]
    if v2-v1 > 0.5: return 2
    elif v2-v1 > 0 and v2-v1 <= 0.5: return 1
    elif v2-v1 == 0: return 0
    elif v2-v1 >= -0.5 and v2-v1 < 0: return -1
    elif v2-v1 < -0.5: return -2

def get_clusters_and_print_outlier_cluster_names(M_info, num_clusters):
    data = []

    for pianist in M_info:
        data.append(pianist.dyn_change)
    k_means_model = KMeans(n_clusters=num_clusters, max_iter=150, n_init=5, verbose=0).fit(data)    
    labels = k_means_model.labels_
    cl_centers = k_means_model.cluster_centers_

    counter = Counter(labels)
    outlier_cluster = min(counter, key=counter.get)
    outlier_pianists = []
    for ind, cluster_point in enumerate(labels):
        if cluster_point == outlier_cluster:
            outlier_pianists.append(get_name_from_ID_map(M_info[ind].id, data_map))
    print ('Pianists in outlier (smallest) cluster: ', outlier_pianists)
    fig, ax = plt.subplots(figsize=(12, 6), dpi= 80)
    for x in list(range(num_clusters)):
        ax.plot(cl_centers[x], label='Cluster' + str(x) + '(' + str(counter[x]) + ')')
    plt.xlabel('Score markings transition', fontsize=14)
    plt.xticks(range(len(M_info[0].dyn_change)), 
                [str(v) for v in values_pairs([v[1] for v in M_info[0].markings_dyn.values()])], 
                rotation='vertical', 
                fontsize=14)     
    
    ax.legend()


####### Features calling for prediction task #######

def get_marking_dyn_value(dyns, idx):
    # Extract the dynamic value in marking position
    # taking into account the dynamic values of the 
    # current position and the following two.
    # this function does not consider the possibility one 
    # marking to be located very close to another. 
    _idx = int(idx) - 1
    if _idx == len(dyns) - 1:
        dyn_value = dyns[_idx]
    elif _idx == len(dyns) - 2:
        dyn_value = np.mean([dyns[_idx], dyns[_idx + 1]])
    else:
        dyn_value = np.mean([dyns[_idx], dyns[_idx + 1], dyns[_idx + 2]])
    return np.round(dyn_value, 3)

def get_dyn_values_per_pianist(markings_positions, p_info):
    pianist_dyn_values = list(map(partial(get_marking_dyn_value, p_info.dyn), markings_positions))
    return (p_info.id, pianist_dyn_values)

def add_dyn_values_for_markings_in_features_info(Mazurka_info, features_info):
    ''' Output: The features info table (rows: marking, columns: features values). 
    Tag 'EN' includes list of tuples of: (pianist_ID, dyn_value) for the 
    particular marking.
    '''
    print ('Adding dynamic values of markings into feature info...')

    # WARNING: remove mazurkas that are not used for the network
    del Mazurka_info['M06-4']
    del Mazurka_info['M63-2']

    # First store tuples (M_ID, [(pianist_ID, list of dyn values for markings), ...])
    mid_pid_dyn_value = []
    row_ind = 0

    for M_ID, ps in Mazurka_info.items():
        # get list of marking positions
        markings_positions = [int(v[0]) for v in [*ps[0].markings_dyn.values()]]
        # get list of marking labels
        markings_labels = [v[1] for v in [*ps[0].markings_dyn.values()]]

        pid_dyn_values = list(map(partial(get_dyn_values_per_pianist, markings_positions), ps))
        mid_pid_dyn_value.append((M_ID, pid_dyn_values))

        # Then create list of tuples [(pianist_ID, dyn value of marking), ...] 
        # and add it to corresponding row of features_info
        for ind, ml in enumerate(markings_labels):
            assert features_info['M'][row_ind] == ml
            assert features_info['M_ID'][row_ind] == M_ID
            features_info['EN'][row_ind] = [(v[0], v[1][ind]) for v in pid_dyn_values]
            row_ind += 1
        
    return features_info

####### Vector processing #######

def replace_Nones_and_make_values_int(values):
    values = [int(x) if x != 'None' else 0 for x in values]
    return values

def replace_names_to_numbers(values, tag):
    if tag == 'N_M' or tag == 'PR_M' or tag == 'M':
        mapping = ['None', 'pp', 'p', 'mf', 'f', 'ff']
    else: 
        mapping = list(set(values))
    new_values = [mapping.index(v) for v in values]

    return mapping, new_values

def modify_continuous_features(features_info, tags):
    '''
    input: 
    features_info: 
    the features info table (rows: marking, columns: features values). 
    Tag 'EN' includes list of tuples of: (pianist_ID, dyn_value) for 
    the particular marking.
    tags: 
    the feature name(s) that we want to modify.
    '''
    for t in tags:
        features_info[t] = norm_by_max(replace_Nones_and_make_values_int(features_info[t]))
    return features_info

def modify_categorical_features(features_info, tags):
    from keras.utils.np_utils import to_categorical
    categories_mapping = {}
    mapping_markings = ['None', 'pp', 'p', 'mf', 'f', 'ff']
    for t in tags:
        mapping, numerical_labels = replace_names_to_numbers(features_info[t], t)
        categories_mapping[t] = mapping 
        categorical_labels = to_categorical(numerical_labels, num_classes=len(mapping))
    
        features_info[t] = categorical_labels

    # from keras.utils.np_utils import to_categorical   
    # categorical_labels = to_categorical(int_labels, num_classes=3)

    ## first encode the features in integers
    # from sklearn.preprocessing import LabelEncoder
    # le = LabelEncoder()
    # X_train_le = le.fit_transform(X_train)
    ## then make them one-hot
    # from sklearn.preprocessing import MultiLabelBinarizer
    # one_hot = MultiLabelBinarizer()
    # one_hot.fit_transform(y)

    return categories_mapping, features_info


######## Plotting tools #########
    
def plot_beat_dyn(M_info_pianist):
    plt.figure(figsize=(18, 12), dpi= 80)

    plt.subplot(211)

    for pianist in M_info_pianist:
        plt.plot(range(len(pianist.beat)-1), norm_by_max(np.diff(pianist.beat)))
        plt.xticks([int(v[0]) for v in [*pianist.markings.values()]], 
                   [m.split('.')[0] for m in list(pianist.markings.keys())], rotation='vertical', fontsize=12) 

    plt.title('Inter-beat-intervals in Mazurka recording', fontsize=14)
    plt.xlabel('Score beats', fontsize=14)
    plt.ylabel('IBIs (normalised)', fontsize=14)

    plt.subplot(212)
    for pianist in M_info_pianist:
        plt.plot(range(len(pianist.dyn)), pianist.dyn)
        plt.xticks([int(v[0]) for v in [*pianist.markings.values()]], 
                   [m.split('.')[0] for m in list(pianist.markings.keys())], rotation='vertical', fontsize=12) 

    plt.title('Dynamics per score beat in Mazurka recording', fontsize=14)
    plt.xlabel('Score beats', fontsize=14)
    plt.ylabel('Dynamics in smoothed sones (normalised)', fontsize=12)
    plt.tight_layout()
    plt.savefig('bubu.png')

def plot_dyn_with_markings_values_boxplots(M_info, idxs_or_names_list):
    # Get dyn.values per marking for data in boxplot

    # get list of marking positions
    markings_positions = [int(v[0])-1 for v in [*M_info[0].markings_dyn.values()]]

    pid_dyn_values = list(map(partial(get_dyn_values_per_pianist, markings_positions), M_info))

    plt.figure(figsize=(12, 6), dpi= 80)
    m=0

    for mp in markings_positions:
        values = [v[1][m] for v in pid_dyn_values]
        plt.boxplot(values, positions=[mp])
        m+=1
    if all(isinstance(n, int) for n in idxs_or_names_list):
        M_info_pianist = [M_info[x] for x in idxs_or_names_list]

        for pianist in M_info_pianist:     
            plt.plot(range(len(pianist.dyn)), pianist.dyn, alpha=0.6)
            plt.title('Dynamics per score beat in Mazurka recording', fontsize=14)
            plt.xlabel('Score beats', fontsize=14)
            plt.xticks([int(v[0])-1 for v in [*pianist.markings.values()]], 
                    [m.split('.')[0] for m in list(pianist.markings.keys())], rotation='vertical', fontsize=12) 
            plt.ylabel('Dynamics in smoothed sones (normalised)', fontsize=12)
            plt.ylim(0, 1)
            plt.xlim(-2, len(pianist.dyn)+2)
        plt.show()    

    else:
        for name in idxs_or_names_list:
            for mip in M_info:
                if get_name_from_ID_map(mip.id, data_map) == name:
                    plt.plot(range(len(mip.dyn)), mip.dyn, alpha=0.6)
                    plt.title('Dynamics per score beat in Mazurka recording', fontsize=14)
                    plt.xlabel('Score beats', fontsize=14)
                    plt.xticks([int(v[0])-1 for v in [*mip.markings.values()]], 
                                [m.split('.')[0] for m in list(mip.markings.keys())], rotation='vertical', fontsize=12) 
                    plt.ylabel('Dynamics in smoothed sones (normalised)', fontsize=12)
                    plt.ylim(0, 1)
                    plt.xlim(-2, len(mip.dyn)+2)
        plt.show()      

def plot_dyn_change_curves(M_info_pianist):
    plt.figure(figsize=(11, 5))
    for pianist in M_info_pianist:
        plt.plot(range(len(pianist.dyn_change)), pianist.dyn_change)
        plt.title('Dynamics transitions', fontsize=14)
        plt.xlabel('Score markings transition', fontsize=14)
        plt.xticks(range(len(pianist.dyn_change)), 
                   [str(v) for v in values_pairs([v[1] for v in pianist.markings_dyn.values()])], 
                   rotation='vertical', 
                   fontsize=14) 
        plt.ylabel('Discrete transition type', fontsize=14)
        plt.yticks([-2, -1, 1, 2], ['a lot softer', 'a bit softer', 'a bit louder', 'a lot louder'])
    plt.tight_layout()
    plt.show()
    return

def plot_sones_with_cp(M_info_with_cp, M_info, idxs_or_names_list):
    plt.figure(figsize=(12, 6), dpi= 80)

    if all(isinstance(n, int) for n in idxs_or_names_list):
        M_info_pianist_cp = [M_info_with_cp[x] for x in idxs_or_names_list]
        M_info_pianist_general = [M_info[x] for x in idxs_or_names_list]
        
        for pianist, pianist_general in zip(M_info_pianist_cp, M_info_pianist_general):
            assert pianist.id == pianist_general.id
            plt.plot(pianist.sones['sone_value'])
            seconds_for_change_points = get_seconds_for_change_points(pianist_general.beat, pianist.cp)
            change_point_positions_in_sone_data = get_change_point_positions_in_sone_data(pianist.sones['time_frame_in_sec'], seconds_for_change_points)
            print (change_point_positions_in_sone_data)
            plt.vlines(change_point_positions_in_sone_data, 0, max(pianist.sones['sone_value']), color='#39D2B4', alpha=0.8, linewidth=1.2, label='Onsets')
            plt.xlabel('Frames')
            plt.ylabel('Sones')
    else:
        for name in idxs_or_names_list:
            for mip in M_info_with_cp:
                if get_name_from_ID_map(mip.id, data_map) == name:
                    plt.plot(mip.sones['sone_value'])
                    seconds_for_change_points = get_seconds_for_change_points(pianist_general.beat, pianist.cp)
                    change_point_positions_in_sone_data = get_change_point_positions_in_sone_data(pianist.sones['time_frame_in_sec'], seconds_for_change_points)
                    plt.vlines(change_point_positions_in_sone_data, 0, max(pianist.sones['sone_value']), color='#39D2B4', alpha=0.8, linewidth=1.2, label='Onsets')
                    plt.xlabel('Frames')
                    plt.ylabel('Sones')
    plt.tight_layout()
    plt.show()               

def plot_total_cps(M_info_with_cp, Mazurka_ID):
    
    total_cps = M_info_with_cp[Mazurka_ID][0].cp_total_in_mazurka
    import ipdb; ipdb.set_trace()
    plt.figure(figsize=(12, 6), dpi= 80)
    plt.bar(total_cps['location'], total_cps['num_cp'], align='center')
    plt.xticks([int(v[0]) for v in [*M_info_with_cp[Mazurka_ID][0].markings.values()]], 
                [m.split('.')[0] for m in list(M_info_with_cp[Mazurka_ID][0].markings.keys())], rotation='vertical', fontsize=12) 
    plt.show()


