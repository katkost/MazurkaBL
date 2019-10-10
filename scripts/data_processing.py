import numpy as np
import pandas as pd
import glob
import os
from collections import namedtuple
import tkinter
import matplotlib
matplotlib.use("macOSX")
import matplotlib.pyplot as plt
from functools import partial

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
    return df.to_dict('list')

######## Data processing ########

def norm_by_max(values):
    return [np.round(x / max(values), 3) for x in values]

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
                        Pianist: 'id beat dyn markings'
    """

    Mazurka_info = {}

    Pianist = namedtuple('Pianist', 'id beat dyn markings markings_dyn')

    files_ID = []
    print ('Retrieving information from the csv files...')
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

                    # round values
                    vals_beat = [np.round(v, 3) for v in vals_beat]
                    vals_dyn = [np.round(v, 3) for v in vals_dyn]

                    tuple_all.append((Pianist(id = id1, 
                                              beat = vals_beat, 
                                              dyn = norm_by_max(vals_dyn), 
                                              markings = make_dict_from_csv(mark), 
                                              markings_dyn = make_dict_from_csv(mark_dyn)
                                              )))
        Mazurka_info[M_ID] = tuple_all

    return Mazurka_info

####### Features calling #######
def get_marking_dyn_value(dyns, idx):
    _idx = int(idx) - 1
    if _idx == len(dyns) - 1:
        dyn_value = dyns[_idx]
    elif _idx == len(dyns) - 2:
        dyn_value = np.mean([dyns[_idx], dyns[_idx + 1]])
    else:
        dyn_value = np.mean([dyns[_idx], dyns[_idx + 1], dyns[_idx + 2]])
    return np.round(dyn_value, 3)

def get_dyn_values_per_pianist(markings_positions, p_info):
    pid_dyn_values = []
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
    plt.figure(figsize=(16, 14), dpi= 80)
    plt.subplot(211)
    for pianist in M_info_pianist:
        plt.plot(range(len(pianist.beat) -1 ), norm_by_max(np.diff(pianist.beat)))
        plt.title('Inter-beat-intervals in Mazurka recording', fontsize=14)
        plt.xlabel('Score beats', fontsize=14)
        plt.xticks([v[0] for v in [*pianist.markings.values()]], 
                   [m.split('.')[0] for m in list(pianist.markings.keys())], rotation='vertical', fontsize=14) 
        plt.ylabel('IBIs (normalised)', fontsize=14)
        plt.tight_layout()

    plt.subplot(212)
    for pianist in M_info_pianist:
        plt.plot(range(len(pianist.dyn)), pianist.dyn)
        plt.title('Dynamics per score beat in Mazurka recording', fontsize=14)
        plt.xlabel('Score beats', fontsize=14)
        plt.xticks([v[0] for v in [*pianist.markings.values()]], 
                   [m.split('.')[0] for m in list(pianist.markings.keys())], rotation='vertical', fontsize=14) 
        plt.ylabel('Dynamics in smoothed sones (normalised)', fontsize=14)
        plt.tight_layout()
    plt.show()

def plot_dyn_with_markings_values_boxplots(M_info, idx1, idx2):

    # Get dyn.values per marking for data in boxplot
    # get list of marking positions
    markings_positions = [int(v[0])-1 for v in [*M_info[0].markings_dyn.values()]]

    pid_dyn_values = list(map(partial(get_dyn_values_per_pianist, markings_positions), M_info))

    plt.figure(figsize=(14, 7), dpi= 80)
    m=0

    for mp in markings_positions:
        
        values = [v[1][m] for v in pid_dyn_values]
        print (values)
        plt.boxplot(values, positions=[mp])
        m+=1
    M_info_pianist = M_info[idx1:idx2]
    for pianist in M_info_pianist:     
        plt.plot(range(len(pianist.dyn)), pianist.dyn)
        plt.title('Dynamics per score beat in Mazurka recording', fontsize=14)
        plt.xlabel('Score beats', fontsize=14)
        plt.xticks([v[0]-1 for v in [*pianist.markings.values()]], 
                   [m.split('.')[0] for m in list(pianist.markings.keys())], rotation='vertical', fontsize=14) 
        plt.ylabel('Dynamics in smoothed sones (normalised)', fontsize=14)
        plt.ylim(0, 1)
        plt.xlim(-2, len(pianist.dyn)+2)
        plt.tight_layout()
    plt.savefig('test_plot.png')
    plt.show()    




###### MODELLING #######

# model_output = L.LSTM(4, return_sequences=False)(model_input)   for returning 4 features for the last timestep
# model = M.Model(input=model_input, output=model_output)
# model.compile('sgd', 'mean_squared_error')