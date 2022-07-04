
import math
import random
import re
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def get_label_conveter(database: str):
    conveters = {
        'iemocap': {'ang': 0, 'neu': 1, 'hap': 2, 'exc': 2, 'sad': 3},
        'meld': {'neutral': 0, 'anger': 1, 'joy': 2, 'sadness': 3, 'surprise': 4, 'disgust': 5, 'fear': 6},
        'pitt': {'Control': 0, 'Dementia': 1},
        'daic_woz': {'not-depressed': 0, 'depressed': 1}
    }

    try:
        label_conveter = conveters[database]
    except KeyError:
        print(f'Can not find the database: {database}, return None.')
        label_conveter = None

    return label_conveter

def iemocap_session_split(fold, dict_list: dict, state='train', strategy='5cv'):
    if strategy == '5cv':
        assert 1<=fold<=5, print('leave-one-session-out 5-fold cross validation, but got fold {}'.format(fold))
    elif strategy == '10cv':
        assert 1<=fold<=10, print('leave-one-speaker-out 10-fold cross validation , but got fold {}'.format(fold))
    else:
        raise KeyError('Wrong cross validation setting')

    name_list = dict_list['name']
    label_list = dict_list['label']
    shape_list = dict_list['shape'] if 'shape' in dict_list.keys() else [None] * len(name_list)

    name_fold, label_fold, shape_fold = [], [], []
    if strategy == '5cv':
        testSes = 'Ses0{}'.format(6-fold)
        for i, name in enumerate(name_list):
            if ((state == 'test') and (testSes in name)) or ((state != 'test') and (testSes not in name)):
                name_fold.append(name)
                label_fold.append(label_list[i])
                shape_fold.append(shape_list[i])
    else:
        gender = 'F' if fold%2 == 0 else 'M'
        fold = math.ceil(fold/2)
        testSes = 'Ses0{}'.format(6-fold)
        for i, name in enumerate(name_list):
            if ((state == 'test') and ((testSes in name) and (gender in name.split('_')[-1]))) \
             or ((state != 'test') and ((testSes not in name) or (gender not in name.split('_')[-1]))):
                name_fold.append(name)
                label_fold.append(label_list[i])
                shape_fold.append(shape_list[i])

    return name_fold, label_fold, shape_fold

def pitt_speaker_independent_split_10fold(fold, dict_list: dict, state='train', seed=2021):
    name_list = dict_list['name']
    label_list = dict_list['label']
    shape_list = dict_list['shape'] if 'shape' in dict_list.keys() else [None] * len(name_list)
    
    speaker = [n[:3] for n in name_list]   # [:3] represent the speaker id 
    speaker = sorted(list(set(speaker)))
    random.seed(seed)        # use the same seed in each fold!!!
    random.shuffle(speaker)
    start_id = math.ceil(len(speaker) / 10) * (fold - 1)
    end_id = math.ceil(len(speaker) / 10) * (fold)
    test_speaker = speaker[start_id:end_id]
        
    pattern = ''
    for sp in test_speaker:
        pattern += f'({sp}.*)|'
    pattern = pattern[:-1]

    train_names, test_names, train_labels, test_labels, train_shape, test_shape = [], [], [], [], [], []
    for name, label, shape in zip(name_list, label_list, shape_list):
        result = re.match(pattern, name)
        if result is not None:
            test_names.append(name)
            test_labels.append(label)
            test_shape.append(shape)
        else:
            train_names.append(name)
            train_labels.append(label)
            train_shape.append(shape)

    name_fold = test_names if state == 'test' else train_names
    label_fold = test_labels if state == 'test' else train_labels
    shape_fold = test_shape if state == 'test' else train_shape

    return name_fold, label_fold, shape_fold

def pitt_random_split_10fold(fold, dict_list: dict, conveter, state='train', seed=2021):
    name_list = dict_list['name']
    label_list = dict_list['label']
    shape_list = dict_list['shape'] if 'shape' in dict_list.keys() else [None] * len(name_list)

    y = [conveter[l] for l in label_list]
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=seed)
    for i, (train_index, test_index) in enumerate(sss.split(y, y)):
        if i + 1 == fold:
            index = train_index if state == 'train' else test_index
            names = [name_list[i] for i in index]
            labels = [label_list[i] for i in index]
            shapes = [shape_list[i] for i in index]
            break

    return names, labels, shapes

def daic_resample_up(dict_list):
    name_list = dict_list['name']
    label_list = dict_list['label']
    shape_list = dict_list['shape'] if 'shape' in dict_list.keys() else [None] * len(name_list)

    num_d = label_list.count('depressed')
    num_nd = label_list.count('not-depressed')

    r = num_nd // num_d
    s = num_nd % num_d
    index_d = np.arange(num_d)
    random.shuffle(index_d)

    d_index = [index for (index,value) in enumerate(label_list) if value == 'depressed']
    d_names = [name_list[index] for index in d_index]
    d_labels = [label_list[index] for index in d_index]
    d_shapes = [shape_list[index] for index in d_index]

    d_names_s = [d_names[index_d[i]] for i in range(s)]
    d_labels_s = [d_labels[index_d[i]] for i in range(s)]
    d_shapes_s = [d_shapes[index_d[i]] for i in range(s)]

    d_names = d_names * (r-1)
    d_labels = d_labels * (r-1)
    d_shapes = d_shapes * (r-1)

    d_names.extend(d_names_s)
    d_labels.extend(d_labels_s)
    d_shapes.extend(d_shapes_s)

    name_list.extend(d_names)
    label_list.extend(d_labels)
    shape_list.extend(d_shapes)

    return name_list, label_list, shape_list
    
def daic_resample_down(dict_list):
    name_list = dict_list['name']
    label_list = dict_list['label']
    shape_list = dict_list['shape'] if 'shape' in dict_list.keys() else [None] * len(name_list)

    num_d = label_list.count('depressed')
    num_nd = label_list.count('not-depressed')

    index_nd = np.arange(num_nd)
    random.shuffle(index_nd)

    nd_index = [index for (index,value) in enumerate(label_list) if value == 'not-depressed']
    nd_names = [name_list[index] for index in nd_index]
    nd_labels = [label_list[index] for index in nd_index]
    nd_shapes = [shape_list[index] for index in nd_index]

    nd_names = [nd_names[index_nd[i]] for i in range(num_d)]
    nd_labels = [nd_labels[index_nd[i]] for i in range(num_d)]
    nd_shapes = [nd_shapes[index_nd[i]] for i in range(num_d)]

    d_index = [index for (index,value) in enumerate(label_list) if value == 'depressed']
    d_names = [name_list[index] for index in d_index]
    d_labels = [label_list[index] for index in d_index]
    d_shapes = [shape_list[index] for index in d_index]

    name_list = d_names + nd_names
    label_list = d_labels + nd_labels
    shape_list = d_shapes + nd_shapes
    
    return name_list, label_list, shape_list
