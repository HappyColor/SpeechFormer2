
import lmdb
import numpy as np
import os
import json
from scipy import io
from tqdm import tqdm
import pickle
import pandas as pd
import librosa
import sys

class LMDBReader():
    def __init__(self, lmdb_path, map_size=1000000, dtype=np.float32):
        '''LMDB database tool. Only the ndarray / str are supported.

        map_size: T * C * 4 (nbytes) * num * 10.
        '''
        isdir = os.path.isdir(lmdb_path)
        self.env = lmdb.Environment(lmdb_path, map_size=map_size, subdir=isdir, readonly=False, meminit=False, map_async=True)
        self.txn = self.env.begin(write=False)  # Avoid opening too many Transactions
        self.dtype = dtype

    def insert(self, key, value):
        txn = self.env.begin(write=True)
        if isinstance(key, list):
            for _key, _value in zip(key, value):
                txn.put(_key.encode(), np.ascontiguousarray(_value))
        else:
            txn.put(key.encode(), np.ascontiguousarray(value))
        txn.commit()

    def delete(self, key: str):
        txn = self.env.begin(write=True)
        txn.delete(key.encode())
        txn.commit()

    def search(self, key: str):
        buf = self.txn.get(key.encode())
        value = np.copy(np.frombuffer(buf, dtype=self.dtype))
        return value

    def display(self):
        txn = self.env.begin(write=False)
        for key, value in txn.cursor():
            value = np.copy(np.frombuffer(value, dtype=self.dtype))
            print(key.decode(), value)
    
    def close(self):
        self.env.close()

def get_info(opt, csv_file):
    df = pd.read_csv(csv_file)
    if opt['database'] == 'iemocap':
        df_sad = df[df.label == 'sad']
        df_neu = df[df.label == 'neu']
        df_ang = df[df.label == 'ang']
        df_hap = df[df.label == 'hap']
        df_exc = df[df.label == 'exc']
        df_list = [df_sad, df_neu, df_ang, df_hap, df_exc]
        df = pd.concat(df_list)
        lmdb_path = os.path.join(opt['lmdb_root'], opt['lmdb_name'])
    elif opt['database'] == 'meld':
        state = opt['state']
        df = df[df.state == state]
        lmdb_path = os.path.join(opt['lmdb_root'], opt['lmdb_name'], state)
    elif opt['database'] == 'pitt':
        df = df[df.valid == True]
        lmdb_path = os.path.join(opt['lmdb_root'], opt['lmdb_name'])
    elif opt['database'] == 'daic_woz':
        state = 'dev' if opt['state'] != 'train' else 'train'
        df = df[df.state == state]
        index_2_label = {0: 'not-depressed', 1: 'depressed'}
        for row_index, row in df.iterrows():
            df.loc[row_index, 'label'] = index_2_label[row['label']]
        lmdb_path = os.path.join(opt['lmdb_root'], opt['lmdb_name'], state)
    else:
        ValueError
    
    return df, lmdb_path

def modify_matdir_sample(opt, matdir, label=None, sample=None):
    if opt['database'] == 'iemocap':
        matdir = matdir
    elif opt['database'] == 'meld':
        matdir = os.path.join(matdir, opt['state'])
    elif opt['database'] == 'pitt':
        matdir = os.path.join(matdir, label, 'cookie')
    elif opt['database'] == 'daic_woz':
        matdir = matdir
        sample = sample[:-10]
    else:
        ValueError
    
    return matdir, sample

def folder2lmdb(opt: dict):
    with open(f"./config/{opt['database']}_feature_config.json", 'r') as f:
        data_json = json.load(f)
        csv_file = data_json['meta_csv_file']
        fea_json = data_json[opt['feature']]
        matdir = fea_json['matdir']
        matkey = fea_json['matkey']
  
    df, lmdb_path = get_info(opt, csv_file)
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)
    else:
        print(f'Watch out! File path is already existed. ({lmdb_path})')
        sys.exit()

    sample_list = df['name'].values
    data_size = fea_json['length'] * fea_json['feature_dim'] * 4
    map_size = data_size * len(sample_list) * 10

    lmdb_reader = LMDBReader(lmdb_path, map_size)
    
    key_list, value_list = [], []
    meta_info = {'shape': [], 'key': [], 'label': []}
    for idx, sample in tqdm(enumerate(sample_list), total=len(sample_list)):
        df_sample = df[df['name'] == sample]
        label = df_sample.iloc[0, :]["label"]

        _matdir, sample = modify_matdir_sample(opt, matdir, label, sample)
        data = io.loadmat(os.path.join(_matdir, sample))[matkey]

        if opt['feature'] == 'spec':
            data = librosa.amplitude_to_db(data, ref=np.max)
            data = data.transpose(1,0)
            
        key_list.append(sample)
        value_list.append(data)

        T, C = data.shape
        meta_info['shape'].append('{:d}_{:d}'.format(T, C))
        meta_info['key'].append(sample)
        meta_info['label'].append(label)
        
        if (idx + 1) % opt['commit_interval'] == 0:
            lmdb_reader.insert(key_list, value_list)
            key_list, value_list = [], []
    
    if len(key_list) > 0:
        lmdb_reader.insert(key_list, value_list)
    
    pickle.dump(meta_info, open(os.path.join(lmdb_path, 'meta_info.pkl'), "wb"))
    print(f'Finish creating lmdb and meta info -> {lmdb_path}')

if __name__ == '__main__':
    opt = {
        'database': 'daic_woz',
        'feature': 'wavlm24',
        'lmdb_name': 'daic_woz_wavlm_L24',
        'lmdb_root': '/148Dataset/data-chen.weidong/lmdb',
        'commit_interval': 100,
        'state': 'train'   # Valid when database is meld or daic_woz.
        }

    folder2lmdb(opt)
