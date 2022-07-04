
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import utils
import pickle
import multiprocessing as mp
import re

def identity(x):
    return x

class DistributedDalaloaderWrapper():
    def __init__(self, dataloader: DataLoader, collate_fn):
        self.dataloader = dataloader
        self.collate_fn = collate_fn
    
    def _epoch_iterator(self, it):
        for batch in it:
            yield self.collate_fn(batch)

    def __iter__(self):
        it = iter(self.dataloader)
        return self._epoch_iterator(it)

    def __len__(self):
        return len(self.dataloader)

    @property
    def dataset(self):
        return self.dataloader.dataset

    def set_epoch(self, epoch: int):
        self.dataloader.sampler.set_epoch(epoch)

def universal_collater(batch):
    all_data = [[] for _ in range(len(batch[0]))]
    for one_batch in batch:
        for i, (data) in enumerate(one_batch):
            all_data[i].append(data)
    return all_data

class LMDB_Dataset(Dataset):
    def __init__(self, corpus, lmdb_root, map_size, label_conveter, state, mode, length, feature_dim, pad_value, fold=0):
        lmdb_path = lmdb_root if os.path.exists(os.path.join(lmdb_root, 'meta_info.pkl')) else os.path.join(lmdb_root, state)
        self.meta_info = pickle.load(open(os.path.join(lmdb_path, 'meta_info.pkl'), "rb"))
        self.LMDBReader = utils.lmdb_kit.LMDBReader(lmdb_path, map_size * len(self.meta_info['key']) * 10)
        self.LMDBReader_km = None
        self.corpus = corpus

        if self.corpus == 'iemocap':
            self.load_name = False
            dict_list = {'name': self.meta_info['key'], 'label': self.meta_info['label'], 'shape': self.meta_info['shape']}
            self.meta_info['key'], self.meta_info['label'], self.meta_info['shape'] = utils.dataset_kit.iemocap_session_split(fold, dict_list, state)
        elif self.corpus == 'meld':
            self.load_name = False
        elif self.corpus == 'pitt':
            self.load_name = True
            dict_list = {'name': self.meta_info['key'], 'label': self.meta_info['label'], 'shape': self.meta_info['shape']}
            self.meta_info['key'], self.meta_info['label'], self.meta_info['shape'] = utils.dataset_kit.pitt_speaker_independent_split_10fold(fold, dict_list, state)
        elif self.corpus == 'daic_woz':
            self.load_name = True
        else:
            raise ValueError(f'Got unknown database: {self.corpus}')

        self.need_pad = True
        self.conveter = label_conveter
        self.kit = utils.speech_kit.Speech_Kit(mode, length, feature_dim, pad_value)

    def load_a_sample(self, idx=0):
        label = self.meta_info['label'][idx]
        x = self.LMDBReader.search(key=self.meta_info['key'][idx])
        T, C = [int(s) for s in self.meta_info['shape'][idx].split('_')]
        x = x.reshape(T, C)
        y = torch.tensor(self.label_2_index(label))
        return x, y

    def load_wav_path(self, idx):
        name = self.load_sample_name(idx)
        if self.corpus == 'iemocap':
            session = int(re.search('\d+', name).group())
            ses = '_'.join(name.split('_')[:-1])
            wav_path = f'/148Dataset/data-chen.weidong/iemocap/Session{session}/sentences/wav/{ses}/{name}.wav'
        else:
            raise ValueError(f'Got unknown database: {self.corpus}')

        return wav_path

    def load_sample_name(self, idx):
        return self.meta_info['key'][idx]
    
    def label_2_index(self, label):
        index = self.conveter[label]
        return index

    def get_need_pad(self):
        return self.need_pad

    def set_need_pad(self, need_pad):
        self.need_pad = need_pad

    def get_load_name(self):
        return self.load_name

    def set_load_name(self, load_name):
        self.load_name = load_name

    def __len__(self):
        return len(self.meta_info['key'])

    def __getitem__(self, idx):
        x, y = self.load_a_sample(idx)
        x = self.kit.pad_input(x) if self.need_pad else x    # ndarray -> Tensor
        name = self.load_sample_name(idx) if self.load_name else None
        return x, y, name

class DataloaderFactory():
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, state, **kwargs):
        corpus = self.cfg.dataset.database
        mode = self.cfg.dataset.padmode
        fold = self.cfg.train.current_fold
        length = kwargs['length']
        feature_dim = kwargs['feature_dim']
        pad_value = kwargs['pad_value']
        lmdb_root = kwargs['lmdb_root']
        
        if corpus == 'daic_woz':
            state = 'dev' if state != 'train' else 'train'   # test refers to dev in daic_woz corpus
            
        map_size = length * feature_dim * 4
        label_conveter = utils.dataset_kit.get_label_conveter(corpus)
        dataset = LMDB_Dataset(corpus, lmdb_root, map_size, label_conveter, state, mode, length, feature_dim, pad_value, fold)
        
        collate_fn = universal_collater
        sampler = DistributedSampler(dataset, shuffle=state == 'train')
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.cfg.train.batch_size, 
            drop_last=False, 
            num_workers=self.cfg.train.num_workers, 
            collate_fn=identity,
            sampler=sampler, 
            pin_memory=True,
            multiprocessing_context=mp.get_context('fork'), # quicker! Used with multi-process loading (num_workers > 0)
        )

        return DistributedDalaloaderWrapper(dataloader, collate_fn)
