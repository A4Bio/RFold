import numpy as np
import os.path as osp
import _pickle as cPickle
from tqdm import tqdm
from torch.utils import data


def get_cut_len(data_len,set_len):
    l = data_len
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16
    return l


class cached_property(object):
    """
    Descriptor (non-data) for building an attribute on-demand on first use.
    """
    def __init__(self, factory):
        """
        <factory> is called such: factory(instance) to build the attribute.
        """
        self._attr_name = factory.__name__
        self._factory = factory

    def __get__(self, instance, owner):
        # Build the attribute.
        attr = self._factory(instance)

        # Cache the value; hide ourselves.
        setattr(instance, self._attr_name, attr)
        return attr


class RNADataset(data.Dataset):
    def __init__(self, path, dataname):
        self.path = path
        self.dataname = dataname
        self.data = self.cache_data

    def __len__(self):
        return len(self.data)

    def get_data(self, dataname):
        filename = dataname + '.pickle'
        pre_data = cPickle.load(open(osp.join(self.path, filename), 'rb'))

        data = []
        for instance in tqdm(pre_data):
            data_x, _, seq_length, name, pairs = instance
            l = get_cut_len(seq_length, 80)
            # contact
            contact = np.zeros((l, l))
            contact[tuple(np.transpose(pairs))] = 1. if pairs != [] else 0.
            # data_seq
            data_seq = np.zeros((l, 4))
            data_seq[:seq_length] = data_x[:seq_length]
            data.append([contact, seq_length, data_seq])
        return data

    @cached_property
    def cache_data(self):
        return self.get_data(self.dataname)

    def __getitem__(self, index):
        return self.data[index]