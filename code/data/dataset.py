import torch
import lmdb
import pyarrow as pa
import numpy as np 
import os
np.random.seed(0)
def read_annotation_file(file_path):
    data = lmdb.open(file_path, subdir=os.path.isdir(file_path),
                         readonly=True, lock=False,
                         readahead=False, meminit=False)
    with data.begin(write=False) as txn:
        length = pa.deserialize(txn.get(b'__len__'))
        keys= pa.deserialize(txn.get(b'__keys__'))
    return data, length, keys
def train_val_split(lmdb_file, val_ratio = 0., train_transform=None,
    val_transform = None):
    data, length, keys = read_annotation_file(lmdb_file)
    if val_ratio == 0.:
        return Sequence_Dataset(data, length, keys), None
    elif val_ratio > 0:
        val_length = int(length*val_ratio)
        keys = np.random.permutation(np.array(keys))
        val_keys = keys[:val_length]
        val_length = len(val_keys)
        train_keys = keys[val_length:]
        train_length = len(train_keys)
        return Sequence_Dataset(data, train_length, train_keys, train_transform), Sequence_Dataset(data, val_length, val_keys, val_transform)
    else:
        raise ValueError("Invalid val ratio")

class Sequence_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, length, keys, transform=None):
        self.data_dict = data_dict
        self.length = length
        self.keys = keys
        assert len(self.keys) == self.length
        self._transform = transform
    def unpack_index(self, index):
        data = self.data_dict
        with data.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)
        return unpacked
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        assert (index < self.length)
        # start_time = time.time()
        feature = None
        label = None
        finished = False
        while not finished:
            try:
                unpacked = self.unpack_index(index)
                finished = True
            except:
                print("Possible file damage. Resample another index")
                index = np.random.choice(self.length, 1)[0]
        # load image
        feature = unpacked[0]
        if self._transform is not None:
            feature = self._transform(feature)

        # load label
        label = unpacked[1]      
        assert len(label.shape) == 2
        # pack data
        sample = {'image': feature,
                  'label': label,
                  'index': index
                  }
        return sample

if __name__=='__main__':
    train_dataset, val_dataset = train_val_split(lmdb_file='../user_data/Train/i3d_features.lmdb', val_ratio = 0.2)
    sample = train_dataset[13]
    sample = val_dataset[9]
