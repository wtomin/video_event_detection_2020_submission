import torch
from data.dataset import train_val_split
class DataLoader:
    def __init__(self, batch_size, train_transform=None, val_transform = None,
        lmdb_file = '', val_ratio = 0.2,
        train_num_workers = 2, val_num_workers = 2,
        pin_memory=True):
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.lmdb_file = lmdb_file
        self.val_ratio = val_ratio
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.pin_memory = pin_memory
        assert 0< self.val_ratio <1

    def create_dataloaders(self):
        train_dataset, val_dataset = train_val_split(self.lmdb_file, self.val_ratio,
        	self.train_transform, self.val_transform)
        train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size = self.batch_size, 
                    shuffle = True,
                    num_workers= int(self.train_num_workers), 
                    drop_last = True,
                    pin_memory = self.pin_memory)
        if val_dataset is not None:
            val_dataloader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size = self.batch_size, 
                        shuffle = False,
                        num_workers= int(self.val_num_workers), 
                        drop_last = False,
                        pin_memory = self.pin_memory)
        else:
            val_dataloader = None
        return train_dataloader, val_dataloader


