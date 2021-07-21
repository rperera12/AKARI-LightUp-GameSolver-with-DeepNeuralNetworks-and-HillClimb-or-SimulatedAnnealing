import numpy as np
import csv 
import os

from numpy import loadtxt, savetxt
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, datasets, models
import torch


class Boardloader(Dataset):
    """Board Initialization dataset."""

    def __init__(self,train_dir, label_dir, transform=None):
        
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train_dir = train_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return 9921

    def __getitem__(self, idx):
    
        # save to csv file
        load_str_train = self.train_dir + str(idx) + '.csv'
        load_str_label = self.label_dir + str(idx) + '.csv'
        train_data = loadtxt(load_str_train, delimiter=',')
        label_data = loadtxt(load_str_label, delimiter=',')

        train_data = train_data.reshape(1, train_data.shape[0], train_data.shape[1])
        label_data = label_data.reshape(label_data.shape[0], label_data.shape[1])
        
        train_data = torch.tensor(train_data)
        label_data = torch.tensor(label_data)
        sample = {'initial': train_data, 'labels': label_data}        


        return sample
    