import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import copy
import random
import matplotlib.pyplot as plt
import os

class DDataset(Dataset):
    def __init__(self, data_dir, train):

        self.data_dir = data_dir
        self.train = train
        self.train_filename = "mm_train.pt"
        self.test_filename = "mm_test.pt"

        if train:
            if not os.path.exists(os.path.join(self.data_dir, self.train_filename)):
                raise RuntimeError(
                    'Dataset not found. Please generate dataset and place it in the data folder.')

            self._t_data, self._m_data, self._f_data = torch.load(os.path.join(data_dir, self.train_filename))

        else:
            if not os.path.exists(os.path.join(self.data_dir, self.test_filename)):
                raise RuntimeError(
                    'Dataset not found. Please generate dataset and place it in the data folder.')

            self._t_data, self._m_data, self._f_data = torch.load(os.path.join(data_dir, self.test_filename))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (t_data, m_data, f_data)
        """
        return self._t_data[index], self._m_data[index], self._f_data[index].permute(1,2,0)

    def __len__(self):
        return len(self._t_data)



class DoubleDataset():
    def __init__(self,
                 data_dir,
                 batch_size=64,
                 eval_samples=10,
                 validation_size=0.1,
                 seed=42):

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.eval_samples = eval_samples
        self.seed = seed

        # Idx
        self.train_idx, self.validation_idx, self.test_idx = None, None, None

        # Data
        self.train_loader, self.val_loader = self.get_training_loaders()
        self.test_loader = self.get_test_loader()


    def get_training_loaders(self):

        train_dataset = DDataset(data_dir=self.data_dir, train=True)
        valid_dataset = DDataset(data_dir=self.data_dir, train=True)

        # Load train and val idx
        self.train_idx, self.validation_idx = self.get_train_val_idx(train_dataset=train_dataset)

        print("Training Dataset Samples = " + str(len(self.train_idx)))
        print("Validation Dataset Samples = " + str(len(self.validation_idx)))

        train_sampler = SubsetRandomSampler(self.train_idx)
        valid_sampler = SubsetRandomSampler(self.validation_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=1,
            pin_memory=True)

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=1,
            pin_memory=True)

        return train_loader, valid_loader

    def get_test_loader(self, bsize=None):

        # load the dataset
        test_dataset = DDataset(data_dir=self.data_dir, train=False)

        if bsize is None:
            bsize = self.eval_samples

        return torch.utils.data.DataLoader(test_dataset,
                                           batch_size=bsize,
                                           num_workers=1,
                                           pin_memory=True)

    def get_train_val_idx(self, train_dataset):

        # Get global indices
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        # Split Training and validation
        split = int(np.floor(self.validation_size * num_train))

        # Training and validation indexes
        return indices[split:], indices[:split]

    def get_test_idx(self, test_dataset):

        # Get global indices
        num_test = len(test_dataset)
        indices = list(range(num_test))

        # Test indexes
        return indices
