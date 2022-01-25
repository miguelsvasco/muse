import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

class StandardClassDataset():
    def __init__(self,
                 dataset,
                 data_dir,
                 d_class,
                 batch_size=64,
                 eval_samples=10,
                 validation_size=0.1,
                 seed=42):

        self.dataset = dataset.lower()
        self.data_dir = data_dir
        self.d_class = d_class
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.eval_samples = eval_samples
        self.seed = seed

        # Transformation
        if self.dataset == "mnist":
            print("Selected MNIST dataset.")
            self.transformation = lambda x: transforms.ToTensor()(x)
        else:
            raise ValueError("No valid dataset selected: " + str(self.dataset))

        # Idx
        self.train_idx, self.validation_idx = None, None

        # Data
        self.train_loader, self.val_loader = self.get_training_loaders()
        self.test_loader = self.get_test_loader()

    def get_training_loaders(self):

        # load the dataset
        if self.dataset == "mnist":
            train_dataset = datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transformation)

            valid_dataset = datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transformation)
        else:
            raise ValueError("[MHVAE] No valid dataset selected: " + str(self.dataset))


        # Get only selected class
        train_idx = train_dataset.targets == self.d_class
        train_dataset.targets = train_dataset.targets[train_idx]
        train_dataset.data = train_dataset.data[train_idx]

        valid_idx = valid_dataset.targets == self.d_class
        valid_dataset.targets = valid_dataset.targets[valid_idx]
        valid_dataset.data = valid_dataset.data[valid_idx]

        # Load train and val idx
        self.train_idx, self.validation_idx = self.get_train_val_idx(
            train_dataset=train_dataset)

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

    def get_test_loader(self, bsize=10):

        # load the dataset
        if self.dataset == "mnist":
            test_dataset = datasets.MNIST(
                root=self.data_dir,
                train=False,
                download=True,
                transform=self.transformation)
        else:
            raise ValueError("No valid dataset selected: " + str(self.dataset))

        # Get only selected class
        test_idx = test_dataset.targets == self.d_class
        test_dataset.targets = test_dataset.targets[test_idx]
        test_dataset.data = test_dataset.data[test_idx]

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=bsize,
            num_workers=1,
            pin_memory=True)
        return self.test_loader

    def get_test_dataset(self):

        # load the dataset
        if self.dataset == "mnist":
            test_dataset = datasets.MNIST(
                root=self.data_dir,
                train=False,
                download=True,
                transform=self.transformation)
        else:
            raise ValueError("No valid dataset selected: " + str(self.dataset))

        return test_dataset

    def get_train_val_idx(self, train_dataset):

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.validation_size * num_train))

        np.random.seed(self.seed)
        np.random.shuffle(indices)

        # Global training and validation indexes
        return indices[split:], indices[:split]
