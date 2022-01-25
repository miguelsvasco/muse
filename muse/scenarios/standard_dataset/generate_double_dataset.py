import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import copy
import random
import matplotlib.pyplot as plt
import os

def get_ordered_dataset(dataset, labels):

    # List to store ordered images
    dataset_images = []

    # Lists to keep idx of classes
    og_dataset_idx_class = [[] for i in range(len(np.unique(labels)))]
    dataset_idx_class = [[] for i in range(len(np.unique(labels)))]
    for k in range(len(dataset_idx_class)):
        og_dataset_idx_class[k] = [i for i, x in enumerate(dataset.labels) if x == k]
        dataset_idx_class[k] = copy.copy(og_dataset_idx_class[k])

    # Sort images of new dataset
    for i in range(len(labels)):

        k = labels[i]

        if len(dataset_idx_class[k]) == 0:
            img_idx = random.choice(og_dataset_idx_class[k])
            dataset_images.append(transforms.ToTensor()(dataset.data[img_idx]))

        else:
            img_idx = dataset_idx_class[k].pop(0)
            dataset_images.append(transforms.ToTensor()(dataset.data[img_idx]))

    return torch.stack(dataset_images, dim=0)


def get_dataset(dataset):

    # List to store ordered images
    dataset_images = []

    # Sort images of new dataset
    for i in range(len(dataset)):
        dataset_images.append(transforms.ToTensor()(dataset[i].numpy()))

    return torch.stack(dataset_images, dim=0)


def get_dataset_class_stats(label, data_dir, train=True):

    # Get total len of dataset
    len_ds = len(label)

    # Get class distribution
    unique, counts = np.unique(label.numpy(), return_counts=True)
    class_dist = dict(zip(unique, 100*counts/len_ds))

    if train:
        print("Train Dataset Stats:\n")
        print(class_dist)
        with open(os.path.join(data_dir, "train_stats.pt"), 'wb') as f:
            torch.save((class_dist,len_ds), f)
    else:

        print("Test Dataset Stats:\n")
        print(class_dist)
        with open(os.path.join(data_dir, "test_stats.pt"), 'wb') as f:
            torch.save((class_dist, len_ds), f)



def main():

    # Dirs
    data_dir = './'

    # Filenames
    train_filename = "mm_train.pt"
    test_filename = "mm_test.pt"

    # Transforms
    m_transform = lambda x: transforms.ToTensor()(x)
    f_transform = lambda x: transforms.ToTensor()(x)

    # Unimodal datasets:
    m_train = datasets.MNIST(root=data_dir,
                             train=True,
                             download=True,
                             transform=m_transform)
    m_test = datasets.MNIST(root=data_dir,
                             train=False,
                             download=True,
                             transform=m_transform)
    f_train = datasets.SVHN(root=data_dir,
                            split='train',
                                    download=True,
                                    transform=f_transform)
    f_test = datasets.SVHN(root=data_dir,
                           split='test',
                            download=True,
                            transform=f_transform)


    # Train Dataset
    train_labels = m_train.targets
    train_m_images = get_dataset(m_train.data)
    train_f_images = get_ordered_dataset(dataset=f_train, labels=train_labels)

    # Save train dataset
    with open(os.path.join(data_dir,train_filename), 'wb') as f:
        torch.save((train_labels,
                    train_m_images.float(),
                    train_f_images.float()), f)

    # Save statistics
    get_dataset_class_stats(train_labels, data_dir, True)

    # Test Dataset
    test_labels = m_test.targets
    test_m_images = get_dataset(m_test.data)
    test_f_images = get_ordered_dataset(dataset=f_test, labels=test_labels)

    # Save train dataset
    with open(os.path.join(data_dir, test_filename), 'wb') as f:
        torch.save((test_labels,
                    test_m_images.float(),
                    test_f_images.float()), f)

    # Save statistics
    get_dataset_class_stats(test_labels, data_dir, False)


if __name__ == '__main__':
    main()

