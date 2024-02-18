import os
import time
import numpy as np
import torch as th
# import tensorflow as tf

# class DataIterator:
#
#     def __init__(self, X, Y=None):
#         self.X = X
#         self.Y = Y
#
#         self.num_examples = self.X.shape[0]
#         self.epochs_completed = 0
#         self.indices = np.arange(self.num_examples)
#         self.reset_iteration()
#
#     def reset_iteration(self):
#         np.random.shuffle(self.indices)
#         self.start_idx = 0
#
#     def get_epoch(self):
#         return self.epochs_completed
#
#     def reset_epoch(self):
#         self.reset_iteration()
#         self.epochs_completed = 0
#
#     def next_batch(self, batch_size, data_type="train", shuffle=True):#
#         assert data_type in ["train", "val", "test"], \
#             "data_type must be 'train', 'val', or 'test'."
#
#         idx = self.indices[self.start_idx:self.start_idx + batch_size]
#
#         batch_x = self.X[idx]
#         batch_y = self.Y[idx] if self.Y is not None else self.Y
#         self.start_idx += batch_size
#
#         if self.start_idx + batch_size > self.num_examples:
#             self.reset_iteration()
#             self.epochs_completed += 1
#
#         return (batch_x, batch_y)
#
#     def sample_random_batch(self, batch_size):
#         start_idx = np.random.randint(0, self.num_examples - batch_size)
#         batch_x = self.X[self.start_idx:self.start_idx + batch_size]
#         batch_y = self.Y[self.start_idx:self.start_idx + batch_size] if self.Y is not None else self.Y
#
#         return (batch_x, batch_y)
#
#
# def get_iterators(file, conv=False, datapoints=0):
#     data = np.load(file)
#     if conv:
#         img_shape = data["train_x"][0,0].shape
#     else:
#         img_shape = data["train_x"][0,0].flatten().shape
#     train_it = DataIterator(X=data["train_x"].reshape(data["train_x"].shape[:2]+img_shape)/255)
#     valid_it = DataIterator(X=data["valid_x"].reshape(data["valid_x"].shape[:2]+img_shape)/255)
#     test_it = DataIterator(X=data["test_x"].reshape(data["test_x"].shape[:2]+img_shape)/255)
#     return train_it, valid_it, test_it


class PAIGDataset(th.utils.data.Dataset):

    def __init__(self, npz_path, split):
        self.npz_path = npz_path
        self.split = split  # either 'train', 'val', or 'test'

        index = 'train_x' if split == 'train' else ('valid_x' if split == 'val' else 'test_x')
        self.data = np.load(self.npz_path)[index]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        sample = th.from_numpy(sample).float() / 255
        sample = th.permute(sample, (0, 3, 1, 2))  # make channel the 2nd dimension
        return sample

