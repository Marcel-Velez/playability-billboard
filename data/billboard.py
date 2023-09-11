import numpy as np
import pandas as pd
from collections import defaultdict
import random

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from utils import process_song
from utils import extract_possible_symbols, split_dict, convert_to_encoding
random.seed(1)


def default_return_val():
    """the default value for unknown symbols in the encoding dictionary.
       as a function because lambda functions are not pickleable."""
    return 1


def not_default_size(encode_dict):
    """
    Returns the number of symbols in the encoding dictionary that
    are not mapped to the default value for unknown symbols.
    :param encode_dict: the default dictionary mapping symbols to their one hot encoding index
    :return:
    """
    not_one = [k for k, v in encode_dict.items() if v != 1]
    return len(not_one)


class ChordDataset(Dataset):
    '''
    Custom Dataset subclass.
    Serves as input to DataLoader to transform X
      into sequence data using rolling window.
    DataLoader using this dataset will output batches
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs.
    '''
    def __init__(self, X: np.ndarray, y: np.ndarray, chord_encoding='char', encode_dict=None, target=None):
        self.X = list(X)
        self.y = torch.tensor(y).float()

        self.chord_encoding = chord_encoding
        self.encode_dict = encode_dict
        # adding 2, one for padding token, one for unk token.
        self.encode_dict_size = not_default_size(self.encode_dict) + 2

        self.target = target

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        file_path = self.X[index]
        label = self.y[index]

        # read the chord annotations file
        with open(file_path, 'r') as f:
            song = f.readlines()

        song = process_song(song, remove_nl=False)
        encoded_song = convert_to_encoding(song, self.chord_encoding, self.encode_dict)
        ohe_song = F.one_hot(torch.tensor(encoded_song).to(torch.int64), self.encode_dict_size).float()
        return ohe_song[ :, :], label


def pad_collate(batch):
    """
    Pads sequences to the same length and returns the padded sequences and their labels.
    :param batch: a batch of different length chord sequences and their labels.
    :return: padded sequences, labels, and the original lengths of sequences before padding.
    """
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    lengths = [len(seq) for seq in x]
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)  # Padding sequences
    return x_padded, torch.stack(y), lengths


def get_train_val_ind(indices):
    """
    Returns the indices of the train and validation subsets belonging to the train and val split indices.
    :param indices: a single index for the train and validation split index offset.
    :return: the indices of the train and validation subsets belonging to the train and val split indices.
    """
    indices_list = np.asarray(list(range(10))) + indices

    train = [split_dict[(i% 10)] for i in indices_list[:-1]]
    validate = [split_dict[(i % 10)] for i in indices_list[-1:]]

    train = [j for i in train for j in i]
    validate = [j for i in validate for j in i]

    return train, validate


class BillBoard(pl.LightningDataModule):
    """
    DataModule for the Billboard playability dataset.
    """
    def __init__(self, batch_size=128, num_workers=0, chord_encoding='char', seed=7,
                 target='shifted_total', target_data_path='./Annotations.csv'):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.chord_encoding = chord_encoding
        self.seed = seed
        self.target = target
        self.encode_dict = None
        self.encode_dict_size = None
        self.target_data_path = target_data_path

    def assign_train_val_ind(self, X, y, train_ind, val_ind):
        """
        Assigns the train and validation subsets belonging to the train and val split indices.
        :param X: the entire 200 song chord dataset.
        :param y: the 200 song labels.
        :param train_ind: the indices of the entire 200 songs to be used for training.
        :param val_ind: the indices of the entire 200 songs to be used for validation.
        """
        self.X_train = X.iloc[train_ind].to_numpy()
        self.y_train = y.iloc[train_ind].to_numpy()
        self.X_val = X.iloc[val_ind].to_numpy()
        self.y_val = y.iloc[val_ind].to_numpy()

        self.X_test = X.iloc[val_ind].to_numpy()
        self.y_test = y.iloc[val_ind].to_numpy()

    def custom_setup(self, indices):
        """
        Sets up the dataset for the given train and validation split indices.
        :param indices:
        :return:
        """
        path = self.target_data_path
        df = pd.read_csv(path, low_memory=False,)
        X = df['chord_locations']
        y = df[self.target]

        # select the train and validation subset belonging to the train and val split indices
        train_ind, val_ind = get_train_val_ind(indices)
        self.assign_train_val_ind(X, y, train_ind, val_ind)

        # go over all song chords in training and extract the set of occuring symbols which is needed for the one hot
        # encoding of the input for the LSTM/ DeepGRU
        possible_symbols = extract_possible_symbols(self.X_train, self.encoding)

        # +2 because we reserve 0 for padding value and 1 for unknown symbols
        encode_dict = {k: i + 2 for i, k in enumerate(set(possible_symbols))}
        self.encode_dict = defaultdict(default_return_val, encode_dict)
        self.encode_dict_size = len(encode_dict)+2


    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if stage == 'fit' and self.X_train is not None:
            return
        if stage == 'test' and self.X_test is not None:
            return
        if stage is None and self.X_train is not None and self.X_test is not None:
            return

        if stage == 'fit' or stage is None:
            self.X_train = self.X_train
            self.y_train = self.y_train.values.reshape((-1, 1))
            self.X_val = self.X_val
            self.y_val = self.y_val.values.reshape((-1, 1))

        if stage == 'test' or stage is None:
            self.X_test = self.X_test
            self.y_test = self.y_test.values.reshape((-1, 1))

    def train_dataloader(self):
        train_dataset = ChordDataset(self.X_train,
                                     self.y_train,
                                     chord_encoding=self.chord_encoding,
                                     encode_dict=self.encode_dict,
                                     target=self.target)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers,
                                  collate_fn=pad_collate)

        return train_loader

    def val_dataloader(self):
        val_dataset = ChordDataset(self.X_val,
                                   self.y_val,
                                   chord_encoding=self.chord_encoding,
                                   encode_dict=self.encode_dict,
                                   target=self.target)

        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                collate_fn=pad_collate)

        return val_loader

    def test_dataloader(self):
        test_dataset = ChordDataset(self.X_test,
                                    self.y_test,
                                    chord_encoding=self.chord_encoding,
                                    encode_dict=self.encode_dict,
                                    target=self.target)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 collate_fn=pad_collate)

        return test_loader

