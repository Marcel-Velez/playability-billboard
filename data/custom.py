import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from utils import process_song, convert_to_encoding

random.seed(1)


def pad_collate(batch):
    xx = batch
    x_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, x_lens


class CustomChordSong(Dataset):
    '''
    '''
    def __init__(self, X: np.ndarray, chord_encoding='char', encode_dict=None):
        self.X = list(X) if not isinstance(X, list) else X
        self.chord_encoding = chord_encoding
        self.encode_dict = encode_dict
        self.encode_dict_size = len(encode_dict) + 2  # one for unk token and one for padd token

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        file_content = self.X[index]
        # read the chord annotations file
        song = process_song(file_content, remove_nl=False)
        encoded_song = convert_to_encoding(song, self.chord_encoding, self.encode_dict)
        if self.chord_encoding == 'guitar_test':
            return encoded_song

        annotations = F.one_hot(torch.tensor(encoded_song).to(torch.int64), self.encode_dict_size).float()
        return annotations

    def loader(self, batch_size=64, shuffle=False, num_workers=0, collate_fn=pad_collate):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)


