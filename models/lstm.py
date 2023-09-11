import torch
import torch.nn as nn
from torch.nn.utils.rnn import  pack_padded_sequence, pad_packed_sequence
import pytorch_lightning as pl


def boundary_loss(input_torch, target):
    """
    apply a function that penalizes the model for predicting a class that is too far from the true class,
    ordinal kind of loss
    :param input_torch:
    :param target:
    :return:
    """
    input_torch = nn.Softmax(dim=1)(input_torch)
    for index in range(input_torch.size(1)):
        if index == 0:
            losses = input_torch[:, index] * torch.abs(target - index)
        else:
            losses += input_torch[:, index] * torch.abs(target - index)

    return torch.mean(losses)


class LSTM(pl.LightningModule):
    def __init__(self, input_size: int,
                 output_size: int,
                 hidden_size: int = 512,
                 dropout: float = 0.5,
                 learning_rate: float = 0.002
                 ):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.lstm = nn.LSTM(input_size, self.hidden_size, dropout=dropout, num_layers=5, batch_first=True)

        self.bn_linear1 = nn.BatchNorm1d((1, self.hidden_size))
        self.dropout1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.relu_linear1 = nn.ReLU()

        self.bn_linear2 = nn.BatchNorm1d(self.hidden_size // 2)
        self.dropout2 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(self.hidden_size // 2, output_size)

    def forward(self, x, lengths):
        packed_seq = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed_seq)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        linear1 = self.bn_linear1(lstm_out[:, -1, :])  # Apply BatchNorm to latest timestep
        linear1 = self.dropout1(linear1)
        linear1 = self.linear1(linear1)
        linear1 = self.relu_linear1(linear1)

        linear2 = self.bn_linear2(linear1)
        linear2 = self.dropout2(linear2)
        linear2 = self.linear2(linear2)

        return linear2

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y, lengths = batch
        preds = self(x, lengths)
        loss = boundary_loss(preds, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, lengths = batch
        preds = self(x, lengths)
        loss = boundary_loss(preds, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y, lengths = batch
        preds = self(x, lengths)
        loss = boundary_loss(preds, y)
        self.log('test_loss', loss)

