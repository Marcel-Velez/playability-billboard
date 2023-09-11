import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as packer, pad_packed_sequence as padder
import pytorch_lightning as pl


def boundary_loss(input_torch, target, class_total=False):
    input_torch = nn.Softmax()(input_torch)
    # if class_total:
    for index in range(input_torch.size(1)):
        if index == 0:
            losses = input_torch[:,index] * abs(target - index)
        else:
            losses += input_torch[:,index] * abs(target - index)

    return torch.mean(losses)


# ----------------------------------------------------------------------------------------------------------------------
# https://github.com/Maghoumi/DeepGRU/blob/master/model.py only diference is the adjustable hidden_size and converting
# it to a pytorch lightning module
class DeepGRU(pl.LightningModule):
    def __init__(self, input_size, hidden_size=512, output_size=4, learning_rate=0.0002):
        super(DeepGRU, self).__init__()
        self.num_features = input_size
        self.criterion = boundary_loss
        self.learning_rate = learning_rate

        # Encoder
        self.gru1 = nn.GRU(self.num_features, hidden_size, 2, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, 2, batch_first=True)
        self.gru3 = nn.GRU(hidden_size//2, hidden_size//4, 1, batch_first=True)

        # Attention
        self.attention = Attention(hidden_size//4)

        # Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_size//2),
            nn.Dropout(0.5),
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size//2),
            nn.Dropout(0.5),
            nn.Linear(hidden_size//2, output_size)
        )

    def forward(self, x_padded, x_lengths):
        x_packed = packer(x_padded, x_lengths, batch_first=True, enforce_sorted=False)

        # Encode
        output, _ = self.gru1(x_packed)
        output, _ = self.gru2(output)
        output, hidden = self.gru3(output)

        # Pass to attention with the original padding
        output_padded, _ = padder(output, batch_first=True)
        attn_output = self.attention(output_padded, hidden[-1:])

        # Classify
        classify = self.classifier(attn_output)
        return classify

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def update_optimizers(self, lr):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y, x_len = batch
        pred = self.forward(x, x_len)
        loss = self.criterion(pred, y.long(), class_total=True)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, x_len = batch
        pred = self.forward(x, x_len)
        loss = self.criterion(pred, y.long(), class_total=True)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, x_len = batch
        pred = self.forward(x, x_len)
        loss = self.criterion(pred, y.long(), class_total=True)
        self.log('test_loss', loss)
        return loss


# ----------------------------------------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, attention_dim):
        super(Attention, self).__init__()
        self.w = nn.Linear(attention_dim, attention_dim, bias=False)
        self.gru = nn.GRU(attention_dim, attention_dim, 1, batch_first=True)

    def forward(self, input_padded, hidden):
        e = torch.bmm(self.w(input_padded), hidden.permute(1, 2, 0))
        context = torch.bmm(input_padded.permute(0, 2, 1), e.softmax(dim=1)) # the attention is after softmax
        context = context.permute(0, 2, 1)

        # Compute the auxiliary context, and concat
        aux_context, _ = self.gru(context, hidden)
        output = torch.cat([aux_context, context], 2).squeeze(1)

        return output



