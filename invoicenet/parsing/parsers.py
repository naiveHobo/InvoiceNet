import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..acp.data import RealData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Conv1dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.cut_last_element = (kernel_size % 2 == 0 and stride == 1 and dilation % 2 == 1)
        self.padding = math.ceil((1 - stride + dilation * (kernel_size-1))/2)
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=self.padding, stride=stride, dilation=dilation)

    def forward(self, x):
        if self.cut_last_element:
            return self.conv(x)[:, :, :-1]
        else:
            return self.conv(x)


class Parser(nn.Module):

    def __init__(self):
        super(Parser, self).__init__()

    def forward(self, x, context):
        raise NotImplementedError()

    def restore(self):
        """
        Must return a tuple of (scope, restore_file_path).
        """
        raise NotImplementedError()


class NoOpParser(Parser):

    def __init__(self):
        super(NoOpParser, self).__init__()

    def forward(self, x, context):
        return x

    def restore(self):
        return None


class OptionalParser(Parser):

    def __init__(self, delegate: Parser, batch_size, context_size, seq_out, n_out, eos_idx):
        super(OptionalParser, self).__init__()
        self.empty_answer = torch.full((batch_size, seq_out), eos_idx, dtype=torch.long).to(device)
        self.empty_answer = F.one_hot(self.empty_answer, n_out).float().permute(0, 2, 1)  # (bs, n_out, seq)
        self.delegate = delegate
        self.linear = nn.Linear(in_features=context_size, out_features=1)

    def restore(self):
        return self.delegate.restore()

    def forward(self, x, context):
        parsed = self.delegate(x, context)
        logit_empty = self.linear(context)  # (bs, 1)
        return parsed + torch.reshape(logit_empty, (-1, 1, 1)) * self.empty_answer


class AmountParser(Parser):
    """
    You should pre-train this parser to parse amount otherwise it's hard to learn jointly.
    """
    seq_in = RealData.seq_in
    seq_out = RealData.seq_amount
    n_out = len(RealData.chars)

    def __init__(self, batch_size):
        super(AmountParser, self).__init__()
        os.makedirs(r"./models/parsers/amount", exist_ok=True)
        self.decoder_input = torch.zeros((batch_size, self.seq_out, 1), dtype=torch.float32).to(device)
        self.encoder = nn.LSTM(self.n_out, 128, bidirectional=True, batch_first=True)
        self.decoder = nn.LSTM(1, 128, batch_first=True)
        self.linear_enc = nn.Linear(256, 128)
        self.linear_dec = nn.Linear(128, 128)
        self.linear_att = nn.Linear(128, 1)
        self.linear_p_gen = nn.Linear(256, 1)
        self.linear_gen = nn.Linear(256, self.n_out)

    def restore(self):
        return r"./models/parsers/amount/best"

    def forward(self, x, context):
        # encoder
        x = x.permute(0, 2, 1)                                      # (bs, seq_in, n_out)
        h_in, _ = self.encoder(x)                                   # (bs, seq_in, 256)
        h_in = torch.reshape(h_in, (-1, self.seq_in, 1, 256))       # (bs, seq_in, 1, 256)

        # decoder
        h_out, _ = self.decoder(self.decoder_input)                 # (bs, seq_out, 128)
        h_out = torch.reshape(h_out, (-1, 1, self.seq_out, 128))    # (bs, 1, seq_out, 128)

        # bahdanau attention
        att = torch.tanh(self.linear_dec(h_out) + self.linear_enc(h_in))        # (bs, seq_in, seq_out, 128)
        att = self.linear_att(att)                                              # (bs, seq_in, seq_out, 1)
        att = torch.softmax(att, dim=1)                                         # (bs, seq_in, seq_out, 1)
        attended_h = torch.sum(att * h_in, dim=1)                               # (bs, seq_out, 256)

        p_gen = self.linear_p_gen(attended_h)                                   # (bs, seq_out, 1)
        p_copy = (1 - p_gen)

        # Generate
        gen = self.linear_gen(attended_h)                                       # (bs, seq_out, n_out)
        gen = torch.reshape(gen, (-1, self.seq_out, self.n_out))

        # Copy
        copy = torch.log(
            torch.sum(att * x.unsqueeze(2).contiguous(), dim=1) + 1e-8)   # (bs, seq_out, n_out)

        output_logits = p_copy * copy + p_gen * gen
        return output_logits.permute(0, 2, 1)


class DateParser(Parser):
    """
    You should pre-train this parser to parse dates otherwise it's hard to learn jointly.
    """
    seq_out = RealData.seq_date
    n_out = len(RealData.chars)

    def __init__(self):
        super(DateParser, self).__init__()
        os.makedirs(r"./models/parsers/date", exist_ok=True)

        self.conv_block = nn.Sequential(
            Conv1dSame(self.n_out, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            Conv1dSame(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            Conv1dSame(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            Conv1dSame(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
        )

        self.linear_block = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
        )

        self.dropout = nn.Dropout()
        self.linear_out = nn.Linear(256, self.seq_out * self.n_out)

    def restore(self):
        return r"./models/parsers/date/best"

    def forward(self, x, context):
        x = self.conv_block(x)                  # (bs, 128, 8)
        x = torch.sum(x, dim=2)                 # (bs, 128)
        x = torch.cat([x, context], dim=1)      # (bs, 256)
        x = self.linear_block(x)                # (bs, 256)
        x = self.dropout(x)                     # (bs, 256)
        x = self.linear_out(x)                  # (bs, seq_out * n_out)
        return torch.reshape(x, (-1, self.n_out, self.seq_out))
