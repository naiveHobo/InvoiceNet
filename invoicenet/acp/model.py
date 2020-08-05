import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import FIELD_TYPES, FIELDS
from .data import RealData
from ..parsing.parsers import DateParser, AmountParser, NoOpParser, OptionalParser


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Conv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.cut_last_element = (kernel_size % 2 == 0 and stride == 1 and dilation % 2 == 1)
        self.padding = math.ceil((1 - stride + dilation * (kernel_size-1))/2)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=self.padding, stride=stride, dilation=dilation)

    def forward(self, x):
        if self.cut_last_element:
            return self.conv(x)[:, :, :-1]
        else:
            return self.conv(x)


class DilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedBlock, self).__init__()
        self.conv1 = Conv2dSame(in_channels, out_channels, 3, dilation=1)
        self.conv2 = Conv2dSame(in_channels, out_channels, 3, dilation=2)
        self.conv3 = Conv2dSame(in_channels, out_channels, 3, dilation=4)
        self.conv4 = Conv2dSame(in_channels, out_channels, 3, dilation=8)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class AttendBlock(nn.Module):

    def __init__(self, batch_size, hidden_dim=32):
        super(AttendBlock, self).__init__()

        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        self.X, self.Y = torch.meshgrid(
            torch.linspace(0.0, 1.0, RealData.im_size[0]),
            torch.linspace(0.0, 1.0, RealData.im_size[1]))
        self.X = self.X.view(1, 1, RealData.im_size[0], RealData.im_size[0]).repeat(batch_size, 1, 1, 1).to(device)
        self.Y = self.Y.view(1, 1, RealData.im_size[1], RealData.im_size[1]).repeat(batch_size, 1, 1, 1).to(device)

        self.word_embed = nn.Embedding(RealData.word_hash_size, hidden_dim)
        self.pattern_embed = nn.Embedding(RealData.pattern_hash_size, hidden_dim)
        self.char_embed = nn.Embedding(RealData.n_output, hidden_dim)

        self.conv_block = nn.Sequential(
            DilatedBlock(110, hidden_dim),
            nn.ReLU(inplace=True),
            DilatedBlock(128, hidden_dim),
            nn.ReLU(inplace=True),
            DilatedBlock(128, hidden_dim),
            nn.ReLU(inplace=True),
            DilatedBlock(128, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(0.5)
        self.conv_out = Conv2dSame(128, RealData.n_memories, 3)

    def forward(self, pixels, word_indices, pattern_indices, char_indices, memory_mask, parses):
        word_indices = torch.reshape(word_indices, (self.batch_size, -1))                       # (bs, h * w)
        pattern_indices = torch.reshape(pattern_indices, (self.batch_size, -1))                 # (bs, h * w)
        char_indices = torch.reshape(char_indices, (self.batch_size, -1))                       # (bs, h * w)

        word_embeddings = self.word_embed(word_indices).permute(0, 2, 1).reshape((
            self.batch_size, self.hidden_dim, RealData.im_size[1], RealData.im_size[0]))        # (bs, 32, h, w)
        pattern_embeddings = self.pattern_embed(pattern_indices).permute(0, 2, 1).reshape((
            self.batch_size, self.hidden_dim, RealData.im_size[1], RealData.im_size[0]))        # (bs, 32, h, w)
        char_embeddings = self.char_embed(char_indices).permute(0, 2, 1).reshape((
            self.batch_size, self.hidden_dim, RealData.im_size[1], RealData.im_size[0]))        # (bs, 32, h, w)

        pixels = torch.reshape(pixels,
                               (self.batch_size, 3, RealData.im_size[1], RealData.im_size[0]))       # (bs, 3, h, w)
        parses = torch.reshape(parses,
                               (self.batch_size, 8, RealData.im_size[1], RealData.im_size[0]))       # (bs, 8, h, w)
        memory_mask = torch.reshape(memory_mask,
                                    (self.batch_size, 1, RealData.im_size[1], RealData.im_size[0]))  # (bs, 1, h, w)

        x = torch.cat([pixels,
                       word_embeddings,
                       pattern_embeddings,
                       char_embeddings,
                       parses,
                       self.X,
                       self.Y,
                       memory_mask], dim=1)                                         # (bs, 110, h, w)

        x = self.conv_block(x)                                                      # (bs, 128, h, w)
        x = self.dropout(x)
        pre_att_logits = x

        att_logits = self.conv_out(x)                                               # (bs, n_memories, h, w)
        att_logits = memory_mask * att_logits - (1.0 - memory_mask) * 1000          # (bs, n_memories, h, w)

        logits = torch.reshape(att_logits, (self.batch_size, -1))                   # (bs, h * w * n_memories)
        logits -= torch.max(logits, dim=1, keepdim=True)[0]
        lp = F.log_softmax(logits, dim=1)                                           # (bs, h * w * n_memories)
        spatial_attention = F.softmax(logits, dim=1)                                # (bs, h * w * n_memories)

        p_uniform = memory_mask / torch.sum(memory_mask, dim=(1, 2, 3), keepdim=True)   # (bs, 1, h, w)

        cross_entropy_uniform = -torch.sum(
            p_uniform * torch.reshape(lp, (-1, RealData.n_memories, RealData.im_size[1], RealData.im_size[0])),
            dim=(1, 2, 3)).unsqueeze(1)                                             # (bs, 1)

        cp = torch.sum(
            torch.reshape(spatial_attention, (-1, RealData.n_memories, RealData.im_size[1], RealData.im_size[0])),
            dim=1, keepdim=True)                                                    # (bs, 1, h, w)

        context = torch.sum(cp * pre_att_logits, dim=(1, 2))                        # (bs, 4 * 32)

        return spatial_attention, cross_entropy_uniform, context


class AttendCopyParseModel(nn.Module):

    def __init__(self, field, batch_size):
        super(AttendCopyParseModel, self).__init__()

        self.field = field
        self.batch_size = batch_size
        self.n_hid = 32

        self.parser = None

        if FIELDS[self.field] == FIELD_TYPES["optional"]:
            noop_parser = NoOpParser()
            self.parser = OptionalParser(
                noop_parser, self.batch_size, self.n_hid * 4, RealData.seq_out, RealData.n_output, RealData.eos_idx)
        elif FIELDS[self.field] == FIELD_TYPES["amount"]:
            self.parser = AmountParser(self.batch_size)
        elif FIELDS[self.field] == FIELD_TYPES["date"]:
            self.parser = DateParser()
        else:
            self.parser = NoOpParser()

        self.attend = AttendBlock(batch_size, self.n_hid)

    @staticmethod
    def sparse_dense_mul(s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(i, v * dv, s.size())

    def forward(self, sparse_memory, pixels, word_indices, pattern_indices, char_indices, memory_mask, parses):
        spatial_attention, cross_entropy_uniform, context = self.attend(pixels,
                                                                        word_indices,
                                                                        pattern_indices,
                                                                        char_indices,
                                                                        memory_mask,
                                                                        parses)
        # copy
        x = torch.sparse.sum(
            self.sparse_dense_mul(sparse_memory, spatial_attention), dim=1).to_dense()  # (bs, n_out, seq_in)

        # parse
        parsed = self.parser(x, context)
        return parsed, cross_entropy_uniform
