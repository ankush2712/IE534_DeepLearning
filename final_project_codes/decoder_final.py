
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StatefulLSTM(nn.Module):
    def __init__(self, in_size, out_size):
        super(StatefulLSTM, self).__init__()

        self.lstm = nn.LSTMCell(in_size, out_size)
        self.out_size = out_size

        self.h = None
        self.c = None

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        batch_size = x.data.size()[0]
        if self.h is None:
            state_size = [batch_size, self.out_size]
            self.c = Variable(torch.zeros(state_size)).cuda()
            self.h = Variable(torch.zeros(state_size)).cuda()
        self.h, self.c = self.lstm(x, (self.h, self.c))

        return self.h,self.c


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout,self).__init__()
        self.m = None

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5, train=True):
        if train==False:
            return x
        if(self.m is None):
            self.m = x.data.new(x.size()).bernoulli_(1 - dropout)
        mask = Variable(self.m, requires_grad=False) / (1 - dropout)

        return mask * x


class NIC_language_model(nn.Module):
    def __init__(self, out_resnet, vocab_size, embed_dim):
        super(NIC_language_model, self).__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        self.img_embedding = nn.Linear(out_resnet, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # ,padding_idx=0)

        self.lstm = StatefulLSTM(embed_dim, embed_dim)
        self.bn_lstm = nn.BatchNorm1d(embed_dim)
        self.dropout = LockedDropout()  # torch.nn.Dropout(p=0.5)

        self.decoder = nn.Linear(embed_dim, vocab_size)

    def reset_state(self):
        self.lstm.reset_state()
        self.dropout.reset_state()

    def forward(self, img, encoded_cap, caption_lengths, train=True):
        self.reset_state()
        vocab_size = self.vocab_size

        batch_s = img.size(0)
        img = img.view(batch_s, -1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        img = img[sort_ind]
        encoded_cap = encoded_cap[sort_ind]

        img = self.img_embedding(img)
        h,c = self.lstm(img)
        h = self.bn_lstm(h)
        h = self.dropout(h, dropout=0.3, train=train)

        # Caption embedding
        embed = self.embedding(encoded_cap)  # batch_size, time_steps, features

        
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_s, max(decode_lengths), vocab_size).to(device)

        # At each time-step, decode
        # then generate a new word in the decoder with the previous word
        for t in range(max(decode_lengths)):
            h,c = self.lstm(embed[:, t, :])
            h = self.bn_lstm(h)
            h = self.dropout(h, dropout=0.3, train=train)

            h = self.decoder(h)

            predictions[:, t, :] = h

        return predictions, encoded_cap, decode_lengths, sort_ind
