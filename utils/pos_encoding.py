import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_len=1000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        # self.drop_out = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        B,L,C = emb.shape
        pe = self.pe[:L].unsqueeze(0).repeat(B,1,1)
        emb = emb + pe
        return emb