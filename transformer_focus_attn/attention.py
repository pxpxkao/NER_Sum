import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None, focus_score=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if focus_score is not None:
        scores = scores + focus_score
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e15)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, with_focus_attention=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        if with_focus_attention:
            self.linear_focus_query = copy.deepcopy(nn.Linear(d_model, d_model))
            self.linear_focus_global = copy.deepcopy(nn.Linear(d_model, d_model))

            up = torch.randn(h, 1, self.d_k)
            self.up = Variable(up, requires_grad=True).cuda()
            torch.nn.init.xavier_uniform_(self.up)

            uz = torch.randn(h, 1, self.d_k)
            self.uz = Variable(uz, requires_grad=True).cuda()
            torch.nn.init.xavier_uniform_(self.uz)
        self.with_focus_attention = with_focus_attention
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            # mask = (nbatch, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        # query,key,value shape: (nbatches, h, seq_len, d_k)
        # query, key, value = \
        #     [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #      for l, x in zip(self.linears, (query, key, value))]
        key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (key, value))]
        query = self.linears[2](query)

        focus_score = None
        if self.with_focus_attention == True:
            glo = torch.mean(query, dim=1, keepdim=True)

            c = torch.tanh(self.linear_focus_query(query) + self.linear_focus_global(glo))
            c = c.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

            p = c * self.up
            p = p.sum(3).squeeze()
            z = c * self.uz
            z = z.sum(3).squeeze()

            P = torch.sigmoid(p) * key.size(2) # key.size(2) == seq_len
            Z = torch.sigmoid(z) * key.size(2)

            j = torch.arange(start=0, end=key.size(2), dtype=P.dtype).unsqueeze(0).unsqueeze(0).unsqueeze(0).to('cuda')
            P = P.unsqueeze(-1)
            Z = Z.unsqueeze(-1)

            G = - (j-P)**2 * 2 / (Z**2)
            focus_score = G

        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout, focus_score=focus_score)
        # print("attn")
        # print(self.attn.size())
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), torch.mean(self.attn, 1)

if __name__ == '__main__':
    attn = MultiHeadedAttention(8, 512, with_focus_attention=True)
    nbatch, seq_len, d_model = 8, 400, 512
    x = torch.randn((nbatch, seq_len, d_model))
    x, attn_dist = attn(x, x, x)
    print('x:', x.size())
    print('attn dist:', attn_dist.size())