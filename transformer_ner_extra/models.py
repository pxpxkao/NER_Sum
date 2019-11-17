import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        
    def forward(self, src, tgt, src_mask, tgt_mask, ner):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask, ner), src_mask,
                            tgt, tgt_mask, src)
    
    def encode(self, src, src_mask, ner, class_num=19):
        print("src_embed:", self.src_embed(src).shape, self.src_embed(src).dtype)
        print('ner:', ner.shape, ner.dtype)
        embed = torch.cat([self.src_embed(src), ner], dim=2)
        print("embed:", embed.shape)
        return self.encoder(embed, src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask, src):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, src)

    def loss_compute(self, out, y):
        true_dist = out.data.clone()
        true_dist.fill_(0.)
        true_dist.scatter_(2, y.unsqueeze(2), 1.)
        true_dist[:,:,0] *= 0.01
        return -(true_dist*out).sum(dim=2).mean()
        #return -torch.gather(out, 2, y.unsqueeze(2)).squeeze(2).mean()


    # def greedy_decode(self, src, src_mask, max_len, start_symbol):
    #     memory = self.encode(src, src_mask)
    #     ys = torch.zeros(1, 1).type_as(src.data) + start_symbol
    #     for i in range(max_len-1):
    #         log_prob = self.decode(memory, src_mask, 
    #                            Variable(ys), 
    #                            Variable(subsequent_mask(ys.size(1))
    #                                     .type_as(src.data)), src)
    #         _, next_word = torch.max(log_prob, dim = -1)
    #         next_word = next_word.data[0,-1]
    #         ys = torch.cat([ys, 
    #                         torch.zeros(1, 1).type_as(src.data)+next_word], dim=1)
    #     return ys


    def greedy_decode(self, src, src_mask, max_len, start_symbol, ner):
        memory = self.encode(src, src_mask, ner)

        #print(memory.size())
        #print(src.size())
        ys = torch.zeros(src.size()[0], 1).type_as(src.data) + start_symbol
        for i in range(max_len-1):
            log_prob = self.decode(memory, src_mask, 
                               Variable(ys), 
                               Variable(subsequent_mask(ys.size(1))
                                        .type_as(src.data).expand((ys.size(0), ys.size(1), ys.size(1)))), src)
            #print(log_prob[:, :, :5])
            _, next_word = torch.max(log_prob, dim = -1)
            #
            next_word = next_word.data[:,-1]
            #print(next_word.view(-1, 1))
            ys = torch.cat([ys, 
                            (torch.zeros(src.size()[0], 1).type_as(src.data)) + (next_word.view(-1, 1))], dim=1)
            

        return ys


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, vocab),
            nn.Dropout(0.1)
            )

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


    
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N, d_model, vocab):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.proj = nn.Linear(d_model, vocab)
        
    def forward(self, x, memory, src_mask, tgt_mask, src):
        # print(x.size())
        # print(memory.size())
        # print(src_mask.size())
        # print(tgt_mask.size())
        index = src
        p_gen = 0.2
        #print(index.size())
        for layer in self.layers[:-1]:
            x, _ = layer(x, memory, src_mask, tgt_mask)
        x, attn_weights = self.layers[-1](x, memory, src_mask, tgt_mask)
        # print(x[:, :, 0])
        dec_dist = self.proj(self.norm(x))
        # print(dec_dist.size())
        # print(dec_dist[:, :, :5])
        index = index.unsqueeze(1).expand_as(attn_weights)
        enc_attn_dist = Variable(torch.zeros(dec_dist.size())).cuda().scatter_(-1, index, attn_weights)
        torch.cuda.synchronize()
        #print(torch.nonzero(enc_attn_dist))

        # attn_weights, index = attn_weights.squeeze(1), index.transpose(0,1)
        # output, attn_weights = (output * p_gen), attn_weights * (1-p_gen)
        # output = output.scatter_add_(dim = 1, index = index, src = attn_weights)

        return (1 - p_gen) * F.log_softmax(dec_dist, dim=-1) + p_gen * enc_attn_dist
        #return F.log_softmax(dec_dist + enc_attn_dist, dim=-1)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)[0])
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)[0])
        _, attn_dist = self.src_attn(x, m, m, src_mask)
        #print(attn_dist.size())
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)[0])
        #x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward), attn_dist     