import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *

class EncoderDecoderOrg(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, entity_encoder):
        super(EncoderDecoderOrg, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.entity_encoder = entity_encoder
            
        
    def forward(self, src, tgt, ner, src_mask, tgt_mask, src_extended, oov_nums):
        "Take in and process masked src and target sequences."
        # return self.decode(self.encode(src, src_mask), src_mask,
        #                     tgt, tgt_mask, src)
        return self.decode(self.encode(src, src_mask, ner), src_mask,
                            tgt, tgt_mask, src_extended, oov_nums)
    
    def encode(self, src, src_mask, ner):
        inp = self.src_embed(src)

        ner_mask = inp.clone().unsqueeze(1)
        ner_mask = ner_mask.expand(inp.size(0), ner.size(1), inp.size(1), inp.size(2))
        ner = ner.unsqueeze(2).expand(inp.size(0), ner.size(1), inp.size(1), inp.size(2))
        ner = (ner * ner_mask).mean(dim = 1)

        return self.encoder(inp+ner, src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask, src, oov_nums=None):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, src, oov_nums)

    def loss_compute(self, out, y):
        true_dist = out.data.clone()
        true_dist.fill_(0.)
        true_dist.scatter_(2, y.unsqueeze(2), 1.)
        true_dist[:, :, 0] = 0  #padding idx


        return -(true_dist*torch.log(out+1e-32)).sum(dim=2).mean()
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


    def greedy_decode(self, src, ner, src_mask, max_len, start_symbol, oov_nums, vocab_size, min_len = 30, unk_idx = 3, pad_idx = 0, eos_idx = 1):
        # print('src', src.size())
        extend_mask = src < vocab_size
        memory = self.encode(src * extend_mask, src_mask, ner)

        #print(memory.size())
        
        ys = torch.zeros(src.size()[0], 1).type_as(src.data) + start_symbol
        ret = ys.data.clone()
        for i in range(max_len-1):
            # print('==============')
            # print(ys.size())
            # print(subsequent_mask(ys.size(1))
            #                             .type_as(src.data).expand((ys.size(0), ys.size(1), ys.size(1))).size())
            # print(subsequent_mask(ys.size(1))
            #                             .type_as(src.data).expand((ys.size(0), ys.size(1), ys.size(1))))
            log_prob = self.decode(memory, src_mask, 
                               Variable(ys), 
                               Variable(subsequent_mask(ys.size(1))
                                        .type_as(src.data).expand((ys.size(0), ys.size(1), ys.size(1)))), src, oov_nums)

            if i <= min_len:
                # print('log_prob', log_prob.size())
                log_prob[:,:,eos_idx] = 0.0
                log_prob[:,:,pad_idx] = 0

            log_prob[:,:, unk_idx] = 0.0
            _, next_word = torch.max(log_prob, dim = -1)
            next_word = next_word.data[:,-1]
            # unk_mask = (next_word == unk_idx).long()

            # _, second = torch.topk(log_prob, 2, dim = -1)
            # second = second[:, -1, 1]

            # next_word = (1-unk_mask) * next_word + unk_mask * second

            
            ret = torch.cat([ret, 
                            (torch.zeros(src.size()[0], 1).type_as(src.data)) + (next_word.view(-1, 1))], dim=1)

            m = next_word < vocab_size
            next_word = m * next_word
            ys = torch.cat([ys, 
                            (torch.zeros(src.size()[0], 1).type_as(src.data)) + (next_word.view(-1, 1))], dim=1)
            
        return ys


# class Generator(nn.Module):
#     "Define standard linear + softmax generation step."
#     def __init__(self, d_model, vocab):
#         super(Generator, self).__init__()
#         self.proj = nn.Sequential(
#             nn.Linear(d_model, vocab),
#             nn.Dropout(0.1)
#             )

#     def forward(self, x):
#         return F.log_softmax(self.proj(x), dim=-1)


    
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# class Encoder_ner(nn.Module):
#     def __init__(self, layer, layer_first, N):
#         super(Encoder_ner, self).__init__()
#         self.layers = clones(layer, N-1)
#         self.first = layer_first
#         self.norm = LayerNorm(layer.size)

#     def forward(self, x, mask):
#         x = self.first(x, mask)
#         print('xxx', x.size())
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)


class DecoderOrg(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N, d_model, vocab, pointer_gen):
        super(DecoderOrg, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.proj = nn.Linear(d_model, vocab)
        if pointer_gen:
            print('pointer_gen')
            self.bptr = nn.Parameter(torch.FloatTensor(1, 1))
            self.Wh = nn.Linear(d_model, 1)
            self.Ws = nn.Linear(d_model, 1)
            self.Wx = nn.Linear(d_model, 1)
            self.pointer_gen = True
        else:
            self.pointer_gen = False
        self.sm = nn.Softmax(dim = -1)        
    def forward(self, x, memory, src_mask, tgt_mask, src_extended, oov_nums):

        # σ(w>h ht* + w>s st + w>x xt + bptr)
        #memory                     [batch, src_len, d_model]
        #x_t                        [batch, dec_len, d_model]
        #s_t                        [batch, dec_len, d_model]
        #attn_weights               [batch, dec_len, src_len]  16, 100, 400
        #h_t = memory * attn_dist   [batch, dec_len, d_model]

        # every decoder step needs a p_gen -> p_gen   [batch, dec_len]
        if self.pointer_gen:
            index = src_extended
            x_t = x 
        for layer in self.layers[:-1]:
            x, _ = layer(x, memory, src_mask, tgt_mask)
        x, attn_weights = self.layers[-1](x, memory, src_mask, tgt_mask)
        dec_dist = self.proj(self.norm(x))
        # print('dec_dist', dec_dist.size())
        dec_dist[:, :, 0] = -float('inf')
        if self.pointer_gen:
            s_t = x
            h_t = torch.bmm(attn_weights, memory)  #context vector
            p_gen = self.Wh(h_t) + self.Ws(s_t) + self.Wx(x_t) + self.bptr.expand_as(self.Wh(h_t))
            p_gen = torch.sigmoid(p_gen)

            dec_dist_extended = torch.cat((dec_dist, torch.zeros((dec_dist.size(0), dec_dist.size(1), oov_nums)).cuda()), dim = -1)

            index = index.unsqueeze(1).expand_as(attn_weights)
            # print('max', index.max())
            # print('oovs',oov_nums)
            enc_attn_dist = Variable(torch.zeros(dec_dist_extended.size())).cuda().scatter_add_(dim = -1, index= index, src=attn_weights)


        torch.cuda.synchronize()

        # print('p_gen * enc_attn_dist', (p_gen * enc_attn_dist).size())
        # return p_gen * F.log_softmax(dec_dist, dim=-1) + (1-p_gen) * enc_attn_dist
        if self.pointer_gen:
            return p_gen * self.sm(dec_dist_extended) + (1-p_gen) * enc_attn_dist
        else:
            return self.sm(dec_dist)


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


# class EncoderLayer_ner(nn.Module):
#     "Encoder is made up of self-attn and feed forward (defined below)"
#     def __init__(self, input_size, size, self_attn, feed_forward, dropout):
#         super(EncoderLayer_ner, self).__init__()
#         self.self_attn = self_attn
#         self.feed_forward = feed_forward
#         self.sublayer0 = SublayerConnection(input_size, dropout)
#         self.sublayer1 = SublayerConnection(size, dropout)
#         self.size = size

#     def forward(self, x, mask):
#         x = self.sublayer0(x, lambda x: self.self_attn(x, x, x, mask)[0])
#         return self.sublayer1(x, self.feed_forward)


class DecoderLayerOrg(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayerOrg, self).__init__()
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