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
    def __init__(self, encoder_sum, encoder_ner, shared_encoder_sum, shared_encoder_ner, decoder, generator, src_embed, tgt_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder_sum = encoder_sum
        self.encoder_ner = encoder_ner
        self.shared_encoder_sum = shared_encoder_sum
        self.shared_encoder_ner = shared_encoder_ner
        self.decoder = decoder
        self.generator = generator
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        
        
    def forward(self, src, src_mask, tgt=None, tgt_mask=None, sum = True):
        "Take in and process masked src and target sequences."  #sum = True => return encoder decoder output; else=>return the output of ner
        
        if sum:
            out_sum = self.encode_sum(src, src_mask)
            out_enc = self.shared_encoder_sum(out_sum, src_mask)
            out = self.decode(out_enc, src_mask,
                            tgt, tgt_mask, src)
        else:
            out_ner = self.encode_ner(src, src_mask)
            out_enc = self.shared_encoder_ner(out_ner, src_mask)
            out = self.generator(out_enc)
        return out
    
    def encode_sum(self, src, src_mask):
        return self.encoder_sum(self.src_embed(src), src_mask)

    def encode_ner(self, src, src_mask):
        return self.encoder_ner(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask, src):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, src)

    def loss_compute(self, out, y):
        true_dist = out.data.clone()
        true_dist.fill_(0.)
        true_dist.scatter_(2, y.unsqueeze(2), 1.)
        true_dist[:,:,0] *= 0.01
        loss = -(true_dist*out).sum(dim=2).mean()
        return loss
        #return -torch.gather(out, 2, y.unsqueeze(2)).squeeze(2).mean()

    def mtl_loss(self):
        sum_params = []
        loss = []
        for name, param in self.shared_encoder_sum.named_parameters():
            if param.requires_grad:
                sum_params.append((name, param.data.clone()))
        i = 0
        for name, param in self.shared_encoder_ner.named_parameters():
            if param.requires_grad:
                if name != sum_params[i][0]:
                    print('ner params: ', name, sum_params[i][0])
                else:
                    loss.append(torch.dist(param, sum_params[i][1])/torch.numel(param))
                i+=1

        return sum(loss)

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


    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        out_sum = self.encode_sum(src, src_mask)
        memory = self.shared_encoder(out_sum, src_mask)
        #memory = self.encode_sum(self.encode_ner(src, src_mask), src_mask)

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
        p_gen = 0.8
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

        #return (1 - p_gen) * F.log_softmax(dec_dist, dim=-1) + p_gen * 
        return torch.log(p_gen * F.softmax(dec_dist, dim=-1) + (1 - p_gen)  * enc_attn_dist)
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