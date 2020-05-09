import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *

class NER_Linear(nn.Module):
    def __init__(self, generator):
        super(NER_Linear, self).__init__()
        self.generator = generator
    
    def forward(self, embedding):
        return self.generator(embedding)

    def loss_compute(self, out, y):
        true_dist = out.data.clone()
        true_dist.fill_(0.)
        true_dist.scatter_(2, y.unsqueeze(2), 1.)
        true_dist[:,:,0] *= 0.01
        return -(true_dist*out).sum(dim=2).mean()

class Linear(nn.Module):
    def __init__(self, d_model, vocab):
        super(Linear, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class NER_CNN(nn.Module):
    def __init__(self, in_=512, out_=128, kernel_=3, num_class=19):
        super(NER_CNN, self).__init__()
        self.kernel_size = kernel_
        self.conv = nn.Conv1d(in_channels=in_, out_channels=out_, kernel_size=kernel_)
        self.fc = nn.Linear(out_, num_class)
    def forward(self, embedding):
        print("Embedding Size:", embedding.size())
        pad_row = torch.zeros([embedding.size(0), self.kernel_size//2, embedding.size(2)])
        emb = torch.cat([pad_row, embedding, pad_row], dim=1)
        emb = emb.permute(0, 2, 1)
        fc = self.conv(emb).permute(0, 2, 1)
        # print(fc.size())
        out = F.log_softmax(self.fc(fc), dim=-1)
        # print("Out Size:", out.size())
        return out
    def loss_compute(self, out, y):
        true_dist = out.data.clone()
        true_dist.fill_(0.)
        true_dist.scatter_(2, y.unsqueeze(2), 1.)
        true_dist[:,:,0] *= 0.01
        return -(true_dist*out).sum(dim=2).mean()

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
        
    def forward(self, src, tgt, src_mask, tgt_mask, src_extended, oov_nums):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask, src_extended, oov_nums)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask, src, oov_nums=None):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, src, oov_nums)

    def loss_compute(self, out, y):
        true_dist = out.data.clone()
        true_dist.fill_(0.)
        true_dist.scatter_(2, y.unsqueeze(2), 1.)
        true_dist[:,:,0] *= 0.01

        return -(true_dist*torch.log(out)).sum(dim=2).mean()

    def greedy_decode(self, src, src_mask, max_len, start_symbol, oov_nums, vocab_size):
        # print('src', src.size())
        extend_mask = (src < vocab_size).long()
        memory = self.encode(src * extend_mask, src_mask)

        #print(memory.size())
        
        ys = torch.zeros(src.size()[0], 1).type_as(src.data) + start_symbol
        ret = ys.data.clone()
        for i in range(max_len-1):

            log_prob = self.decode(memory, src_mask, 
                               Variable(ys), 
                               Variable(subsequent_mask(ys.size(1))
                                        .type_as(src.data).expand((ys.size(0), ys.size(1), ys.size(1)))), src, oov_nums)

            _, next_word = torch.max(log_prob, dim = -1)

            next_word = next_word.data[:,-1]
            ret = torch.cat([ret, 
                            (torch.zeros(src.size()[0], 1).type_as(src.data)) + (next_word.view(-1, 1))], dim=1)

            m = (next_word < vocab_size).long()
            next_word = m * next_word
            ys = torch.cat([ys, 
                            (torch.zeros(src.size()[0], 1).type_as(src.data)) + (next_word.view(-1, 1))], dim=1)
            
        return ys

    
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
    def __init__(self, layer, N, d_model, vocab, pointer_gen):
        super(Decoder, self).__init__()
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

if __name__ == '__main__':
    model = NER_CNN()
    embedding = torch.randn([8, 400, 512])
    model(embedding)