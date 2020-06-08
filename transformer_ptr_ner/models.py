import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *
from attention import MultiHeadedAttention

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, entity_encoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.entity_encoder = entity_encoder
        
    def forward(self, src, tgt, ner, src_mask, tgt_mask, src_extended, oov_nums, ner_mask):
        "Take in and process masked src and target sequences."
        # return self.decode(self.encode(src, src_mask), src_mask,
        #                     tgt, tgt_mask, src)
        return self.decode(self.encode(src, src_mask), ner, src_mask,
                            tgt, tgt_mask, src_extended, oov_nums, ner_mask=ner_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, ner, src_mask, tgt, tgt_mask, src, oov_nums=None, print_gate=False, ner_mask=None):
        return self.decoder(self.tgt_embed(tgt), memory, ner, src_mask, tgt_mask, src, oov_nums, print_gate, ner_mask)

    def loss_compute(self, out, y, padding_idx=0):
        true_dist = out.data.clone()
        true_dist.fill_(0.)
        true_dist.scatter_(2, y.unsqueeze(2), 1.)
        true_dist[:,:,0] *= 0.0

        return -(true_dist*torch.log(out)).sum(dim=2).mean()
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


    def greedy_decode(self, src, ner, src_mask, max_len, start_symbol, oov_nums, vocab_size, print_gate=False, ner_mask=None):
        # print('src', src.size())
        extend_mask = src < vocab_size
        memory = self.encode(src * extend_mask, src_mask)

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
            log_prob = self.decode(memory, ner, src_mask, 
                               Variable(ys), 
                               Variable(subsequent_mask(ys.size(1))
                                        .type_as(src.data).expand((ys.size(0), ys.size(1), ys.size(1)))), src, oov_nums, print_gate, ner_mask)

            _, next_word = torch.max(log_prob, dim = -1)

            next_word = next_word.data[:,-1]
            ret = torch.cat([ret, 
                            (torch.zeros(src.size()[0], 1).type_as(src.data)) + (next_word.view(-1, 1))], dim=1)

            m = next_word < vocab_size
            next_word = m * next_word
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
    def __init__(self, layer, N, d_model, vocab, pointer_gen, ner_last):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        
        
        if pointer_gen:
            print('pointer_gen')
            self.bptr = nn.Parameter(torch.FloatTensor(1, 1))
            self.Wh = nn.Linear(d_model, 1)
            
            self.Wx = nn.Linear(d_model, 1)
            self.pointer_gen = True
        else:
            self.pointer_gen = False
        self.sm = nn.Softmax(dim = -1) 
        self.ner_last = ner_last
        if self.ner_last:   # if last layer ner -> 2 * d_model (ner concat with x)
            self.proj = nn.Linear(2 * d_model, vocab)
            self.norm = LayerNorm(2 * layer.size)
            self.Ws = nn.Linear(2*d_model, 1)
            self.ner_attn = MultiHeadedAttention(1, d_model, 0.3)
        else:
            self.proj = nn.Linear(d_model, vocab)
            self.norm = LayerNorm(layer.size)
            self.Ws = nn.Linear(d_model, 1)

    def forward(self, x, memory, ner, src_mask, tgt_mask, src_extended, oov_nums, print_gate, ner_mask):

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
        if self.ner_last:
            for layer in self.layers[:-1]:
                x, _ = layer(x, memory, src_mask, tgt_mask, ner_mask)
            x, attn_weights = self.layers[-1](x, memory, src_mask, tgt_mask, ner_mask)
            x_ner = self.ner_attn(x, ner, ner)[0]
            x = torch.cat((x, x_ner), dim = -1)
        else:
            # g_list = []
            for layer in self.layers[:-1]:
                x, _, _ = layer(x, memory, ner, src_mask, tgt_mask, ner_mask)
            x, attn_weights, _ = self.layers[-1](x, memory, ner, src_mask, tgt_mask, ner_mask)
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


class DecoderLayer_ner(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, ner_attn, feed_forward, dropout, fusion = 'concat'):
        super(DecoderLayer_ner, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.ner_attn = ner_attn
        self.feed_forward = feed_forward
        # self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.sublayer = clones(SublayerConnection(size, dropout), 4)
        self.fusion = fusion
        if fusion == 'concat':
            self.l = nn.Linear(2*size, size)
        elif fusion == 'gated':
            self.W = nn.Linear(size, 1)
            self.U = nn.Linear(size, 1)

 
    def forward(self, x, memory, ner, src_mask, tgt_mask, ner_mask):
        # print('ner', ner.size())
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)[0])
        _, attn_dist = self.src_attn(x, m, m, src_mask)
        #print(attn_dist.size())
        x_src = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)[0])
        x_ner = self.sublayer[2](x, lambda x: self.ner_attn(x, ner, ner, ner_mask)[0])
        
        if self.fusion == 'concat':
            x = self.l(torch.cat((x_src, x_ner), dim = -1))
            gate = torch.zeros((1, 1))
        elif self.fusion == 'gated':
            gate = torch.sigmoid(self.W(x_src) + self.U(x_ner))
            x = gate * x_src + (1-gate) * x_ner
            # print('x_src', x_src.size())
            # print('x_ner', x_ner.size())
            # print('gate', gate.size())
            # print('x', x.size())

        #x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[3](x, self.feed_forward), attn_dist, gate.mean()


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



from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, embed, input_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.embed = embed

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        return self.tcn(self.embed(inputs))


class Entity_Encoder(nn.Module):
    def __init__(self, embed, encoder):
        super(Entity_Encoder, self).__init__()
        self.embed = embed
        self.encoder = encoder

    def forward(self, inputs, mask=None):
        x = self.embed(inputs)
        x = self.encoder(x, mask)
        return x


class Albert_Encoder(nn.Module):
    def __init__(self, albert, albert_tokenizer, d_model):
        super(Albert_Encoder, self).__init__()
        self.encoder = albert
        self.tokenizer = albert_tokenizer
        self.linear = nn.Linear(768, d_model)

    def forward(self, ner):
        x = self.encoder(ner)[0].detach()
        return self.linear(x)


class Seq_Entity_Encoder(nn.Module):
    def __init__(self, embed, encoder):
        super(Seq_Entity_Encoder, self).__init__()
        self.embed = embed
        self.encoder = encoder

    def forward(self, inputs, mask=None):
        x = self.embed(inputs)
        x = self.encoder(x, mask)
        return x[:,-1]



def gen_mask(lengths, _cuda=True):
    batch_size = lengths.size(0)
    max_len = lengths.max()
    m = torch.arange(max_len).repeat(batch_size, 1).cuda()

    length_mat = lengths.unsqueeze(-1).expand_as(m).cuda()
    mask = torch.where(m<length_mat, torch.ones((batch_size, max_len)).cuda(), torch.zeros((batch_size, max_len)).cuda())
    if _cuda:
        return mask.unsqueeze(1).cuda()
    else:
        return mask.unsqueeze(1)