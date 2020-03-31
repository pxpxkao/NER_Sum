import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import subprocess
from models import *
from utils import *
from attention import *


class Solver():
    def __init__(self, args):
        self.args = args
        self.data_utils = data_utils(args)

        self.model = self.make_model(self.data_utils.vocab_size, self.data_utils.vocab_size, 6)

        self.model_dir = make_save_dir(args.model_dir)


    def make_model(self, src_vocab, tgt_vocab, N=6, 
            d_model=512, d_ff=2048, h=8, dropout=0.1):
            
            "Helper: Construct a model from hyperparameters."
            c = copy.deepcopy
            attn = MultiHeadedAttention(h, d_model)
            ff = PositionwiseFeedForward(d_model, d_ff, dropout)
            position = PositionalEncoding(d_model, dropout)
            word_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
            model = EncoderDecoder(
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                     c(ff), dropout), N, d_model, tgt_vocab),
                word_embed,
                word_embed)
            
            # This was important from their code. 
            # Initialize parameters with Glorot / fan_avg.
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform(p)
            return model.cuda()


    def train(self):
        data_yielder = self.data_utils.data_yielder()
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)#get_std_opt(self.model)
        total_loss = []
        start = time.time()
        for step in range(1000000):
            self.model.train()
            batch = data_yielder.__next__()
            
            out = self.model.forward(batch['src'].long(), batch['tgt'].long(), 
                            batch['src_mask'], batch['tgt_mask'])
            pred = out.topk(1, dim=-1)[1].squeeze().detach().cpu().numpy()
            gg = batch['src'].long().detach().cpu().numpy()
            tt = batch['tgt'].long().detach().cpu().numpy()
            yy = batch['y'].long().detach().cpu().numpy()
            loss = self.model.loss_compute(out, batch['y'].long())
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss.append(loss.detach().cpu().numpy())
            
            if step % 500 == 1:
                elapsed = time.time() - start
                print("Epoch Step: %d Loss: %f Time: %f" %
                        (step, np.mean(total_loss), elapsed))
                print('src:',self.data_utils.id2sent(gg[0]))
                print('tgt:',self.data_utils.id2sent(tt[0]))
                print('pred:',self.data_utils.id2sent(pred[0]))


                pp =  self.model.greedy_decode(batch['src'].long()[:1], batch['src_mask'][:1], 14, self.data_utils.bos)
                pp = pp.detach().cpu().numpy()
                print('pred_greedy:',self.data_utils.id2sent(pp[0]))
                
                print()
                start = time.time()
                total_loss = []


            if step % 1000 == 0:
                print('saving!!!!')
                
                model_name = 'model.pth'
                state = {'step': step, 'state_dict': self.model.state_dict()}
                #state = {'step': step, 'state_dict': self.model.state_dict(),
                #    'optimizer' : optim_topic_gen.state_dict()}
                torch.save(state, os.path.join(self.model_dir, model_name))