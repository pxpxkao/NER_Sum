import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import subprocess
# from models import *
from models import *
from utils import *
from attention import *


class Solver():
    def __init__(self, args):
        self.args = args
        self.data_utils = data_utils(args)

        self.model = self.make_model(self.data_utils.vocab_size, self.data_utils.vocab_size)
        torch.cuda.synchronize()

        self.model_dir = make_save_dir(args.model_dir)


    def make_model(self, src_vocab, tgt_vocab, N1=4, N2 = 2, N = 6,
            d_model=512, d_ff=2048, h=8, dropout=0.1):
            
            "Helper: Construct a model from hyperparameters."
            c = copy.deepcopy
            attn = MultiHeadedAttention(h, d_model)
            ff = PositionwiseFeedForward(d_model, d_ff, dropout)
            position = PositionalEncoding(d_model, dropout)
            word_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
            model = EncoderDecoder(
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N1), 
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N1),  
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N2), 
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N2),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N, d_model, tgt_vocab), 
                Generator(d_model, 8), 
                word_embed,
                word_embed)
            
            
            # This was important from their code. 
            # Initialize parameters with Glorot / fan_avg.
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            return model.cuda()

    def load_ner(self, model, path_n):

        state_dict_n = torch.load(path_n)['state_dict']
        for k, v in enumerate(state_dict_n.keys()):
            print(v[0])
            print(v[1].size())
        #print(state_dict_n)
        #self.model.load_state_dict(state_dict)
        return model

    def load_summary(self, model, path_t):
        state_dict_t = torch.load(path_t)['state_dict']
        for k, v in enumerate(state_dict_t.keys()):
            print(v[0])
            print(v[1].size())
        #print(state_dict_t)
        return model

    def train(self):
        # # for k, v in enumerate(self.model.state_dict().keys()):
        # #     print(k)
        # #     print(v.size())
        # state_dict = torch.load("./train_model/5w_model.pth")['state_dict']
        # for k, v in enumerate(state_dict.keys()):
        #     print(v[0])
        #     print(v[1].size())
        # model = self.load_summary(self.model, "../transformer_ptr/train_model/50w_model.pth")
        # model = self.load_ner(self.model, "../ner/transformer-ner/train_model/model_30.pth")
        
        max_len = self.args.max_len
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)#get_std_opt(self.model)
        total_loss = []
        mtl_loss = []
        start = time.time()     
      
        data_yielder_ner = self.data_utils.data_yielder_ner(num_epoch = 1000)
        data_yielder_sum = self.data_utils.data_yielder_sum()
        for step in range(1000000):
            self.model.train()
            
            batch_ner = data_yielder_ner.__next__()
            
            # print(batch_ner['src'].long().size())
            # print(batch_ner['src_mask'].size())
            # print(batch_sum['src'].long().size())
            # print(batch_sum['src_mask'].size())
            # print(batch_sum['tgt'].long().size())
            # print(batch_sum['tgt_mask'].size())
            if step % 20 == 1: # dataset of ner:sum = 14000:280000 = 1:20
                optim.zero_grad()
                out_ner = self.model.forward(batch_ner['src'].long(), 
                                batch_ner['src_mask'], sum=False)
                pred_ner = out_ner.topk(1, dim=-1)[1].squeeze().detach().cpu().numpy()
                gg = batch_ner['src'].long().detach().cpu().numpy()
                yy = batch_ner['y'].long().detach().cpu().numpy()

                loss = self.model.loss_compute(out_ner, batch_ner['y'].long())
                loss.backward()
                optim.step()
                
                total_loss.append(loss.detach().cpu().numpy())

                if step % 500 == 1:
                    elapsed = time.time() - start
                    print("Epoch Step: %d Loss: %f Time: %f" %
                            (step, np.mean(total_loss), elapsed))
                    print('src:',self.data_utils.id2sent(gg[0]))
                    print('tgt:',self.data_utils.id2label(yy[0]))
                    print('pred:',self.data_utils.id2label(pred_ner[0]))

                    print()

           
            batch_sum = data_yielder_sum.__next__()
            optim.zero_grad()
            
            out_sum = self.model.forward(batch_sum['src'].long(), batch_sum['src_mask'],
                batch_sum['tgt'].long(), batch_sum['tgt_mask'], sum=True)
            pred_sum = out_sum.topk(1, dim=-1)[1].squeeze().detach().cpu().numpy()
            gg = batch_sum['src'].long().detach().cpu().numpy()
            tt = batch_sum['tgt'].long().detach().cpu().numpy()
            yy = batch_sum['y'].long().detach().cpu().numpy()

            lamda = 1000
            loss_mtl = self.model.mtl_loss()
            loss = self.model.loss_compute(out_sum, batch_sum['y'].long()) + lamda * loss_mtl

            loss.backward()
            optim.step()
            
            total_loss.append(loss.detach().cpu().numpy())
            mtl_loss.append(lamda * loss_mtl.detach().cpu().numpy())
            
            if step % 500 == 1:
                print("mtl_loss:", mtl_loss)
                elapsed = time.time() - start
                print("Epoch Step: %d Loss: %f Time: %f" %
                        (step, np.mean(total_loss), np.mean(mtl_loss), elapsed))
                print('src:',self.data_utils.id2sent(gg[0]))
                print('tgt:',self.data_utils.id2sent(tt[0]))
                print('pred:',self.data_utils.id2sent(pred_sum[0]))


                pp =  self.model.greedy_decode(batch_sum['src'].long()[:1], batch_sum['src_mask'][:1], max_len, self.data_utils.bos)
                pp = pp.detach().cpu().numpy()
                print('pred_greedy:',self.data_utils.id2sent(pp[0]))
                
                print()
                start = time.time()
                total_loss = []
                mtl_loss = []

            if step % 10000 == 0:
                print('saving!!!!')
                
                model_name = 'model_'+str(step//10000)+'.pth'
                state = {'step': step, 'state_dict': self.model.state_dict()}
                #state = {'step': step, 'state_dict': self.model.state_dict(),
                #    'optimizer' : optim_topic_gen.state_dict()}
                torch.save(state, os.path.join(self.model_dir, model_name))

    def _test(self):
        #prepare model
        path = self.args.load_model
        max_len = self.args.max_len
        state_dict = torch.load(path)['state_dict']
        model = self.model
        model.load_state_dict(state_dict)
        pred_dir = make_save_dir(self.args.pred_dir)
        filename = self.args.filename

        #start decoding
        data_yielder = self.data_utils.data_yielder_sum()
        total_loss = []
        start = time.time()

        #file
        f = open(os.path.join(pred_dir, filename), 'w')

        self.model.eval()

        index = 0
        for batch in data_yielder:
            #print(batch['src'].data.size())
            out = self.model.greedy_decode(batch['src'].long(), batch['src_mask'], max_len, self.data_utils.bos)
            # out = self.model.forward(batch['src'].long(), batch['src_mask'])
            
            # out = torch.argmax(out, dim = 2)
            print(out.size())

            # for i in range(out.size(0)):
            #     nonz = torch.nonzero(batch['src_mask'][i])
            #     print(nonz)
            #     idx = nonz[-1][1].item()
            #     sentence = self.data_utils.id2label(out[i][:idx], True)
            #     #print(l[1:])
            #     f.write(sentence)
            #     f.write("\n")
            index += out.size(0)
            print(index)
            for l in out:
                sentence = self.data_utils.id2sent(l[1:], True)
                #print(l[1:])
                f.write(sentence)
                f.write("\n")