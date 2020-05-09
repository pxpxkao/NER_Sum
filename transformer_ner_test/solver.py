import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import subprocess
from models import *
from utils import *
from attention import *
from torch.utils.tensorboard import SummaryWriter
from modules import subsequent_mask
    
class Solver():
    def __init__(self, args):
        self.args = args
        self.data_utils = data_utils(args)

        self.emb_model, self.model = self.make_model(self.data_utils.vocab_size, self.data_utils.vocab_size, args.num_layer, args.dropout)
        #print(self.emb_model)
        print(self.model)
        if self.args.train:
            self.outfile = open(self.args.logfile, 'w')
            self.model_dir = make_save_dir(args.model_dir)
            self.logfile = os.path.join(args.logdir, args.exp_name)
            self.log = SummaryWriter(self.logfile)

    def make_model(self, src_vocab, tgt_vocab, N=6, dropout=0.1,
            d_model=512, d_ff=2048, h=8):
            
            "Helper: Construct a model from hyperparameters."
            c = copy.deepcopy
            attn = MultiHeadedAttention(h, d_model)
            ff = PositionwiseFeedForward(d_model, d_ff, dropout)
            position = PositionalEncoding(d_model, dropout)
            word_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
            emb_model = EncoderDecoder(
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                     c(ff), dropout), N, d_model, tgt_vocab, self.args.pointer_gen),
                word_embed,
                word_embed)

            # Linear Model:
            # generator = Linear(d_model, 19)
            # model = NER_Linear(generator)
            # CNN Model
            model = NER_CNN()
            
            # This was important from their code. 
            # Initialize parameters with Glorot / fan_avg.
            for p in emb_model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform(p)
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform(p)
            return emb_model.cuda(), model.cuda()


    def train(self):

        data_yielder = self.data_utils.data_yielder(self.args.train_file, self.args.train_ner_tgt_file)
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-7, betas=(0.9, 0.998), eps=1e-8, amsgrad=True)
        total_loss = []
        start = time.time()
        print('start training...')
        min_loss = 100000000000
        if self.args.load_embmodel:
            state_dict = torch.load(self.args.load_embmodel)['state_dict']
            self.emb_model.load_state_dict(state_dict)
            print("Loading model from " + self.args.load_embmodel + "...")
        if self.args.load_model:
            state_dict = torch.load(self.args.load_model)['state_dict']
            self.model.load_state_dict(state_dict)
            print("Loading model from " + self.args.load_model + "...")

        warmup_steps = 10000
        d_model = 512
        lr = 1e-7

        self.emb_model.eval()
        for step in range(1000002):
            self.model.train()
            batch = data_yielder.__next__()
            # if step % 100 == 1:
            #     lr = (1/(d_model**0.5))*min((1/step**0.5), step * (1/(warmup_steps**1.5)))
            #     for param_group in optim.param_groups:
            #         param_group['lr'] = lr

            embedding = self.emb_model.encode_emb(batch['src'].long())
            # embedding = self.emb_model.encode(batch['src'].long(), batch['src_mask'])
            # print("Embedding Size:", embedding.size())
            out = self.model.forward(embedding)
            k = 100
            pred = out.topk(1, dim=-1)[1].squeeze().detach().cpu().numpy()[0][:k]
            gg = batch['src'].long().detach().cpu().numpy()[0][:k]
            yy = batch['y'].long().detach().cpu().numpy()[0][:k]
            loss = self.model.loss_compute(out, batch['y'].long())
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss.append(loss.detach().cpu().numpy())
            
            if step % 500 == 1:
                elapsed = time.time() - start
                print("Epoch Step: %d Loss: %f Time: %f lr: %6.6f" %
                        (step, np.mean(total_loss), elapsed, optim.param_groups[0]['lr']))
                self.outfile.write("Epoch Step: %d Loss: %f Time: %f\n" %
                        (step, np.mean(total_loss), elapsed))
                try:
                    print('src:\n',self.data_utils.id2sent(gg))
                    print('tgt:\n',self.data_utils.id2label(yy))
                    print('pred:\n',self.data_utils.id2label(pred))
                except:
                    pass
                
                print()
                start = time.time()
                self.log.add_scalar('Loss/train', np.mean(total_loss), step)
                total_loss = []
                
            if step % 10000 == 1:
                self.model.eval()
                val_yielder = self.data_utils.data_yielder(self.args.valid_file, self.args.valid_ner_tgt_file, 1)
                total_loss = []
                for batch in val_yielder:
                    embedding = self.emb_model.encode(batch['src'].long(), batch['src_mask'])
                    out = self.model.forward(embedding)
                    loss = self.model.loss_compute(out, batch['y'].long())
                    total_loss.append(loss.item())
                print('=============================================')
                print('Validation Result -> Loss : %6.6f' %(sum(total_loss)/len(total_loss)))
                print('=============================================')
                self.outfile.write('=============================================\n')
                self.outfile.write('Validation Result -> Loss : %6.6f\n' %(sum(total_loss)/len(total_loss)))
                self.outfile.write('=============================================\n')
                # self.model.train()
                self.log.add_scalar('Loss/valid', sum(total_loss)/len(total_loss), step)

                if min_loss > sum(total_loss)/len(total_loss):
                    min_loss = sum(total_loss)/len(total_loss)
                    print('Saving ' + str(step//10000) + 'w_model.pth!\n')
                    self.outfile.write('Saving ' + str(step//10000) + 'w_model.pth\n')
                    idx_dir = make_save_dir(os.path.join(self.model_dir, self.args.idx))
                    model_name = str(step//10000) + 'w_' + '%6.6f'%(sum(total_loss)/len(total_loss)) + 'model.pth'
                    state = {'step': step, 'state_dict': self.model.state_dict()}

                    torch.save(state, os.path.join(idx_dir, model_name))
                else:
                    print('Valid Loss did not decrease on step', str(step))
                    self.outfile.write('Valid Loss did not decrease on step' + str(step) + '\n')

    def test(self):
        #prepare embedding model
        path = self.args.load_embmodel
        state_dict = torch.load(path)['state_dict']
        emb_model = self.emb_model
        emb_model.load_state_dict(state_dict)

        #prepare NER model
        path = self.args.load_model
        state_dict = torch.load(path)['state_dict']
        model = self.model
        model.load_state_dict(state_dict)

        #make pred directory
        pred_dir = make_save_dir(self.args.pred_dir)
        filename = self.args.filename

        #start decoding
        data_yielder = self.data_utils.data_yielder(self.args.test_file, self.args.test_ner_tgt_file, 1)
        start = time.time()

        #file
        f = open(os.path.join(pred_dir, filename), 'w', encoding='utf-8')

        self.emb_model.eval()
        self.model.eval()
        step = 0
        for batch in data_yielder:
            step += 1
            if step % 100 == 0:
                print('%d batch processed. Time elapsed: %f min.' %(step, (time.time() - start)/60.0))
                start = time.time()
            
            embedding = self.emb_model.encode_emb(batch['src'].long())
            # embedding = self.emb_model.encode(batch['src'].long(), batch['src_mask'])
            out = self.model.forward(embedding)
            
            out = torch.argmax(out, dim = 2)
            #print('out size:', out.size())

            for i in range(out.size(0)):
                nonz = torch.nonzero(batch['src_mask'][i])
                #print(nonz)
                idx = nonz[-1][1].item()+1
                #print('non zero idx:', idx)
                sentence = self.data_utils.id2label(out[i][:idx], True)
                #print(l[1:])
                f.write(sentence)
                f.write("\n")