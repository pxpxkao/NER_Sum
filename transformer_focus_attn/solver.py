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
from beam import Beam
from modules import subsequent_mask

class Solver():
    def __init__(self, args):
        self.args = args
        self.data_utils = data_utils(args)

        self.model = self.make_model(self.data_utils.vocab_size, self.data_utils.vocab_size, args.num_layer, args.dropout)
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
            attn_focus = MultiHeadedAttention(h, d_model, with_focus_attention=True)
            ff = PositionwiseFeedForward(d_model, d_ff, dropout)
            position = PositionalEncoding(d_model, dropout)
            word_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
            model = EncoderDecoder(
                Encoder(EncoderLayer(d_model, attn_focus, c(ff), dropout), N),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                     c(ff), dropout), N, d_model, tgt_vocab, self.args.pointer_gen),
                word_embed,
                word_embed)
            
            # This was important from their code. 
            # Initialize parameters with Glorot / fan_avg.
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform(p)
            return model.cuda()


    def train(self):

        data_yielder = self.data_utils.data_yielder(self.args.train_file, self.args.tgt_file)
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-7, betas=(0.9, 0.998), eps=1e-8, amsgrad=True)#get_std_opt(self.model)
        total_loss = []
        start = time.time()
        print('start training...')
        if self.args.load_model:
            state_dict = torch.load(self.args.load_model)['state_dict']
            self.model.load_state_dict(state_dict)
            print("Loading model from " + self.args.load_model + "...")

        warmup_steps = 10000
        d_model = 512
        lr = 1e-7
        #path = torch.load("./train_model/10w_model.pth")
        #self.model.load_state_dict(path, strict = False)
        for step in range(1000000):
            self.model.train()
            batch = data_yielder.__next__()
            if step % 100 == 1:
                lr = (1/(d_model**0.5))*min((1/step**0.5), step * (1/(warmup_steps**1.5)))
                for param_group in optim.param_groups:
                    param_group['lr'] = lr
                
            # if True:
            batch['src'] = batch['src'].long()
            batch['tgt'] = batch['tgt'].long()
            batch['src_extended'] = batch['src_extended'].long()
            # print(batch['oov_list'])

            out = self.model.forward(batch['src'], batch['tgt'], 
                            batch['src_mask'], batch['tgt_mask'], batch['src_extended'], len(batch['oov_list']))
            pred = out.topk(1, dim=-1)[1].squeeze().detach().cpu().numpy()[0]
            gg = batch['src_extended'].long().detach().cpu().numpy()[0][:100]
            tt = batch['tgt'].long().detach().cpu().numpy()[0]
            yy = batch['y'].long().detach().cpu().numpy()[0]
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
                print('src:\n',self.data_utils.id2sent(gg, False, False, batch['oov_list']))
                print('tgt:\n',self.data_utils.id2sent(yy, False, False, batch['oov_list']))
                print('pred:\n',self.data_utils.id2sent(pred, False, False, batch['oov_list']))
                print('oov_list:\n', batch['oov_list'])

                pp =  self.model.greedy_decode(batch['src_extended'].long()[:1], batch['src_mask'][:1], 100, self.data_utils.bos, len(batch['oov_list']), self.data_utils.vocab_size)
                pp = pp.detach().cpu().numpy()
                print('pred_greedy:\n',self.data_utils.id2sent(pp[0], False, False, batch['oov_list']))
                
                print()
                start = time.time()
                self.log.add_scalar('Loss/train', np.mean(total_loss), step)
                total_loss = []
                
                
            if step % 100000 == 2:
                self.model.eval()
                val_yielder = self.data_utils.data_yielder(self.args.valid_file, self.args.valid_tgt_file, 1)
                total_loss = []
                for batch in val_yielder:
                    batch['src'] = batch['src'].long()
                    batch['tgt'] = batch['tgt'].long()
                    batch['src_extended'] = batch['src_extended'].long()
                    # print(len(batch['oov_list']))
                    out = self.model.forward(batch['src'], batch['tgt'], 
                            batch['src_mask'], batch['tgt_mask'], batch['src_extended'], len(batch['oov_list']))
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

                w_step = int(step/100000)
                if self.args.load_model:
                    w_step += (int(self.args.load_model.split('/')[-1][0]))
                print('Saving ' + str(w_step) + '0w_model.pth!\n')
                self.outfile.write('Saving ' + str(w_step) + '0w_model.pth\n')
                model_name = str(w_step) + '0w_' + '%6.6f'%(sum(total_loss)/len(total_loss)) + 'model.pth'
                state = {'step': step, 'state_dict': self.model.state_dict()}

                torch.save(state, os.path.join(self.model_dir, model_name))


    def test(self):
        #prepare model
        path = self.args.load_model
        max_len = self.args.max_len
        state_dict = torch.load(path)['state_dict']
        model = self.model
        model.load_state_dict(state_dict)
        pred_dir = make_save_dir(self.args.pred_dir)
        filename = self.args.filename

        #start decoding
        data_yielder = self.data_utils.data_yielder(self.args.test_file, self.args.test_file)
        total_loss = []
        start = time.time()

        #file
        f = open(os.path.join(pred_dir, filename), 'w')

        self.model.eval()
            
        step = 0
        for batch in data_yielder:
            #print(batch['src'].data.size())
            step += 1
            if step % 100 == 0:
                print('%d batch processed. Time elapsed: %f min.' %(step, (time.time() - start)/60.0))
                start = time.time()
            if self.args.beam_size == 1:
                out = self.model.greedy_decode(batch['src'].long(), batch['src_mask'], max_len, self.data_utils.bos, len(batch['oov_list']), self.data_utils.vocab_size)
            else:
                out = self.beam_decode(batch, max_len, len(batch['oov_list']))
            #print(out)
            for l in out:
                sentence = self.data_utils.id2sent(l[1:], True, self.args.beam_size!=1, batch['oov_list'])
                #print(l[1:])
                f.write(sentence)
                f.write("\n")


    def beam_decode(self, batch, max_len, oov_nums):

        bos_token = self.data_utils.bos 
        beam_size = self.args.beam_size
        vocab_size = self.data_utils.vocab_size

        src = batch['src'].long()
        src_mask = batch['src_mask']
        src_extended = batch['src_extended'].long()
        memory = self.model.encode(src, src_mask)
        batch_size = src.size(0)

        beam = Beam(self.data_utils.pad, 
                    bos_token, 
                    self.data_utils.eos, 
                    beam_size, 
                    batch_size,
                    self.args.n_best,
                    True,
                    max_len
                    )

        ys = torch.full((batch_size, 1), bos_token).type_as(src.data).cuda()
        log_prob = self.model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data).expand((ys.size(0), ys.size(1), ys.size(1)))), src_extended, oov_nums)

        
        # log_prob = [batch_size, 1, voc_size]
        top_prob, top_indices = torch.topk(input = log_prob, k = beam_size, dim = -1)
        # print(top_indices)
        top_prob = top_prob.view(-1, 1)
        top_indices = top_indices.view(-1, 1)
        beam.update_prob(top_prob.detach().cpu(), top_indices.detach().cpu())
        # [batch_size, 1, beam_size]
        ys = top_indices
        top_indices = None
        # print(ys.size())
        ####### repeat var #######
        src = torch.repeat_interleave(src, beam_size, dim = 0)
        src_mask = torch.repeat_interleave(src_mask, beam_size, dim = 0)
        #[batch_size, src_len, d_model] -> [batch_size*beam_size, src_len, d_model]
        memory = torch.repeat_interleave(memory, beam_size, dim = 0)
        # print('max_len', max_len)
        for t in range(1, max_len):
            log_prob = self.model.decode(memory, src_mask, 
                               Variable(ys), 
                               Variable(subsequent_mask(ys.size(1)).type_as(src.data).expand((ys.size(0), ys.size(1), ys.size(1)))), src)
            # print('log_prob', log_prob.size())
            log_prob = log_prob[:,-1].unsqueeze(1)
            # print(beam.seq)
            real_top = beam.advance(log_prob.detach().cpu())
            # print(real_top.size())
            # print(ys.size())
            # print(real_top.size())
            ys = torch.cat((ys, real_top.view(-1, 1).cuda()), dim = -1)
            # print(ys.size())

        # print(ys.size())
        # print(beam.top_prob)
        # print(len(beam.seq))


        return [beam.seq[0]]
