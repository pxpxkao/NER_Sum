from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import subprocess
from models import *
from org_models import *
from utils import *
from attention import *
# from torch.utils.tensorboard import SummaryWriter
from beam import Beam
from modules import subsequent_mask
# from rouge import FilesRouge 
from rouge_score import rouge_scorer
from transformers import AlbertModel, AlbertTokenizer
import sys
sys.setrecursionlimit(1000000)

def cal_rouge_score(filename1, filename2):
    f1 = open(filename1, 'r')
    f2 = open(filename2, 'r')
    summary = f1.readlines()
    reference = f2.readlines()
    """
    for i in range(len(summary)):
        summary[i] = re.sub('[%s]' % re.escape(string.punctuation), '', summary[i])
        reference[i] = re.sub('[%s]' % re.escape(string.punctuation), '', reference[i])
    """
    for i in range(len(summary)):
        summary[i] = summary[i].strip().replace('<t>', '').replace('</t>', '').strip()
        reference[i] = reference[i].strip().replace('<t>', '').replace('</t>', '').strip()
    print(len(summary))
    print(len(reference))
    #summary = summary[1:1000]
    #reference = reference[1:1000]
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    scores = {"rouge1":[], "rouge2":[],"rougeL":[],"rougeLsum":[]}
    for i in range(len(summary)):
        s = scorer.score(summary[i], reference[i])
        if i%1000==0:
            print(i)
        for k, v in s.items():
            scores[k].append(s[k].fmeasure)

    for k, v in scores.items():
        scores[k] = sum(v)/len(v)
    return scores
        
class Solver():
    def __init__(self, args):
        self.args = args
        self.data_utils = data_utils(args)
        self.disable_comet = args.disable_comet
        self.model = self.make_model(src_vocab = self.data_utils.vocab_size, 
                                                          tgt_vocab = self.data_utils.vocab_size, 
                                                          N = args.num_layer, dropout = args.dropout,
                                                          entity_encoder_type= args.entity_encoder_type
                                                          )
        print(self.model)
        if self.args.train:
            self.outfile = open(self.args.logfile, 'w')
            self.model_dir = make_save_dir(args.model_dir)
            # self.logfile = os.path.join(args.logdir, args.exp_name)
            # self.log = SummaryWriter(self.logfile)
            self.w_valid_file = args.w_valid_file

    def make_model(self, src_vocab, tgt_vocab, N=6, dropout=0.1,
            d_model=512, entity_encoder_type='linear', d_ff=2048, h=8):
            
            "Helper: Construct a model from hyperparameters."
            c = copy.deepcopy
            attn = MultiHeadedAttention(h, d_model)
            attn_ner = MultiHeadedAttention(1, d_model, dropout)
            ff = PositionwiseFeedForward(d_model, d_ff, dropout)
            position = PositionalEncoding(d_model, dropout)
            embed = Embeddings(d_model, src_vocab)
            word_embed = nn.Sequential(embed, c(position))
            print('pgen', self.args.pointer_gen)

            if entity_encoder_type == 'linear':
                entity_encoder = nn.Sequential(embed, nn.Linear(d_model, d_model), nn.ReLU())
                print('linear')
            elif entity_encoder_type == 'MLP':
                entity_encoder = nn.Sequential(embed, nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU())
                print('MLP')
            elif entity_encoder_type == 'transformer':
                # entity_encoder = nn.Sequential(embed, Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), 1))
                print('transformer')
                entity_encoder = Entity_Encoder(embed, Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), 2))
            elif entity_encoder_type == 'albert':
                albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
                albert = AlbertModel.from_pretrained('albert-base-v2')
                entity_encoder = Albert_Encoder(albert, albert_tokenizer, d_model)

            

            if self.args.ner_at_embedding:
                model = EncoderDecoderOrg(
                        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                        DecoderOrg(DecoderLayerOrg(d_model, c(attn), c(attn), 
                                             c(ff), dropout), N, d_model, tgt_vocab, self.args.pointer_gen),
                        word_embed,
                        word_embed, 
                        entity_encoder
                        )
            else:
                if self.args.ner_last:
                    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                         c(ff), dropout), N, d_model, tgt_vocab, self.args.pointer_gen, self.args.ner_last)
                else:
                    decoder = Decoder(DecoderLayer_ner(d_model, c(attn), c(attn), attn_ner,
                                         c(ff), dropout, self.args.fusion), N, d_model, tgt_vocab, self.args.pointer_gen, self.args.ner_last)
                model = EncoderDecoder(
                        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                        decoder,
                        word_embed,
                        word_embed,
                        entity_encoder
                        )
            
            # This was important from their code. 
            # Initialize parameters with Glorot / fan_avg.
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

            
            # levels = 3
            # num_chans = [d_model] * (args.levels)
            # k_size = 5
            # tcn = TCN(embed, d_model, num_channels, k_size, dropout=dropout)

            return model.cuda()


    def train(self):
        if not self.disable_comet:
            # logging
            hyper_params = {
                "num_layer": self.args.num_layer,
                "pointer_gen": self.args.pointer_gen,
                "ner_last": self.args.ner_last,
                "entity_encoder_type": self.args.entity_encoder_type,
                "fusion": self.args.fusion,
                "dropout": self.args.dropout,
            }
            COMET_PROJECT_NAME='summarization'
            COMET_WORKSPACE='timchen0618'


            self.exp = Experiment(api_key='mVpNOXSjW7eU0tENyeYiWZKsl',
                                      project_name=COMET_PROJECT_NAME,
                                      workspace=COMET_WORKSPACE,
                                      auto_output_logging='simple',
                                      auto_metric_logging=None,
                                      display_summary=False,
                                      )
            self.exp.log_parameters(hyper_params)
            self.exp.add_tags(['%s entity_encoder'%self.args.entity_encoder_type, self.args.fusion])
            if self.args.ner_last:
                self.exp.add_tag('ner_last')
            if self.args.ner_at_embedding:
                self.exp.add_tag('ner_at_embedding')
            self.exp.set_name(self.args.exp_name)

        print('ner_last ', self.args.ner_last)
        print('ner_at_embedding', self.args.ner_at_embedding)
        # dataloader & optimizer
        data_yielder = self.data_utils.data_yielder(self.args.train_file, self.args.tgt_file)
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-7, betas=(0.9, 0.998), eps=1e-8, amsgrad=True)#get_std_opt(self.model)
        # entity_optim = torch.optim.Adam(self.entity_encoder.parameters(), lr=1e-7, betas=(0.9, 0.998), eps=1e-8, amsgrad=True)
        total_loss = []
        start = time.time()
        print('*'*50)
        print('Start Training...')
        print('*'*50)
        start_step = 0

        # if loading from checkpoint
        if self.args.load_model:
            state_dict = torch.load(self.args.load_model)['state_dict']
            self.model.load_state_dict(state_dict)
            print("Loading model from " + self.args.load_model + "...")
            # encoder_state_dict = torch.load(self.args.entity_encoder)['state_dict']
            # self.entity_encoder.load_state_dict(encoder_state_dict)
            # print("Loading entity_encoder from %s" + self.args.entity_encoder + "...")
            start_step = int(torch.load(self.args.load_model)['step'])
            print('Resume training from step %d ...'%start_step)

        warmup_steps = 10000
        d_model = 512
        lr = 1e-7
        for step in range(start_step, self.args.total_steps):
            self.model.train()
            batch = data_yielder.__next__()
            optim.zero_grad()
            # entity_optim.zero_grad()

            #update lr
            if step % 400 == 1:
                lr = (1/(d_model**0.5))*min((1/(step/4)**0.5), step * (1/(warmup_steps**1.5)))
                for param_group in optim.param_groups:
                    param_group['lr'] = lr
                # for param_group in entity_optim.param_groups:
                #     param_group['lr'] = lr
                
            batch['src'] = batch['src'].long()
            batch['tgt'] = batch['tgt'].long()
            batch['ner'] = batch['ner'].long()
            batch['src_extended'] = batch['src_extended'].long()

            # forward the model
            if self.args.entity_encoder_type == 'albert':
                d = self.model.entity_encoder.tokenizer.batch_encode_plus(batch['ner_text'], return_attention_masks=True, max_length=10, add_special_tokens=False, pad_to_max_length = True, return_tensors='pt')
                ner_mask = d['attention_mask'].cuda().unsqueeze(1)
                ner = d['input_ids'].cuda()
                # print('ner', ner.size())
                # print('ner_mask', ner_mask.size())
                # print('src_mask', batch['src_mask'].size())
            else:
                ner_mask = None
                ner = batch['ner']

            nnn = self.model.entity_encoder(ner)
            if self.args.ner_at_embedding:
                out = self.model.forward(batch['src'], batch['tgt'], nnn,
                            batch['src_mask'], batch['tgt_mask'], batch['src_extended'], len(batch['oov_list']))
            else:
                out = self.model.forward(batch['src'], batch['tgt'], nnn,
                            batch['src_mask'], batch['tgt_mask'], batch['src_extended'], len(batch['oov_list']), ner_mask)
            # print out info
            pred = out.topk(1, dim=-1)[1].squeeze().detach().cpu().numpy()[0]
            gg = batch['src_extended'].long().detach().cpu().numpy()[0][:100]
            tt = batch['tgt'].long().detach().cpu().numpy()[0]
            yy = batch['y'].long().detach().cpu().numpy()[0]

            #compute loss & update
            loss = self.model.loss_compute(out, batch['y'].long())
            loss.backward()
            optim.step()
            # entity_optim.step()
            
            total_loss.append(loss.detach().cpu().numpy())
            
            # logging information
            if step % self.args.print_every_steps == 1:
                elapsed = time.time() - start
                print("Epoch Step: %d Loss: %f Time: %f lr: %6.6f" %
                        (step, np.mean(total_loss), elapsed, optim.param_groups[0]['lr']))
                self.outfile.write("Epoch Step: %d Loss: %f Time: %f\n" %
                        (step, np.mean(total_loss), elapsed))
                print('src:\n',self.data_utils.id2sent(gg, False, False, batch['oov_list']))
                print('tgt:\n',self.data_utils.id2sent(yy, False, False, batch['oov_list']))
                print('pred:\n',self.data_utils.id2sent(pred, False, False, batch['oov_list']))
                print('oov_list:\n', batch['oov_list'])

                if ner_mask != None and not self.args.ner_at_embedding:
                    pp = self.model.greedy_decode(batch['src_extended'].long()[:1], self.model.entity_encoder(ner[:1]), batch['src_mask'][:1], 100, self.data_utils.bos, len(batch['oov_list']), self.data_utils.vocab_size, True, ner_mask[:1])
                else:
                    pp = self.model.greedy_decode(batch['src_extended'].long()[:1], self.model.entity_encoder(ner[:1]), batch['src_mask'][:1], 100, self.data_utils.bos, len(batch['oov_list']), self.data_utils.vocab_size, True)

                pp = pp.detach().cpu().numpy()
                print('pred_greedy:\n',self.data_utils.id2sent(pp[0], False, False, batch['oov_list']))
                
                print()
                start = time.time()
                if not self.disable_comet:
                    # self.log.add_scalar('Loss/train', np.mean(total_loss), step)
                    self.exp.log_metric('Train Loss', np.mean(total_loss), step =step)
                    self.exp.log_metric('Learning Rate', optim.param_groups[0]['lr'], step=step)

                    self.exp.log_text('Src: ' + self.data_utils.id2sent(gg, False, False, batch['oov_list']))
                    self.exp.log_text('Tgt:' + self.data_utils.id2sent(yy, False, False, batch['oov_list']))
                    self.exp.log_text('Pred:' + self.data_utils.id2sent(pred, False, False, batch['oov_list']))
                    self.exp.log_text('Pred Greedy:' + self.data_utils.id2sent(pp[0], False, False, batch['oov_list']))
                    self.exp.log_text('OOV:' + ' '.join(batch['oov_list']))

                total_loss = []
                
            # validation
            if step % self.args.valid_every_steps == 2:            
                self.model.eval()
                val_yielder = self.data_utils.data_yielder(self.args.valid_file, self.args.valid_tgt_file, 1)
                total_loss = []
                fw = open(self.w_valid_file, 'w')
                for batch in val_yielder:
                    with torch.no_grad():
                        batch['src'] = batch['src'].long()
                        batch['tgt'] = batch['tgt'].long()
                        batch['ner'] = batch['ner'].long()
                        batch['src_extended'] = batch['src_extended'].long()

                        ### ner ###
                        if self.args.entity_encoder_type == 'albert':
                            d = self.model.entity_encoder.tokenizer.batch_encode_plus(batch['ner_text'], return_attention_masks=True, max_length=10, add_special_tokens=False, pad_to_max_length = True, return_tensors='pt')
                            ner_mask = d['attention_mask'].cuda().unsqueeze(1)
                            ner = d['input_ids'].cuda()
                        else:
                            ner_mask = None
                            ner = batch['ner']

                        if self.args.ner_at_embedding:
                            out = self.model.forward(batch['src'], batch['tgt'], self.model.entity_encoder(ner),
                                batch['src_mask'], batch['tgt_mask'], batch['src_extended'], len(batch['oov_list']))
                        else:
                            out = self.model.forward(batch['src'], batch['tgt'], self.model.entity_encoder(ner),
                                batch['src_mask'], batch['tgt_mask'], batch['src_extended'], len(batch['oov_list']), ner_mask)
                        loss = self.model.loss_compute(out, batch['y'].long())
                        total_loss.append(loss.item())

                        if self.args.ner_at_embedding:
                            pred = self.model.greedy_decode(batch['src_extended'].long(), self.model.entity_encoder(ner), batch['src_mask'], self.args.max_len, self.data_utils.bos, len(batch['oov_list']), self.data_utils.vocab_size)
                        else:
                            pred = self.model.greedy_decode(batch['src_extended'].long(), self.model.entity_encoder(ner), batch['src_mask'], self.args.max_len, self.data_utils.bos, len(batch['oov_list']), self.data_utils.vocab_size, ner_mask=ner_mask)
                        
                        for l in pred:
                            sentence = self.data_utils.id2sent(l[1:], True, self.args.beam_size!=1, batch['oov_list'])
                            fw.write(sentence)
                            fw.write("\n")
                fw.close()
                # files_rouge = FilesRouge()
                # scores = files_rouge.get_scores(self.w_valid_file, self.args.valid_tgt_file, avg=True)
                scores = cal_rouge_score(self.w_valid_file, self.args.valid_tgt_file)
                r1_score = scores['rouge1']
                r2_score = scores['rouge2']

                print('=============================================')
                print('Validation Result -> Loss : %6.6f' %(sum(total_loss)/len(total_loss)))
                print(scores)
                print('=============================================')
                self.outfile.write('=============================================\n')
                self.outfile.write('Validation Result -> Loss : %6.6f\n' %(sum(total_loss)/len(total_loss)))
                self.outfile.write('=============================================\n')
                # self.model.train()
                # self.log.add_scalar('Loss/valid', sum(total_loss)/len(total_loss), step)
                # self.log.add_scalar('Score/valid', r1_score, step)
                if not self.disable_comet:
                    self.exp.log_metric('Valid Loss', sum(total_loss)/len(total_loss), step=step)
                    self.exp.log_metric('R1 Score', r1_score, step=step)
                    self.exp.log_metric('R2 Score', r2_score, step=step)

                #Saving Checkpoint
                w_step = int(step/10000)
                print('Saving ' + str(w_step) + 'w_model.pth!\n')
                self.outfile.write('Saving ' + str(w_step) + 'w_model.pth\n')

                model_name = str(w_step) + 'w_' + '%6.6f'%(sum(total_loss)/len(total_loss)) + '%2.3f_'%r1_score + '%2.3f_'%r2_score + 'model.pth'
                state = {'step': step, 'state_dict': self.model.state_dict()}
                torch.save(state, os.path.join(self.model_dir, model_name))

                # entity_encoder_name = str(w_step) + '0w_' + '%6.6f'%(sum(total_loss)/len(total_loss)) + '%2.3f_'%r1_score + 'entity_encoder.pth'
                # state = {'step': step, 'state_dict': self.entity_encoder.state_dict()}
                # torch.save(state, os.path.join(self.model_dir, entity_encoder_name))


    def test(self):
        #prepare model
        path = self.args.load_model
        # entity_encoder_path = self.args.entity_encoder
        state_dict = torch.load(path)['state_dict']
        max_len = self.args.max_len
        model = self.model
        model.load_state_dict(state_dict)

        
        # entity_encoder_dict = torch.load(entity_encoder_path)['state_dict']
        # self.entity_encoder.load_state_dict(entity_encoder_dict)

        pred_dir = make_save_dir(self.args.pred_dir)
        filename = self.args.filename

        #start decoding
        data_yielder = self.data_utils.data_yielder(self.args.test_file, self.args.test_file)
        total_loss = []
        start = time.time()

        #file
        f = open(os.path.join(pred_dir, filename), 'w')

        self.model.eval()

        # decode_strategy = BeamSearch(
        #             self.beam_size,
        #             batch_size=batch.batch_size,
        #             pad=self._tgt_pad_idx,
        #             bos=self._tgt_bos_idx,
        #             eos=self._tgt_eos_idx,
        #             n_best=self.n_best,
        #             global_scorer=self.global_scorer,
        #             min_length=self.min_length, max_length=self.max_length,
        #             return_attention=attn_debug or self.replace_unk,
        #             block_ngram_repeat=self.block_ngram_repeat,
        #             exclusion_tokens=self._exclusion_idxs,
        #             stepwise_penalty=self.stepwise_penalty,
        #             ratio=self.ratio)
            
        step = 0
        for batch in data_yielder:
            #print(batch['src'].data.size())
            step += 1
            if step % 100 == 0:
                print('%d batch processed. Time elapsed: %f min.' %(step, (time.time() - start)/60.0))
                start = time.time()

            ### ner ###
            if self.args.entity_encoder_type == 'albert':
                d = self.model.entity_encoder.tokenizer.batch_encode_plus(batch['ner_text'], return_attention_masks=True, max_length=10, add_special_tokens=False, pad_to_max_length = True, return_tensors='pt')
                ner_mask = d['attention_mask'].cuda().unsqueeze(1)
                ner = d['input_ids'].cuda()
            else:
                ner_mask = None
                ner = batch['ner'].long()

            with torch.no_grad():
                if self.args.beam_size == 1:
                    if self.args.ner_at_embedding:
                        out = self.model.greedy_decode(batch['src_extended'].long(), self.model.entity_encoder(ner), batch['src_mask'], max_len, self.data_utils.bos, len(batch['oov_list']), self.data_utils.vocab_size)
                    else:
                        out = self.model.greedy_decode(batch['src_extended'].long(), self.model.entity_encoder(ner), batch['src_mask'], max_len, self.data_utils.bos, len(batch['oov_list']), self.data_utils.vocab_size, ner_mask=ner_mask)
                else:
                    ret = self.beam_decode(batch, max_len, len(batch['oov_list']))
                    out = ret['predictions']
            for l in out:
                sentence = self.data_utils.id2sent(l[1:], True, self.args.beam_size!=1, batch['oov_list'])
                #print(l[1:])
                f.write(sentence)
                f.write("\n")


    def beam_decode(self, batch, max_len, oov_nums):
        
        src = batch['src'].long()
        src_mask = batch['src_mask']
        src_extended = batch['src_extended'].long()

        bos_token = self.data_utils.bos 
        beam_size = self.args.beam_size
        vocab_size = self.data_utils.vocab_size
        batch_size = src.size(0)

        def rvar(a): return a.repeat(beam_size, 1, 1)
                    
        def rvar2(a): return a.repeat(beam_size, 1)

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        ### ner ###
        if self.args.entity_encoder_type == 'albert':
            d = self.model.entity_encoder.tokenizer.batch_encode_plus(batch['ner_text'], return_attention_masks=True, max_length=10, add_special_tokens=False, pad_to_max_length = True, return_tensors='pt')
            ner_mask = d['attention_mask'].cuda().unsqueeze(1)
            ner = d['input_ids'].cuda()
        else:
            ner_mask = None
            ner = batch['ner'].long()
        ner = self.model.entity_encoder(ner)


        if self.args.ner_at_embedding:
            memory = self.model.encode(src, src_mask, ner)
        else:
            memory = self.model.encode(src, src_mask)
        
        
        assert batch_size == 1

        beam = [Beam(beam_size, 
                    self.data_utils.pad, 
                    bos_token, 
                    self.data_utils.eos,
                    min_length=self.args.min_length
                    )
                for i in range(batch_size)]
        memory = rvar(memory)
        ner = rvar(ner)
        src_mask = rvar(src_mask)
        src_extended = rvar2(src_extended)

        for i in range(self.args.max_len):
            if all((b.done() for b in beam)):
                break
            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = torch.stack([b.get_current_state() for b in beam]).t().contiguous().view(-1, 1)
            #inp -> [1, 3]
            inp_mask = inp < self.data_utils.vocab_size
            inp = inp * inp_mask.long()

            decoder_input = inp

            if self.args.ner_at_embedding:
                final_dist = self.model.decode(memory, ner, src_mask, decoder_input, None, src_extended, oov_nums)
            else:
                final_dist = self.model.decode(memory, ner, src_mask, decoder_input, None, src_extended, oov_nums, ner_mask=ner_mask)
            # final_dist, decoder_hidden, attn_dist_p, p_gen = self.seq2seq_model.model_copy.decoder(
            #                 decoder_input, decoder_hidden,
            #                 post_encoder_outputs, post_enc_padding_mask,
            #                 extra_zeros, post_enc_batch_extend_vocab
            #                 )
            # # Run one step.

            # print('inp', inp.size())

            # decoder_outputs: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            out = unbottle(final_dist)
            out[:, :, 2] = 0  #no unk
            # out.size -> [3, 1, vocab]

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(
                    out[:, j])
                # decoder_hidden = self.beam_update(j, b.get_current_origin(), beam_size, decoder_hidden)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)

        return ret


    def _from_beam(self, beam):
        ret = {
               "predictions": [],
               "scores": []
               }
        for b in beam:
            
            n_best = self.args.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps = []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp = b.get_hyp(times, k)
                hyps.append(hyp)

            ret["predictions"].append(hyps)
            ret["scores"].append(scores)

        return ret


        # ys = torch.full((batch_size, 1), bos_token).type_as(src.data).cuda()
        # log_prob = self.model.decode(memory, src_mask, 
        #                    Variable(ys), 
        #                    Variable(subsequent_mask(ys.size(1)).type_as(src.data).expand((ys.size(0), ys.size(1), ys.size(1)))), src_extended, oov_nums)

        
        # # log_prob = [batch_size, 1, voc_size]
        # top_prob, top_indices = torch.topk(input = log_prob, k = beam_size, dim = -1)
        # # print(top_indices)
        # top_prob = top_prob.view(-1, 1)
        # top_indices = top_indices.view(-1, 1)
        # beam.update_prob(top_prob.detach().cpu(), top_indices.detach().cpu())
        # # [batch_size, 1, beam_size]
        # ys = top_indices
        # top_indices = None
        # # print(ys.size())
        # ####### repeat var #######
        # src = torch.repeat_interleave(src, beam_size, dim = 0)
        # src_mask = torch.repeat_interleave(src_mask, beam_size, dim = 0)
        # #[batch_size, src_len, d_model] -> [batch_size*beam_size, src_len, d_model]
        # memory = torch.repeat_interleave(memory, beam_size, dim = 0)
        # # print('max_len', max_len)
        # for t in range(1, max_len):
        #     log_prob = self.model.decode(memory, src_mask, 
        #                        Variable(ys), 
        #                        Variable(subsequent_mask(ys.size(1)).type_as(src.data).expand((ys.size(0), ys.size(1), ys.size(1)))), src)
        #     # print('log_prob', log_prob.size())
        #     log_prob = log_prob[:,-1].unsqueeze(1)
        #     # print(beam.seq)
        #     real_top = beam.advance(log_prob.detach().cpu())
        #     # print(real_top.size())
        #     # print(ys.size())
        #     # print(real_top.size())
        #     ys = torch.cat((ys, real_top.view(-1, 1).cuda()), dim = -1)
        #     # print(ys.size())

        # # print(ys.size())
        # # print(beam.top_prob)
        # # print(len(beam.seq))


        # return [beam.seq[0]]
