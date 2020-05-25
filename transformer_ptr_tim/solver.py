from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import subprocess
from models import *
from utils import *
from attention import *
# from torch.utils.tensorboard import SummaryWriter
from beam import Beam
from modules import subsequent_mask
# from rouge import FilesRouge 
from rouge_score import rouge_scorer


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
        self.add_ner = args.add_ner
        self.model = self.make_model(self.data_utils.vocab_size, self.data_utils.vocab_size, args.num_layer, args.dropout)
        print(self.model)
        if self.args.train:
            self.outfile = open(self.args.logfile, 'w')
            self.model_dir = make_save_dir(args.model_dir)
            # self.logfile = os.path.join(args.logdir, args.exp_name)
            # self.log = SummaryWriter(self.logfile)
            self.w_valid_file = args.w_valid_file
            if os.path.exists(self.w_valid_file):
                os.remove(self.w_valid_file)

    def make_model(self, src_vocab, tgt_vocab, N=6, dropout=0.1,
            d_model=512, d_ff=2048, h=8, num_class=19):
            
            "Helper: Construct a model from hyperparameters."
            c = copy.deepcopy
            attn = MultiHeadedAttention(h, d_model)
            ff = PositionwiseFeedForward(d_model, d_ff, dropout)
            if self.add_ner:
                ff_first = PositionwiseFeedForward_First(d_model+num_class, d_model, d_ff, dropout)
            position = PositionalEncoding(d_model, dropout)
            word_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
            print('pgen', self.args.pointer_gen)
            encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N) 
            linear = nn.Linear(d_model+num_class, d_model) if self.add_ner else None
            # if not self.add_ner else \
            #           Encoder_ner(EncoderLayer(d_model, c(attn), c(ff), dropout), EncoderLayer_ner(d_model+num_class, d_model, c(attn), c(ff_first), dropout), N)
            model = EncoderDecoder(
                encoder,
                Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                     c(ff), dropout), N, d_model, tgt_vocab, self.args.pointer_gen),
                word_embed,
                word_embed, 
                linear)
            
            # This was important from their code. 
            # Initialize parameters with Glorot / fan_avg.
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            return model.cuda()

    def train(self):
        if not self.args.disable_comet:
            # logging
            COMET_PROJECT_NAME='summarization'
            COMET_WORKSPACE='timchen0618'


            self.exp = Experiment(project_name=COMET_PROJECT_NAME,
                                  workspace=COMET_WORKSPACE,
                                  auto_output_logging='simple',
                                  auto_metric_logging=None,
                                  display_summary=False,
                                  )
            self.exp.add_tag('pure_transformer')
            if self.args.add_ner:
                self.exp.add_tag('add_ner')


        data_yielder = self.data_utils.data_yielder(self.args.train_file, self.args.tgt_file, self.args.ner_file)
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-7, betas=(0.9, 0.998), eps=1e-8, amsgrad=True)#get_std_opt(self.model)
        total_loss = []
        start = time.time()
        print('start training...')
        print('w_valid_file: %s'%self.w_valid_file)
        if self.args.load_model:
            state_dict = torch.load(self.args.load_model)['state_dict']
            self.model.load_state_dict(state_dict)
            print("Loading model from " + self.args.load_model + "...")

        warmup_steps = 8000
        d_model = 512
        lr = 1e-7
        #path = torch.load("./train_model/10w_model.pth")
        #self.model.load_state_dict(path, strict = False)
        for step in range(2000000):
            self.model.train()
            batch = data_yielder.__next__()
            if step % 100 == 1:
                lr = (1/(d_model**0.5))*min((1/float(step/5)**0.5), float(step/5) * (1/(warmup_steps**1.5)))
                for param_group in optim.param_groups:
                    param_group['lr'] = lr
                
            # if True:
            batch['src'] = batch['src'].long()
            batch['tgt'] = batch['tgt'].long()
            batch['src_extended'] = batch['src_extended'].long()

            if self.add_ner:
                out = self.model.forward_ner(batch['src'], batch['tgt'], 
                            batch['src_mask'], batch['tgt_mask'], batch['src_extended'], len(batch['oov_list']), batch['ner'])
            else:
                # print('for')
                out = self.model.forward(batch['src'], batch['tgt'], 
                            batch['src_mask'], batch['tgt_mask'], batch['src_extended'], len(batch['oov_list']))
            pred = out.topk(1, dim=-1)[1].squeeze().detach().cpu().numpy()[0]
            gg = batch['src_extended'].long().detach().cpu().numpy()[0][:100]
            tt = batch['tgt'].long().detach().cpu().numpy()[0]
            yy = batch['y'].long().detach().cpu().numpy()[0]
            loss = self.model.loss_compute(out, batch['y'].long())
            loss.backward()
            nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_value)
            optim.step()
            optim.zero_grad()
            total_loss.append(loss.detach().cpu().numpy())
            
            if step % 5000 == 1:
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
                # self.log.add_scalar('Loss/train', np.mean(total_loss), step)
                if not self.args.disable_comet:
                    self.exp.log_metric('Train Loss', np.mean(total_loss), step=step)
                    self.exp.log_metric('Lr', lr, step=step)
                total_loss = []
                
            
            # if step % 50000 == 1:
                
            if step % 100000 == 2:
                # batch['src'] = batch['src'].detach().cpu()
                # batch['y'] = batch['y'].detach().cpu()
                # batch['src_extended'] = batch['src_extended'].detach().cpu()
                # batch['src_mask'] = batch['src_mask'].detach().cpu()
                # batch['tgt'] = batch['tgt'].detach().cpu()
                # batch['tgt_mask'] = batch['tgt_mask'].detach().cpu()
                # del batch['src']
                # del batch['src_extended']
                # del batch['tgt']
                # del batch['y']
                # del batch['src_mask']
                # del batch['tgt_mask']

                self.model.eval()
                val_yielder = self.data_utils.data_yielder(self.args.valid_file, self.args.valid_tgt_file, self.args.valid_ner_file, 1)
                total_loss = []
                fw = open(self.w_valid_file, 'w')
                for batch in val_yielder:
                    with torch.no_grad():
                        batch['src'] = batch['src'].long()
                        batch['tgt'] = batch['tgt'].long()
                        batch['src_extended'] = batch['src_extended'].long()
                        # print(len(batch['oov_list']))
                        out = self.model.forward(batch['src'], batch['tgt'], 
                                batch['src_mask'], batch['tgt_mask'], batch['src_extended'], len(batch['oov_list']))
                        loss = self.model.loss_compute(out, batch['y'].long())
                        total_loss.append(loss.item())

                        pred = self.model.greedy_decode(batch['src_extended'].long(), batch['src_mask'], self.args.max_len, self.data_utils.bos, len(batch['oov_list']), self.data_utils.vocab_size)
                        
                        for l in pred:
                            sentence = self.data_utils.id2sent(l[1:], True, self.args.beam_size!=1, batch['oov_list'])
                            fw.write(sentence)
                            fw.write("\n")
                fw.close()
                # files_rouge = FilesRouge()
                print('Getting Score from %s'%self.w_valid_file)
                scores = cal_rouge_score(self.w_valid_file, self.args.valid_tgt_file)
                r1_score = scores['rouge1']
                r2_score = scores['rouge2']

                # scores = files_rouge.get_scores(self.w_valid_file, self.args.valid_tgt_file, avg=True)
                # r1_score = scores['rouge-1']['f']
                # r2_score = scores['rouge-2']['f']
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
                if not self.args.disable_comet:
                    self.exp.log_metric('Valid Loss', sum(total_loss)/ len(total_loss), step=step)
                    self.exp.log_metric('R1 Score', r1_score, step = step)
                    self.exp.log_metric('R2 Score', r2_score, step = step)

                w_step = int(step/10000)
                if self.args.load_model:
                    w_step += (int(self.args.load_model.split('/')[-1][0]))
                print('Saving ' + str(w_step) + '0w_model.pth!\n')
                self.outfile.write('Saving ' + str(w_step) + 'w_model.pth\n')
                model_name = str(w_step) + 'w_' + '%6.6f_'%(sum(total_loss)/len(total_loss)) + '%2.3f_'%r1_score + '%2.3f_'%r2_score + 'model.pth'
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
        data_yielder = self.data_utils.data_yielder(self.args.test_file, self.args.test_file, self.args.test_ner_file)
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
            with torch.no_grad():
                if self.args.beam_size == 1:
                    out = self.model.greedy_decode(batch['src_extended'].long(), batch['src_mask'], max_len, self.data_utils.bos, len(batch['oov_list']), self.data_utils.vocab_size)
                else:
                    out = self.beam_decode(batch, max_len, len(batch['oov_list']))

            # print('out',out)
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

        
        memory = self.model.encode(src, src_mask)

        # assert batch_size == 1

        beam = [Beam(beam_size, 
                    self.data_utils.pad, 
                    bos_token, 
                    self.data_utils.eos,
                    min_length=self.args.min_length
                    )
                for i in range(batch_size)]
        memory = rvar(memory)

        src_mask = rvar(src_mask)
        src_extended = rvar2(src_extended)

        for i in range(max_len):
            if all((b.done() for b in beam)):
                break
            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            # [beam * batch, len, dim]
            # inp = torch.stack([b.get_current_state() for b in beam]).t().contiguous().view(-1, 1)
            inp = torch.stack([b.get_current_input() for b in beam]).transpose(0,1).contiguous()
            assert inp.size(0) == beam_size
            assert inp.size(1) == batch_size
            inp = inp.view(-1, inp.size(-1))
            assert inp.size(0) == beam_size * batch_size
            # print(inp.size())
            #inp -> [1, 3]
            inp_mask = inp < self.data_utils.vocab_size
            inp = inp * inp_mask.long()

            decoder_input = inp
            
            final_dist = self.model.decode(memory, src_mask, decoder_input, None, src_extended, oov_nums)
            # # Run one step.

            # decoder_outputs: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            out = final_dist[:, -1].view(beam_size, batch_size, -1)
            
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
            # print('hyps',hyps[0])
            ret["predictions"].append(hyps[0])
            ret["scores"].append(scores)


        return ret['predictions']

    def block_ngram_repeats(self, log_probs):
        """
        We prevent the beam from going in any direction that would repeat any
        ngram of size <block_ngram_repeat> more thant once.
        The way we do it: we maintain a list of all ngrams of size
        <block_ngram_repeat> that is updated each time the beam advances, and
        manually put any token that would lead to a repeated ngram to 0.
        This improves on the previous version's complexity:
           - previous version's complexity: batch_size * beam_size * len(self)
           - current version's complexity: batch_size * beam_size
        This improves on the previous version's accuracy;
           - Previous version blocks the whole beam, whereas here we only
            block specific tokens.
           - Before the translation would fail when all beams contained
            repeated ngrams. This is sure to never happen here.
        """

        # we don't block nothing if the user doesn't want it
        if self.block_ngram_repeat <= 0:
            return

        # we can't block nothing beam's too short
        if len(self) < self.block_ngram_repeat:
            return

        n = self.block_ngram_repeat - 1
        for path_idx in range(self.alive_seq.shape[0]):
            # we check paths one by one

            current_ngram = tuple(self.alive_seq[path_idx, -n:].tolist())
            forbidden_tokens = self.forbidden_tokens[path_idx].get(
                current_ngram, None)
            if forbidden_tokens is not None:
                log_probs[path_idx, list(forbidden_tokens)] = -10e20
                
    # def beam_decode(self, batch, max_len, oov_nums):

    #     bos_token = self.data_utils.bos 
    #     beam_size = self.args.beam_size
    #     vocab_size = self.data_utils.vocab_size

    #     src = batch['src'].long()
    #     src_mask = batch['src_mask']
    #     src_extended = batch['src_extended'].long()
    #     memory = self.model.encode(src, src_mask)
    #     batch_size = src.size(0)

    #     beam = Beam(self.data_utils.pad, 
    #                 bos_token, 
    #                 self.data_utils.eos, 
    #                 beam_size, 
    #                 batch_size,
    #                 self.args.n_best,
    #                 True,
    #                 max_len
    #                 )

    #     ys = torch.full((batch_size, 1), bos_token).type_as(src.data).cuda()
    #     log_prob = self.model.decode(memory, src_mask, 
    #                        Variable(ys), 
    #                        Variable(subsequent_mask(ys.size(1)).type_as(src.data).expand((ys.size(0), ys.size(1), ys.size(1)))), src_extended, oov_nums)

        
    #     # log_prob = [batch_size, 1, voc_size]
    #     top_prob, top_indices = torch.topk(input = log_prob, k = beam_size, dim = -1)
    #     # print(top_indices)
    #     top_prob = top_prob.view(-1, 1)
    #     top_indices = top_indices.view(-1, 1)
    #     beam.update_prob(top_prob.detach().cpu(), top_indices.detach().cpu())
    #     # [batch_size, 1, beam_size]
    #     ys = top_indices
    #     top_indices = None
    #     # print(ys.size())
    #     ####### repeat var #######
    #     src = torch.repeat_interleave(src, beam_size, dim = 0)
    #     src_mask = torch.repeat_interleave(src_mask, beam_size, dim = 0)
    #     #[batch_size, src_len, d_model] -> [batch_size*beam_size, src_len, d_model]
    #     memory = torch.repeat_interleave(memory, beam_size, dim = 0)
    #     # print('max_len', max_len)
    #     for t in range(1, max_len):
    #         log_prob = self.model.decode(memory, src_mask, 
    #                            Variable(ys), 
    #                            Variable(subsequent_mask(ys.size(1)).type_as(src.data).expand((ys.size(0), ys.size(1), ys.size(1)))), src)
    #         # print('log_prob', log_prob.size())
    #         log_prob = log_prob[:,-1].unsqueeze(1)
    #         # print(beam.seq)
    #         real_top = beam.advance(log_prob.detach().cpu())
    #         # print(real_top.size())
    #         # print(ys.size())
    #         # print(real_top.size())
    #         ys = torch.cat((ys, real_top.view(-1, 1).cuda()), dim = -1)
    #         # print(ys.size())

    #     # print(ys.size())
    #     # print(beam.top_prob)
    #     # print(len(beam.seq))


    #     return [beam.seq[0]]
