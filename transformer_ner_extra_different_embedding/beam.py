import torch

class Beam(object):
    """
    Class for managing the internals of the beam search process.
    Takes care of beams, back pointers, and scores.
    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
    """
    def __init__(self, pad, bos, eos, beam_size=5, batch_size = 16,
                 n_best=1, cuda=True,
                 max_length=100):
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.pad = pad
        self.bos = bos
        self.eos = eos

        self.n_best = n_best
        self.use_cuda = cuda

        self.max_length = max_length

        self.top_prob = None
        self.seq = [[self.bos] for i in range(beam_size) for j in range(batch_size)]
        
        
    def advance(self, log_prob):
        # log_prob -> [batch_size * beam_size, 1, voc_size]
        top_prob, top_indices = torch.topk(input = log_prob, k = self.beam_size, dim = -1)
        prev_prob = self.top_prob.repeat(1, self.beam_size).unsqueeze(1)
        top_prob = top_prob + prev_prob

        # [batch_size*beam_size, 1, beam_size]
        # print(top_indices)
        # print(top_prob.size())
        top_prob = top_prob.view(-1, self.beam_size, 1, self.beam_size).squeeze()
        top_prob = top_prob.view(-1, self.beam_size*self.beam_size)
        top_indices = top_indices.view(-1, self.beam_size, 1, self.beam_size).squeeze()
        top_indices = top_indices.view(-1, self.beam_size*self.beam_size)
        # print(top_indices.size())
        # print(top_prob.size())
        # print(top_indices)
        top_prob, select_indices = torch.topk(input = top_prob, k = self.beam_size, dim = -1)
        # print('sel', select_indices.size())
        # print(select_indices)
        # print(select_indices// self.beam_size)

        real_top = torch.gather(input = top_indices, dim = 1, index = select_indices)
        top_prob = top_prob.view(-1, 1)
        # print(real_top.size())
        # print(real_top)
        for i in range(real_top.size(0)):
            for j in range(real_top.size(1)):
                self.seq[real_top.size(1) * i + j].append(int(real_top[i][j].detach().cpu().numpy()))

        for i in range(top_prob.size(0)):
            self.top_prob = top_prob

        return real_top

    def calculate_score(self):
        pass

    def update_prob(self, top_prob, tokens):
        self.top_prob = top_prob   
        for i in range(tokens.size(0)):
            self.seq[i].append(int(tokens[i].detach().cpu().numpy()))