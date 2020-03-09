from __future__ import print_function
import numpy as np
import random
import json
import os
import re
import sys
import torch
from tqdm import tqdm
import operator
import torch.autograd as autograd
from nltk.corpus import stopwords
import time

def read_json(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data


def write_json(filename,data):
    with open(filename, 'w') as fp:
        json.dump(data, fp)


def make_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir



def cc(arr):
    return torch.from_numpy(np.array(arr)).cuda()


def one_hot(indices, depth):
    shape = list(indices.size())+[depth]
    indices_dim = len(indices.size())
    a = torch.zeros(shape,dtype=torch.float).cuda()
    torch.cuda.synchronize()
    return a.scatter_(indices_dim,indices.unsqueeze(indices_dim),1)


def tens2np(tensor):
    return tensor.detach().cpu().numpy()


def make_dict(max_num, dict_path, train_path, target_path):
    #create dict with voc length of max_num
    word_count = dict()
    word2id = dict()
    line_count = 0
    
    for line in tqdm(open(train_path)):
        line_count += 1.0
        for word in line.split():
            word_count[word] = word_count.get(word,0) + 1
    for line in tqdm(open(target_path)):
        line_count += 1.0
        for word in line.split():
            word_count[word] = word_count.get(word,0) + 1

     
    word2id['__EOS__'] = len(word2id)
    word2id['__BOS__'] = len(word2id)
    word2id['__UNK__'] = len(word2id)
    word_count_list = sorted(word_count.items(), key=operator.itemgetter(1))
    for item in word_count_list[-(max_num*2):][::-1]:
        

        if item[1] < word_count_list[-max_num][1]:
            continue
        word = item[0]
        word2id[word] = len(word2id)

    with open(dict_path,'w') as fp:
        json.dump(word2id, fp)

    return word2id


class data_utils():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.train = True if args.train else False
        dict_path = './dictionary.json'
        self.train_path = args.train_file
        self.target_path = args.tgt_file
        if not self.train:
            assert (os.path.exists(dict_path))
        
        if os.path.exists(dict_path):
            self.word2id = read_json(dict_path)
        else:
            self.word2id = make_dict(50000, dict_path, self.train_path, self.target_path)

        self.index2word = [[]]*len(self.word2id)
        for word in self.word2id:
            self.index2word[self.word2id[word]] = word

        self.vocab_size = len(self.word2id)
        print('vocab_size:',self.vocab_size)
        self.eos = self.word2id['__EOS__']
        self.bos = self.word2id['__BOS__']
        self.pad = self.eos
        # self.pad = self.word2id['__UNK__']


    def text2id(self, text, seq_length, oov_list = []):
        vec = np.zeros([seq_length] ,dtype=np.int32)
        vec_extended = np.zeros([seq_length] ,dtype=np.int32)

        # unknown = 0.
        word_list = text.strip().split()
        length = len(word_list)
        for i,word in enumerate(word_list):
            if i >= seq_length:
                break
            if word in self.word2id:
                vec[i] = self.word2id[word]
                vec_extended[i] = self.word2id[word]
            else:
                vec[i] = self.word2id['__UNK__']
                try:
                    vec_extended[i] = self.vocab_size + oov_list.index(word)
                except ValueError:
                    vec_extended[i] = self.vocab_size + len(oov_list)
                    oov_list.append(word)
                
                # unknown += 1

        # if unknown / length > 0.1 or length > seq_length*1.5:
        #     vec = None
        # if self.train:
        #     if length == 0:
        #         vec = None
        #         print('damn') 
        #         return vec
        #     if unknown / length > 0.2 or length > seq_length*1.22:
        #         print('fuck')
        #         print(length, ' ', unknown, ' ' , seq_length)
        #         vec = None
        # print('oov', oov_list)
        # print(vec_extended)
        return vec, vec_extended, oov_list


    def data_yielder(self, src_file, tgt_file, num_epoch = 100):
        print(src_file)
        print(tgt_file)
        print(self.batch_size)
        src_length = 400
        tgt_length = 100
        if self.train:
            batch = {'src':[],'tgt':[],'src_mask':[],'tgt_mask':[],'y':[], 'src_extended':[], 'oov_list':[]}
            for epo in range(num_epoch):
                start_time = time.time()
                print("start epo %d" % (epo))
                oov_list = []
                for line1,line2 in zip(open(src_file),open(tgt_file)):
                    vec1, vec1_extended, oov_list = self.text2id(line1.strip(), src_length, oov_list)
                    vec2, vec2_extended, oov_list = self.text2id(line2.strip(), tgt_length, oov_list)
                    # print('===============')

                    if vec1 is not None and vec2 is not None:
                        batch['src'].append(vec1)
                        batch['src_mask'].append(np.expand_dims(vec1 != self.eos, -2).astype(np.float))
                        batch['src_extended'].append(vec1_extended)
                        batch['tgt'].append(np.concatenate([[self.bos],vec2], axis=0)[:-1])
                        batch['tgt_mask'].append(self.subsequent_mask(vec2))
                        batch['y'].append(vec2_extended)

                        if len(batch['src']) == self.batch_size:
                            batch = {k: (cc(v) if k != 'oov_list' else v) for k, v in batch.items()}
                            batch['oov_list'] = oov_list
                            # print(batch)
                            torch.cuda.synchronize()
                            vec1 = None
                            vec2 = None
                            yield batch
                            batch = {'src':[],'tgt':[],'src_mask':[],'tgt_mask':[],'y':[], 'src_extended':[], 'oov_list':[]}
                            oov_list = []
                            #batch = {'src':[]}
                end_time = time.time()
                print('finish epo %d, time %f' % (epo,end_time-start_time))

        else:
            batch = {'src':[], 'src_mask':[], 'src_extended':[], 'oov_list':[]}
            for epo in range(1):
                start_time = time.time()
                print("start epo %d" % (epo))
                index = 0
                oov_list = []
                for line1 in open(src_file):
                    index += 1
                    vec1, vec1_extended, oov_list = self.text2id(line1.strip(), 400)
                    
                    if vec1 is not None:
                        batch['src'].append(vec1)
                        batch['src_extended'].append(vec1_extended)
                        batch['src_mask'].append(np.expand_dims(vec1 != self.eos, -2).astype(np.float))
                        # batch['oov_list'].append(oov_list)
                        if len(batch['src']) == self.batch_size:
                            batch = {k: (cc(v) if k != 'oov_list' else v) for k, v in batch.items()}
                            batch['oov_list'] = oov_list
                            torch.cuda.synchronize()
                            yield batch
                            batch = {'src':[], 'src_mask':[], 'src_extended':[], 'oov_list':[]}
                            oov_list = []
                batch = {k: cc(v) for k, v in batch.items()}
                torch.cuda.synchronize()
                yield batch
                end_time = time.time()
                print('finish epo %d, time %f' % (epo,end_time-start_time))





    def id2sent(self, indices, test=False, beam_search = False, oov_list = None):
        # print(indices)
        sent = []
        word_dict={}
        if beam_search:
            for index in indices:
                if test and (index == self.word2id['__EOS__'] or index in word_dict):
                    continue
                sent.append(self.index2word[index])
                word_dict[index] = 1
        else:            
            for index in indices:
                if oov_list == None:
                    if test and (index == self.word2id['__EOS__'] or index.item() in word_dict):
                        continue
                    sent.append(self.index2word[index.item()])
                else:
                    if index.item() >= self.vocab_size:
                        if test and index.item() in word_dict:
                            continue
                        sent.append(oov_list[index.item() - self.vocab_size])
                    else:
                        if test and (index.item() == self.word2id['__EOS__'] or index.item() in word_dict):
                            continue
                        sent.append(self.index2word[index.item()])

            word_dict[index.item()] = 1

        return ' '.join(sent)


    def subsequent_mask(self, vec):
        attn_shape = (vec.shape[-1], vec.shape[-1])
        return (np.triu(np.ones((attn_shape)), k=1).astype('uint8') == 0).astype(np.float)



# class data_utils():
#     def __init__(self, args):
#         self.batch_size = args.batch_size

#         dict_path = '../origin/data/dictionary.json'
#         self.train_path = '../origin/data/train.article.txt'
#         if os.path.exists(dict_path):
#             self.word2id = read_json(dict_path)
#         else:
#             self.word2id = make_dict(25000, dict_path, self.train_path)

#         self.index2word = [[]]*len(self.word2id)
#         for word in self.word2id:
#             self.index2word[self.word2id[word]] = word

#         self.vocab_size = len(self.word2id)
#         print('vocab_size:',self.vocab_size)
#         self.eos = self.word2id['__EOS__']
#         self.bos = self.word2id['__BOS__']


#     def text2id(self, text, seq_length=40):
#         vec = np.zeros([seq_length] ,dtype=np.int32)
#         unknown = 0.
#         word_list = text.strip().split()
#         length = len(word_list)

#         for i,word in enumerate(word_list):
#             if i >= seq_length:
#                 break
#             if word in self.word2id:
#                 vec[i] = self.word2id[word]
#             else:
#                 vec[i] = self.word2id['__UNK__']
#                 unknown += 1

#         if unknown / length > 0.1 or length > seq_length*1.5:
#             vec = None

#         return vec


#     def data_yielder(self):
#         batch = {'src':[],'tgt':[],'src_mask':[],'tgt_mask':[],'y':[]}
#         for epo in range(20):
#             start_time = time.time()
#             print("start epo %d" % (epo))
#             for line1,line2 in zip(open('../origin/data/train.article.txt'),open('../origin/data/train.title.txt')):
#                 vec1 = self.text2id(line1.strip(), 45)
#                 vec2 = self.text2id(line2.strip(), 14)

#                 if vec1 is not None and vec2 is not None:
#                     batch['src'].append(vec1)
#                     batch['src_mask'].append(np.expand_dims(vec1 != self.eos, -2).astype(np.float))
#                     batch['tgt'].append(np.concatenate([[self.bos],vec2], axis=0)[:-1])
#                     batch['tgt_mask'].append(self.subsequent_mask(vec2))
#                     batch['y'].append(vec2)

#                     if len(batch['src']) == self.batch_size:
#                         batch = {k: cc(v) for k, v in batch.items()}
#                         torch.cuda.synchronize()
#                         yield batch
#                         batch = {'src':[],'tgt':[],'src_mask':[],'tgt_mask':[],'y':[]}
#             end_time = time.time()
#             print('finish epo %d, time %f' % (epo,end_time-start_time))



#     def id2sent(self,indices, test=False):
#         sent = []
#         word_dict={}
#         for index in indices:
#             if test and (index == self.word2id['__EOS__'] or index in word_dict):
#                 continue
#             sent.append(self.index2word[index])
#             word_dict[index] = 1

#         return ' '.join(sent)


#     def subsequent_mask(self, vec):
#         attn_shape = (vec.shape[-1], vec.shape[-1])
#         return (np.triu(np.ones((attn_shape)), k=1).astype('uint8') == 0).astype(np.float)
