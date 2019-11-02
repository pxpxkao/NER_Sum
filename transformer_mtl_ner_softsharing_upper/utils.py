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
    return a.scatter_(indices_dim,indices.unsqueeze(indices_dim),1)


def tens2np(tensor):
    return tensor.detach().cpu().numpy()


def make_dict(max_num, dict_path, train_ner_path, train_sum_path):
    word_count = dict()
    word2id = dict()
    line_count = 0
    
    for line in tqdm(open(train_ner_path)):
        line_count += 1.0
        for word in line.split():
            word_count[word] = word_count.get(word,0) + 1

    for line in tqdm(open(train_sum_path)):
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

def make_labeldic(label_dic, label_map):
    label2id = {}
    for line in tqdm(open(label_map)):
        label2id[line.split()[0]] = len(label2id)
    with open(label_dic,'w') as fp:
        json.dump(label2id, fp)
    return label2id


class data_utils():
    def __init__(self, args):
        self.batch_size = args.batch_size

        dict_path = args.dict
        labeldic_path = '../data/ner/label_dict.json'
        self.train_ner_path = '../data/ner/train.txt'
        self.tgt_ner_path = '../data/ner/label.txt'
        self.train_sum_path = args.train_sum_file
        self.tgt_sum_path = args.tgt_sum_file
        if os.path.exists(dict_path):
            self.word2id = read_json(dict_path)
        else:
            self.word2id = make_dict(30000, dict_path, self.train_ner_path, self.train_sum_path)

        self.index2word = [[]]*len(self.word2id)
        for word in self.word2id:
            self.index2word[self.word2id[word]] = word

        if os.path.exists(labeldic_path):
            self.label2id = read_json(labeldic_path)
        else:
            self.label2id = make_labeldic(labeldic_path, '../data/ner/label-map.index')
        self.index2label = [[]]*len(self.label2id)
        for word in self.label2id:
            self.index2label[self.label2id[word]] = word


        self.vocab_size = len(self.word2id)
        print('vocab_size:',self.vocab_size)
        self.eos = self.word2id['__EOS__']
        self.bos = self.word2id['__BOS__']

    def text2id(self, text, seq_length=40):
        vec = np.zeros([seq_length] ,dtype=np.int32)
        unknown = 0.
        word_list = text.strip().split()
        length = len(word_list)

        for i,word in enumerate(word_list):
            if i >= seq_length:
                break
            if word in self.word2id:
                vec[i] = self.word2id[word]
            else:
                vec[i] = self.word2id['__UNK__']
                unknown += 1

        # if unknown / length > 0.1 or length > seq_length*1.5:
        #     vec = None

        return vec
    
    def labeltext2id(self, text, seq_length=40):
        vec = np.zeros([seq_length] ,dtype=np.int32) 
        ner_list = text.strip().split()
        length = len(ner_list)
        for i, word in enumerate(ner_list):
            if i >= seq_length:
                break
            if word in self.label2id:
                vec[i] = self.label2id[word]
                if vec[i] > 8:
                    print("==")
                    print(vec[i])
        return vec

    def data_yielder_ner(self, num_epoch = 1):
        batch = {'src':[],'src_mask':[],'y':[]}
        for epo in range(num_epoch):
            start_time = time.time()
            print("start epo %d" % (epo))
            for line1, line2 in zip(open(self.train_ner_path), open(self.tgt_ner_path)):
                vec1 = self.text2id(line1.strip(), 45)
                vec2 = self.labeltext2id(line2.strip(), 45)

                if vec1 is not None and vec2 is not None:
                    batch['src'].append(vec1)
                    batch['src_mask'].append(np.expand_dims(vec1 != self.eos, -2).astype(np.float))
                    batch['y'].append(vec2)

                    if len(batch['src']) == self.batch_size:
                        batch = {k: cc(v) for k, v in batch.items()}
                        yield batch
                        batch = {'src':[], 'src_mask':[],'y':[]}
            end_time = time.time()
            print('finish epo %d, time %f' % (epo,end_time-start_time))

    def data_yielder_sum(self):
        batch = {'src':[],'tgt':[],'src_mask':[],'tgt_mask':[],'y':[]}
        for epo in range(20):
            start_time = time.time()
            print("start epo %d" % (epo))
            # index = 0
            for line1,line2 in zip(open(self.train_sum_path),open(self.tgt_sum_path)):
                vec1 = self.text2id(line1.strip(), 300)
                vec2 = self.text2id(line2.strip(), 60)

                if vec1 is not None and vec2 is not None:
                    batch['src'].append(vec1)
                    batch['src_mask'].append(np.expand_dims(vec1 != self.eos, -2).astype(np.float))
                    batch['tgt'].append(np.concatenate([[self.bos],vec2], axis=0)[:-1])
                    batch['tgt_mask'].append(self.subsequent_mask(vec2))
                    batch['y'].append(vec2)

                    if len(batch['src']) == self.batch_size:
                        batch = {k: cc(v) for k, v in batch.items()}
                        torch.cuda.synchronize()
                        yield batch
                        batch = {'src':[],'tgt':[],'src_mask':[],'tgt_mask':[],'y':[]}
            end_time = time.time()
            print('finish epo %d, time %f' % (epo,end_time-start_time))

    def id2sent(self,indices, test=False):
        sent = []
        word_dict={}
        for index in indices:
            if test and (index == self.word2id['__EOS__'] or index in word_dict):
                continue
            sent.append(self.index2word[index])
            word_dict[index] = 1

        return ' '.join(sent)
    
    def id2label(self, indices, test=False):
        sent = []
        #print(indices.size())
        for index in indices:
            # print(index)
            sent.append(self.index2label[index])

            
        return ' '.join(sent[:-1])

    def subsequent_mask(self, vec):
        attn_shape = (vec.shape[-1], vec.shape[-1])
        return (np.triu(np.ones((attn_shape)), k=1).astype('uint8') == 0).astype(np.float)
