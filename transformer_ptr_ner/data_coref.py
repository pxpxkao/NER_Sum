# {train, val, test}.txt.src.pkl : List[tokens]
# {train, val, test}.sents.pkl : List[sent[tokens]]
# {train, val, test}.convert.pkl : List[cluster[mentions]]

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
from torch.utils.data import Dataset
import pickle

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

    word2id['__PAD__'] = len(word2id)
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
        self.coref_path = args.coref_file
        self.valid_path = args.valid_file
        self.valid_target_path = args.valid_tgt_file
        self.valid_coref_path = args.valid_coref_file
        self.test_path = args.test_file
        self.test_coref_path = args.test_coref_file
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
        self.pad = self.word2id['__PAD__']
        self.pointer_gen = args.pointer_gen
        # self.pad = self.word2id['__UNK__']
        if self.train:
            self.sent = pickle.load(open(self.train_path, 'rb'))
            self.target_sent = pickle.load(open(self.target_path, 'rb'))
            self.coref = pickle.load(open(self.coref_path, 'rb'))

            self.valid_sent = pickle.load(open(args.valid_file, 'rb'))
            self.valid_target_sent = pickle.load(open(args.valid_tgt_file, 'rb'))
            self.valid_coref = pickle.load(open(args.valid_coref_file, 'rb'))
        else:
            self.sent = pickle.load(open(args.test_file, 'rb'))
            self.coref = pickle.load(open(args.test_coref_file, 'rb'))

        self.src_length = 400
        self.tgt_length = 100
        self.max_ner_len = 10
        
        self.max_num_clusters = args.max_num_clusters

    def text2id(self, word_list, seq_length, oov_list = []):
        vec = np.zeros([seq_length] ,dtype=np.int32)
        vec_extended = np.zeros([seq_length] ,dtype=np.int32)

        # unknown = 0.
        # word_list = text.strip().split()
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
                
        return vec, vec_extended, oov_list

    def pad_ner_feature(self, ner_feat, num_clusters, size):
        batch_size = size
        d_model = ner_feat.size(-1)
        max_num_clusters = num_clusters.max()
        m = torch.arange(max_num_clusters).repeat(batch_size, 1).cuda()
        # print('m', m)

        length_mat = num_clusters.unsqueeze(-1).expand_as(m).cuda()
        ner_mask = torch.where(m<length_mat, torch.ones((batch_size, max_num_clusters)).cuda(), torch.zeros((batch_size, max_num_clusters)).cuda())
        ner_mask = ner_mask.unsqueeze(1)
        # print('num_clusters', num_clusters)
        # print('ner_mask', ner_mask)
        # m = m.unsqueeze(-1).expand((-1, -1, d_model))

        nnn = torch.zeros((batch_size, max_num_clusters, d_model)).cuda()
        total = 0
        # print('ner___', ner_feat[:, 0])
        for i in range(batch_size):
            length = num_clusters[i].item()
            # print('t', total, total + length)
            nnn[i][:length] = ner_feat[total:total+length]
            total += num_clusters[i].item()
        # print('nnnn', nnn[:, :, 0])
        return nnn, ner_mask.cuda()


    def data_yielder(self, num_epoch = 100, valid=False):
        sent = self.valid_sent if valid else self.sent
        coref = self.valid_coref if valid else self.coref 
        target_sent = self.valid_target_sent if valid else self.target_sent
        index_list = np.arange(0, len(sent))

        print('Reading source file from %s ...'%(self.valid_path if valid else self.train_path))
        print('Reading target file from %s ...'%(self.valid_target_path if valid else self.target_path))
        print(self.batch_size)

        if self.train:
            batch = {'src':[],'tgt':[],'src_mask':[],'tgt_mask':[],'y':[], 'src_extended':[], 'oov_list':[], 'ner':[], 'cluster_len':[], 'num_clusters':[]}
            oov_list = []
            for epo in range(num_epoch):
                start_time = time.time()
                print("Start Epoch %d" % (epo))
                np.random.shuffle(index_list)

                for i in range(len(sent)):
                    index = index_list[i]
                    vec1, vec1_extended, oov_list = self.text2id(sent[index], self.src_length, oov_list)
                    vec2, vec2_extended, oov_list = self.text2id(target_sent[index], self.tgt_length, oov_list)

                    for j in range(len(coref[index])):
                        cluster = [s[2] for s in coref[index][j]]
                        ner_vec, _, _ = self.text2id(cluster, self.max_ner_len, [])
                        batch['cluster_len'].append(min(len(cluster), self.max_ner_len))
                        batch['ner'].append(ner_vec)
                    # ners = np.array(ners)

                    batch['src'].append(vec1)
                    batch['src_mask'].append(np.expand_dims(vec1 != self.pad, -2).astype(np.float))
                    batch['src_extended'].append(vec1_extended)
                    batch['tgt'].append(np.concatenate([[self.bos],vec2], axis=0)[:-1])
                    batch['tgt_mask'].append(self.subsequent_mask(vec2))
                    if self.pointer_gen:
                        batch['y'].append(vec2_extended)
                    else:
                        batch['y'].append(vec2)
                    # batch['ner'].append(ners)
                    batch['num_clusters'].append(len(coref[index]))


                    if len(batch['src']) == self.batch_size:
                        batch = {k: (cc(v) if (k != 'oov_list' and k != 'num_clusters') else v) for k, v in batch.items()}
                        batch['oov_list'] = oov_list
                        batch['num_clusters'] = torch.from_numpy(np.array(batch['num_clusters']))
                        torch.cuda.synchronize()
                        vec1 = None
                        vec2 = None
                        yield batch
                        batch = {'src':[],'tgt':[],'src_mask':[],'tgt_mask':[],'y':[], 'src_extended':[], 'oov_list':[], 'ner':[], 'cluster_len':[], 'num_clusters':[]}
                        oov_list = []


                if  len(batch['src']) != 0:
                    batch = {k: (cc(v) if (k != 'oov_list' and k != 'num_clusters') else v) for k, v in batch.items()}
                    batch['oov_list'] = oov_list
                    batch['num_clusters'] = torch.from_numpy(np.array(batch['num_clusters']))
                    torch.cuda.synchronize()
                    vec1 = None
                    vec2 = None
                    yield batch
                    batch = {'src':[],'tgt':[],'src_mask':[],'tgt_mask':[],'y':[], 'src_extended':[], 'oov_list':[], 'ner':[], 'cluster_len':[], 'num_clusters':[]}
                    oov_list = []
                end_time = time.time()
                print('Finish Epoch %d, Time Elapsed: %f' % (epo,end_time-start_time))

        else:
            sent = self.sent
            coref = self.coref 
            index_list = np.arange(0, len(sent))
            print('Reading source file from %s ...'%(self.test_path))
            # print('Reading target file from %s ...'%(self.valid_target_path if valid else self.target_path))
            print(self.batch_size)
            batch = {'src':[], 'src_mask':[], 'src_extended':[], 'oov_list':[], 'ner':[], 'cluster_len':[], 'num_clusters':[]}
            oov_list = []
            for epo in range(1):
                start_time = time.time()
                print("start epo %d" % (epo))
                np.random.shuffle(index_list)

                for i in range(len(sent)):
                    index = index_list[i]
                    vec1, vec1_extended, oov_list = self.text2id(sent[index], self.src_length, oov_list)

                    for j in range(len(coref[index])):
                        cluster = [s[2] for s in coref[index][j]]
                        ner_vec, _, _ = self.text2id(cluster, self.max_ner_len, [])
                        batch['cluster_len'].append(min(len(cluster), self.max_ner_len))
                        batch['ner'].append(ner_vec)

                    batch['src'].append(vec1)
                    batch['src_mask'].append(np.expand_dims(vec1 != self.pad, -2).astype(np.float))
                    batch['src_extended'].append(vec1_extended)
                    batch['num_clusters'].append(len(coref[index]))

                    if len(batch['src']) == self.batch_size:
                        batch = {k: (cc(v) if (k != 'oov_list' and k != 'num_clusters') else v) for k, v in batch.items()}
                        batch['oov_list'] = oov_list
                        batch['num_clusters'] = torch.from_numpy(np.array(batch['num_clusters']))
                        torch.cuda.synchronize()
                        vec1 = None
                        yield batch
                        batch = {'src':[], 'src_mask':[], 'src_extended':[], 'oov_list':[], 'ner':[], 'cluster_len':[], 'num_clusters':[]}
                        oov_list = []

                if  len(batch['src']) != 0:
                    batch = {k: (cc(v) if (k != 'oov_list' and k != 'num_clusters') else v) for k, v in batch.items()}
                    batch['oov_list'] = oov_list
                    batch['num_clusters'] = torch.from_numpy(np.array(batch['num_clusters']))
                    torch.cuda.synchronize()
                    vec1 = None
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


