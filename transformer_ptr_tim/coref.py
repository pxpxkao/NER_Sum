import os
import pickle
from stanfordcorenlp import StanfordCoreNLP
import sys
import time

# nlp = StanfordCoreNLP('../stanford-corenlp-4.0.0')

# f = open(sys.argv[1], 'r')
# fw = open(sys.argv[2], 'wb')

# corefs = []
# sent = f.readlines()
# for l in sent:
#     corefs.append(nlp.coref(l))

# pickle.dump(corefs, fw)


files = ['../data/coref/val.coref.full.pkl', '../data/coref/test.coref.full.pkl']
outfiles = [ '../data/coref/val.coref.pkl', '../data/coref/test.coref.pkl']
# files = ['../data/coref/test.coref.src.sm']
# outfiles = ['test.coref.src.sm.simp']
k = 6
for i in range(len(files)):
    corefs = pickle.load(open(files[i], 'rb'))
    new_corefs = []

    for example in corefs:
        new_c = []
        sorted_ex = sorted(example, key = lambda x: len(x), reverse=True)
        new_c, sorted_ex = sorted_ex[:k], sorted_ex[k:]
        # print(new_c)
        for c in sorted_ex:
            for ent in c:
                if ent[0] <= 3:
                    new_c.append(c)
                    break

        # print(new_c)
        new_corefs.append(new_c)


    with open(outfiles[i], 'wb') as fw:
        pickle.dump(new_corefs, fw)
