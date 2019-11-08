import sys
import json, pickle

def read_json(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data

def write_json(filename,data):
    with open(filename, 'w') as fp:
        json.dump(data, fp)

def read_pickle(filename):
    with open(filename, 'rb') as fp:
        data = pickle.load(fp)
    return data

def write_pickle(filename,data):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)

args = sys.argv

with open(args[2], 'r') as fp:
    summary = fp.readlines()
print(len(summary))

import spacy
nlp = spacy.load('en_core_web_lg')

# text = ["He works at Google.", "France is in Europe"]
label_count = {}
label_map = {}
ner_map = {}
for s_idx, s in enumerate(summary):
    print(s_idx)
    doc = nlp(s)
    ner_list = []
    for token_idx, token in enumerate(doc):     
        if token.ent_type_:
            print(token.ent_type_)
            ner_list.append((token_idx, token.text, token.ent_iob_, token.ent_type_))
            label_count[token.ent_type_] = label_count.get(token.ent_type_,0) + 1
    ner_map['P'+str(s_idx)] = ner_list

# label_map['O'] = len(label_map)
for key, value in label_count.items():
    label_map[key] = len(label_map)

if(args[1] == '-train_data'):
    write_json('train.ner.map', ner_map)
if(args[1] == '-test_data'):
    write_json('test.ner.map', ner_map)
write_json('label.map', label_map)


# text = "He works at Google."
# doc = nlp(text)
# spacy.displacy.serve(doc, style='ent')
# for token in doc.ents:
#     print(token.text, token.label_)

# I – Token is inside an entity.
# O – Token is outside an entity.
# B – Token is the beginning of an entity.
# [doc[0].text, doc[0].ent_iob_, doc[0].ent_type_]
