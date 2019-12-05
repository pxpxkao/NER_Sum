import sys
import json

args = sys.argv

with open(args[2], 'r') as fp:
    summary = fp.readlines()
print(len(summary))

import spacy
nlp = spacy.load('en_core_web_lg')

def write_file(filename, data):
    with open(filename, 'w') as f:
        for t in data:
            f.write(t)

text = []
ner = []
label_count = {}
for s_idx, s in enumerate(summary):
    print(s_idx)
    if s_idx%10000 == 2:
        write_file('ner.src', text)
        write_file('ner.tgt', ner)
    doc = nlp(s)
    text_list = []
    ner_list = []
    lable_list = []
    for token in doc:     
        if token.ent_iob_ != 'O':
            text_list.append(token.text)
            ner_list.append(token.ent_type_)
            label_count[token.ent_type_] = label_count.get(token.ent_type_,0) + 1
        elif token.text != '\n':
            text_list.append(token.text)
            ner_list.append('O')
    text.append(' '.join(text_list)+'\n')
    ner.append(' '.join(ner_list)+'\n')

if(args[1] == '-train_data'):
    write_file('train.ner.tgt', ner)
if(args[1] == '-test_data'):
    write_file('test.ner.tgt', ner)

label_map = {}
label_map['O'] = len(label_map)
for key, value in label_count.items():
    label_map[key] = len(label_map)
print(label_map)
# with open('label-index.map', 'w') as fp:
#     json.dump(label_map, fp)
