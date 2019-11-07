import sys
import json

def read_json(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data

def write_json(filename,data):
    with open(filename, 'w') as fp:
        json.dump(data, fp)

args = sys.argv

with open(args[2], 'r') as fp:
    summary = fp.readlines()
print(len(summary))

import spacy
nlp = spacy.load('en_core_web_lg')
# text = "marseille , france -lrb- cnn -rrb- the french prosecutor leading an investigation into the crash of germanwings flight 9525 insisted wednesday that he was not aware of any video footage from on board the plane . "
# doc = nlp(text)
# spacy.displacy.serve(doc, style='ent')
# for token in doc.ents:
#     print(token.text, token.label_)

label_count = {}
label_map = {}
ner_map = {}
for idx, s in enumerate(summary):
    print(idx)
    doc = nlp(s)
    ner_list = {}
    for token in doc.ents:
        ner_list[token.text] = token.label_
        label_count[token.label_] = label_count.get(token.label_,0) + 1
    ner_map['P'+str(idx)] = ner_list

for key, value in label_count.items():
    label_map[key] = len(label_map)

if(args[1] == '-train_data'):
    write_json('train.ner.map', ner_map)
if(args[1] == '-test_data'):
    write_json('test.ner.map', ner_map)
write_json('label.map', label_map)

