import sys
args = sys.argv
if len(args) == 1:
    name = 'pred_dir/pred.txt'
else:
    name = args[1]
print("Testing file:", name)
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

#classes = ['O', 'I-ORG', 'I-LOC', 'I-MISC', 'I-PER', 'B-MISC', 'B-ORG', 'B-LOC']
classes = ["O", "GPE", "ORG", "NORP", "CARDINAL", "DATE", "PERSON", "TIME", "LOC", "PRODUCT", "PERCENT", "FAC", "ORDINAL", "QUANTITY", "WORK_OF_ART", "MONEY", "EVENT", "LAW", "LANGUAGE"]


test = []
with open('/nfs/nas-7.1/pwgao/data/cnndm/test.ner.src') as f:
    for line in f.readlines():
        ner_list = line.split()
        test.append(ner_list)
pred = []
with open(name) as f:
    for line in f.readlines():
        ner_list = line.split()
        pred.append(ner_list)
print(len(pred[0]))
print("Scoring....")
if (len(test) != len(pred)):
    print("Error!")
    print(len(test), len(pred))
y_test, y_pred = [], []
for i in range(min(len(test), len(pred))):
    for j in range(min(400, len(pred[i]))):
        try:
            y_test.append(test[i][j])
            y_pred.append(pred[i][j])
        except:
            print(i, j)
print('y_test length: ', len(y_test))
print('y_pred length: ', len(y_pred))


print(classification_report(y_pred=y_pred, y_true=y_test, labels=classes))