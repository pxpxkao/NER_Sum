import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

classes = ['O', 'I-ORG', 'I-LOC', 'I-MISC', 'I-PER', 'B-MISC', 'B-ORG', 'B-LOC']

test = []
with open('data/label.txt') as f:
    for line in f.readlines():
        ner_list = line.split()
        test.append(ner_list)
pred = []
with open('pred.txt') as f:
    for line in f.readlines():
        ner_list = line.split()
        pred.append(ner_list)
print(len(pred[0]))
print("Testa....")
if (len(test) != len(pred)):
    print("Error!")
    print(len(test), len(pred))
y_test, y_pred = [], []
for i in range(min(len(test), len(pred))):
    for j in range(min(44, len(test[i]))):
        try:
            y_test.append(test[i][j])
            y_pred.append(pred[i][j])
        except:
            print(i, j)
print('y_test length: ', len(y_test))
print('y_pred length: ', len(y_pred))


print(classification_report(y_pred=y_pred, y_true=y_test, labels=classes))