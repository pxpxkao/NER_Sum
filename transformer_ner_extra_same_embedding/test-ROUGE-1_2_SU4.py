import sys
import re, string
from pythonrouge.pythonrouge import Pythonrouge
args = sys.argv

with open(args[1], 'r') as f1:
    summary = f1.readlines()
with open(args[2], 'r') as f2:
    reference = f2.readlines()
"""
for i in range(len(summary)):
    summary[i] = re.sub('[%s]' % re.escape(string.punctuation), '', summary[i])
    reference[i] = re.sub('[%s]' % re.escape(string.punctuation), '', reference[i])
"""
for i in range(len(summary)):
    summary[i] = [summary[i].strip()]
    reference[i] = [[reference[i].strip()]]
print(len(summary))
#summary = summary[1:1000]
#reference = reference[1:1000]

rouge = Pythonrouge(summary_file_exist=False,
                    summary=summary, reference=reference,
                    n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                    recall_only=True, stemming=True, stopwords=True,
                    word_level=True, length_limit=True, length=50,
                    use_cf=False, cf=95, scoring_formula='average',
                    resampling=True, samples=1000, favor=True, p=0.5)
score = rouge.calc_score()
print(score)
