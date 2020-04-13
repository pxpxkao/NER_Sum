import sys
from tqdm import tqdm
from nltk.corpus import stopwords

args = sys.argv
stop_words = set(stopwords.words('english')) 

'''
Usage : 
    python3 preprocess_topk.py [src_file] [ner_file] [output_file] [topK]
'''

def count_topk(src_file, ner_file, K=10):
    topk = []
    for src, ner in tqdm(zip(open(src_file, encoding='utf-8'), open(ner_file, encoding='utf-8'))):
        src, ner = src.split(), ner.split()
        assert len(src) == len(ner)

        # Count ENTS
        topk_dict = dict()
        for (word, ent) in zip(src, ner):
            if ent != 'O': # top5 ent types : PERSON, DATE, GPE, ORG, CARDINAL
                topk_dict[word] = topk_dict.get(word, 0) + 1

        # Delete some words, like stopwords & 'cnn' '-lrb-' '-rrb-' "'s"
        delete = [w for w in topk_dict if w == 'cnn' or w == '-lrb-' or w == '-rrb-' or w == "'s" or len(w) < 2 or w in stop_words]
        for w in delete: del topk_dict[w]

        if len(topk_dict) < K:
            print("Length of topk dict is not enough...")
            t = [ k for k, v in sorted(topk_dict.items(), key=lambda x: x[1], reverse=True)][:len(topk_dict)]
        else:
            t = [ k for k, v in sorted(topk_dict.items(), key=lambda x: x[1], reverse=True)][:K]
        topk.append(' '.join(t))

    return topk

        
if __name__ == '__main__':
    src_file, ner_file, k = args[1], args[2], int(args[4])
    top = count_topk(src_file, ner_file, K=k)
    # print(top)
    with open(args[3], 'w', encoding='utf-8') as f:
        for t in top:
            f.write(t)
            f.write('\n')