import argparse
from solver import Solver

def parse():
    parser = argparse.ArgumentParser(description="tree transformer")
    parser.add_argument('-model_dir',default='train_model',help='output model weight dir')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-data_dir',default='../origin/data/train.article.txt',help='data dir', dest='data')
    #parser.add_argument('-load',action='store_true',help='load pretrained model')
    parser.add_argument('-load', default='./train_model/model.pth', help= 'load: model_dir', dest= 'load_model')
    parser.add_argument('-train', action='store_true',help='whether train whole model')
    parser.add_argument('-test', action='store_true',help='whether test')
    parser.add_argument('-pretrain', action='store_true',help='whether pretrain')
    parser.add_argument('-dict', default='../origin/data/dictionary.json', help='dict_dir')
    parser.add_argument('-max_len', type=int, default=15, help='maximum output length', dest= 'max_len')
    parser.add_argument('-pred_dir', default='./pred_dir/', help='prediction dir', dest='pred_dir')
    parser.add_argument('-filename', default='pred.txt', help='prediction file', dest='filename')
    parser.add_argument('-train_src', default='../data/cnn/train.txt.src', type=str)
    parser.add_argument('-train_tgt', default='../data/cnn/train.txt.tgt.tagged', type=str)
    parser.add_argument('-train_ner', default='../data/conll_data/train.txt')
    parser.add_argument('-train_ner_tgt', default='../data/conll_data/label.txt')
    parser.add_argument('-dict_path', default='../data/dictionary.json', type=str)
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    solver = Solver(args)
    
    if args.train:
        solver.train()
    elif args.test:
        solver._test()