import argparse
from solver import Solver

def parse():
    parser = argparse.ArgumentParser(description="tree transformer")
    parser.add_argument('-model_dir',default='train_model',help='output model weight dir')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-train_sum_file',default='../../data/cnndm/train.txt.src',help='train_sum dir')
    parser.add_argument('-tgt_sum_file',default='../../data/cnndm/train.txt.tgt.tagged',help='tgt_sum dir')
    parser.add_argument('-test_sum_file',default='../../data/cnndm/test.txt.src',help='test_sum dir')
    parser.add_argument('-load', help= 'load: model_dir', dest= 'load_model') #, default='./train_model/model.pth'
    parser.add_argument('-train', action='store_true',help='whether train whole model')
    parser.add_argument('-test', action='store_true',help='whether test')
    parser.add_argument('-pretrain', action='store_true',help='whether pretrain')
    parser.add_argument('-dict', default='./dictionary.json', help='dict_dir')
    parser.add_argument('-max_len', type=int, default=80, help='maximum output length', dest= 'max_len')
    parser.add_argument('-pred_dir', default='./pred_dir/', help='prediction dir', dest='pred_dir')
    parser.add_argument('-filename', default='pred.txt', help='prediction file', dest='filename')
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    solver = Solver(args)
    
    if args.train:
        solver.train()
    elif args.test:
        solver._test()