import argparse
from solver import Solver

def parse():
    parser = argparse.ArgumentParser(description="tree transformer")
    parser.add_argument('-model_dir',default='train_model',help='output model weight dir')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    #parser.add_argument('-load',action='store_true',help='load pretrained model')
    parser.add_argument('-train_file',default='../data/cnn/train.txt.src',help='data dir')
    parser.add_argument('-tgt_file',default='../data/cnn/train.txt.tgt.tagged',help='data dir')
    parser.add_argument('-test_file',default='../data/cnn/test.txt.src',help='data dir')
    parser.add_argument('-load', default='./train_model/40w_model.pth', help= 'load: model_dir', dest= 'load_model')
    parser.add_argument('-train', action='store_true',help='whether train whole model')
    parser.add_argument('-test', action='store_true',help='whether test')
    parser.add_argument('-pretrain', action='store_true',help='whether pretrain')
    parser.add_argument('-dict', default='../origin/data/dictionary.json', help='dict_dir')
    parser.add_argument('-max_len', type=int, default=15, help='maximum output length', dest= 'max_len')
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
        solver.test()
