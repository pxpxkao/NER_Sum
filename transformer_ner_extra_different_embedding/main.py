import argparse
from solver import Solver

def parse():
    parser = argparse.ArgumentParser(description="tree transformer")
    parser.add_argument('-model_dir',default='train_model',help='output model weight dir')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-num_layer', type=int, default=6)
    #parser.add_argument('-load',action='store_true',help='load pretrained model')
    parser.add_argument('-train_file',default='../../data/cnndm/train.txt.src',help='data dir')
    parser.add_argument('-tgt_file',default='../../data/cnndm/train.txt.tgt.tagged',help='data dir')
    parser.add_argument('-valid_file',default='../../data/cnndm/val.txt.src',help='data dir')
    parser.add_argument('-valid_tgt_file',default='../../data/cnndm/val.txt.tgt.tagged',help='data dir')
    parser.add_argument('-test_file',default='../../data/cnndm/test.txt.src',help='data dir')
    parser.add_argument('-train_ner_tgt_file',default='../../data/ents/train.ner.src',help='data dir')
    parser.add_argument('-valid_ner_tgt_file',default='../../data/ents/val.ner.src',help='data dir')
    parser.add_argument('-test_ner_tgt_file',default='../../data/ents/test.ner.src',help='data dir')
    parser.add_argument('-load', help= 'load: model_dir', dest= 'load_model')
    parser.add_argument('-train', action='store_true',help='whether train whole model')
    parser.add_argument('-test', action='store_true',help='whether test')
    parser.add_argument('-pretrain', action='store_true',help='whether pretrain')
    parser.add_argument('-dict', default='./dictionary.json', help='dict_dir')
    parser.add_argument('-max_len', type=int, default=100, help='maximum output length', dest= 'max_len')
    parser.add_argument('-pred_dir', default='./pred_dir/', help='prediction dir', dest='pred_dir')
    parser.add_argument('-filename', default='pred.txt', help='prediction file', dest='filename')
    parser.add_argument('-logfile', help='logfile', type=str)
    parser.add_argument('-logdir', type=str, default='./log/')
    parser.add_argument('-exp_name', type=str)
    parser.add_argument('-pointer_gen', default=False, type=bool)
    parser.add_argument('-dropout', default = 0.5, type=float)
    parser.add_argument('-beam_size', default=1, type=int)
    parser.add_argument('-n_best', default=1, type=int)
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    solver = Solver(args)
    
    if args.train:
        solver.train()
    elif args.test:
        solver.test()
