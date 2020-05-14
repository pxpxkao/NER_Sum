import argparse
from solver import Solver

def parse():
    parser = argparse.ArgumentParser(description="tree transformer")
    parser.add_argument('-model_dir',default='train_model',help='output model weight dir')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-num_layer', type=int, default=4)
    parser.add_argument('-num_epoch', type=int, default=100)
    parser.add_argument('-train_file',default='/nfs/nas-7.1/pwgao/data/cnndm/train.txt.src',help='data dir')
    parser.add_argument('-tgt_file',default='/nfs/nas-7.1/pwgao/data/cnndm/train.txt.tgt.tagged',help='data dir')
    parser.add_argument('-valid_file',default='/nfs/nas-7.1/pwgao/data/cnndm/val.txt.src',help='data dir')
    parser.add_argument('-valid_tgt_file',default='/nfs/nas-7.1/pwgao/data/cnndm/val.txt.tgt.tagged',help='data dir')
    parser.add_argument('-test_file',default='/nfs/nas-7.1/pwgao/data/cnndm/test.txt.src',help='data dir')
    parser.add_argument('-train_ner_tgt_file',default='/nfs/nas-7.1/pwgao/data/cnndm/train.ner.src',help='data dir')
    parser.add_argument('-valid_ner_tgt_file',default='/nfs/nas-7.1/pwgao/data/cnndm/val.ner.src',help='data dir')
    parser.add_argument('-test_ner_tgt_file',default='/nfs/nas-7.1/pwgao/data/cnndm/test.ner.src',help='data dir')
    parser.add_argument('-load_emb', default="/nfs/nas-7.1/pwgao/data/90w_1.561824model.pth", help= 'load: emb_model_dir', dest= 'load_embmodel')
    parser.add_argument('-load', help= 'load: model_dir', dest= 'load_model')
    parser.add_argument('-only_emb', action='store_true',help='whether only word embedding')
    parser.add_argument('-train', action='store_true',help='whether train whole model')
    parser.add_argument('-test', action='store_true',help='whether test')
    parser.add_argument('-dict', default='../origin/data/dictionary.json', help='dict_dir')
    parser.add_argument('-max_len', type=int, default=100, help='maximum output length', dest= 'max_len')
    parser.add_argument('-pred_dir', default='./pred_dir/', help='prediction dir', dest='pred_dir')
    parser.add_argument('-filename', default='pred.txt', help='prediction file', dest='filename')
    parser.add_argument('-logfile', help='logfile', type=str, default='train.log')
    parser.add_argument('-logdir', type=str, default='./log/')
    parser.add_argument('-exp_name', type=str, default='train')
    parser.add_argument('-pointer_gen', default=True, type=bool)
    parser.add_argument('-dropout', default = 0.5, type=float)
    parser.add_argument('-beam_size', default=1, type=int)
    parser.add_argument('-n_best', default=1, type=int)
    parser.add_argument('-idx', default='linear', type=str)
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    solver = Solver(args)
    
    if args.train:
        solver.train()
    elif args.test:
        solver.test()
