import argparse
from solver import Solver
from coref_solver import CorefSolver

def parse():
    parser = argparse.ArgumentParser(description="tree transformer")
    parser.add_argument('-train', action='store_true',help='whether train whole model')
    parser.add_argument('-test', action='store_true',help='whether test')
    parser.add_argument('-pretrain', action='store_true',help='whether pretrain')
    #train/test config
    parser.add_argument('-batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-beam_size', default=1, type=int)
    parser.add_argument('-n_best', default=1, type=int)
    parser.add_argument('-max_len', type=int, default=100, help='maximum output length', dest= 'max_len')
    parser.add_argument('-min_len', type=int, default=50, help='minimum output length', dest= 'min_length')
    parser.add_argument('-total_steps', type=int, default=1000000)
    parser.add_argument('-print_every_steps', type=int, default=5000)
    parser.add_argument('-valid_every_steps', type=int, default=100000)
    parser.add_argument('-max_num_clusters', type=int, default=12)
    # parser.add_argument('-n_best', type=int, default=1)

    #model config
    parser.add_argument('-num_layer', type=int, default=6)
    parser.add_argument('-pointer_gen', action = 'store_true')
    parser.add_argument('-ner_last', action = 'store_true')
    parser.add_argument('-ner_at_embedding', action='store_true')
    parser.add_argument('-coref', action='store_true')
    parser.add_argument('-dropout', default = 0.5, type=float)
    parser.add_argument('-entity_encoder_type', type=str, default='linear')
    parser.add_argument('-fusion', type = str, default='concat')
    #data
    parser.add_argument('-w_valid_file',default='./valid.out.txt',help='data dir')
    parser.add_argument('-train_file',default='../data/new_cnndm/train.txt.src_ner_upperbound',help='data dir')
    parser.add_argument('-tgt_file',default='../data/new_cnndm/train.txt.tgt.tagged',help='data dir')
    parser.add_argument('-valid_file',default='../data/new_cnndm/val.txt.src_ner_upperbound',help='data dir')
    parser.add_argument('-valid_tgt_file',default='../data/new_cnndm/val.txt.tgt.tagged',help='data dir')
    parser.add_argument('-valid_ref_file',default='../data/new_cnndm/val.txt.tgt.tagged',help='data dir')
    parser.add_argument('-test_file',default='../data/new_cnndm/test.txt.src_ner_upperbound',help='data dir')
    parser.add_argument('-coref_file',default='../data/coref_convert/train.convert.pkl',help='data dir')
    parser.add_argument('-valid_coref_file',default='../data/coref_convert/val.convert.pkl',help='data dir')
    parser.add_argument('-test_coref_file',default='../data/coref_convert/test.convert.pkl',help='data dir')
    #load model
    parser.add_argument('-load', help= 'load: model_dir', dest= 'load_model')
    parser.add_argument('-entity_encoder', help='load: entity_encoder_dir')
    # model/ output file location
    parser.add_argument('-model_dir',default='train_model',help='output model weight dir')
    parser.add_argument('-pred_dir', default='./pred_dir/', help='prediction dir', dest='pred_dir')
    parser.add_argument('-filename', default='pred.txt', help='prediction file', dest='filename')
    #keep track
    parser.add_argument('-logfile', help='logfile', type=str)
    parser.add_argument('-logdir', type=str, default='./log/')
    parser.add_argument('-exp_name', type=str)

    parser.add_argument('-disable_comet', action='store_true')

    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    if args.coref:
        solver = CorefSolver(args)
    else:
        solver = Solver(args)
    
    if args.train:
        solver.train()
    elif args.test:
        solver.test()
