python3 main.py -test -batch_size 8 -pointer_gen True -load $1
rouge -f pred_dir/pred.txt ../../../../nfs/nas-7.1/pwgao/data/cnndm/test.txt.tgt.tagged --avg