import os
import sys

def main(f, fw, line_limit):
    sentences = f.readlines()[:line_limit]
    for l in sentences:
        fw.write(l)


def check_space(f):
    data = f.readlines()
    for i, l in enumerate(data):
        if len(l) < 5:
            print('sesefe', i)


if __name__ == '__main__':
    f = open(sys.argv[1], 'r')
    fw = open(sys.argv[2], 'w')
    line_limit = 10000
    main(f, fw, line_limit)
    # check_space(f)