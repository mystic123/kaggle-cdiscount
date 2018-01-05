import io
import os
import sys
import threading

import bson
from skimage.data import imread
from random import randint

from skimage.io import imsave

NUM_EXAMPLES = 12371293
VALID_SET_SIZE = 100
NUM_THREADS = 16

train_set_prods = set()
valid_set_prods = set()
valid_set_classes = set()


class WorkerThread(threading.Thread):
    def __init__(self, args, id):
        super(WorkerThread, self).__init__()
        self.id = id
        self.args = args
        self.data = bson.decode_file_iter(open(args.train_bson, 'rb'))

    def run(self):
        for i, d in enumerate(self.data):
            if i % NUM_THREADS == self.id:
                if self.id == NUM_THREADS - 1:
                    sys.stderr.write('{}\r'.format(i))
                category_id = int(d['category_id'])
                prod_id = int(d['_id'])
                if (category_id not in valid_set_classes) or (randint(0, VALID_SET_SIZE) == 1):
                    path_ = args.valid_dir
                    valid_set_prods.add((prod_id, category_id))
                    valid_set_classes.add(category_id)
                else:
                    path_ = args.train_dir
                    train_set_prods.add((prod_id, category_id))
                for j, pic in enumerate(d['imgs']):
                    pic = imread(io.BytesIO(pic['picture']))
                    img_name = '{}_{}.jpg'.format(prod_id, j)
                    imsave(os.path.join(path_, img_name), pic)


def main(args):
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    if not os.path.exists(args.valid_dir):
        os.makedirs(args.valid_dir)

    threads = []
    for i in range(NUM_THREADS):
        t = WorkerThread(args, i)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    with open(os.path.join(args.train_dir, 'meta.csv'), 'w') as meta_file:
        meta_file.write('_id, category_id\n')
        for prod, cls in train_set_prods:
            meta_file.write('{},{}\n'.format(prod, cls))

    with open(os.path.join(args.valid_dir, 'meta.csv'), 'w') as meta_file:
        meta_file.write('_id, category_id\n')
        for prod, cls in valid_set_prods:
            meta_file.write('{},{}\n'.format(prod, cls))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # Data handling parameters
    parser.add_argument('train_bson', type=str, default=None, help='train bson file')
    parser.add_argument('train_dir', type=str, default=None, help='train dir')
    parser.add_argument('valid_dir', type=str, default=None, help='valid dir')

    args = parser.parse_args()

    main(args)

    exit(0)
