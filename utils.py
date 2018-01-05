import json
import os
import threading
from random import shuffle

import numpy as np
from skimage.io import imread

PATH = 'PATH'


def load_dict(name):
    with open(os.path.join(PATH, name), 'r') as f:
        d = json.load(f)
    return d


LABEL_TO_CAT = load_dict('label_to_cat.json')
CAT_TO_LABEL = load_dict('cat_to_label.json')
CAT_TO_LVL2 = load_dict('cat_to_level2_label.json')
CAT_TO_LVL1 = load_dict('cat_to_level1_label.json')


class PrefetchingThread(threading.Thread):
    def __init__(self, queue, data_path):
        super(PrefetchingThread, self).__init__()
        self.queue = queue
        self.data_path = data_path
        self.products = {}

    def run(self):
        files = [f for f in os.listdir(self.data_path) if f.endswith('.jpg')]
        with open(os.path.join(self.data_path, 'meta.csv'), 'r') as f:
            header = 0
            for line in f:
                if header > 0:
                    prod_id, cat_id = line.strip().split(',')
                    self.products[str(prod_id)] = cat_id
                else:
                    header += 1
        while True:
            shuffle(files)
            for f in files:
                prod_id, _ = f.split('_')
                pic = imread(os.path.join(self.data_path, f))
                cat_id = str(self.products[str(prod_id)])
                self.queue.put(
                    ((CAT_TO_LVL1[cat_id], CAT_TO_LVL2[cat_id], CAT_TO_LABEL[cat_id]), np.array(pic))
                )
