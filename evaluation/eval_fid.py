import os
import sys
import argparse
import numpy as np
import logging
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import frechet_inception_distance as FID

sys.path.insert(0, './../source')
from utils import mkdir

Img_W = Img_H = 28
Img_C = 1
DATA_ROOT = './../data'
STAT_DIR = './stats'


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-data', type=str, default='mnist', choices=['mnist', 'fashionmnist'],
                        help=' dataset name')
    parser.add_argument('--gen_data', type=str,
                        default='./../results/mnist/main/ResNet_default/gen_data.npz',
                        help='path of file that store the generated data')
    parser.add_argument('--save_dir', type=str,
                        help='output folder name; will be automatically save to the folder of gen_data if not specified')
    parser.add_argument('--num_eval_samples', type=int, default=20000,
                        help="number of samples to be evaluated")
    args = parser.parse_args()
    return args


##########################################################################
### helper functions
##########################################################################
def convert_data(data, Img_W, Img_H, Img_C):
    shape = data.shape
    if len(shape) == 2:
        data = np.reshape(data, [-1, Img_W, Img_H, Img_C])
    elif len(shape) == 3:
        data = np.reshape(data, [-1, Img_W, Img_H, Img_C])
    elif len(shape) == 4:
        if shape[1] == Img_C:
            data = np.transpose(data, [0, 2, 3, 1])

    if Img_C == 1:
        data = np.tile(data, [1, 1, 1, 3])
    return data * 255


def load_mnist(selection='train_val', one_hot=True):
    mnist = read_data_sets(os.path.join(DATA_ROOT, 'MNIST/raw'), one_hot=one_hot)
    if selection == 'train':
        x = mnist.train.images
        y = mnist.train.labels
    elif selection == 'train_val':
        x = np.concatenate([mnist.train.images, mnist.validation.images])
        y = np.concatenate([mnist.train.labels, mnist.validation.labels])
    elif selection == 'test':
        x = mnist.test.images
        y = mnist.test.labels
    elif selection == 'val':
        x = mnist.validation.images
        y = mnist.validation.labels
    print('data shape', x.shape, y.shape)
    print('data range:', np.min(x), np.max(x))
    return x, y


def load_fashionmnist(selection='train_val', one_hot=True):
    mnist = read_data_sets(os.path.join(DATA_ROOT, 'FashionMNIST/raw'), one_hot=one_hot)
    if selection == 'train':
        x = mnist.train.images
        y = mnist.train.labels
    elif selection == 'train_val':
        x = np.concatenate([mnist.train.images, mnist.validation.images])
        y = np.concatenate([mnist.train.labels, mnist.validation.labels])
    elif selection == 'test':
        x = mnist.test.images
        y = mnist.test.labels
    elif selection == 'val':
        x = mnist.validation.images
        y = mnist.validation.labels
    print('data shape', x.shape, y.shape)
    print('data range:', np.min(x), np.max(x))
    return x, y


##########################################################################
### main
##########################################################################
def main(args):
    ### Get real data statistics
    stat_file = os.path.join(STAT_DIR, args.dataset, 'stat.npz')
    if not os.path.exists(stat_file):
        if args.dataset == 'mnist':
            real_data, _ = load_mnist('train_val')
        elif args.dataset == 'fashionmnist':
            real_data, _ = load_fashionmnist('train_val')
        real_data = convert_data(real_data, Img_W, Img_H, Img_C)
        m1, s1, real_act = FID.calculate_activation(real_data)

        ## Save real statistics
        mkdir(os.path.join(STAT_DIR, args.dataset))
        np.savez(stat_file, mu=m1, sigma=s1, real_act=real_act)
    else:
        ## Load pre-computed statistics
        f = np.load(stat_file)
        m1, s1 = f['mu'][:], f['sigma'][:]
        real_act = f['real_act']
    print(m1.shape)
    print(s1.shape)
    print(real_act.shape)

    ### load gen data
    gen_data = np.load(args.gen_data)
    x_gen = gen_data['data_x']
    x_gen = convert_data(x_gen, Img_W, Img_H, Img_C)
    rand_perm = np.random.permutation(len(x_gen))
    x_gen = x_gen[rand_perm]
    x_gen = x_gen[:args.num_eval_samples]
    print(x_gen.shape)
    print(np.min(x_gen), np.max(x_gen))

    ### Get fake data statistics and compute FID
    m2, s2, fake_act = FID.calculate_activation(x_gen)
    fid_value = FID.calculate_frechet_distance(m1, s1, m2, s2)

    ### Save results
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = os.path.join(os.path.dirname(args.gen_data), 'eval', 'FID')
    mkdir(save_dir)
    logging.basicConfig(filename=os.path.join(save_dir, 'FID.txt'), level=logging.INFO, filemode='a',
                        format='%(message)s')
    infostr = 'fid value: {}'.format(fid_value)
    print(infostr)
    logging.info('=' * 30)
    logging.info(infostr)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
