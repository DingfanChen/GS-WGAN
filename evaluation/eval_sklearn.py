import os, sys
import numpy as np
import pickle
import argparse
import logging
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn import linear_model, ensemble, naive_bayes, svm, tree, discriminant_analysis, neural_network
import xgboost
import torch
from torchvision import datasets

sys.path.insert(0, './../source')
from utils import mkdir, savefig

NUM_CLASSES = 10
IMG_W = IMG_H = 28
IMG_C = 1
DATA_ROOT = './../data'
RESULT_DIR = './../results'


##########################################################################
### config
##########################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_data', type=str,
                        default='./../results/mnist/main/ResNet_default/gen_data.npz',
                        help='path of file that store the generated data')
    parser.add_argument('--save_dir', type=str,
                        help='output folder name; will be automatically save to the folder of gen_data if not specified')
    parser.add_argument('--dataset', '-data', type=str, default='mnist', choices=['mnist', 'fashionmnist'],
                        help='dataset name')
    parser.add_argument('--if_print_conf_mat', action='store_true', default=False,
                        help='print confusion matrix')
    parser.add_argument('--if_skip_slow_models', action='store_true', default=False,
                        help='skip models that take longer')
    parser.add_argument('--if_only_slow_models', action='store_true', default=False, help='only do slower the models')
    parser.add_argument('--if_compute_real_to_real', action='store_true', default=False,
                        help='add train:real,test:real')
    parser.add_argument('--if_skip_gen_to_real', action='store_true', default=False,
                        help='skip train:gen,test:real setting')
    parser.add_argument('--if_skip_real_to_gen', action='store_true', default=False,
                        help='add train:real,test:gen')
    args = parser.parse_args()
    return args


##########################################################################
### helper functions
##########################################################################
def test_model(model, x_trn, y_trn, x_tst, y_tst):
    model.fit(x_trn, y_trn)
    y_pred = model.predict(x_tst)
    acc = accuracy_score(y_pred, y_tst)
    f1 = f1_score(y_true=y_tst, y_pred=y_pred, average='macro')
    conf = confusion_matrix(y_true=y_tst, y_pred=y_pred)
    return acc, f1, conf


def vis_data(samples, save_name, save_dir):
    # requires IMG_C = 1
    ncols = 10
    nrows = len(samples) // ncols
    samples = np.reshape(samples, [-1, IMG_W, IMG_H])
    zeros = np.zeros((nrows * ncols - len(samples), IMG_W, IMG_H))

    data_mat = np.concatenate([samples, zeros])
    data_mat_list = [data_mat[i * ncols: (i + 1) * ncols] for i in range(nrows)]
    data_mat_flat = np.concatenate([np.reshape(k, [IMG_W * ncols, IMG_H]) for k in data_mat_list], axis=1)
    plt.imsave(os.path.join(save_dir, '{}.png'.format(save_name)), data_mat_flat, cmap='gray', vmin=0., vmax=1.)


##########################################################################
### main
##########################################################################
def main(args):
    dataset = args.dataset
    if args.save_dir is not None:
        save_dir = os.path.join(args.save_dir)
    else:
        ## set up the output folder based on the extracted path
        save_dir = os.path.join(os.path.dirname(args.gen_data), 'eval', 'sklearn')
    mkdir(save_dir)

    ### set up logger
    logging.basicConfig(filename=os.path.join(save_dir, 'log.txt'), level=logging.INFO, format='%(message)s')
    header = '=' * 100 + '\n' + 'evaluate ' + args.gen_data
    header += '=' * 100
    logging.info(header)

    ### load real data
    if dataset == 'mnist':
        train_data = datasets.MNIST(os.path.join(DATA_ROOT, 'MNIST'), download=True, train=True)
        test_data = datasets.MNIST(os.path.join(DATA_ROOT, 'MNIST'), train=False)
    elif dataset == 'fashionmnist':
        train_data = datasets.FashionMNIST(os.path.join(DATA_ROOT, 'FashionMNIST'), download=True, train=True)
        test_data = datasets.FashionMNIST(os.path.join(DATA_ROOT, 'FashionMNIST'), train=False)
    else:
        raise ValueError

    try:
        x_real_train, y_real_train = train_data.train_data.numpy(), train_data.train_labels.numpy()
        x_real_test, y_real_test = test_data.test_data.numpy(), test_data.test_labels.numpy()
    except:
        x_real_train, y_real_train = train_data.data.numpy(), train_data.targets.numpy()
        x_real_test, y_real_test = test_data.data.numpy(), test_data.targets.numpy()

    x_real_train = np.reshape(x_real_train, (-1, IMG_H * IMG_W)) / 255
    x_real_test = np.reshape(x_real_test, (-1, IMG_H * IMG_W)) / 255

    ### load generated data
    gen_data = np.load(args.gen_data)
    x_gen = gen_data['data_x']
    y_gen = gen_data['data_y']
    x_gen = np.reshape(x_gen, (-1, IMG_H * IMG_W * IMG_C))

    ### visualize
    vis_data(x_gen[:100], 'vis', save_dir)

    ### shuffle data
    rand_perm = np.random.permutation(len(y_gen))
    x_gen, y_gen = x_gen[rand_perm], y_gen[rand_perm]
    print('data ranges: [{},{}], [{},{}],[{},{}]'.format(np.min(x_real_test), np.max(x_real_test),
                                                         np.min(x_real_train), np.max(x_real_train),
                                                         np.min(x_gen), np.max(x_gen)))
    print('label ranges: [{},{}], [{},{}],[{},{}]'.format(np.min(y_real_test), np.max(y_real_test),
                                                          np.min(y_real_train), np.max(y_real_train),
                                                          np.min(y_gen), np.max(y_gen)))

    ### eval classifiers
    models = {'logistic_reg': linear_model.LogisticRegression,
              'random_forest': ensemble.RandomForestClassifier,
              'gaussian_nb': naive_bayes.GaussianNB,
              'bernoulli_nb': naive_bayes.BernoulliNB,
              'linear_svc': svm.LinearSVC,
              'decision_tree': tree.DecisionTreeClassifier,
              'lda': discriminant_analysis.LinearDiscriminantAnalysis,
              'adaboost': ensemble.AdaBoostClassifier,
              'mlp': neural_network.MLPClassifier,
              'bagging': ensemble.BaggingClassifier,
              'gbm': ensemble.GradientBoostingClassifier,
              'xgboost': xgboost.XGBClassifier}
    slow_models = {'bagging', 'gbm', 'xgboost'}

    model_specs = defaultdict(dict)
    model_specs['logistic_reg'] = {'solver': 'lbfgs', 'max_iter': 5000, 'multi_class': 'auto'}
    model_specs['random_forest'] = {'n_estimators': 100, 'class_weight': 'balanced'}
    model_specs['linear_svc'] = {'max_iter': 10000, 'tol': 1e-8, 'loss': 'hinge'}
    model_specs['bernoulli_nb'] = {'binarize': 0.5}
    model_specs['lda'] = {'solver': 'eigen', 'n_components': 9, 'tol': 1e-8, 'shrinkage': 0.5}
    model_specs['decision_tree'] = {'class_weight': 'balanced', 'criterion': 'gini', 'splitter': 'best',
                                    'min_samples_split': 2, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0,
                                    'min_impurity_decrease': 0.0}
    model_specs['adaboost'] = {'n_estimators': 100, 'algorithm': 'SAMME.R'}
    model_specs['bagging'] = {'max_samples': 0.1, 'n_estimators': 20}
    model_specs['gbm'] = {'subsample': 0.1, 'n_estimators': 50}
    model_specs['xgboost'] = {'colsample_bytree': 0.1, 'objective': 'multi:softprob', 'n_estimators': 50}

    ### models to be considered
    if args.if_skip_slow_models:
        run_keys = [k for k in models.keys() if k not in slow_models]
    elif args.if_only_slow_models:
        run_keys = [k for k in models.keys() if k in slow_models]
    else:
        run_keys = models.keys()
    run_keys = sorted(run_keys)

    ### store results
    result_dict = {}
    for key in run_keys:
        print('Model: {}'.format(key))
        logging.info('Model: {}'.format(key))

        if not args.if_skip_gen_to_real:
            model = models[key](**model_specs[key])
            g_to_r_acc, g_to_r_f1, g_to_r_conf = test_model(model, x_gen, y_gen, x_real_test, y_real_test)
        else:
            g_to_r_acc, g_to_r_f1, g_to_r_conf = -1, -1, -np.ones((10, 10))

        if args.if_compute_real_to_real:
            model = models[key](**model_specs[key])
            base_acc, base_f1, base_conf = test_model(model, x_real_train, y_real_train, x_real_test, y_real_test)
        else:
            base_acc, base_f1, base_conf = -1, -1, -np.ones((10, 10))

        if not args.if_skip_real_to_gen:
            model = models[key](**model_specs[key])
            r_to_g_acc, r_to_g_f1, r_to_g_conv = test_model(model, x_real_train, y_real_train, x_gen[:10000],
                                                            y_gen[:10000])
        else:
            r_to_g_acc, r_to_g_f1, r_to_g_conv = -1, -1, -np.ones((10, 10))

        result_dict[key + '_g2r_acc'] = g_to_r_acc
        result_dict[key + '_r2g_acc'] = r_to_g_acc
        result_dict[key + '_g2r_f1'] = g_to_r_f1
        result_dict[key + '_r2g_f1'] = r_to_g_f1

        print('acc: real {}, gen to real {}, real to gen {}'.format(base_acc, g_to_r_acc, r_to_g_acc))
        print('f1:  real {}, gen to real {}, real to gen {}'.format(base_f1, g_to_r_f1, r_to_g_f1))
        logging.info('acc: real {}, gen to real {}, real to gen {}'.format(base_acc, g_to_r_acc, r_to_g_acc))
        logging.info('f1:  real {}, gen to real {}, real to gen {}'.format(base_f1, g_to_r_f1, r_to_g_f1))

        if args.if_print_conf_mat:
            print('gen to real confusion matrix:')
            logging.info('gen to real confusion matrix:')
            print(g_to_r_conf)
            logging.info(str(g_to_r_conf))

    ### writ csv
    field_names = sorted(result_dict.keys())
    csv_path = os.path.join(save_dir, 'sklearn_results.csv')
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a+', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_dict)
    return result_dict


if __name__ == '__main__':
    main(parse_arguments())
