import os
import sys
import argparse
import random
import time
import shutil
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import classification_networks as models

sys.path.insert(0, './../source')
from utils import savefig, mkdir
from classifier_utils import train, test, adjust_learning_rate, Logger

DATA_ROOT = './../data'
RESULT_DIR = './../results'


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_data', type=str,
                        default='./../results/mnist/main/ResNet_default/gen_data.npz',
                        help='path of file that store the generated data')
    parser.add_argument('--save_dir', type=str,
                        help='output folder name; will be automatically save to the folder of gen_data if not specified')
    parser.add_argument('--model_architecture', '-arch', type=str, default='lenet', help='model architecture')
    parser.add_argument('--dataset', '-data', type=str, default='mnist', choices=['mnist', 'fashionmnist'],
                        help='dataset')
    parser.add_argument('--random_seed', '-s', type=int, default=1000, help='random seed')
    parser.add_argument('--lr', '-lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--schedule_milestone', type=int, nargs='+', default=[40],
                        help='when to decrease the learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate step gamma')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--train_batchsize', '-trbs', type=int, default=128, help='training batch size')
    parser.add_argument('--test_batchsize', '-tebs', type=int, default=128, help='testing batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--num_epochs', '-ep', type=int, default=50, help='number of epochs')
    args = parser.parse_args()
    return args


def check_args(args):
    '''
    check and store the arguments as well as set up the save_dir
    :param args: arguments
    :return:
    '''
    ## set up save_dir
    name = '{}_lr{}'.format(args.model_architecture, args.lr)
    if args.save_dir is not None:
        save_dir = os.path.join(args.save_dir, name)
    else:
        save_dir = os.path.join(os.path.dirname(args.gen_data), 'eval', 'cnn', name)
    mkdir(save_dir)

    ## store the parameters
    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.writelines(k + ":" + str(v) + "\n")
            print(k + ":" + str(v))
    pickle.dump(vars(args), open(os.path.join(save_dir, 'params.pkl'), 'wb'), protocol=2)

    ## store this script
    shutil.copy(os.path.realpath(__file__), save_dir)
    return args, save_dir


#############################################################################################################
# main
#############################################################################################################
def main():
    ### config
    args, save_dir = check_args(parse_arguments())
    dataset = args.dataset
    device = torch.device('cuda')

    ### random seed
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    ### load testdata
    num_classes = 10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    if dataset == 'mnist':
        testset = datasets.MNIST(DATA_ROOT, download=True, transform=transform, train=False)
    elif dataset == 'fashionmnist':
        testset = datasets.FashionMNIST(DATA_ROOT, download=True, transform=transform, train=False)
    else:
        raise ValueError
    testloader = data.DataLoader(testset, batch_size=args.test_batchsize, shuffle=False, num_workers=args.num_workers)

    ### load generated data
    gen_data = np.load(args.gen_data)
    x_gen = gen_data['data_x']
    x_gen = torch.from_numpy(np.transpose(x_gen, [0, 3, 1, 2]))  # totensor
    y_gen = torch.from_numpy(gen_data['data_y'])

    train_x = torch.stack([normalize(x) for x in x_gen])
    train_y = y_gen
    print(train_x.shape)
    print(train_y.shape)

    full_indices = np.arange(len(train_x))
    np.random.shuffle(full_indices)
    tensor_x = train_x[full_indices]
    tensor_y = train_y[full_indices]

    trainset = data.TensorDataset(tensor_x, tensor_y)  # create your datset
    trainloader = data.DataLoader(trainset, batch_size=args.train_batchsize, shuffle=True, num_workers=args.num_workers)

    ### Model
    model = models.__dict__[args.model_architecture](num_classes=num_classes)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    cudnn.benchmark = True

    ### Loss
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ### Set up logger
    title = args.dataset + '_' + args.model_architecture
    start_epoch = 0
    logger = Logger(os.path.join(save_dir, 'log.txt'), title=title)
    logger.set_names(['Epoch', 'Train Acc', 'Val Acc', 'Train Loss', 'Val Loss', 'Learning Rate'])

    ### Training
    for epoch in range(start_epoch, args.num_epochs):
        adjust_learning_rate(optimizer, epoch, args.gamma, args.schedule_milestone)

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, device)
        test_loss, test_acc = test(testloader, model, criterion, epoch, device)

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        logger.append([epoch, train_acc, test_acc, train_loss, test_loss, lr])
        print('Epoch %d, Train acc: %f, Test acc: %f, lr: %f' % (epoch, train_acc, test_acc, lr))

        save_dict = {'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'opt_state_dict': optimizer.state_dict()}

        torch.save(save_dict, os.path.join(save_dir, 'checkpoint.pkl'))
        torch.save(model, os.path.join(save_dir, 'model.pt'))

        logger.plot(['Train Loss', 'Val Loss'])
        savefig(os.path.join(save_dir, 'loss.png'))

        logger.plot(['Train Acc', 'Val Acc'])
        savefig(os.path.join(save_dir, 'acc.png'))

    logger.close()


if __name__ == '__main__':
    main()
