import os
import sys
import argparse
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import classification_networks

sys.path.insert(0, './../source')
from utils import mkdir, savefig

Img_W = Img_H = 28
Img_C = 1
DATA_ROOT = './../data'


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
    parser.add_argument('--pretrain_dir', type=str,
                        help='path to the pre-trained classifier')
    parser.add_argument('--dataset', '-data', default='mnist',
                        help='dataset name')
    parser.add_argument('--if_real', action='store_true', default=False,
                        help='if evaluate the real data')
    args = parser.parse_args()
    return args


##########################################################################
### helper functions
##########################################################################
def load_realdata(dataset, batch_size=100):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    if dataset == 'mnist':
        testset = datasets.MNIST(DATA_ROOT, download=True, transform=transform, train=False)
    elif dataset == 'fashionmnist':
        testset = datasets.FashionMNIST(DATA_ROOT, download=True,
                                        transform=transform, train=False)
    else:
        raise ValueError
    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return testloader


def get_inception_score(data_loader, model_dir, device=torch.device('cuda')):
    ### load model
    model = torch.load(os.path.join(model_dir, 'model.pt'))
    model.to(device)
    model.eval()

    ### data
    softmax = nn.Softmax(dim=1)
    preds = []

    for i, (data, _) in enumerate(data_loader):
        data = data.to(device)
        y_final = model(data)
        pred_softmax = softmax(y_final)
        preds.append(pred_softmax.detach().cpu().numpy())

    preds = np.concatenate(preds, 0)
    scores = []
    splits = 10
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


############################################################################################
# main
############################################################################################
def main(args):
    ### config
    dataset = args.dataset
    device = torch.device('cuda')
    if args.pretrain_dir is not None:
        pretrain_dir = args.pretrain_dir
    else:
        pretrain_dir = os.path.join(os.path.dirname(__file__), 'models_IS', args.dataset, 'vgg11')

    ### eval
    if args.if_real:
        data_loader = load_realdata(dataset)
        save_dir = args.save_dir if args.save_dir is not None else pretrain_dir

    else:
        ### load gen data
        gen_data = np.load(args.gen_data)
        x_gen = gen_data['data_x']
        y_gen = torch.from_numpy(gen_data['data_y'])
        x_gen = torch.from_numpy(np.transpose(x_gen, [0, 3, 1, 2]))
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        x_gen = torch.stack([normalize(x) for x in x_gen])

        ### shuffle data
        rand_perm = np.random.permutation(len(x_gen))
        tensor_x = x_gen[rand_perm]
        tensor_y = y_gen[rand_perm]

        testset = data.TensorDataset(tensor_x, tensor_y)  # create your datset
        data_loader = data.DataLoader(testset, batch_size=100, shuffle=False)

        ### set up save dir
        if args.save_dir is not None:
            save_dir = args.save_dir
        else:
            save_dir = os.path.join(os.path.dirname(args.gen_data), 'eval', 'IS')

    ### inception score
    mean, std = get_inception_score(data_loader, pretrain_dir, device=device)
    infostr = 'mnist inception score (model: {}) : mean  {} std {}'.format(pretrain_dir, mean, std)
    print(infostr)

    ### Save results
    mkdir(save_dir)
    logging.basicConfig(filename=os.path.join(save_dir, 'IS.txt'), level=logging.INFO, filemode='a',
                        format='%(message)s')
    logger = logging.getLogger(os.path.join(save_dir, 'IS.txt'))
    logger.info('=' * 30)
    logger.info(infostr)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
