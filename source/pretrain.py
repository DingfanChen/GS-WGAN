import os, sys
import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

from config import *
from models import *
from utils import *

IMG_DIM = 784
NUM_CLASSES = 10
DATA_ROOT = './../data'


##########################################################
### main
##########################################################
def main(args):
    dataset = args.dataset
    num_discriminators = args.num_discriminators
    z_dim = args.z_dim
    model_dim = args.model_dim
    batchsize = args.batchsize
    L_gp = args.L_gp
    L_epsilon = args.L_epsilon
    critic_iters = args.critic_iters
    latent_type = args.latent_type
    save_dir = args.save_dir
    net_ids = args.net_ids
    gen_arch = args.gen_arch

    ### CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    ### Random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    ### Fix noise for visualization
    if latent_type == 'normal':
        fix_noise = torch.randn(10, z_dim)
    elif latent_type == 'bernoulli':
        p = 0.5
        bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
        fix_noise = bernoulli.sample((10, z_dim)).view(10, z_dim)
    else:
        raise NotImplementedError

    ### Set up models
    netD_list = []
    for i in range(len(net_ids)):
        netD = DiscriminatorDCGAN()
        netD_list.append(netD)
    netD_list = [netD.to(device) for netD in netD_list]

    ### Set up optimizers
    optimizerD_list = []
    for i in range(len(net_ids)):
        netD = netD_list[i]
        optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optimizerD_list.append(optimizerD)

    ### Data loaders
    transform_train = transforms.ToTensor()
    if dataset == 'mnist':
        dataloader = datasets.MNIST
        trainset = dataloader(root=os.path.join(DATA_ROOT, 'MNIST'), train=True, download=True,
                              transform=transform_train)
    elif dataset == 'fashionmnist':
        dataloader = datasets.FashionMNIST
        trainset = dataloader(root=os.path.join(DATA_ROOT, 'FashionMNIST'), train=True, download=True,
                              transform=transform_train)
    else:
        raise NotImplementedError

    if os.path.exists(os.path.join(save_dir, 'indices.npy')):
        print('load indices from disk')
        indices_full = np.load(os.path.join(save_dir, 'indices.npy'), allow_pickle=True)
    else:
        print('creat indices file')
        indices_full = np.arange(len(trainset))
        np.random.shuffle(indices_full)
        indices_full.dump(os.path.join(save_dir, 'indices.npy'))
    trainset_size = int(len(trainset) / num_discriminators)
    print('Size of the dataset: ', trainset_size)

    ### Input pipelines
    input_pipelines = []
    for i in net_ids:
        start = i * trainset_size
        end = (i + 1) * trainset_size
        indices = indices_full[start:end]
        trainloader = data.DataLoader(trainset, batch_size=batchsize, drop_last=False,
                                      num_workers=args.num_workers, sampler=SubsetRandomSampler(indices))
        input_data = inf_train_gen(trainloader)
        input_pipelines.append(input_data)

    ### Training Loop
    for idx, netD_id in enumerate(net_ids):

        ### stop the process if finished
        if netD_id >= num_discriminators:
            print('ID {} exceeds the num of discriminators'.format(netD_id))
            sys.exit()

        ### Discriminator
        netD = netD_list[idx]
        optimizerD = optimizerD_list[idx]
        input_data = input_pipelines[idx]

        ### Train (non-private) Generator for each Discriminator
        if gen_arch == 'DCGAN':
            netG = GeneratorDCGAN(z_dim=z_dim, model_dim=model_dim, num_classes=10).to(device)
        elif gen_arch == 'ResNet':
            netG = GeneratorResNet(z_dim=z_dim, model_dim=model_dim, num_classes=10).to(device)
        optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

        ### Save dir for each discriminator
        save_subdir = os.path.join(save_dir, 'netD_%d' % netD_id)

        if os.path.exists(save_subdir):
            print("netD %d already pre-trained" % netD_id)
        else:
            mkdir(save_subdir)

            for iter in range(args.pretrain_iterations + 1):
                #########################
                ### Update D network
                #########################
                for p in netD.parameters():
                    p.requires_grad = True

                for iter_d in range(critic_iters):
                    real_data, real_y = next(input_data)
                    real_data = real_data.view(-1, IMG_DIM)
                    real_data = real_data.to(device)
                    real_y = real_y.to(device)
                    real_data_v = autograd.Variable(real_data)

                    ### train with real
                    netD.zero_grad()
                    D_real_score = netD(real_data_v, real_y)
                    D_real = -D_real_score.mean()

                    ### train with fake
                    batchsize = real_data.shape[0]
                    if latent_type == 'normal':
                        noise = torch.randn(batchsize, z_dim).to(device)
                    elif latent_type == 'bernoulli':
                        noise = bernoulli.sample((batchsize, z_dim)).view(batchsize, z_dim).to(device)
                    else:
                        raise NotImplementedError
                    noisev = autograd.Variable(noise)
                    fake = autograd.Variable(netG(noisev, real_y).data)
                    inputv = fake
                    D_fake = netD(inputv, real_y)
                    D_fake = D_fake.mean()

                    ### train with gradient penalty
                    gradient_penalty = netD.calc_gradient_penalty(real_data_v.data, fake.data, real_y, L_gp, device)
                    D_cost = D_fake + D_real + gradient_penalty

                    ### train with epsilon penalty
                    logit_cost = L_epsilon * torch.pow(D_real_score, 2).mean()
                    D_cost += logit_cost

                    ### update
                    D_cost.backward()
                    Wasserstein_D = -D_real - D_fake
                    optimizerD.step()

                ############################
                # Update G network
                ###########################
                for p in netD.parameters():
                    p.requires_grad = False
                netG.zero_grad()

                if latent_type == 'normal':
                    noise = torch.randn(batchsize, z_dim).to(device)
                elif latent_type == 'bernoulli':
                    noise = bernoulli.sample((batchsize, z_dim)).view(batchsize, z_dim).to(device)
                else:
                    raise NotImplementedError
                label = torch.randint(0, NUM_CLASSES, [batchsize]).to(device)
                noisev = autograd.Variable(noise)
                fake = netG(noisev, label)
                G = netD(fake, label)
                G = - G.mean()

                ### update
                G.backward()
                G_cost = G
                optimizerG.step()

                ############################
                ### Results visualization
                ############################
                if iter < 5 or iter % args.print_step == 0:
                    print('G_cost:{}, D_cost:{}, Wasserstein:{}'.format(G_cost.cpu().data,
                                                                        D_cost.cpu().data,
                                                                        Wasserstein_D.cpu().data
                                                                        ))
                if iter == args.pretrain_iterations:
                    generate_image(iter, netG, fix_noise, save_subdir, device, num_classes=10)

            torch.save(netD.state_dict(), os.path.join(save_subdir, 'netD.pth'))


if __name__ == '__main__':
    args = parse_arguments()
    save_config(args)
    main(args)
