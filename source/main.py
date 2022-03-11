import os, sys
import numpy as np
import random
import copy
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

from ops import exp_mov_avg

IMG_DIM = 784
NUM_CLASSES = 10
CLIP_BOUND = 1.
SENSITIVITY = 2.
DATA_ROOT = './../data'


##########################################################
### hook functions
##########################################################
def master_hook_adder(module, grad_input, grad_output):
    '''
    global hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    global dynamic_hook_function
    return dynamic_hook_function(module, grad_input, grad_output)


def dummy_hook(module, grad_input, grad_output):
    '''
    dummy hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    pass


def modify_gradnorm_conv_hook(module, grad_input, grad_output):
    '''
    gradient modification hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    ### get grad wrt. input (image)
    grad_wrt_image = grad_input[0]
    grad_input_shape = grad_wrt_image.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize  # account for the 'sum' operation in GP

    grad_wrt_image = grad_wrt_image.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

    ### clip
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_image = clip_coef * grad_wrt_image
    grad_input_new = [grad_wrt_image.view(grad_input_shape)]
    for i in range(len(grad_input) - 1):
        grad_input_new.append(grad_input[i + 1])
    return tuple(grad_input_new)


def dp_conv_hook(module, grad_input, grad_output):
    '''
    gradient modification + noise hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    global noise_multiplier
    ### get grad wrt. input (image)
    grad_wrt_image = grad_input[0]
    grad_input_shape = grad_wrt_image.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize

    grad_wrt_image = grad_wrt_image.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

    ### clip
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
    clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_image = clip_coef * grad_wrt_image

    ### add noise
    noise = clip_bound_ * noise_multiplier * SENSITIVITY * torch.randn_like(grad_wrt_image)
    grad_wrt_image = grad_wrt_image + noise
    grad_input_new = [grad_wrt_image.view(grad_input_shape)]
    for i in range(len(grad_input) - 1):
        grad_input_new.append(grad_input[i + 1])
    return tuple(grad_input_new)


##########################################################
### main
##########################################################
def main(args):
    ### config
    global noise_multiplier
    dataset = args.dataset
    num_discriminators = args.num_discriminators
    noise_multiplier = args.noise_multiplier
    z_dim = args.z_dim
    model_dim = args.model_dim
    batchsize = args.batchsize
    L_gp = args.L_gp
    L_epsilon = args.L_epsilon
    critic_iters = args.critic_iters
    latent_type = args.latent_type
    load_dir = args.load_dir
    save_dir = args.save_dir
    if_dp = (noise_multiplier > 0.)
    gen_arch = args.gen_arch
    num_gpus = args.num_gpus

    ### CUDA
    use_cuda = torch.cuda.is_available()
    devices = [torch.device("cuda:%d" % i if use_cuda else "cpu") for i in range(num_gpus)]
    device0 = devices[0]
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
    if gen_arch == 'DCGAN':
        netG = GeneratorDCGAN(z_dim=z_dim, model_dim=model_dim, num_classes=10)
    elif gen_arch == 'ResNet':
        netG = GeneratorResNet(z_dim=z_dim, model_dim=model_dim, num_classes=10)
    else:
        raise ValueError

    netGS = copy.deepcopy(netG)
    netD_list = []
    for i in range(num_discriminators):
        netD = DiscriminatorDCGAN()
        netD_list.append(netD)

    ### Load pre-trained discriminators
    if load_dir is not None:
        for netD_id in range(num_discriminators):
            print('Load NetD ', str(netD_id))
            network_path = os.path.join(load_dir, 'netD_%d' % netD_id, 'netD.pth')
            netD = netD_list[netD_id]
            netD.load_state_dict(torch.load(network_path))

    netG = netG.to(device0)
    for netD_id, netD in enumerate(netD_list):
        device = devices[get_device_id(netD_id, num_discriminators, num_gpus)]
        netD.to(device)

    ### Set up optimizers
    optimizerD_list = []
    for i in range(num_discriminators):
        netD = netD_list[i]
        optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optimizerD_list.append(optimizerD)
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

    ### Data loaders
    transform_train = transforms.ToTensor()
    if dataset == 'mnist':
        dataloader = datasets.MNIST
        trainset = dataloader(root=DATA_ROOT, train=True, download=True,
                              transform=transform_train)
    elif dataset == 'fashionmnist':
        dataloader = datasets.FashionMNIST
        trainset = dataloader(root=DATA_ROOT, train=True, download=True,
                              transform=transform_train)
    else:
        raise NotImplementedError

    if load_dir is not None:
        assert os.path.exists(os.path.join(load_dir, 'indices.npy'))
        print('load indices from disk')
        indices_full = np.load(os.path.join(load_dir, 'indices.npy'), allow_pickle=True)
    else:
        print('creat indices file')
        indices_full = np.arange(len(trainset))
        np.random.shuffle(indices_full)
        indices_full.dump(os.path.join(save_dir, 'indices.npy'))
    trainset_size = int(len(trainset) / num_discriminators)
    print('Size of the dataset: ', trainset_size)

    input_pipelines = []
    for i in range(num_discriminators):
        start = i * trainset_size
        end = (i + 1) * trainset_size
        indices = indices_full[start:end]
        trainloader = data.DataLoader(trainset, batch_size=args.batchsize, drop_last=False,
                                      num_workers=args.num_workers, sampler=SubsetRandomSampler(indices))
        input_data = inf_train_gen(trainloader)
        input_pipelines.append(input_data)

    ### Register hook
    global dynamic_hook_function
    for netD in netD_list:
        netD.conv1.register_backward_hook(master_hook_adder)

    for iter in range(args.iterations + 1):
        #########################
        ### Update D network
        #########################
        netD_id = np.random.randint(num_discriminators, size=1)[0]
        device = devices[get_device_id(netD_id, num_discriminators, num_gpus)]
        netD = netD_list[netD_id]
        optimizerD = optimizerD_list[netD_id]
        input_data = input_pipelines[netD_id]

        for p in netD.parameters():
            p.requires_grad = True

        for iter_d in range(critic_iters):
            real_data, real_y = next(input_data)
            real_data = real_data.view(-1, IMG_DIM)
            real_data = real_data.to(device)
            real_y = real_y.to(device)
            real_data_v = autograd.Variable(real_data)

            ### train with real
            dynamic_hook_function = dummy_hook
            netD.zero_grad()
            D_real_score = netD(real_data_v, real_y)
            D_real = -D_real_score.mean()

            ### train with fake
            batchsize = real_data.shape[0]
            if latent_type == 'normal':
                noise = torch.randn(batchsize, z_dim).to(device0)
            elif latent_type == 'bernoulli':
                noise = bernoulli.sample((batchsize, z_dim)).view(batchsize, z_dim).to(device0)
            else:
                raise NotImplementedError
            noisev = autograd.Variable(noise)
            fake = autograd.Variable(netG(noisev, real_y.to(device0)).data)
            inputv = fake.to(device)
            D_fake = netD(inputv, real_y.to(device))
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

        del real_data, real_y, fake, noise, inputv, D_real, D_fake, logit_cost, gradient_penalty
        torch.cuda.empty_cache()

        ############################
        # Update G network
        ###########################
        if if_dp:
            ### Sanitize the gradients passed to the Generator
            dynamic_hook_function = dp_conv_hook
        else:
            ### Only modify the gradient norm, without adding noise
            dynamic_hook_function = modify_gradnorm_conv_hook

        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()

        ### train with sanitized discriminator output
        if latent_type == 'normal':
            noise = torch.randn(batchsize, z_dim).to(device0)
        elif latent_type == 'bernoulli':
            noise = bernoulli.sample((batchsize, z_dim)).view(batchsize, z_dim).to(device0)
        else:
            raise NotImplementedError
        label = torch.randint(0, NUM_CLASSES, [batchsize]).to(device0)
        noisev = autograd.Variable(noise)
        fake = netG(noisev, label)
        fake = fake.to(device)
        label = label.to(device)
        G = netD(fake, label)
        G = - G.mean()

        ### update
        G.backward()
        G_cost = G
        optimizerG.step()

        ### update the exponential moving average
        exp_mov_avg(netGS, netG, alpha=0.999, global_step=iter)

        ############################
        ### Results visualization
        ############################
        if iter < 5 or iter % args.print_step == 0:
            print('G_cost:{}, D_cost:{}, Wasserstein:{}'.format(G_cost.cpu().data,
                                                                D_cost.cpu().data,
                                                                Wasserstein_D.cpu().data
                                                                ))

        if iter % args.vis_step == 0:
            generate_image(iter, netGS, fix_noise, save_dir, device0, num_classes=10)

        if iter % args.save_step == 0:
            ### save model
            torch.save(netGS.state_dict(), os.path.join(save_dir, 'netGS_%d.pth' % iter))

        del label, fake, noisev, noise, G, G_cost, D_cost
        torch.cuda.empty_cache()

    ### save model
    torch.save(netG.state_dict(), os.path.join(save_dir, 'netG.pth'))
    torch.save(netGS.state_dict(), os.path.join(save_dir, 'netGS.pth'))

    ### save generate samples
    save_gen_data(os.path.join(save_dir, 'gen_data.npz'), netGS, z_dim, device0, latent_type=latent_type)


if __name__ == '__main__':
    args = parse_arguments()
    save_config(args)
    main(args)
