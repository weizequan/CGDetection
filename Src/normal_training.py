from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import myDataset
import torch.backends.cudnn as cudnn
import os
import numpy as np
from tqdm import tqdm
from model import ENet
import networks

TRAIN_STEP = 100  # used for snapshot, and adjust learning rate

# Training settings
parser = argparse.ArgumentParser(description='PyTorch NI vs CG')
parser.add_argument('--dataroot', type=str, 
                    help='path to images')
parser.add_argument('--input_nc', type=int, default=3, 
                    help='# of input image channels')
parser.add_argument('--img_mode', type=str, default='RGB', 
                    help='chooses how image are loaded')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--log-dir', default='/logs',
                    help='folder to output model checkpoints')
parser.add_argument('--epochs', type=int, default=TRAIN_STEP*3, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: sgd)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
srm_trainable = False  # SRM trainable or not

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

data_root = '/home/wzquan/publicData/NIvsCG/RRVData/RRVNature-Corona'
project_root = '/home/wzquan/Project/NIvsCG/RRVNature-project/RRVNature-Corona/SrcCode'

LOG_DIR = project_root + args.log_dir
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

args.dataroot = os.path.join(data_root, 'train')
train_loader = myDataset.DataLoaderHalf(
    myDataset.MyDataset(args,
                   transforms.Compose([
                       transforms.RandomCrop(233),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       normalize
                   ])),
    batch_size=args.batch_size, shuffle=True, half_constraint=True, sampler_type='RandomBalancedSampler', **kwargs)
print('The number of train data:{}'.format(len(train_loader.dataset)))

def main():
    # instantiate model and initialize weights
    model = ENet()
    networks.print_network(model)
    networks.init_weights(model, init_type='normal')
    model.init_convFilter(trainable=srm_trainable)

    if args.cuda:
        model.cuda()

    torch.save({'epoch': 0,
            'state_dict': model.state_dict()},
            '{}/checkpoint_{}.pth'.format(LOG_DIR, 0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    L1_criterion = nn.L1Loss(reduction='sum').cuda()
    
    if not srm_trainable:
        params = []
        for name, param in model.named_parameters():
            if name.find('convFilter1') == -1:
                params += [param]

        optimizer = create_optimizer(params, args.lr)
    else:
        optimizer = create_optimizer(model.parameters(), args.lr)

    for epoch in range(1, args.epochs+1):
        # update the optimizer learning rate
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, optimizer, criterion, L1_criterion, epoch)

def train(train_loader, model, optimizer, criterion, L1_criterion, epoch):
    # switch to train mode
    model.train()

    pbar = tqdm(enumerate(train_loader))

    for batch_idx, (data, target) in pbar:
        if args.cuda:
            data_var, target_var = data.cuda(), target.cuda()

        # compute output
        prediction = model(data_var)
        L1_loss = 0
        for name, param in model.named_parameters():
            # L1 don't add bias
            if not srm_trainable and name.find('convFilter1') != -1:
            	continue
            if name.find('bias') == -1:
                L1_loss += L1_criterion(param, torch.zeros_like(param))

        loss = criterion(prediction, target_var) + args.wd * L1_loss
            
        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))

    if epoch % TRAIN_STEP == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                    '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    lr = args.lr * (0.1 ** ((epoch - 1) // TRAIN_STEP))
    print('lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def create_optimizer(params, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=new_lr,
                              momentum=0.9,
                              weight_decay=0)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=new_lr,
                               weight_decay=args.wd, betas=(args.beta1, 0.999))
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(params,
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer

if __name__ == '__main__':
    main()