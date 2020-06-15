from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import myDataset

import torch.backends.cudnn as cudnn
import os
import numpy as np
from tqdm import tqdm
from model import ENet
from PIL import Image
import collections
from shutil import copytree, rmtree
import networks
import shutil

TRAIN_STEP = 100  # used for snapshot, and adjust learning rate
THRESHOLD_MAX = 4

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
parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                    help='input batch size for testing (default: 4)')
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

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
srm_trainable = False  # SRM trainable or not

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

data_root = '/home/wzquan/publicData/NIvsCG/RRVData/RRVNature-Corona'
project_root = '/home/wzquan/Project/NIvsCG/RRVNature-project/RRVNature-Corona/SrcCode'

LOG_DIR = project_root + args.log_dir

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

src_dir = os.path.join(data_root, 'train')
dst_dir = os.path.join(data_root, 'data_centric')
if os.path.exists(dst_dir):
    rmtree(dst_dir)

copytree(src_dir, dst_dir)

pn_data_dir = os.path.join(data_root, 'unpairLinear')

# construct negative examples from CG and add into CG
img_des_dir = os.path.join(dst_dir,'CGG')

image_name = os.listdir(img_des_dir)
all_image_num = len(image_name)
print(all_image_num)

args.dataroot = '/home/wzquan/publicData/NIvsCG/RRVData/natural_validation_dataset'
val_loader = torch.utils.data.DataLoader(
    myDataset.MyDataset(args,
                    transforms.Compose([
                        transforms.TenCrop(233),
                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),  # returns a 4D tensor
                        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                   ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)
print('The number of val data:{}'.format(len(val_loader.dataset)))

def main():
    
    # instantiate model and initialize weights
    model = ENet()
    networks.print_network(model)
    networks.init_weights(model, init_type='normal')
    model.init_convFilter(trainable=srm_trainable)

    if args.cuda:
        model.cuda()

    print('using pretrained model')
    checkpoint = torch.load(project_root + args.log_dir + '/checkpoint_300.pth')
    model.load_state_dict(checkpoint['state_dict'])
    args.lr = args.lr * 0.001
    threshold = THRESHOLD_MAX

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

    nature_error_itr_global = []
    for itr in np.arange(1, 11):
        args.dataroot = dst_dir
        nature_error_itr_local = []

        # adding negative samples into the original training dataset
        construct_negative_samples(itr)

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
        args.epochs = 15

        train_multi(train_loader, optimizer, model, criterion, L1_criterion, val_loader, itr, \
            nature_error_itr_local, nature_error_itr_global)

        # start from itr = 1  
        if len(nature_error_itr_local) > 0:
            adv_model_num, adv_model_idx = adv_model_selection(nature_error_itr_local, threshold, itr)
            if adv_model_num < 1:
                break
        
    print(nature_error_itr_global)
    print(len(nature_error_itr_global)/(args.epochs - args.epochs//2))
    final_model_selection(nature_error_itr_global, threshold)

def train_multi(train_loader, optimizer, model, criterion, L1_criterion, val_loader, itr, \
    nature_error_itr_local, nature_error_itr_global):

    for epoch in range(1, args.epochs+1):
        # update the optimizer learning rate
        adjust_learning_rate(optimizer, epoch, itr)

        train(train_loader, model, optimizer, criterion, L1_criterion, epoch, itr)

        nature = test_nature(val_loader, model)
        
        if epoch > args.epochs//2:
            nature_error_itr_global.append(nature)
            nature_error_itr_local.append(nature)

def train(train_loader, model, optimizer, criterion, L1_criterion, epoch, itr):
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
        
    if epoch > args.epochs//2:
        tmp = (args.epochs - args.epochs//2) * (itr - 1) + (epoch - args.epochs//2)  # the model index
        torch.save({'epoch': tmp,
                    'state_dict': model.state_dict()},
                    '{}/checkpoint_pn_{}.pth'.format(LOG_DIR, tmp))

def test_nature(test_loader, model):
    # switch to evaluate mode
    model.eval()

    oriImageLabel = []  # one dimension list, store the original label of image
    oriTestLabel = []  # one dimension list, store the predicted label of image

    pbar = tqdm(enumerate(test_loader))

    with torch.no_grad():
        for batch_idx, (data, target) in pbar:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            
            # data is a 5d tensor, target is 2d
            bs, ncrops, c, h, w = data.size()
            # compute output
            result = model(data.view(-1, c, h, w)) # fuse batch size and ncrops
            result = F.softmax(result, dim=1)
            output = result.view(bs, ncrops, -1).mean(1) # avg over crops
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            oriTestLabel.extend(pred.squeeze().cpu().numpy())
            oriImageLabel.extend(target.cpu().numpy())

    #  Computing average accuracy
    nature_result = np.array(oriImageLabel) == np.array(oriTestLabel)
    return (len(nature_result) - nature_result.sum())*100.0/len(nature_result)    

# Construct negative samples using linear interpolation
def construct_negative_samples(itr):
    assert(itr > 0)
    print('Adding negative samples ...')

    # add negative sample
    for line in image_name:
        filename, file_extension = os.path.splitext(line)
        new_image_name = filename + '-linear-' + str(itr) +'.png'
        shutil.copy(os.path.join(pn_data_dir, new_image_name), img_des_dir)

def adv_model_selection(nature_error_itr_local, threshold, itr):
    assert(itr > 0)
    nature_np = np.array(nature_error_itr_local)
    boundary = nature_np < threshold
    boundary_idx = np.where(boundary)
    print(boundary_idx[0] + 1)
    adv_model_idx = 0
    if len(boundary_idx[0]) > 0:
        nature_idx = np.argmax(nature_np[boundary_idx[0]])
        model_idx = boundary_idx[0][nature_idx]  # start from 0
        adv_model_idx = (args.epochs - args.epochs//2) * (itr - 1) + model_idx + 1 # the model index
        print('The adv model is checkpoint_pn_{}.pth'.format(adv_model_idx))
    return len(boundary_idx[0]), adv_model_idx

def final_model_selection(nature_error_itr_global, threshold):

    nature_np = np.array(nature_error_itr_global)
    boundary = nature_np < threshold
    boundary_idx = np.where(boundary)
    print(boundary_idx[0] + 1)
    nature_idx = np.argmax(nature_np[boundary_idx[0]])
    model_idx = boundary_idx[0][nature_idx]
    print('The final model is checkpoint_pn_{}.pth'.format(model_idx+1))


def adjust_learning_rate(optimizer, epoch, itr):
    """Sets the learning rate"""
    if itr > 4:
        itr = 4  
    lr = args.lr * (0.1 ** (itr - 1))
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
