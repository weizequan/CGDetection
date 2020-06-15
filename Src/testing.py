import numpy as np
from PIL import Image
import sys, os
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from model import ENet
import argparse
import myDataset
from tqdm import tqdm
import torch.nn.functional as F

# Testing settings
parser = argparse.ArgumentParser(description='PyTorch NI vs CG')
parser.add_argument('--dataroot', type=str, 
                    help='path to images')
parser.add_argument('--input_nc', type=int, default=3, 
                    help='# of input image channels')
parser.add_argument('--img_mode', type=str, default='RGB', 
                    help='chooses how image are loaded')
parser.add_argument('--test-batch-size', type=int, default=5, metavar='N',
                    help='input batch size for testing (default: 5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

# data_root = '/home/wzquan/publicData/NIvsCG/RRVData/RRVNature-Artlantis/'
# data_root = '/home/wzquan/publicData/NIvsCG/RRVData/RRVNature-Autodesk/'
data_root = '/home/wzquan/publicData/NIvsCG/RRVData/RRVNature-Corona/'
# data_root = '/home/wzquan/publicData/NIvsCG/RRVData/RRVNature-VRay/'

project_root = '/home/wzquan/Project/NIvsCG/RRVNature-project/RRVNature-Corona/SrcCode'

kCgNum = 360

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

args.dataroot = data_root + 'test'
test_loader = torch.utils.data.DataLoader(
    myDataset.MyDataset(args,
                    transforms.Compose([
                        transforms.TenCrop(233),
                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),  # returns a 4D tensor
                        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                   ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)
print('The number of test data:{}'.format(len(test_loader.dataset)))

# instantiate model and initialize weights
model = ENet()
model.cuda()

checkpoint = torch.load(project_root + '/logs/checkpoint_300.pth')
model.load_state_dict(checkpoint['state_dict'])

# switch to evaluate mode
model.eval()

oriImageLabel = []  # one dimension list, store the original label of image
oriTestLabel = []  # one dimension list, store the predicted label of image

pbar = tqdm(enumerate(test_loader))

with torch.no_grad():
    for batch_idx, (data, target) in pbar:
        data, target = data.cuda(), target.cuda()
        
        # data is a 5d tensor, target is 2d
        bs, ncrops, c, h, w = data.size()
        # compute output
        result = model(data.view(-1, c, h, w)) # fuse batch size and ncrops
        result = F.softmax(result, dim=1)
        output = result.view(bs, ncrops, -1).mean(1) # avg over crops
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        oriTestLabel.extend(pred.squeeze().cpu().numpy())
        oriImageLabel.extend(target.data.cpu().numpy())

result = np.array(oriImageLabel) == np.array(oriTestLabel)
cg_result = result[:kCgNum]
ni_result = result[kCgNum:]
print('HTER: ', ((len(cg_result) - cg_result.sum())*100.0/len(cg_result) + (len(ni_result) - ni_result.sum())*100.0/len(ni_result))/2)
