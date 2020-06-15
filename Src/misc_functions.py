import copy
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

def pil_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if mode == 'L':
            return img.convert('L')  # convert image to grey
        elif mode == 'RGB':
            return img.convert('RGB')  # convert image to rgb image
        elif mode == 'HSV':
            return img.convert('HSV')

def preprocess_image(img):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    
    # define transform
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    img_trans = transforms.Compose([
                           transforms.RandomCrop(233),
                           transforms.ToTensor(),
                           normalize
                       ])

    img_data = torch.zeros([1, 3, 233, 233])

    img_data[0, ...] = img_trans(img)

    return img_data


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable

    Args:
        im_as_var (torch variable): Image to recreate

    returns:
        recreated_im: PIL image
    """

    image_tensor = (torch.clamp(im_as_var, -1., 1.) + 1.0) / 2.0

    image_numpy = image_tensor[0].detach().cpu().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    return Image.fromarray(np.around(image_numpy).astype(np.uint8), mode='RGB')  
