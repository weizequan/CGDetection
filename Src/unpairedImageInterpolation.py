import sys
from PIL import Image
import numbers
import os
import random
import numpy as np
from shutil import copytree, rmtree

def _is_pil_image(img):
    return isinstance(img, Image.Image)

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

def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        The crop rectangle, as a (left, upper, right, lower)-tuple
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))

def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)

# out = image1 * (1.0 - alpha) + image2 * alpha
def unpaired_image_interpolation(img_a, img_b, alpha):
	th = min(img_a.height, img_b.height)
	tw = min(img_a.width, img_b.width)
	img_a_crop = center_crop(img_a, (th, tw))  # im1
	img_b_crop = center_crop(img_b, (th, tw))  # im2
	img_out = Image.blend(img_a_crop, img_b_crop, alpha)
	return img_out

data_root = '/home/wzquan/publicData/NIvsCG/RRVData/RRVNature-Corona'
src_dir = os.path.join(data_root, 'train/CGG')
ref_dir = os.path.join(data_root, 'train/Real')
dst_dir = os.path.join(data_root, 'unpairLinear')

if os.path.exists(dst_dir):
    rmtree(dst_dir)

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

image_name = os.listdir(src_dir)
print(len(image_name))
NI_image_name = os.listdir(ref_dir)
print(len(NI_image_name))

def main():

    for itr in np.arange(1, 11):
        construct_negative_samples(itr)

def construct_negative_samples(itr):
    print('Constructing negative samples ...', itr)

    NI_index = list(range(len(NI_image_name)))
    random.shuffle(NI_index)
    random.shuffle(NI_index)
    idx_tmp = 0
    for line in image_name:
        ref_image_name = NI_image_name[NI_index[idx_tmp]]
        idx_tmp += 1
        ref_img = pil_loader(os.path.join(ref_dir, ref_image_name))

        img_path = os.path.join(src_dir, line)
        img = pil_loader(img_path)

        if itr < 10:
            negative_sample = unpaired_image_interpolation(img, ref_img, 0.1*itr)  # img: CG; ref_img: NI
        else:
            negative_sample = unpaired_image_interpolation(img, ref_img, 0.99)  # img: CG; ref_img: NI

        filename, file_extension = os.path.splitext(line)
        new_image_name = filename + '-linear-' + str(itr) +'.png' 
        negative_sample.save(os.path.join(dst_dir, new_image_name))

if __name__ == '__main__':
    main()
