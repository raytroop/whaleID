# Ignore warnings
import warnings
import numpy as np

from skimage import io
from skimage import color
from skimage import transform

import torch
from torch.utils import data

import imgaug as ia
from imgaug import augmenters as iaa
warnings.filterwarnings("ignore")


STD = 0.225
AVG = 0.757
LENGTH = 384

def bounding_rectangle(coord, length):
    x0, y0 = coord[0]
    x1, y1 = coord[0]
    for x, y in coord[1:]:
        x0 = min(x, x0)
        y0 = min(y, y0)
        x1 = max(x, x1)
        y1 = max(y, y1)
    return max(x0, 0), max(y0, 0), min(x1, length), min(y1, length)


def readimage_resize(fnm, length):
    img = io.imread(fname=fnm)
    img = color.rgb2gray(img)
    h, w = img.shape[:2]
    s_h = length / float(h)
    s_w = length / float(w)
    img = transform.resize(img, (length, length))
    return img, (s_w, s_h), (w, h)

def trn_tfms(image, keypoints):
    points = [ia.Keypoint(x, y) for x, y in keypoints]
    keypoints = ia.KeypointsOnImage(points, shape=image.shape)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.Flipud(0.5),
        iaa.Affine(
            rotate=(-15, 15)
        ),
        iaa.Multiply((0.5, 1.5)),
        iaa.GaussianBlur((0, 2.0))
    ])
    seq_det = seq.to_deterministic()
    image_aug = seq_det.augment_images([image])[0]
    keypoints_aug = seq_det.augment_keypoints([keypoints])[0]
    coord = [(p.x, p.y) for p in keypoints_aug.keypoints]
    return image_aug, coord


class Dataset(data.Dataset):
    def __init__(self, fnms, argumentation=None):
        self.x = fnms
        self.transfrom = argumentation
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        fnm, coord = self.x[index]
        img, (s_w, s_h), _ = readimage_resize('rawdata/bboxes/'+fnm, LENGTH)
        coord = [(x*s_w, y*s_h) for x, y in coord]
        if self.transfrom is not None:
            img, coord = self.transfrom(img, coord)
        coord = bounding_rectangle(coord, LENGTH)
        img = (img - AVG) / STD
        coord = np.array(coord, dtype=np.float32)
        img = img.astype(np.float32)
        return torch.tensor(img).unsqueeze(0), torch.tensor(coord)
