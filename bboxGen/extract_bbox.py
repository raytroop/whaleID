import sys
import os
import numpy as np
import gc
import pickle
from tqdm import tqdm

import torch

# np.random.seed(42)
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
    __package__ = "bboxGen"
from .datagen import readimage_resize, bounding_rectangle
from .bboxModel import bbox_model
from utils import draw_bbox

STD = 0.225
AVG = 0.757
LENGTH = 384

def predict(model, imgnm):
    img, (s_w, s_h), (width, height) = readimage_resize(imgnm, LENGTH)
    img = np.expand_dims(np.expand_dims((img - AVG) / STD, 0), 0)
    img = torch.from_numpy(img.astype(np.float32))
    img = img.to(device)
    bbox = model(img)
    bbox = bbox.cpu().numpy()[0]
    scales = np.array([s_w, s_h]*2)
    bbox = bbox / scales
    bbox = bbox.astype(np.int32).tolist()
    bbox = [max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], width), min(bbox[3], height)]

    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        bbox = [0, 0, width, height]
    return bbox


def test_predict():
    from sklearn.model_selection import train_test_split
    import skimage.io as io

    with open('bboxGen/cropping.txt', 'rt') as f:
        df = f.read().split('\n')[:-1]
        df = [line.split(',') for line in df]
        coords = []
        for p, *coord in df:
            coords.append((p, [(int(coord[i]), int(coord[i+1]))
                            for i in range(0, len(coord), 2)]))

    trn_fnms, val_fnms = train_test_split(coords, test_size=200, random_state=42)

    model = bbox_model()
    model.load_state_dict(torch.load('bboxGen/logs/model-74.pth'))
    model.to(device)
    model.eval()
    imgnm, gtbox = val_fnms[np.random.randint(0, high=199)]
    gtbox = bounding_rectangle(gtbox, 9999999)
    with torch.no_grad():
        pred = predict(model, 'rawdata/bboxes/'+imgnm)
    image = io.imread('rawdata/bboxes/'+imgnm)
    draw_bbox(image, pred, gtbox)
    print(gtbox)
    print(pred)

    fnms = os.listdir('rawdata/train')
    trn_img = os.path.join('rawdata/train', fnms[np.random.randint(0, len(fnms))])
    with torch.no_grad():
        pred = predict(model, trn_img)
    image = io.imread(trn_img)
    draw_bbox(image, pred)

def extract_all():
    model = bbox_model()
    model.load_state_dict(torch.load('bboxGen/logs/model-74.pth'))
    model.to(device)
    model.eval()

    trn_bboxes = {}
    trn_fnms = os.listdir('rawdata/train')
    for fnm in tqdm(trn_fnms):
        img = os.path.join('rawdata/train', fnm)
        with torch.no_grad():
            pred = predict(model, img)
        trn_bboxes[fnm] = pred
    with open('bboxGen/bbOut/trn_fnms2bboxs.pkl', 'wb') as f:
        pickle.dump(trn_bboxes, f)
    del trn_bboxes
    gc.collect()

    test_bboxes = {}
    test_fnms = os.listdir('rawdata/test')
    for fnm in tqdm(test_fnms):
        img = os.path.join('rawdata/test', fnm)
        with torch.no_grad():
            pred = predict(model, img)
        test_bboxes[fnm] = pred
    with open('bboxGen/bbOut/test_fnms2bboxs.pkl', 'wb') as f:
        pickle.dump(test_bboxes, f)
    del test_bboxes
    gc.collect()

if __name__ == '__main__':
    # test_predict()
    extract_all()
