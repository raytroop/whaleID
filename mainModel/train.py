import os
import sys
import time
import functools
import pickle
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
from imgaug import augmenters as iaa
import warnings

warnings.filterwarnings("ignore")

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
    __package__ = "mainModel"

from utils import set_logger, RunningAverage
from .datagen import get_trnDataset
from .resnet import resnet34_wotop
from .head_model import headCONV, headFC

# setting device on GPU if available, else CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
IN_FEATURES = 512

class topModel(nn.Module):
    def __init__(self, in_features=IN_FEATURES, mid=32):
        super().__init__()
        self.branch_model = resnet34_wotop(pretrained=True)
        self.head_model = headCONV(in_features, mid)
    def forward(self, imgsA, imgsB):
        featsA = self.branch_model(imgsA)
        featsB = self.branch_model(imgsB)
        out = self.head_model(featsA, featsB)
        return out


def train_one_epoch(model, datagen, loss_fn, optimizer):
    model.train()
    loss_avg = RunningAverage()
    with tqdm(total=len(datagen)) as t:
        for imgsA, imgsB, labels in datagen:
            imgsA, imgsB, labels = imgsA.to(DEVICE), imgsB.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgsA, imgsB)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

            t.set_postfix(loss=loss.cpu().item())
            t.update()
            loss_avg.update(loss.cpu().item())
    return loss_avg()


def train_model(get_trnDataset, noise, start, epochs, model, optimizer, logger):
    logger.info('train epochs [{:0d} {:3d})'.format(start, start+epochs))
    loss_fn = torch.nn.BCELoss()

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.Flipud(0.5),
        iaa.Affine(
            rotate=(-15, 15)
        ),
        iaa.Multiply((0.5, 1.5)),
        iaa.GaussianBlur((0, 2.0))
    ])
    dataset = get_trnDataset(noise=noise, model=model, transformer=seq)

    for e in range(start, start+epochs):
        logger.info('epoch {:03d}; update train dataset pairs...'.format(e))
        dataset.update_pairs()
        datagen = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        logger.info('start training...')
        loss_avg = train_one_epoch(model, datagen, loss_fn, optimizer)
        logger.info('loss avg: {:03.3f}'.format(loss_avg))
        torch.save(model.branch_model.state_dict(), 'mainModel/checkpoints/branch_model-{:02d}.pth'.format(e))
        torch.save(model.head_model.state_dict(), 'mainModel/checkpoints/head_model-{:02d}.pth'.format(e))
        torch.save(model.state_dict(), 'mainModel/checkpoints/model-{:02d}.pth'.format(e))


if __name__ == '__main__':
    logger = set_logger('mainModel', log_path='mainModel/logs/trn_' + time.strftime("%m%d-%H%M%S"))
    TRN_DIR = 'rawdata/train'
    with open('bboxGen/bbOut/trn_fnms2bboxs.pkl', 'rb') as f:
        fnms2bboxs = pickle.load(f)
    with open('mainModel/metadata/trn_fnms.pkl', 'rb') as f:
        fnms = pickle.load(f)
    with open('mainModel/metadata/trn_kls2idxs.pkl', 'rb') as f:
        kls2idxs = pickle.load(f)
    get_trnDataset = functools.partial(
        get_trnDataset, trn_fnms=fnms, trn_kls2idxs=kls2idxs, directory=TRN_DIR, fnms2bboxes=fnms2bboxs, logger=logger)

    model = topModel()
    model.to(DEVICE)

    learning_rate = 1e-4
    l2_reg = 1e-5
    optimizer = torch.optim.Adam([{'params': model.branch_model.parameters(), 'lr': learning_rate*0.5},
                                  {'params': model.head_model.parameters(), 'lr': learning_rate}], weight_decay=l2_reg)
    train_model(get_trnDataset, noise=100, start=0, epochs=30, model=model, optimizer=optimizer, logger=logger)
