import sys
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
from tqdm import tqdm
import numpy as np
import logging
import time

EPOCHS = 120
BATCH_SIZE = 32
LR = 0.03
OnPlateau_Patience = 5
EarlyStopping_Patience = 15

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
    __package__ = "bboxGen"

from utils import set_logger, RunningAverage, iou_sim
from .datagen import trn_tfms, Dataset
from .bboxModel import bbox_model

set_logger('bboxes', log_path='bboxGen/logs/logs_' +
           time.strftime("%m%d-%H%M%S"))
log = logging.getLogger('bboxes')
log.info('Train Begin')
np.random.seed(42)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('bboxGen/cropping.txt', 'rt') as f:
    df = f.read().split('\n')[:-1]
    df = [line.split(',') for line in df]
    coords = []
    for p, *coord in df:
        coords.append((p, [(int(coord[i]), int(coord[i+1]))
                           for i in range(0, len(coord), 2)]))

trn_fnms, val_fnms = train_test_split(coords, test_size=200, random_state=42)
trn_loader = data.DataLoader(Dataset(fnms=trn_fnms, argumentation=trn_tfms), batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4)
val_loader = data.DataLoader(
    Dataset(fnms=val_fnms, argumentation=None), batch_size=BATCH_SIZE, num_workers=4)


model = bbox_model()
model.to(device)
loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

OnPlateau = 0
EarlyStopping = 0
best_loss = float('inf')
for i in range(EPOCHS):
    log.info('epoch {}'.format(i))
    model.train()
    loss_trn_avg = RunningAverage()
    with tqdm(total=len(trn_loader)) as t:
        for imgs, labels in trn_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss=loss.cpu().item())
            t.update()
            loss_trn_avg.update(loss.cpu().item())

    model.eval()
    loss_val_avg = RunningAverage()
    iou_val_avg = RunningAverage()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss_val_avg.update(loss.cpu().item())
            iou_val_avg.update(
                iou_sim(labels.cpu().detach().numpy(), out.cpu().detach().numpy()))

    log.info('trn:{:05.3f}, val:{:05.3f}, iou_val: {:.3f}'.format(
        loss_trn_avg(), loss_val_avg(), iou_val_avg()))
    if(loss_val_avg() < best_loss):
        torch.save(model.state_dict(),
                   'bboxGen/logs/model-{:02}.pth'.format(i))
        best_loss = loss_val_avg()
        log.info('save model with val loss {:05.3f}'.format(best_loss))
        OnPlateau = 0
        EarlyStopping = 0
    else:
        OnPlateau += 1
        EarlyStopping += 1
        if(OnPlateau == OnPlateau_Patience):
            OnPlateau = 0
            for g in optimizer.param_groups:
                if g['lr'] < 0.0001:
                    break
                g['lr'] = g['lr'] * 0.5
            log.info('reduce lr with {:.5f}'.format(g['lr']))

        if(EarlyStopping == EarlyStopping_Patience):
            log.info('Early stop epoch {:02}'.format(i))
            break
