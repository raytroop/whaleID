from collections import defaultdict
import os
import sys
import time
import pickle
import torch
from torch.utils import data
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# setting device on GPU if available, else CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 384
# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
    __package__ = "mainModel"

from .train import topModel
from .datagen import imgsDataset, compute_score


def sub_score():
    with open('mainModel/metadata/ref_fnms.pkl', 'rb') as f:
        ref_fnms = pickle.load(f)
    with open('bboxGen/bbOut/trn_fnms2bboxs.pkl', 'rb') as f:
        ref_fnms2bboxs = pickle.load(f)

    with open('bboxGen/bbOut/test_fnms2bboxs.pkl', 'rb') as f:
        test_fnms2bboxs = pickle.load(f)

    test_df = pd.read_csv('rawdata/sample_submission.csv')
    test_fnms = test_df.Image.tolist()

    test_dir = 'rawdata/test'
    ref_dir = 'rawdata/train'
    imgsGen_test = data.DataLoader(imgsDataset(test_fnms, test_dir, test_fnms2bboxs), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    imgsGen_ref = data.DataLoader(imgsDataset(ref_fnms, ref_dir, ref_fnms2bboxs), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = topModel()
    model.load_state_dict(torch.load('mainModel/checkpoints/model-29.pth'))
    model.to(DEVICE)

    score = compute_score(model.branch_model, model.head_model, imgsGen_test, imgsGen_ref, batch_size=BATCH_SIZE*8)
    assert score.shape[0] == len(test_fnms)
    assert score.shape[1] == len(ref_fnms)
    np.save('mainModel/metadata/sub_score', score)


def submission(threshold):
    with open('mainModel/metadata/ref_ids.pkl', 'rb') as f:
        ref_ids = pickle.load(f)
    ref_ids += ['new_whale']    # append 'new_whale'

    test_df = pd.read_csv('rawdata/sample_submission.csv')
    test_fnms = test_df.Image.tolist()

    score = np.load('mainModel/metadata/sub_score.npy')
    score = np.concatenate((score, np.full(shape=(score.shape[0], 1), fill_value = threshold, dtype=np.float32)), axis=1)

    test_ids = []
    for s in tqdm(score):
        rank = []
        idxs = reversed(np.argsort(s))
        viewed = set()
        num = 0
        for ix in idxs:
            if ref_ids[ix] not in viewed:
                viewed.add(ref_ids[ix])
                rank.append(ref_ids[ix])
                num += 1
            if num == 5:
                rank = ' '.join(rank)
                test_ids.append(rank)
                break
    assert len(test_fnms) == len(test_ids)

    sub = pd.DataFrame()
    sub['Image'] = test_fnms
    sub['Id'] = test_ids
    sub.to_csv('mainModel/submissions/sub-' + time.strftime("%m%d-%H%M%S" + '.csv'), index=False)


if __name__ == '__main__':
    # sub_score()
    submission(threshold=0.99)
