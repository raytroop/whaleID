import random
import os
import copy
import numpy as np
import torch
from torch.utils import data
from skimage import io, transform
from lapjv import lapjv # pylint: disable=no-name-in-module

EXPANSION_RATIO = 0.1
LENGTH = 384
IN_FEATURES = 512
# setting device on GPU if available, else CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def readImage(fnmPs, bbox):
    img = io.imread(fnmPs)
    x0, y0, x1, y1 = bbox
    Height, Width = img.shape[:2]
    h = y1 - y0
    w = x1 - x0
    dx = int(w*EXPANSION_RATIO / 2)
    dy = int(h*EXPANSION_RATIO / 2)
    x0, x1 = x0 - dx, x1 + dx
    y0, y1 = y0 - dy, y1 + dy
    x0 = max(x0, 0)
    x1 = min(x1, Width)
    y0 = max(y0, 0)
    y1 = min(y1, Height)
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)
    img_crop = img[y0:y1, x0:x1, :]
    return transform.resize(img_crop, (LENGTH, LENGTH))


class imgsDataset(data.Dataset):
    """only used in select pair for training
    """
    def __init__(self, fnms, directory, bboxes, transformer=None):
        self.fnms = fnms
        self.directory = directory
        self.bboxes = bboxes
        self.tfms = transformer

    def __len__(self):
        return len(self.fnms)

    def __getitem__(self, index):
        fnm = self.fnms[index]
        fnmPs = os.path.join(self.directory, fnm)
        img = readImage(fnmPs, self.bboxes[fnm])  # (h, w, 3)
        if self.tfms is not None:
            img = self.tfms.augment_images(img)
        avg_ = np.mean(img, axis=(0, 1), keepdims=True)
        std_ = np.std(img, axis=(0, 1), keepdims=True)
        out = ((img - avg_) / std_).astype(np.float32)
        return torch.FloatTensor(np.transpose(out, (2, 0, 1)))   # channel first


class featureDataset(data.Dataset):
    def __init__(self, x, y=None):
        """
        Args:
            x: features
            y: features
        """
        self.x = x
        self.y = y
        if y is None:
            self.y = self.x
            self.ix, self.iy = np.triu_indices(x.shape[0], 1)
        else:
            self.ix, self.iy = np.indices((x.shape[0], y.shape[0]))
            self.ix = self.ix.flatten()
            self.iy = self.iy.flatten()

    def __getitem__(self, index):
        return torch.FloatTensor(self.x[self.ix[index]]), torch.FloatTensor(self.y[self.iy[index]])

    def __len__(self):
        return len(self.ix)


def score_reshape(score, x, y=None):
    """
    Tranformed the packed matrix 'score' into a square matrix.
        @param score the packed matrix
        @param x the first image feature tensor
        @param y the second image feature tensor if different from x
        @result the square matrix
    """
    if y is None:
        # When y is None, score is a packed upper triangular matrix.
        # Unpack, and transpose to form the symmetrical lower triangular matrix.
        m = np.zeros((x.shape[0], x.shape[0]), dtype=np.float32)
        m[np.triu_indices(x.shape[0], 1)] = score.flatten()
        m += m.transpose()
    else:
        m = np.zeros((x.shape[0], y.shape[0]), dtype=np.float32)
        ix, iy = np.indices((x.shape[0], y.shape[0]))
        ix = ix.flatten()
        iy = iy.flatten()
        m[ix, iy] = score.flatten()
    return m


# imgsGen_trn = data.DataLoader(imgsDataset(*imgsDatasetArgs), batch_size=batch_size, shuffle=False, num_workers=4)
def compute_score(branch_model, head_model, imgsXGen, imgsYGen=None, batch_size=32):
    """
    Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
    """
    branch_model.eval()
    head_model.eval()
    featuresX = []
    print('generating features with branch_model...')
    with torch.no_grad():
        for imgs in imgsXGen:
            imgs = imgs.to(DEVICE)
            featuresX.append(branch_model(imgs).cpu().numpy())
    featuresX = np.concatenate(featuresX, axis=0)
    assert featuresX.shape[1] == IN_FEATURES

    featuresY = []
    if imgsYGen is not None:
        with torch.no_grad():
            for imgs in imgsYGen:
                imgs = imgs.to(DEVICE)
                featuresY.append(branch_model(imgs).cpu().numpy())
    featuresY = np.concatenate(featuresY, axis=0) if featuresY else None

    featureGen = data.DataLoader(featureDataset(
        featuresX, featuresY), batch_size=batch_size, shuffle=False, num_workers=4)
    score = []
    print('generating scores with head_model ...')
    with torch.no_grad():
        for featsA, featsB in featureGen:
            featsA = featsA.to(DEVICE)
            featsB = featsB.to(DEVICE)
            score.append(head_model(featsA, featsB).detach().cpu().numpy())
    score = np.concatenate(score, axis=0)
    return score_reshape(score, featuresX, featuresY)


class trnDataset(data.Dataset):
    def __init__(self, trn_fnms, kls2idxs, score, directory, fnms2bboxes, transformer=None):
        self.fnms = trn_fnms
        self.tfms = transformer
        self.directory = directory
        self.fnms2bboxes = fnms2bboxes
        self.kls2idxs = kls2idxs
        self.score = -score # Maximizing the score is the same as minimuzing -score.
        for idxs in kls2idxs.values():
            for i in idxs:
                for j in idxs:
                    # Set a large value for matching whales -- eliminates this potential pairing
                    self.score[i, j] = float('inf')

    def update_pairs(self):
        match = []
        for idxs_ in self.kls2idxs.values():
            idxs = copy.copy(idxs_)
            while True:
                random.shuffle(idxs)
                if all(i != j for i, j in zip(idxs, idxs_)):
                    match.extend([(i, j) for i, j in zip(idxs, idxs_)])
                    break
        assert len(match) == len(self.fnms)

        print(f'score shape: {self.score.shape}')
        diff = []
        row_ind, col_ind, _ = lapjv(self.score)  # Solve the linear assignment problem
        badpair = 0
        for i, j in zip(row_ind, col_ind):
            if (i == j) or (self.score[i, j] == float('inf')):  # avoid same image or same class images misclassified as diff
                badpair += 1
                print(f'the {badpair}th BAD PAIR ({i}, {j}) with score {self.score[i, j]}')
                avoidIdxs = None
                for _, idxs in self.kls2idxs.items():
                    if i in idxs:
                       avoidIdxs = idxs
                       break
                while j in avoidIdxs:
                    j = random.randint(0, self.score.shape[0]-1)
                print(f'        Replaced with ({i}, {j}) with score {self.score[i, j]}')

            diff.append((i, j))
            self.score[i, j] = 1e6
        assert len(diff) == len(self.fnms)

        idxPairs = []
        for idxsA, idxsB in zip(match, diff):
            idxPairs.append(idxsA)
            idxPairs.append(idxsB)  # match, diff, match, diff ...

        # shuffle data
        ys = [1, 0] * len(self.fnms)
        shuffle_idxs = list(range(len(self.fnms) * 2))
        random.shuffle(shuffle_idxs)
        self.idxPairs = [idxPairs[ix] for ix in shuffle_idxs]
        self.ys = [ys[ix] for ix in shuffle_idxs]

    def __len__(self):
        return len(self.idxPairs)

    def __getitem__(self, index):
        idx1, idx2 = self.idxPairs[index]
        fnm1 = self.fnms[idx1]
        fnm2 = self.fnms[idx2]
        img1 = readImage(os.path.join(self.directory, fnm1), bbox=self.fnms2bboxes[fnm1])
        img2 = readImage(os.path.join(self.directory, fnm2), bbox=self.fnms2bboxes[fnm2])
        if self.tfms is not None:
            img1, img2 = self.tfms.augment_images(img1), self.tfms.augment_images(img2)

        avg1 = np.mean(img1, axis=(0, 1), keepdims=True)
        std1 = np.std(img1, axis=(0, 1), keepdims=True)
        img1 = (img1 - avg1) / std1

        avg2 = np.mean(img1, axis=(0, 1), keepdims=True)
        std2 = np.std(img1, axis=(0, 1), keepdims=True)
        img2 = (img2 - avg2) / std2

        img1 = np.transpose(img1, (2, 0, 1))    # channel first
        img2 = np.transpose(img2, (2, 0, 1))

        y = self.ys[index]
        return torch.FloatTensor(img1), torch.FloatTensor(img2), torch.FloatTensor([y])


def get_trnDataset(trn_fnms, directory, fnms2bboxes, trn_kls2idxs, noise, model, batch_size=384, transformer=None, logger=None):
    if logger is not None:
        logger.info('preparing imgages dataset to compute score...')
    imgsXGen = data.DataLoader(imgsDataset(trn_fnms, directory, fnms2bboxes), batch_size=batch_size, shuffle=False, num_workers=4)
    if logger is not None:
        logger.info('computing score...')
    score = compute_score(model.branch_model, model.head_model, imgsXGen, batch_size=batch_size*8)
    score = score + noise*np.random.random_sample(size=score.shape)
    if logger is not None:
        logger.info("preparing train dataset with 'lapjv' 'and random.shuffle'...")
    dataset = trnDataset(trn_fnms, trn_kls2idxs, score, directory, fnms2bboxes, transformer=transformer)
    if logger is not None:
        logger.info("training dataset is Ready ^_^")
    return dataset
