from collections import defaultdict
import pickle
import pandas as pd
from tqdm import tqdm

exclude = set()
with open('mainModel/metadata/exclude.txt', 'r') as f:
    for fnm in f:
        exclude.add(fnm.strip())

trn_df = pd.read_csv('rawdata/train.csv')
# at least 2 images in train dataset
trn_kls = trn_df.groupby('Id').size()[trn_df.groupby('Id').size()>1].index.tolist()
trn_kls.remove('new_whale')

trn_kls = set(trn_kls)
trn_kls2fnms = defaultdict(list)
for _, row in tqdm(trn_df.iterrows(), total=trn_df.shape[0]):
    fnm = row['Image']
    theId = row['Id']
    if (theId in trn_kls) and (fnm not in exclude):
        trn_kls2fnms[theId].append(fnm)

delete_fnms = []
for theId, fnms in trn_kls2fnms.items():
    if len(fnms) < 2:
        delete_fnms.append(theId)
# assure at least 2 image per class
print(f'{len(delete_fnms)} image removed')
for fnm in delete_fnms:
    trn_kls2fnms.pop(fnm, None)

trn_fnms = []
trn_kls2idxs = dict()    # idx for score matrix
start = 0
for kls, fnms in trn_kls2fnms.items():
    trn_fnms.extend(fnms)
    idxs = list(range(start, start+len(fnms)))
    trn_kls2idxs[kls] = idxs
    start = start+len(fnms)

with open('mainModel/metadata/trn_kls2fnms.pkl', 'wb') as f:
    pickle.dump(trn_kls2fnms, f)

with open('mainModel/metadata/trn_kls2idxs.pkl', 'wb') as f:
    pickle.dump(trn_kls2idxs, f)

with open('mainModel/metadata/trn_fnms.pkl', 'wb') as f:
    pickle.dump(trn_fnms, f)


