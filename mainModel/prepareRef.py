import pickle
from tqdm import tqdm
import pandas as pd

exclude = set()
with open('mainModel/metadata/exclude.txt', 'r') as f:
    for fnm in f:
        exclude.add(fnm.strip())

trn_df = pd.read_csv('rawdata/train.csv')

# used for prediction and submission
ref_fnms = []
ref_ids = []
for _, row in tqdm(trn_df.iterrows(), total=trn_df.shape[0]):
    fnm = row['Image']
    theId = row['Id']
    if (theId != 'new_whale') and (fnm not in exclude):
        ref_fnms.append(fnm)
        ref_ids.append(theId)

with open('mainModel/metadata/ref_fnms.pkl', 'wb') as f:
    pickle.dump(ref_fnms, f)

with open('mainModel/metadata/ref_ids.pkl', 'wb') as f:
    pickle.dump(ref_ids, f)