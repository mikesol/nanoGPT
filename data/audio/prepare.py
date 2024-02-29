"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
from scipy.io import wavfile
import numpy as np
import sys

import os

data_path = '../data/unsilenced-aligned/'

D1 = [
    os.path.join(data_path, "day1", x)
    for x in os.listdir(os.path.join(data_path, "day1"))
]
D2 = [
    os.path.join(data_path, "day2", x)
    for x in os.listdir(os.path.join(data_path, "day2"))
]


def make_pairings(i):
    i = [x for x in i if "67_near" in x or "nt1_middle" in x]
    D = {}
    for x in i:
        sp = x.split("/")[-1].split(".")[0].split("_")[-1]
        if not (sp in D):
            D[sp] = []
        D[sp].append(x)
    o = []
    for x in D.values():
        ii = 0 if "nt1_middle" in x[0] else 1
        ti = 0 if ii == 1 else 1
        o.append((x[ii], x[ti]))
    return o


FILES = make_pairings(D1) + make_pairings(D2)
FILES=[:2]
print(FILES)

ipts = np.array([])
tgts = np.array([])
for ipt, tgt in FILES:
    _, data_x = wavfile.read(ipt)
    _, data_y = wavfile.read(tgt)
    mlen = min(data_x.shape[0], data_y.shape[0])
    data_x, data_y = data_x[:mlen], data_y[:mlen]
    ipts = np.concatenate([ipts, data_x])
    tgts = np.concatenate([tgts, data_y])

print(f"length of dataset in samples: {len(data_x):,}")

assert ipts.dtype == np.int16
assert tgts.dtype == np.int16
ipts, tgts = ipts.astype(np.int32), tgts.astype(np.int32)
assert ipts.min() < 0
assert tgts.min() < 0
ipts, tgts = ipts + 32768, tgts + 32768
assert ipts.min() >= 0
assert tgts.min() >= 0

# create the train and test splits
n = len(data_x)
train_data_x = ipts[:int(n*0.9)]
train_data_y = tgts[:int(n*0.9)]
val_data_x = ipts[int(n*0.9):]
val_data_y = tgts[int(n*0.9):]

# export to bin files
train_ids_x = np.array(train_data_x, dtype=np.uint32)
train_ids_y = np.array(train_data_y, dtype=np.uint32)
val_ids_x = np.array(val_data_x, dtype=np.uint32)
val_ids_y = np.array(val_data_y, dtype=np.uint32)
train_ids_x.tofile(os.path.join(os.path.dirname(__file__), 'train_x.bin'))
train_ids_y.tofile(os.path.join(os.path.dirname(__file__), 'train_y.bin'))
val_ids_x.tofile(os.path.join(os.path.dirname(__file__), 'val_x.bin'))
val_ids_y.tofile(os.path.join(os.path.dirname(__file__), 'val_y.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': 2**16
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
