# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 14:27:15 2023

@author: VuralBayraklii
"""

from io import open
import unicodedata
import string
import re
import random
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator
from collections import Counter 

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

from tr_tokenizer import tokenize, analyze

from save_pickle_format import SavePickle

MAX_SENTENCE_LENGTH = 20
FILTER_TO_BASIC_PREFIXES = False
SAVE_DIR = os.path.join(".", "models")

ENCODER_EMBEDDING_DIM = 256
ENCODER_HIDDEN_SIZE = 256
DECODER_EMBEDDING_DIM = 256
DECODER_HIDDEN_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('tur-eng/tur.txt', encoding='utf-8') as f:
    lines = f.read().strip().split('\n')

print(f"{len(lines):,} English-Turkish phrase pairs.\n")
print("~~~~~ Examples: ~~~~~")
for example in random.choices(lines, k=5):
    pair = example.split('\t')
    print(f"English:  {pair[0]}")
    print(f"Turkish:   {pair[1]}")
    print()


def unicodeToAscii(s, language):
    s = ''.join(
        c for c in unicodedata.normalize('NFD', s) 
        if unicodedata.category(c) != 'Mn'
    )
    
    if language.lower() == 'tr':
        turkish_mapping = {'ı': 'i', 'İ': 'I', 'ğ': 'g', 'Ğ': 'G', 'ü': 'u', 'Ü': 'U', 'ş': 's', 'Ş': 'S', 'ö': 'o', 'Ö': 'O', 'ç': 'c', 'Ç': 'C'}
        s = ''.join(turkish_mapping.get(c, c) for c in s)
    
    return s

def normalizeString(s):
    #s = unicodeToAscii(s.lower().strip(), language)
    s = s.lower()
    s = re.sub(r"[^a-zA-ZıİÇçğÜüÖöŞş.!?]+", " ", s)
    return s

def filterPair(p, max_length, prefixes):
    good_length = (len(p[0].split(' ')) < max_length) and (len(p[1].split(' ')) < max_length)
    if len(prefixes) == 0:
        return good_length
    else:
        return good_length and p[0].startswith(prefixes)

def filterPairs(pairs, max_length, prefixes=()):
    return [pair for pair in pairs if filterPair(pair, max_length, prefixes)]

def prepareData(lines, filter=False, reverse=False, max_length=10, prefixes=()):
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    print(f"Given {len(pairs):,} sentence pairs.")

    if filter:
        pairs = filterPairs(pairs, max_length=max_length, prefixes=prefixes)
        print(f"After filtering, {len(pairs):,} remain.")

    return pairs

basic_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re ",
    'are you', 'am i ', 
    'were you', 'was i ', 
    'where are', 'where is',
    'what is', 'what are'
)

pairs = prepareData(lines, 
                    filter=True, 
                    max_length=MAX_SENTENCE_LENGTH, 
                    prefixes=basic_prefixes if FILTER_TO_BASIC_PREFIXES else ())

en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

def tr_tokenizerr(text):
    words = []
    tokens = tokenize(text)
    for word in tokens:
        token = word.token
        words.append(token)
    
    return words

SPECIALS = ['<unk>', '<pad>', '<bos>', '<eos>']

en_list = []
tr_list = []
en_counter = Counter()
tr_counter = Counter()
en_lengths = []
tr_lengths = []
for idx, (en, tr, info) in enumerate(pairs):
    en_toks = en_tokenizer(en)
    print(en_toks)
    tr_toks = tr_tokenizerr(tr)
    
    en_list += [en_toks]
    tr_list += [tr_toks]
    en_counter.update(en_toks)
    tr_counter.update(tr_toks)
    en_lengths.append(len(en_toks))
    tr_lengths.append(len(tr_toks))
    if idx % 1000 == 0:
        print(idx)
en_vocab = build_vocab_from_iterator(en_list, specials=SPECIALS)
tr_vocab = build_vocab_from_iterator(tr_list, specials=SPECIALS)

VALID_PCT = 0.1
TEST_PCT  = 0.1

train_data = []
valid_data = []
test_data = []

random.seed(6547)
for (en, tr, info) in pairs:
    en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(en)])
    tr_tensor_ = torch.tensor([tr_vocab[token] for token in tr_tokenizerr(tr)])
    random_draw = random.random()
    if random_draw <= VALID_PCT:
        valid_data.append((en_tensor_, tr_tensor_))
    elif random_draw <= VALID_PCT + TEST_PCT:
        test_data.append((en_tensor_, tr_tensor_))
    else:
        train_data.append((en_tensor_, tr_tensor_))


print(f"""
  Training pairs: {len(train_data):,}
Validation pairs: {len(valid_data):,}
      Test pairs: {len(test_data):,}""")

PAD_IDX = en_vocab['<pad>']
BOS_IDX = en_vocab['<bos>']
EOS_IDX = en_vocab['<eos>']

for en_id, tr_id in zip(en_vocab.lookup_indices(SPECIALS), tr_vocab.lookup_indices(SPECIALS)):
  assert en_id == tr_id

def generate_batch(data_batch):
    '''
    Prepare English and French examples for batch-friendly modeling by appending
    BOS/EOS tokens to each, stacking the tensors, and filling trailing spaces of
    shorter sentences with the <pad> token. To be used as the collate_fn in the
    English-to-French DataLoader.

    Input: 
    - data_batch, an iterable of (English, French) tuples from the datasets 
      created above

    Outputs
    - en_batch: a (max length X batch size) tensor of English token IDs
    - fr_batch: a (max length X batch size) tensor of French token IDs 
    '''
    en_batch, tr_batch = [], []
    for (en_item, tr_item) in data_batch:
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        tr_batch.append(torch.cat([torch.tensor([BOS_IDX]), tr_item, torch.tensor([EOS_IDX])], dim=0))

    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX, batch_first=False)
    tr_batch = pad_sequence(tr_batch, padding_value=PAD_IDX, batch_first=False)

    return en_batch, tr_batch


BATCH_SIZE = 16

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)

for i, (en_id, tr_id) in enumerate(train_iter):
    print('English:', ' '.join([en_vocab.lookup_token(idx) for idx in en_id[:, 0]]))
    print('Turkish:', ' '.join([tr_vocab.lookup_token(idx) for idx in tr_id[:, 0]]))
    if i == 4: 
        break
    else:
        print()


save_pairs = SavePickle("pairs", pairs)
save_pairs.save_process()

save_en_vocab = SavePickle("en_vocab", en_vocab)
save_en_vocab.save_process()

save_tr_vocab =SavePickle("tr_vocab", tr_vocab)
save_tr_vocab.save_process()

save_train_data = SavePickle("train_data", train_data)   
save_train_data.save_process()

save_valid_data = SavePickle("valid_data", valid_data)
save_valid_data.save_process()

save_test_data = SavePickle("test_data", test_data) 
save_test_data.save_process()
    
save_train_iter = SavePickle("train_iter", train_iter)
save_train_iter.save_process()

save_test_iter = SavePickle("test_iter", test_iter)
save_test_iter.save_process()
    
save_valid_iter = SavePickle("valid_iter", valid_iter)
save_valid_iter.save_process()
