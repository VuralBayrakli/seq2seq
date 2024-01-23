# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 08:36:20 2023

@author: VuralBayraklii
"""

from tqdm import tqdm
from seq2seqmodel import (BahdanauAttentionQKV, 
                          BahdanauDecoder, 
                          BahdanauEncoder, 
                          BahdanauSeq2Seq, 
                          MultipleOptimizer)
import pickle
import torch.nn.functional as F

ENCODER_EMBEDDING_DIM = 256
ENCODER_HIDDEN_SIZE = 256
DECODER_EMBEDDING_DIM = 256
DECODER_HIDDEN_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

with open(os.path.join("pickles", "en_vocab.pkl"), "rb") as f:
    en_vocab = pickle.load(f)
    
with open(os.path.join("pickles", "tr_vocab.pkl"), "rb") as f:
    tr_vocab = pickle.load(f)    

enc = BahdanauEncoder(input_dim=len(en_vocab),
                      embedding_dim=ENCODER_EMBEDDING_DIM, 
                      encoder_hidden_dim=ENCODER_HIDDEN_SIZE, 
                      decoder_hidden_dim=DECODER_HIDDEN_SIZE, 
                      dropout_p=0.15)

attn = BahdanauAttentionQKV(DECODER_HIDDEN_SIZE)

dec = BahdanauDecoder(output_dim=len(tr_vocab), 
                      embedding_dim=DECODER_EMBEDDING_DIM,
                      encoder_hidden_dim=ENCODER_HIDDEN_SIZE, 
                      decoder_hidden_dim=DECODER_HIDDEN_SIZE,
                      attention=attn,
                      dropout_p=0.15)

seq2seq = BahdanauSeq2Seq(enc, dec, device)


PAD_IDX = en_vocab['<pad>']
BOS_IDX = en_vocab['<bos>']
EOS_IDX = en_vocab['<eos>']

def train(model, iterator, optimizer, loss_fn, device, clip=None):
    model.train()
    if model.device != device:
        model = model.to(device)
        
    epoch_loss = 0
    with tqdm(total=len(iterator), leave=False) as t:
        for i, (src, tgt) in enumerate(iterator):
            src_mask = (src != PAD_IDX).to(device)
            src = src.to(device)
            tgt = tgt.to(device)
            
            optimizer.zero_grad()
            
            output = model(src, tgt, src_mask)
            
            loss = loss_fn(output[1:].view(-1, output.shape[2]),
                           tgt[1:].view(-1))
            
            loss.backward()
            
            if clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (i+1)
            t.set_postfix(loss='{:05.3f}'.format(avg_loss),
                          ppl='{:05.3f}'.format(np.exp(avg_loss)))
            t.update()
            
    return epoch_loss / len(iterator)

def evaluate(model, iterator, loss_fn, device):
    model.eval()
    if model.device != device:
        model = model.to(device)
        
    epoch_loss = 0
    with torch.no_grad():
        with tqdm(total=len(iterator), leave=False) as t:
            for i, (src, tgt) in enumerate(iterator):
                src_mask = (src != PAD_IDX).to(device)
                src = src.to(device)
                tgt = tgt.to(device)
                
                output = model(src, tgt, src_mask, teacher_forcing_ratio=0)
                loss = loss_fn(output[1:].view(-1, output.shape[2]),
                               tgt[1:].view(-1))
                
                epoch_loss += loss.item()
                
                avg_loss = epoch_loss / (i+1)
                t.set_postfix(loss='{:05.3f}'.format(avg_loss),
                              ppl='{:05.3f}'.format(np.exp(avg_loss)))
                t.update()
    
    return epoch_loss / len(iterator)

def count_params(model, return_int=False):
    params = sum([torch.prod(torch.tensor(x.shape)).item() for x in model.parameters() if x.requires_grad])
    if return_int:
        return params
    else:
        print("There are {:,} trainable parameters in this model.".format(params))
        
count_params(seq2seq)

enc_optim = torch.optim.AdamW(seq2seq.encoder.parameters(), lr=1e-4)
dec_optim = torch.optim.AdamW(seq2seq.decoder.parameters(), lr=1e-4)
optims = MultipleOptimizer(enc_optim, dec_optim)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

with open(os.path.join("pickles", "train_iter.pkl"), "rb") as f:
    train_iter = pickle.load(f)
    
with open(os.path.join("pickles", "test_iter.pkl"), "rb") as f:
    test_iter = pickle.load(f)
    
with open(os.path.join("pickles", "valid_iter.pkl"), "rb") as f:
    valid_iter = pickle.load(f)

N_EPOCHS = 20
CLIP = 10 # clipping value, or None to prevent gradient clipping
EARLY_STOPPING_EPOCHS = 2

    
model_path = os.path.join('.', 'bahdanau_en_tr.pt')
bahdanau_metrics = {}
best_valid_loss = float("inf")
early_stopping_count = 0
for epoch in tqdm(range(N_EPOCHS), leave=False, desc="Epoch"):
    train_loss = train(seq2seq, train_iter, optims, loss_fn, device, clip=CLIP)
    valid_loss = evaluate(seq2seq, valid_iter, loss_fn, device)
    
    if valid_loss < best_valid_loss:
        tqdm.write(f"Checkpointing at epoch {epoch + 1}")
        best_valid_loss = valid_loss
        torch.save(seq2seq.state_dict(), model_path)
        early_stopping_count = 0
    else:
        early_stopping_count += 1
    
    bahdanau_metrics[epoch+1] = dict(
        train_loss = train_loss,
        train_ppl = np.exp(train_loss),
        valid_loss = valid_loss,
        valid_ppl = np.exp(valid_loss)
    )
    
    if early_stopping_count == EARLY_STOPPING_EPOCHS:
        tqdm.write(f"Early stopping triggered in epoch {epoch + 1}")
        break
