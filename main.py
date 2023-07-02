from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel # import class BaseModel của thư viện pydantic
import modelLSTM
import pickle
import numpy as np
import torch

with open('vocab.pkl', 'rb') as file:
    vocab = pickle.load(file)

def tokenize(x_set, vocab):    
    final_set= []
    for sent in x_set:
        final_set.append([vocab[word] for word in sent if vocab.get(word) != None])
    return np.array(final_set, dtype=object)

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

init_h = modelLSTM.model.init_hidden(1)
h = tuple([each.data for each in init_h])

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World 234"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.post("/items/")
async def create_item(item: str): # khai báo dưới dạng parameter
    item = [item.split()]   
    print(item)
    input = tokenize(item, vocab) #trả về np.array chứa list
    input_pad = padding_(input, 500)
    input_pad_torch = torch.from_numpy(input_pad).to(modelLSTM.device)
    print(input_pad_torch.shape)
    output, _ = modelLSTM.model(input_pad_torch, h)    
    return 'Positive' if torch.round(output.squeeze()) == 1 else 'Negative'