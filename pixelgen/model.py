import torch
import torch.nn as nn
import torch.nn.functional as F
from data import get_data 
import random


pixel_list, size = get_data()

print(size)
batch_size = 64
block_size = (size**2)-1
max_iters = 5000
eval_interval = 500
learning_rate = 0.0006
device = 'cuda' 
eval_iters =  200
embd_d = 384
n_head = 6
n_layer = 6
dropout = 0.2
head_size = embd_d//n_head

vocab_size = 256

print(len(pixel_list))
'''
lst = []
for i in range(vocab_size):
    lst.append(i)
print(lst)
'''
#enc = dict(zip(result,lst))
# dec = {v: k for k, v in enc.items()}

#pixls = [[enc[value] for value in row] for row in pixel_list]
pixel_list = torch.tensor(pixel_list)
splitnum = int(0.9*(len(pixel_list)))
print(splitnum)

train = pixel_list[:splitnum]
val_data = pixel_list[splitnum:]

def get_batch(split):
  data = train if split == 'train' else val_data
  index = random.randint(0,(len(data))-1)
  b = torch.randint((len(data[index]) - block_size), (batch_size,))
  x = (torch.stack([data[index][i:i+block_size] for i in b])).to(device)
  y = (torch.stack([data[index][i+1:i+block_size+1] for i in b])).to(device)
  return x,y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(embd_d, head_size, bias=False) 
        self.query = nn.Linear(embd_d, head_size, bias=False)
        self.value = nn.Linear(embd_d, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x) 
        qk = q@k.transpose(-2,-1)*k.shape[-1]**-0.5 
        zeromask = torch.tril(qk)
        zeromask.masked_fill_(zeromask==0, (float('-inf')))
        scores = F.softmax(zeromask, dim=-1)
        scores = self.dropout(scores)
        fin = scores @ v
        return fin

class MultiHeadAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.multihead = nn.ModuleList([Head() for _ in range(n_head)])
        self.lproj = nn.Linear(head_size*n_head, embd_d)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        mheadout = torch.cat([head(x)for head in self.multihead], dim=-1)
        out = self.dropout(self.lproj(mheadout))
        return out
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd_d,4*  embd_d),
            nn.ReLU(),
            nn.Linear(4* embd_d, embd_d),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ma = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(embd_d)
        self.ln2 = nn.LayerNorm(embd_d)

    def forward(self, x):
        x = x + self.ma(self.ln1(x))
        out = x + self.ffwd(self.ln2(x))
        return out
class Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_table = nn.Embedding(vocab_size, embd_d)
        self.pos_emb = nn.Embedding(block_size, embd_d)
        self.decoders = nn.Sequential(*[Decoder() for _ in range (n_layer)])
        self.fproj = nn.Linear(embd_d, vocab_size)
        self.ln1 = nn.LayerNorm(embd_d)
    def forward(self, x, targets=None):
        B,T = x.shape
        
        tokens = self.token_table(x)
        positional = self.pos_emb(torch.arange(T, device=device))
        embeddings = tokens+ positional
        dout = self.decoders(embeddings)
        lnout = self.ln1(dout)
        logits = self.fproj(lnout)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx   
model = Transformer()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def train_model():
    for iter in range(max_iters):
        
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        torch.cuda.empty_cache()
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
   
    torch.save(model.state_dict(), f"models/{size},{losses['val']:.4f}.pth")
#train_model()
