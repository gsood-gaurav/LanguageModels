import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = "mps" if torch.backends.mps.is_available() else "cpu"
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)

text = Path("./input.txt").read_text(encoding="utf-8")
vocab = sorted(set(text))
vocab_size = len(vocab)

stoi = {v:k for k, v in enumerate(vocab)}
itos = {v:k for k, v in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda s: "".join(itos[i] for i in s)

# Traning data into tokens
data = encode(text)
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]


def get_batches(dt):
    data = train_data if (dt == "train") else val_data
    ix = torch.randint(0, len(data)-8, (batch_size,))
    train_batch = torch.stack([torch.tensor(data[i:i+block_size]) for i in ix])
    target_batch = torch.stack([torch.tensor(data[i+1:i+block_size+1]) for i in ix])
    # If we want to run on gpu, make sure when we load data move it to gpu
    train_batch, target_batch = train_batch.to(device), target_batch.to(device)

    return train_batch, target_batch


# This decorator tells that I am not going to call `.backwards`
# This way pytorch can be more efficient, it doesn't need to store intermediate variables.
@torch.no_grad()
def estimate_loss():
    out = {}
    # Some layer like dropout, BatchNorm has different behavior during inference or training time
    # Layers which are stateful like BatchNorm or dropout.
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batches(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of self attention"""
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril is not a parameter
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ffwd = nn.Sequential(nn.Linear(n_embd, n_embd), nn.ReLU())

    def forward(self, x):
        # It happens at token level
        return self.ffwd(x)


class BiGramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Each position will get an embedding vector.
        # we are learning embeddings for each position
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.sa_head = MultiHeadAttention(4, n_embd//4)
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, xb, target=None):
        B, T = xb.shape
        token_emb = self.token_embedding_table(xb) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_emb + pos_emb #(B, T, C)
        x = self.sa_head(x)
        x = self.ffwd(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        B, T, C = logits.shape
        # cross entropy takes input a flat structure as follows in case of logits.
        if target is not None:
            loss = F.cross_entropy(logits.view((B*T), C), target.view(-1))
        else:
            loss = None
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # consider only block size tokens
            # because we have position embeddngs till that point.
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on last time step, becomes B, C
            logits = logits[:,-1,:]
            # apply softmax to get probabilites
            probs = F.softmax(logits, dim=1)
            # sample from the distribution, returns indices
            idx_next = torch.multinomial(probs, num_samples=1)
            # we are building a context here.
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)

        return idx

model = BiGramLanguageModel()

# when we create the model, to run on gpu move it to gpu
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and test sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss{losses['val']:.4f}")
    

    xb, yb = get_batches("train_data")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, 500)[0].tolist()))
