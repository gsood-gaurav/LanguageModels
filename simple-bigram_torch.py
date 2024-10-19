import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "mps" if torch.backends.mps.is_available() else "cpu"
eval_iters = 200

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


class BiGramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, xb, target=None):
        logits = self.emb(xb)
        B, T, C = logits.shape
        # cross entropy takes input a flat structure as follows in case of logits.
        if target is not None:
            loss = F.cross_entropy(logits.view((B*T), C), target.view(-1))
        else:
            loss = None
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on last time step, becomes B, C
            logits = logits[:,-1,:]
            # apply softmax to get probabilites
            probs = F.softmax(logits, dim=1)
            # sample from the distribution, returns indices
            idx_next = torch.multinomial(probs, num_samples=1)
            # we are building a context here.
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)

        return idx

model = BiGramLanguageModel(vocab_size)

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
