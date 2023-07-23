import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

batch_size = 64 # how many independent sequences will be processes in parallel.
block_size = 256  # what is maximum context length for predictions.
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

tf.random.set_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i , c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ls: ''.join([itos[ix] for ix in ls])


data = tf.constant(encode(text), dtype=tf.dtypes.int64)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[:n]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = tf.random.uniform(maxval=len(data)-block_size, shape=(batch_size,), dtype=tf.dtypes.int64)
    x = tf.stack([data[i:i+block_size] for i in ix])
    y = tf.stack([data[i+1:i+1+block_size] for i in ix])

    return x, y


def estimate_loss():
    out = {}
    # model.eval()
    for split in ["train", "val"]:
        losses = tf.Variable(tf.zeros(eval_iters))
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            # losses[k] = loss.numpy()
            losses[k].assign(loss.numpy())
        # out[split] = losses.mean()
        out[split] = tf.reduce_mean(losses)
    # model.train()

    return out


class Head(tf.keras.layers.Layer):
    """One Head self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = tf.keras.layers.Dense(head_size, input_shape=(n_embed,), use_bias=False)
        self.query = tf.keras.layers.Dense(head_size, input_shape=(n_embed,), use_bias=False)
        self.value = tf.keras.layers.Dense(head_size, input_shape=(n_embed,), use_bias=False)

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute Attention scores
        wei = q @ tf.transpose(k, perm=[0, 2, 1]) * C**-0.5
        wei = tf.linalg.band_part(wei, -1, 0)
        mask = tf.equal(wei, 0.0)
        wei = tf.where(mask, -np.inf, wei)
        wei = tf.keras.activations.softmax(wei, axis=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out


class MultiHeadAttention(tf.keras.layers.Layer):
    """Mulitiple heads of self attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.projection = tf.keras.layers.Dense(n_embed, input_shape=(n_embed,))
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        out = tf.concat([h(x) for h in self.heads], axis=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(tf.keras.layers.Layer):
    """A Simple Linear Feed Forward Layer."""

    def __init__(self, n_embed):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(4*n_embed, activation="relu"),
            tf.keras.layers.Dense(n_embed),
            tf.keras.layers.Dropout(dropout)
        ]
        )

    def call(self, x):
        return self.net(x)


class Block(tf.keras.layers.Layer):

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


# class LayerNorm(tf.keras.layers.Layer):

#     def __init__(self, dim, eps=1e-5, momentum=0.1):
#         super().__init__()
#         self.eps = eps
#         self.momentum = momentum
#         self.gamma = tf.Variable(tf.ones(dim), trainable=True)
#         self.beta = tf.Variable(tf.zeros(dim), trainable=True)

#     def call(self, x):
#         xmean = x.mean(axis=1, keepdims=True)
#         xvar = x.var(axis=1, keepdims=True)
#         xhat = (x-xmean) / tf.sqrt(xvar+self.eps)
#         self.out = self.gamma * xhat + self.beta

#         return self.out

#     def parameters(self):
#         return [self.gamma, self.beta]


class BigramLanguageModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = tf.keras.layers.Embedding(vocab_size, n_embed)
        self.position_embedding_table = tf.keras.layers.Embedding(block_size, n_embed)
        # self.sa_head = Head(n_embed)
        # self.sa_heads = MultiHeadAttention(4, n_embed//4)
        # self.ffwd = FeedForward(n_embed)
        # self.blocks = tf.keras.Sequential([
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        #     tf.keras.layers.LayerNormalization()
        # ]
        # )
        self.blocks = tf.keras.Sequential([Block(n_embed, n_head)] * n_layer)
        self.ln_f = tf.keras.layers.LayerNormalization()
        self.lm_head = tf.keras.layers.Dense(vocab_size,
                                             input_shape=(n_embed,),
                                             activation=None)

    def call(self, idx, targets=None):

        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(tf.range(T))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        # x = self.sa_head(x)
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = tf.reshape(logits, shape=(B*T, C))
            targets = tf.reshape(targets, shape=(B*T,))
            # loss = tf.keras.losses.Categorical_Crossentropy(tf.one_hot(targets, depth=vocab_size), logits, from_logits=True)
            scce = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True)
            loss = scce(targets, logits)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = tf.keras.activations.softmax(logits, axis=-1)
            multinomial = tfd.Categorical(probs=probs, dtype=tf.dtypes.int64)
            idx_next = multinomial.sample(1)
            idx = tf.concat([idx, idx_next], axis=1)

        return idx


model = BigramLanguageModel()
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
for steps in range(max_iters):

    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        logits, loss = model(xb, yb)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

context = tf.zeros(shape=(1, 1), dtype=tf.dtypes.int64)
print(decode(model.generate(context, max_new_tokens=500)[0].numpy().tolist()))
