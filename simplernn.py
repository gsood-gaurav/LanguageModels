import tensorflow as tf
# import numpy as np

batch_size = 8


def data_gen(file_name):
    with open(file_name, 'r') as f:
        xs, ys = [], []
        for line in f:
            line = line.strip()
            if not line:
                yield (xs, ys)
                xs, ys = [], []
            else:
                word, tag = line.split()
                xs.append(word)
                ys.append(tag)


dataset = tf.data.Dataset.from_generator(
    data_gen,
    args = ["eng.train.iob"],
    output_signature= (
        tf.TensorSpec(shape=(None,), dtype=tf.string),
        tf.TensorSpec(shape=(None,), dtype=tf.string)
    )
)

sentences = dataset.map(lambda xs, _ : xs)
tags = dataset.map(lambda _, ys : ys)
layer_sen = tf.keras.layers.TextVectorization()
layer_tag = tf.keras.layers.TextVectorization(standardize=None, max_tokens=7)
layer_sen.adapt(sentences)
layer_tag.adapt(tags)
# print(layer_sen.get_vocabulary())
vocab_size = layer_sen.vocabulary_size()
# print(vocab_size)
num_tags = layer_tag.vocabulary_size()

print("vocab_size: ", vocab_size)

d = dataset.map(lambda xs, ys: (xs, ys))
train_dataset = dataset.map(lambda xs, ys:
                            (tf.reshape(layer_sen(xs), tf.shape(xs)),
                             tf.reshape(layer_tag(ys),
                                        tf.shape(ys)))).padded_batch(batch_size)


class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=vocab_size, num_tags=num_tags):
        super().__init__()
        # self.minput = tf.keras.Input(type_spec=tf.TensorSpec(shape=(None, None), dtype=tf.int32))
        self.embedding = tf.keras.layers.Embedding(vocab_size, 64, mask_zero=True)
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, return_state=True))
        self.dense = tf.keras.layers.Dense(num_tags)

    def call(self, inputs):
        # x = self.minput(inputs)
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.dense(x)
        return x


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
nll = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model = MyModel(vocab_size=vocab_size, num_tags=num_tags)
model.compile(loss=nll, optimizer=optimizer)
model.fit(train_dataset, epochs=3)

tf.keras.activations.softmax