import numpy as np
import tensorflow as tf

# Hyperparameters
# vocab_size = 27
# emb_dim = 384
# n_heads = 6
# dk = emb_dim // n_heads
context_length = 128
# Nx = 6
dropout = 0.2


class PositionEncoding(tf.keras.layers.Layer):
    def __init__(self, context_length, d_model):
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model

        position = tf.range(context_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, d_model, 2, dtype=tf.float32)
            * (-tf.math.log(10000.0) / d_model)
        )

        pe = np.zeros((context_length, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position.numpy() * div_term.numpy())
        pe[:, 1::2] = np.cos(position.numpy() * div_term.numpy())
        self.pe = tf.constant(pe)[tf.newaxis, ...]

    def call(self, inputs):
        return self.pe[:, : tf.shape(inputs)[1], :]


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, dk):
        super().__init__()
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.WK = tf.Variable(self.initializer(shape=(emb_dim, dk)), trainable=True)
        self.WQ = tf.Variable(self.initializer(shape=(emb_dim, dk)), trainable=True)
        self.WV = tf.Variable(self.initializer(shape=(emb_dim, dk)), trainable=True)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.mask = 1 - tf.linalg.band_part(
            tf.ones((context_length, context_length)), -1, 0
        )
        self.mask = self.mask * -1e9

    def call(self, x, training=False):
        B, T, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        key = x @ self.WK
        query = x @ self.WQ
        value = x @ self.WV

        wei = query @ tf.transpose(key, [0, 2, 1])
        wei = wei / tf.math.sqrt(tf.cast(dk, tf.float32))
        wei = wei + self.mask[:T, :T]
        wei = tf.nn.softmax(wei, axis=-1)
        wei = self.dropout(wei, training=training)
        out = wei @ value
        return out


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, n_heads, dk):
        super().__init__()
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.heads = [SelfAttention(dk) for _ in range(n_heads)]
        self.WO = tf.Variable(
            self.initializer(shape=(emb_dim, emb_dim)), trainable=True
        )
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        out = tf.concat([h(x) for h in self.heads], axis=-1)
        out = out @ self.WO
        return out


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, emb_dim):
        super().__init__()
        self.network = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(4 * emb_dim),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(emb_dim),
                tf.keras.layers.Dropout(dropout),
            ]
        )

    def call(self, x):
        return self.network(x)


class Block(tf.keras.layers.Layer):
    def __init__(self, emb_dim, dk, n_heads):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, dk)
        self.ffwd = FeedForward(emb_dim)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderTransformer(tf.keras.Model):
    def __init__(self , vocab_size , emb_dim , n_heads ,  Nx):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(vocab_size, emb_dim)
        self.pos_emb = PositionEncoding(context_length, emb_dim)
        self.blocks = tf.keras.Sequential(
            [Block(emb_dim, emb_dim // n_heads, n_heads) for _ in range(Nx)]
        )
        self.ln = tf.keras.layers.LayerNormalization()
        self.lm_head = tf.keras.layers.Dense(vocab_size)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, xb):
        te = self.token_emb(xb)
        pe = self.pos_emb(xb)
        emb = te + pe
        x = self.blocks(emb)
        x = self.ln(x)
        logits = self.lm_head(x)  # (B , T , vocab_size)
        return logits

    def compute_loss(self, xb, yb):
        logits = self(xb)
        logits_flat = tf.reshape(logits, (-1, logits.shape[2]))  # (B * T , vocab_size)
        target_flat = tf.reshape(yb, (-1))  # (B * T)
        loss = self.loss_fn(target_flat, logits_flat)
        loss = tf.reduce_mean(loss)
        return loss


def create_transformer(vocab_size, emb_dim, n_heads, Nx):
    return DecoderTransformer(vocab_size, emb_dim, n_heads, Nx)


def compute_loss(model, xb, targets):
    return model.compute_loss(xb, targets)
