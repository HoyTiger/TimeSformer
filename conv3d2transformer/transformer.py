import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow import keras
# 位置编码信息
def positional_embedding(maxlen, model_size):
    PE = np.zeros((maxlen, model_size))
    for i in range(maxlen):
        for j in range(model_size):
            if j % 2 == 0:
                PE[i, j] = np.sin(i / 10000 ** (j / model_size))
            else:
                PE[i, j] = np.cos(i / 10000 ** ((j-1) / model_size))
    PE = tf.constant(PE, dtype=tf.float32)
    return PE



class MultiHeadAttention(keras.Model):
    def __init__(self, model_size, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.model_size = model_size
        self.num_heads = num_heads
        self.head_size = model_size // num_heads
        self.WQ = keras.layers.Dense(model_size, name="dense_query")
        self.WK = keras.layers.Dense(model_size, name="dense_key")
        self.WV = keras.layers.Dense(model_size, name="dense_value")
        self.dense = keras.layers.Dense(model_size)

    def call(self, query, key, value, mask):
        # query: (batch, maxlen, model_size)
        # key  : (batch, maxlen, model_size)
        # value: (batch, maxlen, model_size)
        batch_size = tf.shape(query)[0]

        # shape: (batch, maxlen, model_size)
        query = self.WQ(query)
        key = self.WK(key)
        value = self.WV(value)

        def _split_heads(x):
            x = tf.reshape(x, shape=[batch_size, -1, self.num_heads, self.head_size])
            return tf.transpose(x, perm=[0, 2, 1, 3])

        # shape: (batch, num_heads, maxlen, head_size)
        query = _split_heads(query)
        key = _split_heads(key)
        value = _split_heads(value)

        # shape: (batch, num_heads, maxlen, maxlen)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        # 缩放 matmul_qk
        dk = tf.cast(query.shape[-1], tf.float32)
        score = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            # mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)
            score += (1 - mask) * -1e9

        alpha = tf.nn.softmax(score)
        context = tf.matmul(alpha, value)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.model_size))
        output = self.dense(context)

        return output


# position-wise feed forward network
class FeedForwardNetwork(keras.Model):
    def __init__(self, dff_size, model_size):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(dff_size, activation="relu")
        self.dense2 = keras.layers.Dense(model_size)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


# Encoder Layer层
class EncoderLayer(keras.layers.Layer):
    def __init__(self, model_size, num_heads, dff_size, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_size, num_heads)
        self.ffn = FeedForwardNetwork(dff_size, model_size)

        # Layer Normalization
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # multi head attention
        attn_output = self.attention(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # residual connection
        out1 = self.layernorm1(x + attn_output)
        # ffn layer
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Residual connection
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


# 多层Encoder
class Encoder(keras.Model):
    def __init__(self, num_layers, model_size, num_heads, dff_size, vocab_size, maxlen, rate=0.1):
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers

        self.embedding = keras.layers.Embedding(vocab_size, model_size)
        self.pos_embedding = positional_embedding(maxlen, model_size)

        self.encoder_layers = [EncoderLayer(model_size, num_heads, dff_size, rate) for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)

    def call(self, x, training, padding_mask):
        # input embedding + positional embedding
        x = self.embedding(x) + self.pos_embedding
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training, padding_mask)
        return x


# Decoder Layer
class DecoderLayer(keras.layers.Layer):
    def __init__(self, model_size, num_heads, dff_size, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mask_attention = MultiHeadAttention(model_size, num_heads)
        # self.mask_attention = MultiHeadAttention(model_size, num_heads, causal=True)
        self.attention = MultiHeadAttention(model_size, num_heads)
        self.ffn = FeedForwardNetwork(dff_size, model_size)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn_decoder = self.mask_attention(x, x, x, look_ahead_mask)
        attn_decoder = self.dropout1(attn_decoder, training=training)
        out1 = self.layernorm1(x + attn_decoder)

        attn_encoder_decoder = self.attention(out1, enc_output, enc_output, padding_mask)
        attn_encoder_decoder = self.dropout2(attn_encoder_decoder, training=training)
        out2 = self.layernorm2(out1 + attn_encoder_decoder)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3


# 多层Decoder
class Decoder(keras.Model):
    def __init__(self, num_layers, model_size, num_heads, dff_size, vocab_size, maxlen, rate=0.1):
        super(Decoder, self).__init__()

        self.model_size = model_size
        self.num_layers = num_layers

        self.embedding = keras.layers.Embedding(vocab_size, model_size)
        self.pos_embedding = positional_embedding(maxlen, model_size)

        self.decoder_layers = [DecoderLayer(model_size, num_heads, dff_size, rate) for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)

    def call(self, enc_output, x, training, look_ahead_mask, padding_mask):
        # input embedding + positional embedding
        x = self.embedding(x) + self.pos_embedding
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.decoder_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

        return x

# padding填充mask
def padding_mask(seq):
    mask = tf.cast(tf.math.not_equal(seq, 0), dtype=tf.float32)
    mask = mask[:, tf.newaxis, tf.newaxis, :]
    return mask

# decode mask
def look_ahead_mask(size):
    ahead_mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    ahead_mask = tf.cast(ahead_mask, dtype=tf.float32)
    return ahead_mask


def create_mask(inp, tar):
    enc_padding_mask = padding_mask(inp)
    dec_padding_mask = padding_mask(tar)
    ahead_mask = look_ahead_mask(tf.shape(tar)[1])
    combined_mask = tf.minimum(dec_padding_mask, ahead_mask)
    return enc_padding_mask, dec_padding_mask, combined_mask


# Encoder和Decoder组合成Transformer 本文只使用了Encoder
class Transformer(keras.Model):
    def __init__(self, num_layers, model_size, num_heads, dff_size, vocab_size, maxlen, training=True, rete=0.1):
        super(Transformer, self).__init__()
        self.training = training

        self.encoder = Encoder(num_layers, model_size, num_heads, dff_size, vocab_size, maxlen)
        self.decoder = Decoder(num_layers, model_size, num_heads, dff_size, vocab_size, maxlen)
        self.final_dense = keras.layers.Dense(vocab_size, name="final_output")

    def call(self, inputs):

        enc_padding_mask = padding_mask(inputs)

        enc_output = self.encoder(inputs, self.training, enc_padding_mask)

        return enc_output
