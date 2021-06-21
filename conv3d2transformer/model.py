from einops import rearrange, repeat
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D, Conv3D, Input, AveragePooling2D, \
    multiply, Dense, Dropout, Flatten, AveragePooling3D, Layer, LayerNormalization, Embedding
from tensorflow.python.keras.models import Model, Sequential
from einops.layers.tensorflow import Rearrange, Reduce

from transformer import Transformer

num_layers = 4
model_size = 768
num_heads = 12
dff_size = 1024
def CAN(n_frame, nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3, 3), dropout_rate1=0.25, dropout_rate2=0.5,
           pool_size=(2, 2, 2), nb_dense=128):
    rawf_input = Input(shape=input_shape)

    r1 = Conv3D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv3D(nb_filters1, kernel_size, activation='tanh')(r1)
    r3 = AveragePooling3D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)
    r5 = Conv3D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv3D(nb_filters2, kernel_size, activation='tanh')(r5)
    r7 = AveragePooling3D(pool_size)(r6)
    r8 = Dropout(dropout_rate1)(r7)
    r9 = Flatten()(r8)
    r10 = Dense(nb_dense, activation='relu')(r9)
    r11 = Dropout(dropout_rate2)(r10)
    x = Transformer(num_layers=num_layers,
                    model_size=model_size,
                    num_heads=num_heads,
                    dff_size=dff_size,
                    vocab_size=nb_dense+1,
                    maxlen=nb_dense)(r11)
    x = Flatten()(x)
    out = Dense(n_frame)(x)
    model = Model(rawf_input, out)
    model.summary()
    return model

class PreNorm(keras.Model):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = LayerNormalization()

    def call(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

@tf.function(experimental_relax_shapes=True)
def gelu(x):
    return 0.5 * x * (1 + K.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))


class GEGLU(keras.Model):
    def __init__(self):
        super(GEGLU, self).__init__()
    def call(self, x):
        x, gates = tf.split(x, 2, -1)
        return x * gelu(gates)


# position-wise feed forward network
class FeedForward(keras.Model):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super(FeedForward, self).__init__()
        self.dense1 = keras.layers.Dense(dim * mult * 2, activation="relu")
        self.dense2 = keras.layers.Dense(dim)

    def call(self, x):
        x = self.dense1(x)
        x = GEGLU()(x)
        x = Dropout(0.5)(x)
        x = self.dense2(x)
        return x

# class FeedForward(keras.Model):
#     def __init__(self, dim, mult = 4, dropout = 0.):
#         super(FeedForward).__init__()
#         self.net = Sequential(
#             Dense(dim * mult * 2),
#             GEGLU(),
#             Dropout(dropout),
#             Dense(dim)
#         )
#
#     def call(self, x):
#         return self.net(x)


def attn(q, k, v):
    sim = tf.einsum('b i d, b j d -> b i j', q, k)
    attn = tf.nn.softmax(sim, axis = -1)
    out = tf.einsum('b i j, b j d -> b i d', attn, v)
    return out

class MultiHeadAttention(keras.Model):
    def __init__(self, dim, model_size, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.model_size = model_size
        self.num_heads = num_heads
        self.head_size = model_size // num_heads
        self.WQ = keras.layers.Dense(model_size, name="dense_query")
        self.WK = keras.layers.Dense(model_size, name="dense_key")
        self.WV = keras.layers.Dense(model_size, name="dense_value")
        self.dense = keras.layers.Dense(dim)

    def call(self, query, key, value, mask=None):
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
        output = Dropout(0.2)(output)

        return output


class TimeSformer(keras.Model):
    def __init__(
        self,
        *,
        dim,
        num_frames,
        num_classes,
        image_size = 224,
        patch_size = 16,
        channels = 3,
        depth = 12,
        heads = 8,
        dim_head = 64,
        ff_dropout = 0.
    ):
        super(TimeSformer, self).__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches

        self.patch_size = patch_size
        self.to_patch_embedding = Dense(dim)
        self.pos_emb = Embedding(num_positions + 1, dim)
        self.cls_token = tf.Variable(K.truncated_normal([1,dim]))

        self.layerslist = []
        for _ in range(depth):
            self.layerslist.append([
                PreNorm(dim,
                        MultiHeadAttention(dim, model_size=heads * dim_head, num_heads=heads)),
                PreNorm(dim,
                        MultiHeadAttention(dim, model_size=heads * dim_head, num_heads=heads)),
                PreNorm(dim, FeedForward(dim, dropout=ff_dropout))
            ])

        self.to_out = Sequential([
            LayerNormalization(name='out_norm'),
            Dense(num_classes, name='out')
        ], name='out_dense')

    def call(self, video):
        b, _, f, h, w, p = *video.shape, self.patch_size
        if b==None:
            b = 1
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

        n = (h // p) * (w // p)

        video = Rearrange( 'b c f (h p1) (w p2) -> b (c h w) (p1 p2 f)', p1 = p, p2 = p)(video)
        tokens = self.to_patch_embedding(video)

        cls_token = repeat(self.cls_token, 'n d -> b n d', b = b)
        x =  tf.concat((cls_token, tokens), 1)
        x += self.pos_emb(K.arange(x.shape[1]))

        for (time_attn, spatial_attn, ff) in self.layerslist:
            x = time_attn(x,x,x) + x
            x = spatial_attn(x,x,x) + x
            x = ff(x) + x

        cls_token = x[:, 0]
        print(x.shape)
        return self.to_out(cls_token)

def build_mdoel_1(n_class,nb_filters1,nb_filters2,input_shape,  kernel_size=(3, 3, 3), dropout_rate1=0.25, dropout_rate2=0.5,
           pool_size=(2, 2, 2)):
    input = Input(shape=input_shape)
    r1 = Conv3D(nb_filters1, kernel_size, padding='same', activation='tanh', data_format='channels_first')(input)
    r2 = Conv3D(nb_filters1, kernel_size, padding='same', activation='tanh', data_format='channels_first')(r1)
    r3 = AveragePooling3D(pool_size, data_format='channels_first')(r2)
    r4 = Dropout(dropout_rate1)(r3)
    r5 = Conv3D(nb_filters2, kernel_size, padding='same', activation='tanh', data_format='channels_first')(r4)
    r6 = Conv3D(nb_filters2, kernel_size, padding='same', activation='tanh',data_format='channels_first')(r5)
    r8 = Dropout(dropout_rate1)(r6)

    a = TimeSformer(dim=256,
                    image_size=18,
                    patch_size=9,
                    num_frames=64,
                    num_classes=n_class,
                    depth=12,
                    heads=8,
                    dim_head=64,
                    ff_dropout=0.1)(r8)
    model = Model(input, a)
    model.summary()
    return model

def build_mdoel_2(n_frame, input_shape, n_class):
    input = Input(shape=input_shape)

    a = TimeSformer(dim=256,
                    image_size=36,
                    patch_size=9,
                    num_frames=n_frame,
                    num_classes=n_class,
                    depth=12,
                    heads=8,
                    dim_head=64,
                    ff_dropout=0.1)(input)
    model = Model(input, a)
    model.summary()
    return model

if __name__ == '__main__':
    build_mdoel_1(10,32,64, (3,10,36,36))
    build_mdoel_2(10, (3,10,36,36))
    CAN(10,32,64,(36,36,10,3))
