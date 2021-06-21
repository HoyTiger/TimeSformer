
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D, Conv3D, Input, AveragePooling2D, \
    multiply, Dense, Dropout, Flatten, AveragePooling3D
from tensorflow.python.keras.models import Model



def CAN(n_frame, nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3, 3), dropout_rate1=0.25, dropout_rate2=0.5,
           pool_size=(2, 2, 2), nb_dense=128):
    rawf_input = Input(shape=input_shape)


    r1 = Conv3D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    print(rawf_input.shape, r1.shape)
    r2 = Conv3D(nb_filters1, kernel_size, activation='tanh')(r1)
    r3 = AveragePooling3D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)
    r5 = Conv3D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv3D(nb_filters2, kernel_size, activation='tanh')(r5)
    r7 = AveragePooling3D(pool_size)(r6)
    r8 = Dropout(dropout_rate1)(r7)
    r9 = Flatten()(r8)
    r10 = Dense(nb_dense, activation='tanh')(r9)
    r11 = Dropout(dropout_rate2)(r10)
    out = Dense(n_frame)(r11)
    model = Model(inputs=[rawf_input], outputs=out)
    return model


