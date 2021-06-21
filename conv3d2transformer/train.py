# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from model import *
from inference_preprocess import preprocess_raw_video,detrend
import random
import h5py
import scipy
from scipy.signal import butter
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras import losses as LOSSES
import tensorflow as tf
from scipy.signal import resample
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


n_frame       = 10
nb_filters1   = 32 
nb_filters2   = 64
input_shape_1 = (3,n_frame,36,36)
kernel_size   = (3, 3)
dropout_rate1 = 0.25
dropout_rate2 = 0.5
pool_size     = (2, 2)
nb_dense      = 128
lr            = 1.0
batch_size1    = 1
epochs        = 1000
steps_per_epoch = 10
dataset_type  = 1
train_path    = './train.txt'

test_path     = './train.txt'
dataset_path  = './data/'
sample_list   = None
start         = 0
end           = 5000
cv            = 10

with open(train_path) as f:
    sample_list = [dataset_path+i.replace('\n','') for i in f.readlines()]



def gen(video_path):
    data ,label_y ,label_r = [],[],[]
    for temp_path in video_path:
        try:
            dXsub = preprocess_raw_video(temp_path+'.avi')
            f1 = h5py.File(temp_path+'.hdf5', 'r')
            
            dXsub_len = (dXsub.shape[0] // n_frame)  * n_frame
            dXsub_len = dXsub.shape[0] 
            dysub = np.array(f1['pulse'])       #脉搏、心率

            num_window = 100
            
            for f in range(num_window):
                data.append(dXsub[f:f+n_frame,:,:,:])
            

                label_y.append(dysub[f:f+n_frame])
        except Exception as e:
            continue

    data = np.array(data).reshape((-1,n_frame,36,36,6))
    data = data.transpose((0,4,1,2,3))
    # data = np.swapaxes(data, 1, 2)
    label_y = np.array(label_y).reshape(-1,n_frame)
    output = (data[:, :3, :, :, :], data[:, -3:, :, :, :])
    return output,label_y

kf = KFold(n_splits=cv)
flag = 1
His = {'val_mae':[]}

tf.config.experimental_run_functions_eagerly(True)
for train_index,vali_index in kf.split(sample_list):
    train_sample = [sample_list[i] for i in train_index]
    vali_sample  = [sample_list[i] for i in vali_index]


    output_tr,label_tr = gen(train_sample)
    output_va,label_va = gen(vali_sample)
    model_name = '1'
    Model = build_mdoel_1(n_frame,nb_filters1,nb_filters2,input_shape_1)
    # model_name = '2'
    # Model = build_mdoel_2(n_frame, input_shape_1, n_frame)
    optimizer = optimizers.Adadelta(learning_rate=lr)
    Model.compile(loss="huber_loss", optimizer=optimizer,metrics=['mae','mse'])

    checkpoint = ModelCheckpoint(filepath=f'./ZZ-cv-{model_name}'+str(flag)+'-{epoch:02d}-{loss:.2f}.hdf5',\
                                 monitor='val_loss',
                                 save_best_only=False,
                                 save_weights_only=True)
    history = Model.fit(output_tr[0],label_tr,batch_size=12,epochs=50,verbose=1,\
                        callbacks=[checkpoint],validation_data=(output_va[0],label_va))

    His['val_mae'].append(history.history['val_mae'])
    flag+=1


