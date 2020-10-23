import numpy as np
import pylab
import tensorflow as tf
from tensorflow import losses, optimizers, nn, keras
from tensorflow.keras import Input, Model, models, layers, metrics, initializers
import os
import skimage
from skimage import transform
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model, model_to_dot
import h5py
from tensorflow.keras.datasets.mnist import load_data
import math


def encoder(im_shape, relu_before_bn=True):
    conv2d = lambda x, filters: layers.Conv2D(filters, (3, 3), strides=(2, 2), padding='same')(x)
    lrelu = lambda x: layers.LeakyReLU(alpha=0.2)(x)
    norm = lambda x: layers.BatchNormalization()(x)
    if relu_before_bn:
        block = lambda x, filters: norm(lrelu(conv2d(x, filters)))
    else:
        block = lambda x, filters: lrelu(norm(conv2d(x, filters)))

    inputs = Input(shape=im_shape)
    l1 = block(inputs, 32)
    l2 = block(l1, 16)
    l3 = block(l2, 8)
    encoder_out = block(l3, 8)


    encoder_model = Model(inputs=inputs, outputs=encoder_out)
    return encoder_model

def decoder(gen_input_size, relu_before_bn=True):
    trans2d = lambda x, filters: layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(x)
    lrelu = lambda x: layers.LeakyReLU(alpha=0.2)(x)
    norm = lambda x: layers.BatchNormalization()(x)
    if relu_before_bn:
        block = lambda x, filters: norm(lrelu(trans2d(x, filters)))
    else:
        block = lambda x, filters: lrelu(norm(trans2d(x, filters)))

    inputs = Input(shape=gen_input_size)
    l1 = block(inputs, 8)
    l2 = block(l1, 16)
    l3 = block(l2, 32)
    decoder_out = layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', activation='tanh')(l3)

    decoder_model = Model(inputs=inputs, outputs=decoder_out)
    return decoder_model


def autoencoder(encoder_model, decoder_model):
    decoder_output = decoder_model(encoder_model.output)

    model = Model(inputs=encoder_model.input, outputs=decoder_output)
    opt = optimizers.Adam()
    model.compile(loss='mse', optimizer=opt)
    return model


def load_data(filename, tanh=True):
    dataset = np.load(filename)
    dataset.astype('float32')
    if tanh:
        dataset = (dataset-127.5)/127.5
    else:
        dataset = dataset/255
    return dataset
