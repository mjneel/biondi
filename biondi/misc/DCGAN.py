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


def generator(gen_input_size, relu_before_bn=True):
    trans2d = lambda x, filters: layers.Conv2DTranspose(filters, (5, 5), strides=(2, 2), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.02))(x)
    dense = lambda x, nodes: layers.Dense(nodes, kernel_initializer=initializers.RandomNormal(stddev=0.02))(x)
    reshape = lambda x, shape: layers.Reshape(shape)(x)
    relu = lambda x: layers.ReLU()(x)
    norm = lambda x: layers.BatchNormalization()(x)
    if relu_before_bn:
        block = lambda x, filters: norm(relu(trans2d(x, filters)))
    else:
        block = lambda x, filters: relu(norm(trans2d(x, filters)))

    inputs = Input(shape=gen_input_size)
    l1 = reshape(dense(inputs, 4 * 4 * 1024), (4, 4, 1024))
    l2 = block(l1, 512)
    l3 = block(l2, 256)
    l4 = block(l3, 128)
    g_out = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh')(l4)

    g_model = Model(inputs=inputs, outputs=g_out)
    return g_model


def discriminator(im_shape, relu_before_bn=True):
    conv2d = lambda x, filters: layers.Conv2D(filters, (5, 5), strides=(2, 2), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.02))(x)
    flatten = lambda x: layers.Flatten()(x)
    lrelu = lambda x: layers.LeakyReLU(alpha=0.2)(x)
    norm = lambda x: layers.BatchNormalization()(x)
    if relu_before_bn:
        block = lambda x, filters: norm(lrelu(conv2d(x, filters)))
    else:
        block = lambda x, filters: lrelu(norm(conv2d(x, filters)))

    inputs = Input(shape=im_shape)
    l1 = block(inputs, 128)
    l2 = block(l1, 256)
    l3 = block(l2, 512)
    l4 = flatten(block(l3, 1024))
    d_out = layers.Dense(1, activation='sigmoid')(l4)

    d_model = Model(inputs=inputs, outputs=d_out)
    opt = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    d_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return d_model


def gan(g_model, d_model):
    d_model.trainable = False
    d_output = d_model(g_model.output)

    gan_model = Model(inputs=g_model.input, outputs=d_output)
    opt = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    gan_model.compile(loss='binary_crossentropy', optimizer=opt)
    return gan_model


def load_real_samples():
    (trainX, _), (_, _) = load_data()
    X = np.expand_dims(trainX, axis=-1)
    X = X.astype('float32')
    X = (X-127.5) / 127.5
    return X


def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))
    return X, y


def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y


def save_plot(examples, epoch, n=10):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, :])
    filename = 'WSI_DCGAN_generated_plot_e%03d.pdf' % (epoch + 1)
    plt.savefig(filename)
    plt.close()


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    save_plot(x_fake, epoch)
    filename = 'WSI_DCGAN_generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # need to separate real and fake minibatches
            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
            # note that the underscore will cause problems if d_model is compiles without accuracy metric
            d_loss, _ = d_model.train_on_batch(X, y)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i + 1, j + 1, bat_per_epo, d_loss, g_loss))
        if (i + 1) % 5 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)
