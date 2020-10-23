import numpy as np
import pylab
import tensorflow as tf
from tensorflow import losses, optimizers, nn, keras
from tensorflow.keras import Input, Model, models, layers, metrics
import os
import skimage
from skimage import transform
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model, model_to_dot
import h5py
from tensorflow.keras.datasets.mnist import load_data
import math


def discriminator(n_cat):
    conv2d = lambda x, filters, strides: layers.Conv2D(filters, (4, 4), strides=strides, padding='same')(x)
    leakyrelu = lambda x: layers.LeakyReLU(alpha=0.2)(x)
    norm = lambda x: layers.BatchNormalization()(x)
    dense = lambda x, nodes: layers.Dense(nodes)(x)
    flatten = lambda x : layers.Flatten()(x)

    inputs = Input(shape=(64, 64, 3))
    l1 = leakyrelu(conv2d(inputs, 64, (2, 2)))
    l2 = norm(leakyrelu(conv2d(l1, 128, (2, 2))))
    l3 = norm(leakyrelu(conv2d(l2, 256, (2, 2))))
    l4 = norm(leakyrelu(conv2d(l3, 256, (1, 1))))
    l5 = norm(leakyrelu(conv2d(l4, 256, (1, 1))))
    l6 = norm(leakyrelu(dense(flatten(l5), 1024)))
    d_out = layers.Dense(1, activation='sigmoid')(l6)

    # not sure if i should include the continuous variables.
    # for now, I will not implement continuous variables to keep things simple.
    aux1 = leakyrelu(norm(dense(l6, 128)))
    q_out = dense(aux1, n_cat)
    d_model = Model(inputs=inputs, outputs=d_out)
    opt = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    d_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    q_model = Model(inputs=inputs, outputs=q_out)
    return d_model, q_model


def generator():
    relu = lambda x: layers.ReLU()(x)
    norm = lambda x: layers.BatchNormalization()(x)
    trans2d = lambda x, filters, strides: layers.Conv2DTranspose(filters, (4, 4), strides=strides, padding='same')(x)
    dense = lambda x, nodes: layers.Dense(nodes)(x)

    inputs = Input(shape=gen_input_size)
    l1 = norm(relu(dense(inputs, 1024)))
    l2 = norm(relu(dense(l1, 8 * 8 * 256)))
    l3 = layers.Reshape((8,8,256))(l2)
    l4 = norm(relu(trans2d(l3, 256, (1, 1))))
    l5 = norm(relu(trans2d(l4, 256, (1, 1))))
    l6 = norm(relu(trans2d(l5, 128, (2, 2))))
    l7 = norm(relu(trans2d(l6, 64, (2, 2))))
    g_out = layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='sigmoid')(l7)

    model = Model(inputs=inputs, outputs=g_out)
    return model


def gan(g_model, d_model, q_model):
    d_model.trainable = False
    d_output = d_model(g_model.output)
    q_output = q_model(g_model.output)

    model = Model(inputs=g_model.input, outputs=[d_output, q_output])

    opt = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=opt)
    return model


def load_real_samples():
    (trainX, _), (_, _) = load_data()
    X = np.expand_dims(trainX, axis=-1)
    X = X.astype('float32')
    X = X / 255.0
    return X


def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))
    return X, y


def generate_latent_points(latent_dim, n_samples, n_cat):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    cat_codes = np.random.randint(0, n_cat, n_samples)
    cat_codes = keras.utils.to_categorical(cat_codes, num_classes=n_cat)
    x_input = np.hstack((x_input, cat_codes))
    return [x_input, cat_codes]


def generate_fake_samples(g_model, latent_dim, n_samples, n_cat):
    x_input = generate_latent_points(latent_dim, n_samples, n_cat)
    X = g_model.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y


def save_plot(examples, epoch, n=10):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, :])
    filename = 'batchnorm_generated_plot_e%03d.png' % (epoch + 1)
    plt.savefig(filename)
    plt.close()


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples, n_cat)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    save_plot(x_fake, epoch)
    filename = 'batchnorm_generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch, n_cat)
            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
            d_loss, _ = d_model.train_on_batch(X, y)
            X_gan, cat_codes = generate_latent_points(latent_dim, n_batch, n_cat)
            y_gan = np.ones((n_batch, 1))
            _, g_loss, q_loss = gan_model.train_on_batch(X_gan, [y_gan, cat_codes])
            print('>%d, %d/%d, d=%.3f, g=%.3f, q=%.3f' % (i + 1, j + 1, bat_per_epo, d_loss, g_loss, q_loss))
        if (i + 1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


def generate_latent_points2(latent_dim, n_cat, n_samples, digit):
    # generate points in the latent space
    z_latent = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_latent = z_latent.reshape(n_samples, latent_dim)
    # define categorical codes
    cat_codes = np.asarray([digit for _ in range(n_samples)])
    # one hot encode
    cat_codes = keras.utils.to_categorical(cat_codes, num_classes=n_cat)
    # concatenate latent points and control codes
    z_input = np.hstack((z_latent, cat_codes))
    return [z_input, cat_codes]


# create and save a plot of generated images
def save_plot2(examples, n_examples):
    # plot images
    for i in range(n_examples):
        # define subplot
        plt.subplot(math.sqrt(n_examples), math.sqrt(n_examples), 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    plt.show()


n_cat = 2
latent_dim = 128
gen_input_size = latent_dim + n_cat
# create the discriminator
d_model, q_model = discriminator(n_cat=10)
# create the generator
g_model = generator()
# create the gan
gan_model = gan(g_model, d_model, q_model)
# load image data
dataset = load_real_samples()

train(g_model, d_model, gan_model, dataset, latent_dim)

digit = 9
n_samples2 = 25
inputs, _ = generate_latent_points2(latent_dim, n_cat, n_samples2, digit)
X = g_model.predict(inputs)
save_plot2(X, n_samples2)
