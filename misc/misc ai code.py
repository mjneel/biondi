from tensorflow import losses, optimizers, nn, keras
from tensorflow.keras import Input, Model, models, layers, metrics
import os
import skimage
from skimage import transform
import matplotlib.pyplot as plt
import h5py
from tensorflow.keras.utils import plot_model, model_to_dot
import numpy as np
import pylab


def generate_3D_VGG(L2_constant, dropout, learning_rate, kernel_initializer='glorot_uniform', model_summary=True):
    # --- Define kwargs dictionary
    kwargs = {
        'kernel_size': (3, 3, 3),
        'padding': 'same',
        'kernel_regularizer': keras.regularizers.l2(L2_constant),
        'kernel_initializer': kernel_initializer}

    # --- Define block components
    conv = lambda x, filters, strides: layers.Conv3D(filters=filters, strides=strides, **kwargs)(x)
    relu = lambda x: layers.LeakyReLU()(x)
    norm = lambda x: layers.BatchNormalization()(x)
    # --- Define stride-1, stride-2 blocks
    conv1 = lambda filters, x: relu(norm(conv(x, filters, strides=1)))
    conv2 = lambda filters, x: relu(norm(conv(x, filters, strides=(2, 2, 2))))

    # --- Architecture
    inputs = Input(shape=(64, 64, 64, 3))
    l1 = conv2(16, conv1(16, inputs))
    l2 = conv2(32, conv1(32, l1))
    l3 = conv2(64, conv1(64, l2))
    l4 = conv2(128, conv1(128, l3))
    f0 = layers.Flatten()(l4)
    h1 = relu(layers.Dense(128)(f0))
    h2 = relu(norm(h1))
    h2 = layers.Dropout(dropout)
    logits = layers.Dense(2)(h1)

    model = Model(inputs=inputs, outputs=logits)

    if model_summary:
        model.summary()

    # --- Define a categorical cross-entropy loss
    loss = losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    # --- Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'])

    return model


def test_hyperparameters(dropout, LR, L2_reg, batch_size, epochs):
    model = generate_3D_VGG(L2_reg, dropout, LR, model_summary=False)
    hist = model.fit(x1, y, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x1_test, y_test))
    return hist


def grid_search_optimizaiton(dropout, LR, L2_reg, batch_size, epochs):
    history = []
    parameters = []
    import itertools
    for i, j, k, l in itertools.product(batch_size, dropout, LR, L2_reg):
        hist = test_hyperparameters(j, k, l, i, epochs)
        history.append(hist.history)
        np.save('properly_shuffled_WSI_vgg_' + str(j) + '_' + str(k) + '_' + str(l) + '_' + str(i) + '.npy',
                hist.history)
        parameters.append([i, j, k, l])
        print('batch size', i, 'dropout', j, 'learning rate', k, 'L2_constant', l)
        plot_training(hist)
    legend_acc1 = []
    print('batch_size', 'initializer', 'learning_rate', 'L2_constant')
    for m in range(len(history)):
        plt.plot(history[m]['accuracy'])
        i, j, k, l = parameters[m]
        legend_acc1.append(str(i) + ',' + str(j) + ',' + str(k) + ',' + str(l))
    plt.title('Model train accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(legend_acc1, loc='best')
    plt.show()

    legend_acc2 = []
    for m in range(len(history)):
        plt.plot(history[m]['val_accuracy'])
        i, j, k, l = parameters[m]
        legend_acc2.append(str(i) + ',' + str(j) + ',' + str(k) + ',' + str(l))
    plt.title('Model val accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(legend_acc2, loc='best')
    plt.show()

    legend_acc3 = []
    for m in range(len(history)):
        plt.plot(history[m]['loss'])
        i, j, k, l = parameters[m]
        legend_acc3.append(str(i) + ',' + str(j) + ',' + str(k) + ',' + str(l))
    plt.title('Model train loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(legend_acc3, loc='best')
    plt.show()

    legend_acc4 = []
    for m in range(len(history)):
        plt.plot(history[m]['val_loss'])
        i, j, k, l = parameters[m]
        legend_acc4.append(str(i) + ',' + str(j) + ',' + str(k) + ',' + str(l))
    plt.title('Model val loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(legend_acc4, loc='best')
    plt.show()
    return history, parameters


def ai_comparison_graphs(history, parameters):
    legend_acc1 = []
    print('batch_size', 'initializer', 'learning_rate', 'L2_constant')
    for m in range(len(history)):
        plt.plot(history[m]['accuracy'])
        i, j, k, l = parameters[m]
        legend_acc1.append(str(i) + ',' + str(j) + ',' + str(k) + ',' + str(l))
    plt.title('Model train accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(legend_acc1, loc='best')
    plt.show()

    legend_acc2 = []
    for m in range(len(history)):
        plt.plot(history[m]['val_accuracy'])
        i, j, k, l = parameters[m]
        legend_acc2.append(str(i) + ',' + str(j) + ',' + str(k) + ',' + str(l))
    plt.title('Model val accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(legend_acc2, loc='best')
    plt.show()

    legend_acc3 = []
    for m in range(len(history)):
        plt.plot(history[m]['loss'])
        i, j, k, l = parameters[m]
        legend_acc3.append(str(i) + ',' + str(j) + ',' + str(k) + ',' + str(l))
    plt.title('Model train loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(legend_acc3, loc='best')
    plt.show()

    legend_acc4 = []
    for m in range(len(history)):
        plt.plot(history[m]['val_loss'])
        i, j, k, l = parameters[m]
        legend_acc4.append(str(i) + ',' + str(j) + ',' + str(k) + ',' + str(l))
    plt.title('Model val loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(legend_acc4, loc='best')
    plt.show()

