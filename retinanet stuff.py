import numpy as np, pandas as pd
from tensorflow import losses, optimizers
from tensorflow.keras import Input, Model, models, layers, metrics
from jarvis.train import datasets, custom
from jarvis.train.box import BoundingBox
from jarvis.utils.display import imshow
from tensorflow import keras
import tensorflow as tf
import re


def retinanet_resnet(inputs, K, A):
    # --- Define kwargs dictionary
    kwargs1 = {
        'kernel_size': (1, 1, 1),
        'padding': 'valid'}
    kwargs3 = {
        'kernel_size': (1, 3, 3),
        'padding': 'same'}
    kwargs7 = {
        'kernel_size': (1, 7, 7),
        'padding': 'valid'}
    # --- Define block components
    conv1 = lambda x, filters, strides: layers.Conv3D(filters=filters, strides=strides, **kwargs1)(x)
    conv3 = lambda x, filters, strides: layers.Conv3D(filters=filters, strides=strides, **kwargs3)(x)
    relu = lambda x: layers.LeakyReLU()(x)
    conv7 = lambda x, filters, strides: layers.Conv3D(filters=filters, strides=strides, **kwargs7)(x)
    max_pool = lambda x, pool_size, strides: layers.MaxPooling3D(pool_size=pool_size, strides=strides, padding='valid')(
        x)
    norm = lambda x: layers.BatchNormalization()(x)
    add = lambda x, y: layers.Add()([x, y])
    zeropad = lambda x, padding: layers.ZeroPadding3D(padding=padding)(x)
    upsamp2x = lambda x: layers.UpSampling3D(size=(1, 2, 2))(x)
    # --- Define stride-1, stride-2 blocks
    # conv1 = lambda filters, x : relu(conv(x, filters, strides=1))
    # conv2 = lambda filters, x : relu(conv(x, filters, strides=(2, 2)))
    # --- Residual blocks
    # conv blocks
    conv_1 = lambda filters, x, strides: relu(norm(conv1(x, filters, strides=strides)))
    conv_2 = lambda filters, x: relu(norm(conv3(x, filters, strides=1)))
    conv_3 = lambda filters, x: norm(conv1(x, filters, strides=1))
    conv_sc = lambda filters, x, strides: norm(conv1(x, filters, strides=strides))
    conv_block = lambda filters1, filters2, x, strides: relu(
        add(conv_3(filters2, conv_2(filters1, conv_1(filters1, x, strides))), conv_sc(filters2, x, strides)))
    # identity blocks
    identity_1 = lambda filters, x: relu(norm(conv1(x, filters, strides=1)))
    identity_2 = lambda filters, x: relu(norm(conv3(x, filters, strides=1)))
    identity_3 = lambda filters, x: norm(conv1(x, filters, strides=1))
    identity_block = lambda filters1, filters2, x: relu(
        add(identity_3(filters2, identity_2(filters1, identity_1(filters1, x))), x))
    # --- feature pyramid blocks
    fp_block = lambda x, y: add(upsamp2x(x), conv1(y, 256, strides=1))
    # --- class subnet blocks
    conv_class1 = lambda filters, x: relu(conv3(x, filters, strides=1))
    # --- classification head
    class_subnet = classificaiton_head(K, A)
    # --- regression head
    box_subnet = regression_head(A)
    # --- ResNet-50 backbone
    # stage 1 c2 1/4
    res1 = max_pool(zeropad(relu(norm(conv7(zeropad(inputs['dat'], (0, 3, 3)), 64, strides=(1, 2, 2)))), (0, 1, 1)), (1, 3, 3),
                    strides=(1, 2, 2))
    # stage 2 c2 1/4
    res2 = identity_block(64, 256, identity_block(64, 256, conv_block(64, 256, res1, strides=1)))
    # stage 3 c3 1/8
    res3 = identity_block(128, 512, identity_block(128, 512, identity_block(128, 512, conv_block(128, 512, res2,
                                                                                                 strides=(1, 2, 2)))))
    # stage 4 c4 1/16
    res4 = identity_block(256, 1024, identity_block(256, 1024, identity_block(256, 1024, identity_block(256, 1024,
                                                                                                        identity_block(
                                                                                                            256, 1024,
                                                                                                            conv_block(
                                                                                                                256,
                                                                                                                1024,
                                                                                                                res3,
                                                                                                                strides=(
                                                                                                                1, 2,
                                                                                                                2)))))))
    # stage 5 c5 1/32
    res5 = identity_block(512, 2048, identity_block(512, 2048, conv_block(512, 2048, res4, strides=(1, 2, 2))))
    # --- Feature Pyramid Network architecture
    # p5 1/32
    fp5 = conv1(res5, 256, strides=1)
    # p4 1/16
    fp4 = fp_block(fp5, res4)
    p4 = conv3(fp4, 256, strides=1)
    # p3 1/8
    fp3 = fp_block(fp4, res3)
    p3 = conv3(fp3, 256, strides=1)
    # p6 1/4
    #p6 = conv3(fp5, 256, strides=(2, 2))
    # p7 1/2
    #p7 = conv3(relu(p6), 256, strides=(2, 2))
    feature_pyramid = [p3, p4, fp5]
    # lambda layer that allows multiple outputs from a shared model to have specific names
    # layers.Lambda(lambda x:x, name=name)()
    # --- Class subnet
    class_outputs = [class_subnet(features) for features in feature_pyramid]
    # --- Box subnet
    box_outputs = [box_subnet(features) for features in feature_pyramid]
    # --- put class and box outputs in dictionary
    logits = {'cls-c3': layers.Lambda(lambda x: x, name='cls-c3')(class_outputs[0]),
              'reg-c3': layers.Lambda(lambda x: x, name='reg-c3')(box_outputs[0]),
              'cls-c4': layers.Lambda(lambda x: x, name='cls-c4')(class_outputs[1]),
              'reg-c4': layers.Lambda(lambda x: x, name='reg-c4')(box_outputs[1]),
              'cls-c5': layers.Lambda(lambda x: x, name='cls-c5')(class_outputs[2]),
              'reg-c5': layers.Lambda(lambda x: x, name='reg-c5')(box_outputs[2])}

    model = Model(inputs=inputs, outputs=logits)
    return model


def classificaiton_head(K, A):
    kwargs3 = {
        'kernel_size': (1, 3, 3),
        'padding': 'same'}
    conv3 = lambda x, filters, strides: layers.Conv3D(filters=filters, strides=strides, **kwargs3)(x)
    relu = lambda x: layers.LeakyReLU()(x)
    conv_class1 = lambda filters, x: relu(conv3(x, filters, strides=1))

    inputs = Input(shape=(None, None, None, 256))
    c1 = conv_class1(256, conv_class1(256, conv_class1(256, conv_class1(256, inputs))))
    c2 = conv3(c1, K*A, strides=1)
    class_subnet = Model(inputs=inputs, outputs=c2)
    return class_subnet


def regression_head(A):
    kwargs3 = {
        'kernel_size': (1, 3, 3),
        'padding': 'same'}
    conv3 = lambda x, filters, strides: layers.Conv3D(filters=filters, strides=strides, **kwargs3)(x)
    relu = lambda x: layers.LeakyReLU()(x)
    conv_class1 = lambda filters, x: relu(conv3(x, filters, strides=1))

    inputs = Input(shape=(None, None, None, 256))
    b1 = conv_class1(256, conv_class1(256, conv_class1(256, conv_class1(256, inputs))))
    b2 = conv3(b1, 4*A, strides=1)

    box_subnet = Model(inputs=inputs, outputs=b2)
    return box_subnet


def retinanet(inputs, K, A):
    kwargs = {
        'kernel_size': (1, 3, 3),
        'padding': 'same'
    }
    conv = lambda x, filters, strides : layers.Conv3D(filters=filters, strides=strides, **kwargs)(x)
    norm = lambda x : layers.BatchNormalization()(x)
    relu = lambda x : layers.LeakyReLU()(x)

    conv1 = lambda filters, x: relu(norm(conv(x, filters, strides=1)))
    conv2 = lambda filters, x: relu(norm(conv(x, filters, strides=(1, 2, 2))))

    l1 = conv1(8, inputs['dat'])
    l2 = conv1(16, conv2(16, l1))
    l3 = conv1(24, conv2(24, l2))
    l4 = conv1(32, conv2(32, l3))
    l5 = conv1(48, conv2(48, l4))
    l6 = conv1(64, conv2(64, l5))

    zoom = lambda x: layers.UpSampling3D(
        size=(1, 2, 2))(x)

    proj = lambda filters, x: layers.Conv3D(
        filters=filters,
        strides=1,
        kernel_size=(1, 1, 1),
        padding='same',
        kernel_initializer='he_normal')(x)

    l7 = proj(64, l6)
    l8 = conv1(64, zoom(l7) + proj(64, l5))
    l9 = conv1(64, zoom(l8) + proj(64, l4))

    logits = {}
    K = K
    A = A

    # --- C2
    c3_cls = conv1(64, conv1(64, l9))
    c3_reg = conv1(64, conv1(64, l9))
    logits['cls-c3'] = layers.Conv3D(filters=(A * K), name='cls-c3', **kwargs)(c3_cls)
    logits['reg-c3'] = layers.Conv3D(filters=(A * 4), name='reg-c3', **kwargs)(c3_reg)

    # --- C3
    c4_cls = conv1(64, conv1(64, l8))
    c4_reg = conv1(64, conv1(64, l8))
    logits['cls-c4'] = layers.Conv3D(filters=(A * K), name='cls-c4', **kwargs)(c4_cls)
    logits['reg-c4'] = layers.Conv3D(filters=(A * 4), name='reg-c4', **kwargs)(c4_reg)

    # --- C4
    c5_cls = conv1(64, conv1(64, l7))
    c5_reg = conv1(64, conv1(64, l7))
    logits['cls-c5'] = layers.Conv3D(filters=(A * K), name='cls-c5', **kwargs)(c5_cls)
    logits['reg-c5'] = layers.Conv3D(filters=(A * 4), name='reg-c5', **kwargs)(c5_reg)

    model = Model(inputs=inputs, outputs=logits)
    return model


def generator(data, batchsize = 1):
    i = 0
    keys = data.keys()
    while True:
        if i == (len(data['dat']) // batchsize):
            i = 0
            p = np.random.permutation(len(data['dat']))
            for key in keys:
                data[key] = data[key][p]
        start = i * batchsize
        stop = start + batchsize
        xbatch = {}
        ybatch = {}
        for key in keys:
            if 'dat' in key:
                xbatch[key] = data[key][start:stop]
            elif 'msk' in key:
                xbatch[key] = data[key][start:stop]
            else:
                ybatch[key] = data[key][start:stop]
        i += 1
        yield xbatch, ybatch


def compare_specific_results(data, i, iou_nms, model, boundingbox):
    test_im_x = {}
    test_im_y = {}
    for key in data.keys():
        if 'dat' in key:
            test_im_x[key] = np.expand_dims(data[key][i], axis=0)
        elif 'msk' in key:
            test_im_x[key] = np.expand_dims(data[key][i], axis=0)
        else:
            test_im_y[key] = np.expand_dims(data[key][i], axis=0)
    test1 = model.predict(test_im_x)
    test1_dict = {name: pred for name, pred in zip(model.output_names, test1)}

    # --- Convert box to mask (for visualization)
    msk = boundingbox.convert_box_to_msk(test1_dict, iou_nms=iou_nms, apply_deltas=True)
    imshow(data['dat'][i, :, :, :, 2], msk, title='AI-generated boxes')

    # box = bb.convert_anc_to_box(bbox[300], np.ones((bbox[300].shape[0],1)))
    # --- Convert box to mask (for visualization)
    msk = boundingbox.convert_box_to_msk(test_im_y, apply_deltas=True)
    imshow(data['dat'][i, :, :, :, 2], msk, title='Ground-truth template boxes')
    return test1_dict
