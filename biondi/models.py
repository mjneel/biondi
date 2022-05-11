import numpy as np
from tensorflow.keras import Input, Model, layers, activations
from tensorflow import keras, optimizers
from jarvis.utils.display import imshow
from jarvis.train import custom


def retinanet_resnet50_3d_legacy(inputs, K, A, filter_ratio=1, n=2, include_fc_layer=False, shared_weights=False):
    """Generates retinanet with resnet backbone. Can specify if classification and regression networks share weights"""
    r_model = resnet50_3d(inputs=inputs['dat'],
                          filter_ratio=filter_ratio,
                          n=n,
                          include_fc_layer=include_fc_layer,
                          kernal1=(1, 1, 1),
                          kernal3=(1, 3, 3),
                          kernal7=(1, 7, 7))
    backbone_output = [r_model.get_layer(layer_name).output for layer_name in ['c3-output', 'c4-output', 'c5-output']]
    fp_out = feature_pyramid_3d(inputs=backbone_output,
                                filter_ratio=filter_ratio)
    logits = class_and_reg_subnets(feature_pyramid=fp_out,
                                   K=K,
                                   A=A,
                                   filter_ratio=filter_ratio,
                                   shared_weights=shared_weights)
    model = Model(inputs=inputs, outputs=logits)
    return model


def retinanet_resnet50_3d(inputs, K, A, filter_ratio=1, n=2, include_fc_layer=False, shared_weights=False, tahn=False,
                          lr=2e-4, feature_maps=('c3')):
    """Generates retinanet with resnet backbone. Can specify if classification and regression networks share weights"""
    r_model = resnet50_3d(inputs=inputs['dat'],
                          filter_ratio=filter_ratio,
                          n=n,
                          include_fc_layer=include_fc_layer,
                          kernal1=(1, 1, 1),
                          kernal3=(1, 3, 3),
                          kernal7=(1, 7, 7))
    backbone_output = [r_model.get_layer(layer_name).output for layer_name in ['c3-output', 'c4-output', 'c5-output']]
    fp_out = feature_pyramid_3d(inputs=backbone_output,
                                filter_ratio=filter_ratio)
    logits = class_and_reg_subnets(feature_pyramid=fp_out,
                                   K=K,
                                   A=A,
                                   filter_ratio=filter_ratio,
                                   shared_weights=shared_weights,
                                   tahn=tahn,
                                   feature_maps=feature_maps)
    preds = LogisticEndpoint1()(logits, inputs)
    model = Model(inputs=inputs, outputs=preds)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        experimental_run_tf_function=False,
    )
    return model


def resnet50_3d(inputs, filter_ratio=1, n=2, include_fc_layer=False, logits=True, kernal1=(1, 1, 1), kernal3=(1, 3, 3),
                kernal7=(1, 7, 7), num_layers=None):
    """

    :param inputs: Keras Input object with desire shape
    :type inputs:
    :param filter_ratio:
    :type filter_ratio:
    :param n: # of categories
    :type n: integer
    :param include_fc_layer:
    :type include_fc_layer:
    :return:
    :rtype:
    """
    # --- Define kwargs dictionary
    kwargs1 = {
        'kernel_size': kernal1,
        'padding': 'valid',
    }
    kwargs3 = {
        'kernel_size': kernal3,
        'padding': 'same',
    }
    kwargs7 = {
        'kernel_size': kernal7,
        'padding': 'valid',
    }
    # --- Define block components
    conv1 = lambda x, filters, strides: layers.Conv3D(filters=filters,
                                                      strides=strides,
                                                      **kwargs1)(x)
    conv3 = lambda x, filters, strides: layers.Conv3D(filters=filters,
                                                      strides=strides,
                                                      **kwargs3)(x)
    relu = lambda x: layers.LeakyReLU()(x)
    conv7 = lambda x, filters, strides: layers.Conv3D(filters=filters,
                                                      strides=strides,
                                                      **kwargs7)(x)
    max_pool = lambda x, pool_size, strides: layers.MaxPooling3D(pool_size=pool_size,
                                                                 strides=strides,
                                                                 padding='valid')(x)
    norm = lambda x: layers.BatchNormalization()(x)
    add = lambda x, y: layers.Add()([x, y])
    zeropad = lambda x, padding: layers.ZeroPadding3D(padding=padding)(x)
    # --- Residual blocks
    # conv blocks
    conv_1 = lambda filters, x, strides: relu(norm(conv1(x,
                                                         filters,
                                                         strides=strides)))
    conv_2 = lambda filters, x: relu(norm(conv3(x,
                                                filters,
                                                strides=1)))
    conv_3 = lambda filters, x: norm(conv1(x,
                                           filters,
                                           strides=1))
    conv_sc = lambda filters, x, strides: norm(conv1(x,
                                                     filters,
                                                     strides=strides))
    conv_block = lambda filters1, filters2, x, strides: relu(add(conv_sc(filters2,
                                                                         x,
                                                                         strides),
                                                                 conv_3(filters2,
                                                                        conv_2(filters1,
                                                                               conv_1(filters1,
                                                                                      x,
                                                                                      strides)))))
    # identity blocks
    identity_1 = lambda filters, x: relu(norm(conv1(x,
                                                    filters,
                                                    strides=1)))
    identity_2 = lambda filters, x: relu(norm(conv3(x,
                                                    filters,
                                                    strides=1)))
    identity_3 = lambda filters, x: norm(conv1(x,
                                               filters,
                                               strides=1))
    identity_block = lambda filters1, filters2, x: relu(add(identity_3(filters2,
                                                                       identity_2(filters1,
                                                                                  identity_1(filters1,
                                                                                             x))),
                                                            x))
    # --- ResNet-50 backbone
    # stage 1 c2 1/4
    res1 = max_pool(zeropad(relu(norm(conv7(zeropad(inputs,
                                                    (0, 3, 3)),
                                            int(64 * filter_ratio),
                                            strides=(1, 2, 2)))),
                            (0, 1, 1)),
                    (1, 3, 3),
                    strides=(1, 2, 2))
    # stage 2 c2 1/4
    res2 = layers.Lambda(lambda x: x, name='c2-output')(
        identity_block(int(64 * filter_ratio),
                       int(256 * filter_ratio),
                       identity_block(int(64 * filter_ratio),
                                      int(256 * filter_ratio),
                                      conv_block(int(64 * filter_ratio),
                                                 int(256 * filter_ratio),
                                                 res1,
                                                 strides=1)))
    )
    # stage 3 c3 1/8
    res3 = layers.Lambda(lambda x: x, name='c3-output')(
        identity_block(int(128 * filter_ratio),
                       int(512 * filter_ratio),
                       identity_block(int(128 * filter_ratio),
                                      int(512 * filter_ratio),
                                      identity_block(int(128 * filter_ratio),
                                                     int(512 * filter_ratio),
                                                     conv_block(int(128 * filter_ratio),
                                                                int(512 * filter_ratio),
                                                                res2,
                                                                strides=(1, 2, 2)))))
    )
    # stage 4 c4 1/16
    res4 = layers.Lambda(lambda x: x, name='c4-output')(
        identity_block(int(256 * filter_ratio),
                       int(1024 * filter_ratio),
                       identity_block(int(256 * filter_ratio),
                                      int(1024 * filter_ratio),
                                      identity_block(int(256 * filter_ratio),
                                                     int(1024 * filter_ratio),
                                                     identity_block(int(256 * filter_ratio),
                                                                    int(1024 * filter_ratio),
                                                                    identity_block(int(256 * filter_ratio),
                                                                                   int(1024 * filter_ratio),
                                                                                   conv_block(int(256 * filter_ratio),
                                                                                              int(1024 * filter_ratio),
                                                                                              res3,
                                                                                              strides=(1, 2, 2)))))))
    )
    # stage 5 c5 1/32
    res5 = layers.Lambda(lambda x: x, name='c5-output')(
        identity_block(int(512 * filter_ratio),
                       int(2048 * filter_ratio),
                       identity_block(int(512 * filter_ratio),
                                      int(2048 * filter_ratio),
                                      conv_block(int(512 * filter_ratio),
                                                 int(2048 * filter_ratio),
                                                 res4,
                                                 strides=(1, 2, 2))))
    )
    if num_layers:
        avg_pool = layers.GlobalAveragePooling3D()([res1, res2, res3, res4, res5][num_layers - 1])
    else:
        avg_pool = layers.GlobalAveragePooling3D()(res5)
    flatten = layers.Flatten()(avg_pool)
    if logits:
        logits = layers.Dense(n)(flatten)
    else:
        logits = layers.Dense(n, activation='softmax')
    if include_fc_layer:
        model = Model(inputs=inputs, outputs=logits)
    else:
        model = Model(inputs=inputs, outputs=res5)
    return model


def feature_pyramid_3d(inputs, filter_ratio):
    kwargs1 = {
        'kernel_size': (1, 1, 1),
        'padding': 'valid',
    }
    kwargs3 = {
        'kernel_size': (1, 3, 3),
        'padding': 'same',
    }
    conv1 = lambda x, filters, strides: layers.Conv3D(filters=filters, strides=strides, **kwargs1)(x)
    add = lambda x, y: layers.Add()([x, y])
    upsamp2x = lambda x: layers.UpSampling3D(size=(1, 2, 2))(x)
    fp_block = lambda x, y: add(upsamp2x(x), conv1(y, int(256 * filter_ratio), strides=1))
    conv3 = lambda x, filters, strides: layers.Conv3D(filters=filters, strides=strides, **kwargs3)(x)
    relu = lambda x: layers.LeakyReLU()(x)

    p5 = conv1(inputs[2], int(256 * filter_ratio), strides=1)
    fp4 = fp_block(p5, inputs[1])
    p4 = conv3(fp4, int(256 * filter_ratio), strides=1)
    fp3 = fp_block(fp4, inputs[0])
    p3 = conv3(fp3, int(256 * filter_ratio), strides=1)
    p6 = conv3(p5, int(256 * filter_ratio), strides=(1, 2, 2))
    p7 = conv3(relu(p6), int(256 * filter_ratio), strides=(1, 2, 2))
    return [p3, p4, p5, p6, p7]


def class_and_reg_subnets(feature_pyramid, K, A, filter_ratio, shared_weights=False, tahn=False, feature_maps=('c3', 'c4', 'c5'),):
    if shared_weights:
        class_subnet = classification_head(K, A, filter_ratio)
        box_subnet = regression_head(A, filter_ratio, tahn=tahn)
        class_outputs = [class_subnet(features) for features in feature_pyramid]
        box_outputs = [box_subnet(features) for features in feature_pyramid]
        logits = {}
        if 'c3' in feature_maps:
            logits['cls-c3'] = layers.Lambda(lambda x: x, name='cls-c3')(class_outputs[0])
            logits['reg-c3'] = layers.Lambda(lambda x: x, name='reg-c3')(box_outputs[0])
        if 'c4' in feature_maps:
            logits['cls-c4'] = layers.Lambda(lambda x: x, name='cls-c4')(class_outputs[1])
            logits['reg-c4'] = layers.Lambda(lambda x: x, name='reg-c4')(box_outputs[1])
        if 'c5' in feature_maps:
            logits['cls-c5'] = layers.Lambda(lambda x: x, name='cls-c5')(class_outputs[2])
            logits['reg-c5'] = layers.Lambda(lambda x: x, name='reg-c5')(box_outputs[2])
        return logits
    else:
        class_models = []
        reg_models = []
        for _ in feature_pyramid:
            class_models.append(classification_head(K, A, filter_ratio))
            reg_models.append(regression_head(A, filter_ratio, tahn=tahn))
        logits = {}
        if 'c3' in feature_maps:
            logits['cls-c3'] = layers.Lambda(lambda x: x, name='cls-c3')(class_models[0](feature_pyramid[0]))
            logits['reg-c3'] = layers.Lambda(lambda x: x, name='reg-c3')(reg_models[0](feature_pyramid[0]))
        if 'c4' in feature_maps:
            logits['cls-c4'] = layers.Lambda(lambda x: x, name='cls-c4')(class_models[1](feature_pyramid[1]))
            logits['reg-c4'] = layers.Lambda(lambda x: x, name='reg-c4')(reg_models[1](feature_pyramid[1]))
        if 'c5' in feature_maps:
            logits['cls-c5'] = layers.Lambda(lambda x: x, name='cls-c5')(class_models[2](feature_pyramid[2]))
            logits['reg-c5'] = layers.Lambda(lambda x: x, name='reg-c5')(reg_models[2](feature_pyramid[2]))
        return logits


def retinanet_resnet_2(inputs, K, A):
    """Retinanet with resnet backbone. Classification and regression networks have different weights for each feature
    pyramid layer"""
    # --- Define kwargs dictionary
    kwargs1 = {
        'kernel_size': (1, 1, 1),
        'padding': 'valid',
    }
    kwargs3 = {
        'kernel_size': (1, 3, 3),
        'padding': 'same',
    }
    kwargs7 = {
        'kernel_size': (1, 7, 7),
        'padding': 'valid',
    }
    # --- Define block components
    conv1 = lambda x, filters, strides: layers.Conv3D(filters=filters,
                                                      strides=strides,
                                                      **kwargs1)(x)
    conv3 = lambda x, filters, strides: layers.Conv3D(filters=filters,
                                                      strides=strides,
                                                      **kwargs3)(x)
    relu = lambda x: layers.LeakyReLU()(x)
    conv7 = lambda x, filters, strides: layers.Conv3D(filters=filters,
                                                      strides=strides,
                                                      **kwargs7)(x)
    max_pool = lambda x, pool_size, strides: layers.MaxPooling3D(pool_size=pool_size,
                                                                 strides=strides,
                                                                 padding='valid')(x)
    norm = lambda x: layers.BatchNormalization()(x)
    add = lambda x, y: layers.Add()([x, y])
    zeropad = lambda x, padding: layers.ZeroPadding3D(padding=padding)(x)
    upsamp2x = lambda x: layers.UpSampling3D(size=(1, 2, 2))(x)
    # --- Define stride-1, stride-2 blocks
    # conv1 = lambda filters, x : relu(conv(x, filters, strides=1))
    # conv2 = lambda filters, x : relu(conv(x, filters, strides=(2, 2)))
    # --- Residual blocks
    # conv blocks
    conv_1 = lambda filters, x, strides: relu(norm(conv1(x,
                                                         filters,
                                                         strides=strides)))
    conv_2 = lambda filters, x: relu(norm(conv3(x,
                                                filters,
                                                strides=1)))
    conv_3 = lambda filters, x: norm(conv1(x,
                                           filters,
                                           strides=1))
    conv_sc = lambda filters, x, strides: norm(conv1(x,
                                                     filters,
                                                     strides=strides))
    conv_block = lambda filters1, filters2, x, strides: relu(add(conv_3(filters2,
                                                                        conv_2(filters1,
                                                                               conv_1(filters1,
                                                                                      x,
                                                                                      strides))),
                                                                 conv_sc(filters2,
                                                                         x,
                                                                         strides)))
    # identity blocks
    identity_1 = lambda filters, x: relu(norm(conv1(x,
                                                    filters,
                                                    strides=1)))
    identity_2 = lambda filters, x: relu(norm(conv3(x,
                                                    filters,
                                                    strides=1)))
    identity_3 = lambda filters, x: norm(conv1(x,
                                               filters,
                                               strides=1))
    identity_block = lambda filters1, filters2, x: relu(add(identity_3(filters2,
                                                                       identity_2(filters1,
                                                                                  identity_1(filters1,
                                                                                             x))),
                                                            x))
    # --- feature pyramid blocks
    fp_block = lambda x, y: add(upsamp2x(x), conv1(y,
                                                   256,
                                                   strides=1))
    # --- class subnet blocks
    conv_class1 = lambda filters, x: relu(conv3(x,
                                                filters,
                                                strides=1))
    # --- classification head
    class_subnet3 = classification_head(K, A)
    class_subnet4 = classification_head(K, A)
    class_subnet5 = classification_head(K, A)
    # --- regression head
    box_subnet3 = regression_head(A)
    box_subnet4 = regression_head(A)
    box_subnet5 = regression_head(A)
    # --- ResNet-50 backbone
    # stage 1 c2 1/4
    res1 = max_pool(zeropad(relu(norm(conv7(zeropad(inputs['dat'],
                                                    (0, 3, 3)),
                                            64,
                                            strides=(1, 2, 2)))),
                            (0, 1, 1)),
                    (1, 3, 3),
                    strides=(1, 2, 2))
    # stage 2 c2 1/4
    res2 = identity_block(64,
                          256,
                          identity_block(64,
                                         256,
                                         conv_block(64,
                                                    256,
                                                    res1,
                                                    strides=1)))
    # stage 3 c3 1/8
    res3 = identity_block(128,
                          512,
                          identity_block(128,
                                         512,
                                         identity_block(128,
                                                        512,
                                                        conv_block(128,
                                                                   512,
                                                                   res2,
                                                                   strides=(1, 2, 2)))))
    # stage 4 c4 1/16
    res4 = identity_block(256,
                          1024,
                          identity_block(256,
                                         1024,
                                         identity_block(256,
                                                        1024,
                                                        identity_block(256,
                                                                       1024,
                                                                       identity_block(256,
                                                                                      1024,
                                                                                      conv_block(256,
                                                                                                 1024,
                                                                                                 res3,
                                                                                                 strides=(1, 2, 2)))))))
    # stage 5 c5 1/32
    res5 = identity_block(512,
                          2048,
                          identity_block(512,
                                         2048,
                                         conv_block(512,
                                                    2048,
                                                    res4,
                                                    strides=(1, 2, 2))))
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
    # p6 = conv3(fp5, 256, strides=(2, 2))
    # p7 1/2
    # p7 = conv3(relu(p6), 256, strides=(2, 2))
    # feature_pyramid = [p3, p4, fp5]
    # lambda layer that allows multiple outputs from a shared model to have specific names
    # layers.Lambda(lambda x:x, name=name)()
    # --- Class subnet
    # class_outputs = [class_subnet(features) for features in feature_pyramid]
    # --- Box subnet
    # box_outputs = [box_subnet(features) for features in feature_pyramid]
    # --- put class and box outputs in dictionary
    logits = {'cls-c3': layers.Lambda(lambda x: x, name='cls-c3')(class_subnet3(p3)),
              'reg-c3': layers.Lambda(lambda x: x, name='reg-c3')(box_subnet3(p3)),
              'cls-c4': layers.Lambda(lambda x: x, name='cls-c4')(class_subnet4(p4)),
              'reg-c4': layers.Lambda(lambda x: x, name='reg-c4')(box_subnet4(p4)),
              'cls-c5': layers.Lambda(lambda x: x, name='cls-c5')(class_subnet5(fp5)),
              'reg-c5': layers.Lambda(lambda x: x, name='reg-c5')(box_subnet5(fp5))}

    model = Model(inputs=inputs, outputs=logits)
    return model


def retinanet_resnet(inputs, K, A):
    """Retinanet with resnet backbone. Classification and regression networks share weights across feature pyramid
     layers"""
    # --- Define kwargs dictionary
    kwargs1 = {
        'kernel_size': (1, 1, 1),
        'padding': 'valid',
    }
    kwargs3 = {
        'kernel_size': (1, 3, 3),
        'padding': 'same',
    }
    kwargs7 = {
        'kernel_size': (1, 7, 7),
        'padding': 'valid',
    }
    # --- Define block components
    conv1 = lambda x, filters, strides: layers.Conv3D(filters=filters,
                                                      strides=strides,
                                                      **kwargs1)(x)
    conv3 = lambda x, filters, strides: layers.Conv3D(filters=filters,
                                                      strides=strides,
                                                      **kwargs3)(x)
    relu = lambda x: layers.LeakyReLU()(x)
    conv7 = lambda x, filters, strides: layers.Conv3D(filters=filters,
                                                      strides=strides,
                                                      **kwargs7)(x)
    max_pool = lambda x, pool_size, strides: layers.MaxPooling3D(pool_size=pool_size,
                                                                 strides=strides,
                                                                 padding='valid')(x)
    norm = lambda x: layers.BatchNormalization()(x)
    add = lambda x, y: layers.Add()([x, y])
    zeropad = lambda x, padding: layers.ZeroPadding3D(padding=padding)(x)
    upsamp2x = lambda x: layers.UpSampling3D(size=(1, 2, 2))(x)
    # --- Define stride-1, stride-2 blocks
    # conv1 = lambda filters, x : relu(conv(x, filters, strides=1))
    # conv2 = lambda filters, x : relu(conv(x, filters, strides=(2, 2)))
    # --- Residual blocks
    # conv blocks
    conv_1 = lambda filters, x, strides: relu(norm(conv1(x,
                                                         filters,
                                                         strides=strides)))
    conv_2 = lambda filters, x: relu(norm(conv3(x,
                                                filters,
                                                strides=1)))
    conv_3 = lambda filters, x: norm(conv1(x,
                                           filters,
                                           strides=1))
    conv_sc = lambda filters, x, strides: norm(conv1(x,
                                                     filters,
                                                     strides=strides))
    conv_block = lambda filters1, filters2, x, strides: relu(add(conv_3(filters2,
                                                                        conv_2(filters1,
                                                                               conv_1(filters1,
                                                                                      x,
                                                                                      strides))),
                                                                 conv_sc(filters2,
                                                                         x,
                                                                         strides)))
    # identity blocks
    identity_1 = lambda filters, x: relu(norm(conv1(x,
                                                    filters,
                                                    strides=1)))
    identity_2 = lambda filters, x: relu(norm(conv3(x,
                                                    filters,
                                                    strides=1)))
    identity_3 = lambda filters, x: norm(conv1(x,
                                               filters,
                                               strides=1))
    identity_block = lambda filters1, filters2, x: relu(add(identity_3(filters2,
                                                                       identity_2(filters1,
                                                                                  identity_1(filters1,
                                                                                             x))),
                                                            x))
    # --- feature pyramid blocks
    fp_block = lambda x, y: add(upsamp2x(x), conv1(y, 256, strides=1))
    # --- classification head
    class_subnet = classification_head(K, A)
    # --- regression head
    box_subnet = regression_head(A)
    # --- ResNet-50 backbone
    # stage 1 c2 1/4
    res1 = max_pool(zeropad(relu(norm(conv7(zeropad(inputs['dat'],
                                                    (0, 3, 3)),
                                            64,
                                            strides=(1, 2, 2)))),
                            (0, 1, 1)),
                    (1, 3, 3),
                    strides=(1, 2, 2))
    # stage 2 c2 1/4
    res2 = identity_block(64,
                          256,
                          identity_block(64,
                                         256,
                                         conv_block(64,
                                                    256,
                                                    res1,
                                                    strides=1)))
    # stage 3 c3 1/8
    res3 = identity_block(128,
                          512,
                          identity_block(128,
                                         512,
                                         identity_block(128,
                                                        512,
                                                        conv_block(128,
                                                                   512,
                                                                   res2,
                                                                   strides=(1, 2, 2)))))
    # stage 4 c4 1/16
    res4 = identity_block(256,
                          1024,
                          identity_block(256,
                                         1024,
                                         identity_block(256,
                                                        1024,
                                                        identity_block(256,
                                                                       1024,
                                                                       identity_block(256,
                                                                                      1024,
                                                                                      conv_block(256,
                                                                                                 1024,
                                                                                                 res3,
                                                                                                 strides=(1, 2, 2)))))))
    # stage 5 c5 1/32
    res5 = identity_block(512,
                          2048,
                          identity_block(512,
                                         2048,
                                         conv_block(512,
                                                    2048,
                                                    res4,
                                                    strides=(1, 2, 2))))
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
    # p6 = conv3(fp5, 256, strides=(2, 2))
    # p7 1/2
    # p7 = conv3(relu(p6), 256, strides=(2, 2))
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


def classification_head(K, A, filter_ratio=1):
    kwargs3 = {
        'kernel_size': (1, 3, 3),
        'padding': 'same',
    }
    conv3 = lambda x, filters, strides: layers.Conv3D(filters=filters,
                                                      strides=strides,
                                                      **kwargs3)(x)
    relu = lambda x: layers.LeakyReLU()(x)
    conv_class1 = lambda filters, x: relu(conv3(x,
                                                filters,
                                                strides=1))

    inputs = Input(shape=(None, None, None, int(256 * filter_ratio)))
    c1 = conv_class1(int(256 * filter_ratio),
                     conv_class1(int(256 * filter_ratio),
                                 conv_class1(int(256 * filter_ratio),
                                             conv_class1(int(256 * filter_ratio),
                                                         inputs))))
    c2 = conv3(c1, K * A, strides=1)
    class_subnet = Model(inputs=inputs, outputs=c2)
    return class_subnet


def regression_head(A, filter_ratio=1, tahn=False):
    kwargs3 = {
        'kernel_size': (1, 3, 3),
        'padding': 'same',
    }
    conv3 = lambda x, filters, strides: layers.Conv3D(filters=filters,
                                                      strides=strides,
                                                      **kwargs3)(x)
    # added tahn activation
    conv3_final = lambda x, filters, strides: layers.Conv3D(filters=filters,
                                                            strides=strides,
                                                            activation=activations.tanh,
                                                            **kwargs3)(x)
    relu = lambda x: layers.LeakyReLU()(x)
    conv_class1 = lambda filters, x: relu(conv3(x,
                                                filters,
                                                strides=1))

    inputs = Input(shape=(None, None, None, int(256 * filter_ratio)))
    b1 = conv_class1(int(256 * filter_ratio),
                     conv_class1(int(256 * filter_ratio),
                                 conv_class1(int(256 * filter_ratio),
                                             conv_class1(int(256 * filter_ratio),
                                                         inputs))))
    # changed to use tahn activation
    if tahn:
        b2 = conv3_final(b1, 4 * A, strides=1)
    else:
        b2 = conv3(b1, 4 * A, strides=1)

    box_subnet = Model(inputs=inputs, outputs=b2)
    return box_subnet


def retinanet(inputs, K, A):
    """Retinanet architecture with peter's simple backbone architecture."""
    kwargs = {
        'kernel_size': (1, 3, 3),
        'padding': 'same'
    }
    conv = lambda x, filters, strides: layers.Conv3D(filters=filters,
                                                     strides=strides,
                                                     **kwargs)(x)
    norm = lambda x: layers.BatchNormalization()(x)
    relu = lambda x: layers.LeakyReLU()(x)

    conv1 = lambda filters, x: relu(norm(conv(x,
                                              filters,
                                              strides=1)))
    conv2 = lambda filters, x: relu(norm(conv(x,
                                              filters,
                                              strides=(1, 2, 2))))

    l1 = conv1(8, inputs['dat'])
    l2 = conv1(16, conv2(16, l1))
    l3 = conv1(24, conv2(24, l2))
    l4 = conv1(32, conv2(32, l3))
    l5 = conv1(48, conv2(48, l4))
    l6 = conv1(64, conv2(64, l5))

    zoom = lambda x: layers.UpSampling3D(size=(1, 2, 2))(x)

    proj = lambda filters, x: layers.Conv3D(filters=filters,
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


def generator(data, batchsize=1):
    """Data generator for retinanet"""
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


def unet(inputs, filter_ratio=1, logits_num=2, num_layers=6, class_num=1, _3d=False, compile=False, lr=2e-4):
    # --- Define kwargs dictionary
    if _3d:
        kwargs = {
            'kernel_size': (3, 3, 3),
            'padding': 'same'}
    else:
        kwargs = {
            'kernel_size': (1, 3, 3),
            'padding': 'same'}

    # --- Define lambda functions
    conv = lambda x, filters, strides: layers.Conv3D(filters=filters,
                                                     strides=strides,
                                                     **kwargs)(x)
    norm = lambda x: layers.BatchNormalization()(x)
    relu = lambda x: layers.LeakyReLU()(x)
    tran = lambda x, filters, strides: layers.Conv3DTranspose(filters=filters,
                                                              strides=strides,
                                                              **kwargs)(x)

    # --- Define stride-1, stride-2 blocks
    conv1 = lambda filters, x: relu(norm(conv(x, filters, strides=1)))
    conv2 = lambda filters, x: relu(norm(conv(x, filters, strides=(1, 2, 2))))
    tran2 = lambda filters, x: relu(norm(tran(x, filters, strides=(1, 2, 2))))

    # --- Define simple layers
    c_layer = lambda filters, x: conv1(filters, conv2(filters, x))
    e_layer = lambda filters1, filters2, x: tran2(filters1, conv1(filters2, x))

    contracting_layers = []
    for i in range(num_layers):  # 0,1,2,3,4,5
        if i == 0:
            contracting_layers.append(conv1(int(4*filter_ratio), inputs))
        else:
            contracting_layers.append(c_layer(int(8*filter_ratio)*i, contracting_layers[i-1]))
    expanding_layers = []
    for j in reversed(range(num_layers - 1)):  # 4,3,2,1,0
        if j == num_layers - 2:
            expanding_layers.append(tran2(int(8 * filter_ratio) * j, contracting_layers[j + 1]))
        else:
            expanding_layers.append(e_layer(int(8 * filter_ratio) * j if j != 0 else int(4 * filter_ratio),
                                            int(8 * filter_ratio) * (j + 1),
                                            expanding_layers[-1] + contracting_layers[j + 1]))
        last_layer = conv1(int(4 * filter_ratio),
                           conv1(int(4 * filter_ratio), expanding_layers[-1] + contracting_layers[0]))

    # --- Create logits
    logits = {}
    for k in range(class_num):
        logits[f'zones{k}'] = layers.Conv3D(filters=logits_num, name=f'zones{k}', **kwargs)(last_layer)

    # --- Create model
    model = Model(inputs=inputs, outputs=logits)
    if compile:
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss_weights={i: keras.losses.SparseCategoricalCrossentropy(from_logits=True) for i in model.output_names},
            metrics={i: custom.dsc(cls=1) for i in model.output_names},
            # TODO: Check if leaving this parameter out affects model training.
            experimental_run_tf_function=False,
        )
    return model


class LogisticEndpoint1(keras.layers.Layer):
    def __init__(self, name=None):
        super(LogisticEndpoint1, self).__init__(name=name)
        self.loss1_fn = custom.focal_sigmoid_ce
        self.loss2_fn = custom.sl1
        self.ppv_fn = custom.sigmoid_ce_ppv()
        self.sens_fn = custom.sigmoid_ce_sens()
    def call(self, logits, targets=None):
        if targets is not None:
            # Compute the training-time loss value and add it
            # to the layer using `self.add_loss()`.
            loss1 = self.loss1_fn(targets['cls-c3-msk'])(targets['cls-c3'], logits['cls-c3'])
            loss2 = self.loss2_fn(targets['reg-c3-msk'])(targets['reg-c3'], logits['reg-c3'])
            self.add_loss(loss1)
            self.add_loss(loss2)

            # Log the accuracy as a metric (we could log arbitrary metrics,
            # including different metrics for training and inference.
            metric1 = custom.sigmoid_ce_ppv()(y_true=targets['cls-c3'], y_pred=logits['cls-c3'])
            metric2 = custom.sigmoid_ce_sens()(y_true=targets['cls-c3'], y_pred=logits['cls-c3'])
            #self.add_metric(self.accuracy_fn(targets, logits, sample_weight))
            self.add_metric(metric1, name='cls-c3_ppv', aggregation='mean')
            self.add_metric(metric2, name='cls-c3_sens', aggregation='mean')

        # Return the inference-time prediction tensor (for `.predict()`).
        return logits
