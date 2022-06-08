'''
This is the ResNet implementation in TensorFlow.
Reference: https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/TensorFlow/CNN_architectures/ResNet
'''
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Activation,
    Add,
    BatchNormalization,
    Input,
    ZeroPadding2D,
    Flatten,
    GlobalAveragePooling2D,
    AveragePooling2D,
    MaxPooling2D
)
from tensorflow.keras.models import Model
import typing

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@tf.function
def residual_block(
    X: tf.Tensor,
    filters: typing.List[int],
    base_name: str,
    downsample: bool) -> tf.Tensor:

    # Filters
    F1, F2 = filters
    x = Conv2D(filters=F1, kernel_size=3, strides=1, padding='same', name=f'{base_name}_a')
    x = BatchNormalization(name=f'bn_{base_name}_a')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=F2, kernel_size=3, strides=1, padding='same', name=f'{base_name}_b')
    x = BatchNormalization(name=f'bn_{base_name}_b')
    x = Activation('relu')(x)
    if downsample:
        residual = Conv2D(filters=F2, kernel_size=3, strides=1, padding='same', name=f'{base_name}_downsample')(x)
        x = Add()([residual, x])
        x = BatchNormalization(name=f'bn_{base_name}_downsample')(x)
        x = Activation('relu')(x)
    return x

def _make_layer(
    X: tf.Tensor,
    base_name: str,
    filters: typing.List[int],
    layers: int) -> tf.Tensor:
    curr_filters = X.shape[-1]
    t = 0
    if curr_filters != filters[0]:
        X = residual_block(X,
            filters=filters,
            base_name=f'{base_name}_{t}',
            downsample=True)
        t += 1
    else:
        X = residual_block(X,
            filters=filters,
            base_name=f'{base_name}_{t}',
            downsample=False)
        t += 1
    for _ in range(layers - 1):
        X = residual_block(X, base_name=f'{base_name}_{t}', downsample=False, filters=filters)
    return X

@tf.function
def resnet(
    layers: typing.List[int],
    name: str,
    out_classes: int = 1000,
    input_shape: tuple = (224, 224, 3)) -> Model:
    X_input = Input(shape=input_shape)

    # Conv1
    X = ZeroPadding2D(padding=(3, 3))(X_input)
    X = Conv2D(filters=64, kernel_size=3, strides=2, name='conv1')(X)
    X = BatchNormalization(name="bn_conv1")(X)
    X = Activation('relu')(X)
    print(f'X after Conv1: {X.shape}')
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)
    print(f'X after MaxPooling2D: {X.shape}')

    # Conv2
    X = _make_layer(X, base_name='conv2', filters=[64, 64], layers=layers[0])
    print(f'X after Conv2: {X.shape}')

    # Conv3
    X = _make_layer(X, base_name='conv3', filters=[64, 64], layers=layers[0])
    print(f'X after Conv2: {X.shape}')

    # Conv4
    X = _make_layer(X, base_name='conv4', filters=[64, 64], layers=layers[0])
    print(f'X after Conv2: {X.shape}')

    # Conv5
    X = _make_layer(X, base_name='conv5', filters=[64, 64], layers=layers[0])
    print(f'X after Conv2: {X.shape}')

    # Global Average Pooling
    X = GlobalAveragePooling2D()(X)
    print(f'X after GlobalAveragePooling2D: {X.shape}')

    X = Flatten()(X)
    print(f'X after Flatten: {X.shape}')

    X = Dense(units=out_classes, activation='softmax', name='output_layer')

    model = Model(inputs=[X_input], outputs=[X])
    return model
