import keras

#
# ResNet 20, 36, 44, 56, 110, 1202 in Keras 3
#
# He, Kaiming, et al. "Deep residual learning for image recognition." (2016) [1]
#  - https://arxiv.org/abs/1512.03385
#


def resnet_basic_block(inputs, num_filters, strides=1, conv_shortcut=False, name=None):
    # Following ResNet building block from Figure 2 [1].
    # Using the [3x3 , 3x3] x n convention. Section 4.2 [1]

    # Between stacks, the subsampling is performed by convolutions with
    # a stride of 2. Section 4.2 [1]
    x = keras.layers.Conv2D(
        num_filters,
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name=name + "_1_conv2d",
    )(inputs)
    # We adopt batch normalization right after each convolution and before
    #   activation. Section 3.4 [1]
    x = keras.layers.BatchNormalization(name=name + "_1_bn")(x)
    x = keras.layers.Activation("relu", name=name + "_1_relu")(x)

    x = keras.layers.Conv2D(
        num_filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name=name + "_2_conv2d",
    )(x)
    # We adopt batch normalization right after each convolution and before
    #   activation. Section 3.4 [1]
    x = keras.layers.BatchNormalization(name=name + "_2_bn")(x)

    #
    # conv_shortcut = False:
    #   The identity shortcuts (Eqn.(1)) can be directly used when the input
    #   and output are of the same dimensions (solid line shortcuts in Fig. 3).
    # conv_shortcut = True:
    #   When the dimensions increase (dotted line shortcuts in Fig. 3), we consider
    #   two options: ... (B) The projection shortcut in Eqn.(2) is used to match
    #   dimensions (done by 1×1 convolutions).
    # Section 3.3 "Residual Network" [1]
    #
    if conv_shortcut:
        # The projection shortcut in Eqn.(2) is used to match dimensions
        # (done by 1×1 convolutions). Section 3.3 "Residual Network" [1]
        y = keras.layers.Conv2D(
            num_filters,
            kernel_size=1,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(1e-4),
            name=name + "_0_conv2d",
        )(inputs)
    else:
        # The identity shortcuts (Eqn.(1)) can be directly used when the input
        # and output are of the same dimensions. Section 3.3 "Residual Network" [1]
        y = inputs

    out = keras.layers.add([x, y], name=name + "_add")
    out = keras.layers.Activation("relu", name=name + "_out")(out)

    return out


def resnet(input_shape, num_blocks=3, num_classes=10):
    #
    # The following table summarizes the architecture: Section 4.2 [1]
    # | output map size | 32×32 | 16×16 | 8×8 |
    # |-----------------|-------|-------|-----|
    # | # layers        | 1+2n  | 2n    | 2n  |
    # | # filters       | 16    | 32    | 64  |
    #
    # n = num_blocks
    #
    # num_blocks = 3: ResNet20
    # num_blocks = 5: ResNet32
    # num_blocks = 7: ResNet44
    # num_blocks = 9: ResNet56
    # num_blocks = 18: ResNet110
    # num_blocks = 200: ResNet1202
    #

    inputs = keras.layers.Input(shape=input_shape)

    # The first layer is 3×3 convolutions. Section 4.2 [1]
    name = "conv1"
    x = keras.layers.Conv2D(
        16,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name=name + "_conv2d",
    )(inputs)
    # We adopt batch normalization right after each convolution and before
    # activation. Section 3.4 [1]
    x = keras.layers.BatchNormalization(name=name + "_bn")(x)
    x = keras.layers.Activation("relu", name=name + "_relu")(x)

    # The numbers of filters are {16, 32, 64} respectively. Section 4.2 [1]

    name = "conv2"
    # The first stack uses 16 filters. Section 4.2 [1]
    # First block of the first stack uses identity shortcut (strides=1) since the
    #   input size matches the output size of the first layer is 3×3 convolutions.
    #   Section 3.3 "Residual Network" [1]
    x = resnet_basic_block(
        inputs=x, num_filters=16, strides=1, conv_shortcut=False, name=name + "_block1"
    )
    for blocks in range(num_blocks - 1):
        # All other blocks use identity shortcut (strides=1)
        x = resnet_basic_block(
            inputs=x,
            num_filters=16,
            strides=1,
            conv_shortcut=False,
            name=name + "_block" + str(blocks + 2),
        )

    name = "conv3"
    # The second stack uses 32 filters. Section 4.2 [1]
    # First block of the second stack uses projection shortcut (strides=2)
    #   Section 3.3 "Residual Network" [1]
    x = resnet_basic_block(
        inputs=x, num_filters=32, strides=2, conv_shortcut=True, name=name + "_block1"
    )
    for blocks in range(num_blocks - 1):
        # All other blocks use identity shortcut (strides=1)
        x = resnet_basic_block(
            inputs=x,
            num_filters=32,
            strides=1,
            conv_shortcut=False,
            name=name + "_block" + str(blocks + 2),
        )

    name = "conv4"
    # The third stack uses 64 filters. Section 4.2 [1]
    # First block of the third stack uses projection shortcut (strides=2)
    #   Section 3.3 "Residual N etwork" [1]
    x = resnet_basic_block(
        inputs=x, num_filters=64, strides=2, conv_shortcut=True, name=name + "_block1"
    )
    for blocks in range(num_blocks - 1):
        # All other blocks use identity shortcut (strides=1)
        x = resnet_basic_block(
            inputs=x,
            num_filters=64,
            strides=1,
            conv_shortcut=False,
            name=name + "_block" + str(blocks + 2),
        )

    # The network ends with a global average pooling, a 10-way fully-connected
    # layer, and softmax. Section 4.2 [1]
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)

    # Build the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def resnet20(input_shape, num_classes=10):
    return resnet(input_shape, 3, num_classes=num_classes)


def resnet32(input_shape, num_classes=10):
    return resnet(input_shape, 5, num_classes=num_classes)


def resnet44(input_shape, num_classes=10):
    return resnet(input_shape, 7, num_classes=num_classes)


def resnet56(input_shape, num_classes=10):
    return resnet(input_shape, 9, num_classes=num_classes)


def resnet110(input_shape, num_classes=10):
    return resnet(input_shape, 18, num_classes=num_classes)


def resnet1202(input_shape, num_classes=10):
    return resnet(input_shape, 200, num_classes=num_classes)


if __name__ == "__main__":
    model = resnet20(input_shape=(32, 32, 3))

    model.summary()

    plot_model_filename = "resnet20_plot.png"
    keras.utils.plot_model(model, to_file=plot_model_filename, show_shapes=True)
