import tensorflow as tf

import numpy as np
import os
from keras.backend import int_shape
from tensorflow.keras.models import save_model, load_model, Model
from tensorflow.keras.layers import Input, Dropout, BatchNormalization, concatenate, Activation, Concatenate, Cropping2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import UpSampling2D
from config import Config


def conv2d_BN(input, nb_filter, filter_size, str=1, padding='same', use_bias=False, bn=True, act=None,
              name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(filters=nb_filter, kernel_size=filter_size, padding=padding, strides=str, activation='relu', use_bias=use_bias,
               name=None)(input)

    if bn:
        x = BatchNormalization(axis=3, name=bn_name)(x)

    if act is not None:
        x = Activation(act)(x)

    return x


def shortcut():
    pass


def bottle_block(input, nb_filters, stride):
    x = conv2d_BN(input=input, nb_filter=nb_filters, filter_size=1, padding='same', act='relu')
    #print('bottlecov1={}'.format(x.shape))
    x = conv2d_BN(input=x, nb_filter=nb_filters,  filter_size=3, padding='same', str=stride, act=None)
    #print('bottleconv2={}'.format(x.shape))
    #short_bn = tf.add(input, x)
    #x = Activation('relu')(short_bn)
    return x


def encoder_block(input, encoder_depth, encoder_filters, block):
    x=input
    for i in range(encoder_depth[block]):
        x = bottle_block(x, nb_filters=encoder_filters[block], stride=2 if i == 0 and block != 0 else 1)
    print('Encoder block', block, x.shape)
    return x


def decoder_block(input, concat_input, decoder_depth, decoder_filters, block, up_mode='upsample'):
    global decon_bn

    # 1. crop the concat_input with 2*converse_input tensor
    if up_mode == 'upconv':
        x = Conv2DTranspose(filters=decoder_filters[block], kernel_size=2, strides=2)(input)
        _, x_height, x_width, _ = int_shape(x)
        _, concat_input_height, concat_input_width, _ = int_shape(concat_input)
        h_crop = concat_input_height-x_height
        w_crop = concat_input_width-x_width
        assert h_crop >= 0
        assert w_crop >= 0
        print('h_crop={},w_crop={}'.format(h_crop, w_crop))
        if h_crop == 0 and w_crop == 0:
            concat_input1 = concat_input
        else:
            cropping = (h_crop // 2, h_crop - h_crop // 2), (w_crop // 2, w_crop - w_crop // 2)
            concat_input1 = Cropping2D(cropping=cropping)(concat_input)


    # 2. 2*upsample_input to match the concat_input, without cropping all the padding should be 'same'
    if up_mode == 'upsample':
        x = input
        x = UpSampling2D(size=(2,2), data_format=None, interpolation='bilinear')(x)
        x = bottle_block(input=x, nb_filters=decoder_filters[block], stride=1)
        #concat_input1 = conv2d_BN(input=concat_input, nb_filter=concat_input.shape[1], filter_size=1, act='relu')
        concat_input1 = concat_input

    decon_con = concatenate([x, concat_input1], axis=-1)
    for i in range(decoder_depth[block]):
        decon_bn = bottle_block(input=decon_con, nb_filters=decoder_filters[block], stride=1)
    print('decoder block', block, decon_bn.shape)
    return decon_bn


def unet(img, cfg):

    encoder_depth = [2, 2, 2, 2]
    encoder_filters = [128, 256, 512, 1024]
    decoder_depth = [1, 1, 1, 1]
    decoder_filters = [512, 256, 128, 64]

    # Encoder
    start = Input(img)
    start_cov = conv2d_BN(start, nb_filter=64, filter_size=3, padding='same', bn=False, act='relu')
    x = conv2d_BN(start_cov, nb_filter=64, filter_size=3, padding='same', str=1, act='relu')
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    print('start convlution', x.shape)
    conv0 = encoder_block(x, encoder_depth, encoder_filters, block=0)
    # conv0maxpooling
    conv1 = encoder_block(conv0, encoder_depth, encoder_filters, block=1)
    conv2 = encoder_block(conv1, encoder_depth, encoder_filters, block=2)
    conv3 = encoder_block(conv2, encoder_depth, encoder_filters, block=3)

    # Decoder
    decode_conv1 = decoder_block(conv3, conv2, decoder_depth, decoder_filters, block=0)
    decode_conv2 = decoder_block(decode_conv1, conv1, decoder_depth, decoder_filters, block=1)
    decode_conv3 = decoder_block(decode_conv2, conv0, decoder_depth, decoder_filters, block=2)
    decode_conv4 = decoder_block(decode_conv3, start, decoder_depth, decoder_filters, block=3)

    # Output

    #conv5 = conv2d_BN(input=decode_conv4, nb_filter=16, filter_size=3, padding='same', act='relu')
    last_conv = conv2d_BN(input=decode_conv4, nb_filter=cfg.NUM_CLASSES, filter_size=1, padding='same', act='softmax')
    print('end convolution', last_conv.shape)
    model = Model(inputs=start, outputs=last_conv)
    return model

'''class Unet():
    def __init__(self, config):
        super(Unet, self).__init__()
        img = (768,256,3)
        x = unet(img, config.NUM_CLASSES)'''



#unet(img=(768,256,3),cfg=Config).summary()

