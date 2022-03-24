import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

'''
VGG3D model with 3 Vgg blocks
'''
def vgg_block(layer_in,initial_fs,conv_layers):
    
    # loop for creating vgg block
    for _ in range (conv_layers):
        layer_in = layers.Conv3D(filters=initial_fs, kernel_size=(3, 3, 3), activation='relu')(layer_in)
    layer_in = layers.MaxPool3D(pool_size=(2, 2, 2))(layer_in)
    layer_in = layers.Dropout(0.4)(layer_in)
    return layer_in


def Vgg3D_3blocks(input_shape):
    initial_filtersize= 64           
    input_layer = keras.Input((input_shape[0],input_shape[1],input_shape[2],1))   

    # create 3 VGG blocks
    layer = vgg_block(input_layer,initial_filtersize,2)
    layer = vgg_block(layer, initial_filtersize*2, 2)
    layer = vgg_block(layer, initial_filtersize*3, 3)

    bn_layer = layers.BatchNormalization()(layer)
    flt_layer = layers.Flatten()(bn_layer)

    dense1 = layers.Dense(units=512, activation='relu')(flt_layer)
    dense1 = layers.Dropout(0.4)(dense1)


    dense2 = layers.Dense(units=512, activation='relu')(dense1)
    dense2 = layers.Dropout(0.4)(dense2)

    dense3 = layers.Dense(units=256, activation='relu')(dense2)
    dense3 = layers.Dropout(0.4)(dense3)

    dense4 = layers.Dense(units=128, activation='relu')(dense3)
    dense4 = layers.Dropout(0.4)(dense4)

    output_layer = layers.Dense(units=1, activation='tanh')(dense4)


    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model 
