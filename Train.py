import keras
import numpy as np
from keras.layers import Input, Dense, Dropout, Activation, Concatenate, BatchNormalization, Flatten
from keras.models import Model
from keras.layers import Conv2D, GlobalAveragePooling2D, AveragePooling2D, ZeroPadding2D, MaxPooling2D
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
import math

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.1
    epochs_drop = 7.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def DenseNet(input_shape=None, dense_blocks=3, dense_layers=-1, growth_rate=12, nb_classes=None, dropout_rate=None,
             bottleneck=False, compression=1.0, weight_decay=1e-4, depth=40):
    
    if nb_classes==None:
        raise Exception('Please define number of classes (e.g. num_classes=10). This is required for final softmax.')
    
    if compression <=0.0 or compression > 1.0:
        raise Exception('Compression have to be a value between 0.0 and 1.0. If you set compression to 1.0 it will be turn off.')
    
    if type(dense_layers) is list:
        if len(dense_layers) != dense_blocks:
            raise AssertionError('Number of dense blocks have to be same length to specified layers')
    elif dense_layers == -1:
        if bottleneck:
            dense_layers = (depth - (dense_blocks + 1))/dense_blocks // 2
        else:
            dense_layers = (depth - (dense_blocks + 1))//dense_blocks
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]
    else:
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]
        
    img_input = Input(shape=input_shape)
    nb_channels = growth_rate * 2
    
    # Initial convolution layer
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(nb_channels, (7,7), strides=2, use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=((1,1), (1, 1)))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    
    # Building dense blocks
    for block in range(dense_blocks):
        x, nb_channels = dense_block(x, dense_layers[block], nb_channels, growth_rate, dropout_rate, bottleneck, weight_decay)
        
        if block < dense_blocks - 1:  # if it's not the last dense block
            x = transition_layer(x, nb_channels, dropout_rate, compression, weight_decay)
            nb_channels = int(nb_channels * compression)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
    
    model = Model(img_input, x, name='densenet')
    return model

def dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
    x_list = [x]
    for i in range(nb_layers):
        cb = convolution_block(x, growth_rate, dropout_rate, bottleneck, weight_decay)
        x_list.append(cb)
        x = Concatenate(axis=-1)(x_list)
        nb_channels += growth_rate
    return x, nb_channels

def convolution_block(x, growth_rate, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
    if bottleneck:
        x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Conv2D(4 * growth_rate, (1, 1), use_bias=False, kernel_regularizer=l2(weight_decay))(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
    
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    return x

def transition_layer(x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4):
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_channels*compression), (1, 1), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x

if __name__ == '__main__':
    # Define the class labels
    class_labels = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
    nb_classes = len(class_labels)

    # Create the model
    model = DenseNet(input_shape=(64, 64, 1), dense_blocks=2, dense_layers=6, growth_rate=32, 
                     nb_classes=nb_classes, bottleneck=True, depth=27, weight_decay=1e-5)
    
    print(model.summary())

    # Compile the model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Set up data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2  # 20% of the data will be used for validation
    )

    train_generator = train_datagen.flow_from_directory(
        'D:\\PCB Defect Classification\\PCB_DATASET\\images',
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical',
        subset='training',
        classes=class_labels
    )

    validation_generator = train_datagen.flow_from_directory(
        'D:\\PCB Defect Classification\\PCB_DATASET\\images',
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        classes=class_labels
    )

    # Set up callbacks
    lrate = LearningRateScheduler(step_decay, verbose=1)
    callbacks_list = [lrate]

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=50,
        callbacks=callbacks_list,
        verbose=1
    )

    # Save the model
    model.save("D:\\PCB Defect Classification\\Model\\Model.h5")

    print("Training completed. Model saved as 'Model.h5'")