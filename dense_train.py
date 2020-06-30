import tensorflow as tf 
import pickle
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras import regularizers
from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model
import os 

#training parameters 
categories = ['bridge','childs','downwarddog','mountain','plank','seatedforwardbend','tree','trianglepose','warrior1','warrior2']

img_height = img_width = 224 
batch_size = 32 
epochs = 10
n_classes = len(categories)
nb_validation_samples = 119 # number of images in test_set

train_dir = os.path.join('yoga','training_set')
test_dir = os.path.join('yoga',"test_set")


def data_gen(train_dir,test_dir):
    '''make the data generators, does transforms'''
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        zoom_range=0.2,
        rotation_range = 5,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    return train_generator, validation_generator

def build_model(im_size=224,n_classes=10):
    '''build a densenet121 model with new classification layer, all layers trainable, imagenet starting weights'''
    base_model = DenseNet121(include_top=False,pooling='avg',input_shape=(im_size,im_size,3))
    #for layer in base_model.layers:
    #    layer.trainable = True
    
    x = base_model.output
    x = Dense(1000, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = Dense(500, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

if __name__ == "__main__":
    pass

    
    #initialize the model, and generators
    train_generator, validation_generator = data_gen(train_dir,test_dir)
    model = build_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])
    early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
    callbacks_list = [early_stop, reduce_lr]

    #training
    model_history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=callbacks_list)
    
    # save the model so I dont have to re-train
    model_json = model.to_json()
    with open("model.json", "w") as json_file :
        json_file.write(model_json)
    model.save_weights("yoga/model/model.h5")
    model.save('yoga/model/CNN.model')
    print("\nSaved model to disk\n")

