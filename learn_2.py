import tensorflow as tf 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras.applications import ResNet50V2
from keras import regularizers
import os 


img_size = 224
batch_size = 128
epochs = 10

train_dir = os.path.join(os.getcwd(),'yoga_pose','82_data')
n_classes = len(os.listdir(train_dir))




train_data_gen = ImageDataGenerator(rescale=1./255,zoom_range=0.2,horizontal_flip=True,rotation_range=5,
    validation_split= 0.2)

train_gen = train_data_gen.flow_from_directory(directory=train_dir,target_size=(img_size,img_size),
        batch_size=batch_size,class_mode='categorical',subset='training')


val_gen = train_data_gen.flow_from_directory(directory=train_dir,target_size=(img_size,img_size),
        batch_size=batch_size,class_mode='categorical',subset='validation')

base_model = ResNet50V2(include_top=False, input_shape=(img_size,img_size,3))

mid_model = Sequential()
mid_model.add(Dense(1000,activation='relu',kernel_regularizer=regularizers.l1_l2(0.01), 
        activity_regularizer=regularizers.l2(0.01)))
mid_model.add(Dropout(0.4))
mid_model.add(Dense(500,activation='relu',kernel_regularizer=regularizers.l1_l2(0.01), 
        activity_regularizer=regularizers.l2(0.01)))
mid_model.add(Dropout(0.2))
mid_model.add(Dense(n_classes,activation='sigmoid'))


pooling = tf.keras.layers.GlobalAveragePooling2D()
model = Sequential([
    base_model,
    pooling,
    mid_model]
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])
early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
callbacks_list = [early_stop, reduce_lr]

#training
model_history = model.fit_generator(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=callbacks_list)