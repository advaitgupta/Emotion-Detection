
from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNet
import matplotlib.pyplot as plt

train_dir='/Users/advaitgupta/Desktop/Projects & Study/Projects/Personal/Emotion Detection Model/archive-2/train'
val_dir='/Users/advaitgupta/Desktop/Projects & Study/Projects/Personal/Emotion Detection Model/archive-2/test'


train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=25,
					zoom_range=0.25,
                    shear_range=0.25,
                    height_shift_range=0.3,
					width_shift_range=0.3,
					horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                        train_dir,
                        target_size = (224,224),
                        batch_size = 32,
                        class_mode = 'categorical'
                        )

val_set = val_datagen.flow_from_directory(
                            val_dir,
                            target_size=(224,224),
                            batch_size=32,
                            class_mode='categorical')

MobileNet = MobileNet(weights='imagenet',include_top=False,input_shape=(224,224,3))

for layer in MobileNet.layers:
    layer.trainable = True

for (i,layer) in enumerate(MobileNet.layers):
    print(str(i),layer.__class__.__name__,layer.trainable)

def Top(bottom_model, classes):

    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)

    top_model = Dense(1024, activation='relu')(top_model)

    top_model = Dense(512, activation='relu')(top_model)

    top_model = Dense(classes, activation='softmax')(top_model)

    return top_model


g = Top(MobileNet, 7)
model = Model(inputs=MobileNet.input, outputs=g)
print(model.summary())

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint('Emotion_Model1.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=5,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

model.compile(loss='categorical_crossentropy',
              optimizer = "adam",
              metrics=['accuracy'])

callbacks = [earlystop,checkpoint,reduce_lr]

history=model.fit(
                training_set,
                validation_data=val_set,
                epochs=25,
                callbacks=callbacks,
                shuffle=True)

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('emotion_acc.png')
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('emotion_loss.png')
plt.show()

