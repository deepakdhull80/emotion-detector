
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout,Dense,Activation,Flatten,BatchNormalization,Conv2D,MaxPooling2D
import os

num_class=5
img_row,img_col=48,48
batch_size=32


os.chdir("D:\python programs\machine learning\emotion detector")

train_dir="data/train"
valid_dir="data/validation"

""" data augmentation """

train_datagen= ImageDataGenerator(rescale=1/255,
                                  rotation_range=30,
                                  shear_range=0.3,
                                  zoom_range=0.3,
                                  width_shift_range=0.4,
                                  height_shift_range=0.4,
                                  horizontal_flip=True,
                                  fill_mode='nearest')


valid_datagen=ImageDataGenerator(rescale=1/255)

train_generator= train_datagen.flow_from_directory(train_dir,color_mode='grayscale',
                                                   target_size=(img_row,img_col),
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   shuffle=True)


valid_generator= train_datagen.flow_from_directory(valid_dir,color_mode='grayscale',
                                                   target_size=(img_row,img_col),
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   shuffle=True)

"""Define model - Convolutional Neural Network"""



model=Sequential()

#block 1

model.add(Conv2D(32,(3,3),padding='same',input_shape=(img_row,img_col,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',input_shape=(img_row,img_col,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#block 2

model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


#block 3


model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


#block 4


model.add(Conv2D(256,(3,3),padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


#block 5

model.add(Flatten())
model.add(Dense(64,activation='elu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#block 6

model.add(Dense(64,activation='elu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#block 7

model.add(Dense(num_class,activation='softmax',kernel_initializer='he_normal'))



model.summary()


from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau

checkpoint=ModelCheckpoint('model_vgg.h5',
                           monitor='val_loss',
                           mode='min',
                           save_best_only=True,
                           verbose=1)

earlystop= EarlyStopping(monitor='val_loss',
                         min_delta=0,
                         patience=15,
                         verbose=1,
                         restore_best_weights=True)

reduce_lr= ReduceLROnPlateau(monitor='val_loss',
                             factor=0.2,
                             min_delta=0.0001,
                             patience=15,
                             verbose=1)

callback=[checkpoint,earlystop,reduce_lr]


"""Train model"""

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

nb_train=24176
nb_valid=3006
epoch=25


history=model.fit_generator(train_generator,
                            steps_per_epoch=nb_train//batch_size,
                            epochs=epoch,
                            callbacks=callback,
                            validation_data=valid_generator,
                            validation_steps=nb_valid//batch_size)


""" model train with 62.77 acc and 57.40 val_acc"""
