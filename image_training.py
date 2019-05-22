# from keras.applications.inception_v3 import InceptionV3
from keras import Sequential
from keras.layers import Conv3D, Conv2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling3D, MaxPooling2D

def create_model(max_frames):
    #base_model = InceptionV3(weights='imagenet', include_top=False, input_shape = (256, 256, max_frames))
    model = Sequential([
        Conv3D(32, kernel_size = (3,3,1), padding = 'same' , activation = 'relu', input_shape = (256,256,max_frames,1)),
        GlobalAveragePooling3D(),
        Dense(1024, activation='relu'),
        Dense(5, activation='softmax') # 5 is the number of classes in the data, since TICI is 0,1,2a,2b,3
    ])
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, epochs, x_train, y_train, x_test, y_test, batch_size = 50):
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
    return model

# is not working for the 3D data because you can only send 2D data to the Image Data Generator
# if we want to use this we will need to do it in a different way...
'''def create_generators(x_train, y_train, x_test, y_test):
    train_d_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    
    train_d_gen.fit(x_train)
    
    train_gen = train_d_gen.flow(x_train, y_train)
    
    validation_gen = ImageDataGenerator().flow(x_test, y_test)
    
    return train_gen, validation_gen

def train_model(model, nb_epoch, generators, train_len):
    train_generator, validation_generator = generators
    model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        validation_steps=10,
        epochs=nb_epoch)
    return model
'''