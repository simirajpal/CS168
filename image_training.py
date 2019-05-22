# from keras.applications.inception_v3 import InceptionV3
from keras import Sequential
from keras.layers import Conv3D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling3D

def create_model(max_frames):
    #base_model = InceptionV3(weights='imagenet', include_top=False, input_shape = (256, 256, max_frames))
    model = Sequential([
         Conv3D(32, kernel_size = (3,3,1), padding = 'same' , activation = 'relu', input_shape = (256,256, max_frames)),
    
    # DNN portion
    # average pooling layer
    GlobalAveragePooling3D(),
    # fully-connected layer
    Dense(1024, activation='relu'),
    # logistic layer
    Dense(5, activation='softmax') # 5 is the number of classes in the data, since TICI is 0,1,2a,2b,3
    ])
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_generators(x_train, y_train, batch_size):
    train_d_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    
    train_d_gen.fit(x_train)
    
    train_gen = train_d_gen.flow(x_train, y_train, batch_size)
    
    validation_gen = ImageDataGenerator().flow(x_test, y_test)
    
    return train_generator, validation_generator

def train_model(model, nb_epoch, generators, train_len, batch_size):
    train_generator, validation_generator = generators
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_len // batch_size,
        validation_data=validation_generator,
        validation_steps=10,
        epochs=nb_epoch)
    return model