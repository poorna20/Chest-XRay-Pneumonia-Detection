from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

test_path="D:\Desktop\Chest xRay\chest_xray\test"
model=Sequential()
model.add(Conv2D(32,(3,3), activation='relu',input_shape=(64,64,1),padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory("D:\\Desktop\\Chest xRay\\chest_xray\\train",
        target_size=(64,64),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory("D:\\Desktop\\Chest xRay\\chest_xray\\test",
        target_size=(64,64),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=3,
        validation_data=test_generator,
        validation_steps=800)
