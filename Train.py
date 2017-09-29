import keras 
from keras.models import Sequential 
from keras.layers import Dense, Activation, Flatten 
from keras.layers import Convolution2D, MaxPooling2D, Dropout 
import tnfix 
# from keras import backend as K 
# K.set_image_dim_ordering('th')
'''
  from keras.utils import plot_model
    plot_model(model, to_file='model.png')

'''


import os 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split 

PATH = os.getcwd()
# Define data path
data_path = PATH + '/images/test_data'
data_dir_list = os.listdir(data_path)

img_rows=128
img_cols=128
num_channel=1
num_epoch=20

# Define the number of classes
num_classes = 4

img_data_list=[]

for dataset in data_dir_list:
  img_list=os.listdir(data_path+'/'+ dataset)
  print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
  for img in img_list:
    input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
    input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    input_img_resize=cv2.resize(input_img,(128,128))
    img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255

# img_data = np.expand_dims(img_data, axis=1)
img_data = img_data.reshape(img_data.shape[0],1,128,128)

# print(img_data.shape)

num_classes = 2 
number_of_images = img_data.shape[0]
labels = np.ones((number_of_images,),dtype='int64')


labels[1:600]=0 
labels[600:1200]=1 
names = ['Cats','Dogs']

Y = keras.utils.to_categorical(labels, num_classes)
x,y = shuffle(img_data, Y, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0, random_state=2)
input_shape = X_train.shape

model = Sequential()

model.add(Convolution2D(32, (3, 3), padding="same", input_shape=(1,128,128), dim_ordering='th'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
#model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
  )

hist = model.fit(X_train, y_train, batch_size=5, nb_epoch=70, verbose=1, validation_data=(X_test,y_test))
model.save('models/new_test_data.h5')