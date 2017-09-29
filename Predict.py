import keras 
from keras.models import Sequential 
from keras.models import load_model
import matplotlib.pyplot as plt 
import numpy as np 
import cv2 
import tnfix 
from keras import backend as K
K.set_image_dim_ordering('th')

num_channel = 1 
loaded_model = load_model('models/new_test_data.h5')
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

test_image_first = cv2.imread('images/dog.jpg')
test_image=cv2.cvtColor(test_image_first, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(128,128))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)
if num_channel==1:
	if K.image_dim_ordering()=='th':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=3) 
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
else:
	if K.image_dim_ordering()=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
# Predicting the test image
score = ((loaded_model.predict(test_image)))
data = (loaded_model.predict_classes(test_image))

if data == np.array([0]):
	print "It's a cat"
elif data == np.array([1]):
	print "It's a Dog"
	print score
	plt.imshow(test_image_first)
	plt.show()