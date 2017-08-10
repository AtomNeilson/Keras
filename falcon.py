import keras 
from keras.models import Sequential
from keras.layers import Dense , Activation, Flatten,Dropout 
from keras.datasets import mnist 

(X_train, y_train),(X_test, y_test) = mnist.load_data()
