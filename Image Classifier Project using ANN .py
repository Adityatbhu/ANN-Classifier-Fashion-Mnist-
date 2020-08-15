# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
""" 
import keras
import tensorflow as tf
from tensorflow import keras
keras.__version__
tf.__version__
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
fashion_mnist= keras.datasets.fashion_mnist
(X_train_full, Y_train_full) , (X_test,Y_test)=fashion_mnist.load_data()
plt.imshow(X_train_full[9])
class_order=['T-shirt','Trousers','Pullovers','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

#%% Data Normalisation
train_n=X_train_full/255.0
test_n= X_test/255.0
#%% Validation Data set
x_valid,x_train= train_n[:5000],train_n[5000:]
y_valid,y_train=Y_train_full[:5000],Y_train_full[5000:]
x_test=test_n
#%% model making
np.random.seed(42)
tf.random.set_seed(42)
model= keras.models.Sequential()
m=model.add(keras.layers.Flatten(input_shape=[28,28]))
n=model.add(keras.layers.Dense(300, activation='relu'))
o=model.add(keras.layers.Dense(100, activation='relu'))
p=model.add(keras.layers.Dense(10,activation='softmax'))

model.summary()

#%% mlodel compilation
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])
#%%
checkpoint_cb=keras.callbacks.ModelCheckpoint("model.{epoch:02d}.h5")
#%%
model_history= model.fit(x_train,y _train,epochs=50,
                         validation_data=(x_valid,y_valid),callbacks=[checkpoint_cb])
#%% ploting of accuracy, error,val_accuracy,val_error
history=model_history.history
pd.DataFrame(model_history.history).plot()

plt.gca().set_ylim(0,1)
plt.show()
#%%
model.evaluate(x_test,Y_test)
x_new=x_test[:5]
y_prob=model.predict(x_new)
y_prob.round(2)
y_prob=model.predict_classes(x_new)
y_prob
np.array(class_order)[y_prob]
weights=model.get_weights()
#%%
model.save("my model.h5")
%pwd
