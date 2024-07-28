import numpy as np
import tensorflow as tf


mnist = tf.keras.datasets.mnist

(X_train, y_train),(X_test,y_test)= tf.keras.datasets.mnist.load_data()

X_train = tf.keras.utils.normalize(X_train,axis = 1)

X_test = tf.keras.utils.normalize(X_test,axis = 1)
"""
#model
model =tf.keras.models.Sequential()


# adding layers
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=3)

model.save('handwritten.h5')
"""

model = tf.keras.models.load_model('handwritten.h5')

loss, accuracy = model.evaluate(X_test,y_test)

print('Loss : ',loss)
print('Accuracy : ',accuracy)
