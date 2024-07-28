#implementation of the model
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('handwritten.h5')

img_num = 1

while os.path.isfile(f"digits/digit{img_num}.png"):
    try:
        img = cv2.imread(f"digits/digit{img_num}.png")[:,:,0]
        img = np.invert(np.array([img]))

        pred =model.predict(img)

        print(f"this is digit is probably {np.argmax(pred)}")

        plt.imshow(img[0],cmap=plt.cm.binary)

    except:
        print('Error')

    finally:
        img_num +=1



