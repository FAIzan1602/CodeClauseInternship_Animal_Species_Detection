import cv2
import tensorflow_hub as hub
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

img_path = 'image.jpg'

image = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
image = keras.preprocessing.image.img_to_array(image)
image = image / 255.0
image = image.reshape(1,224,224,3)

model = keras.models.load_model('Animal_spices.h5',custom_objects={'KerasLayer':hub.KerasLayer})

pridiction = model.predict(image)

dicts = {0:"Buffalo", 1:"Elephant", 2:"Rhino", 3:"Zebra"}

i = cv2.imread(img_path)
plt.imshow(i)
plt.show()

print(dicts[np.argmax(pridiction)])