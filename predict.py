import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Add, LeakyReLU, UpSampling2D
from keras.models import Model, load_model

from keras.backend import argmax

height = 512
width = 512

data = np.array(Image.open('predict.jpg'))

data = np.array(Image.fromarray(data.astype('uint8')).resize((width, height)))

data = np.expand_dims(data,axis=0)

print(data.shape)

model = load_model('allnight.h5')

pdata = model.predict(data)

pred = np.zeros((height,width))


for h in range(0,height):
    for w in range(0,width):

        temp = pdata[0][h][w][0]
        pred[h][w] = 0

        if(pdata[0][h][w][1] > temp) :
            temp = pdata[0][h][w][1]
            pred[h][w] = 1

        if(pdata[0][h][w][2] > temp) :
            temp = pdata[0][h][w][2]
            pred[h][w] = 2
        

fig = plt.figure()
rows = 1
cols = 2

ax1 = fig.add_subplot(rows, cols, 1)
ax1.imshow(data.squeeze(),cmap='gray')
ax1.set_title('before')
ax1.axis("off")
 
ax2 = fig.add_subplot(rows, cols, 2)
ax2.imshow(pred.squeeze(),cmap='gray')
ax2.set_title('after')
ax2.axis("off")
 
plt.show()
