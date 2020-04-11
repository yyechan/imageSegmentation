import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Add, LeakyReLU, UpSampling2D
from keras.models import Model, load_model

from keras.backend import argmax


data = np.array(Image.open('22.jpg'))

data = np.array(Image.fromarray(data.astype('uint8')).resize((512, 512)))

data = np.expand_dims(data,axis=0)




model = load_model('weight200.h5')

pdata = model.predict(data)

print(pdata.shape)

pred = np.zeros((512,512))


for i in range(0,512):
    for j in range(0,512):

        temp = pdata[0][i][j][0]
        pred[i][j] = 0

        if(pdata[0][i][j][1] > temp) :
            temp = pdata[0][i][j][1]
            pred[i][j] = 1

        if(pdata[0][i][j][2] > temp) :
            temp = pdata[0][i][j][2]
            pred[i][j] = 2
        

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
