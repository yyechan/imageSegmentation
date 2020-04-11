from model import unet
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model




xpath_list = glob.glob('./data/trainX/*.jpg')
ypath_list = glob.glob('./data/trainY/*.png')

xpath_list = [filename.replace('\\','/') for filename in xpath_list]
ypath_list = [filename.replace('\\','/') for filename in ypath_list]

xpath_list.sort()
ypath_list.sort()

Xdata = []
Ydata = []

# Xdata = np.array(Xdata.append(np.array(Image.open(path)) for path in xpath_list))

for path in xpath_list:
    img = np.array(Image.open(path))
    Xdata.append(img)

for path in ypath_list:

    img = np.array(Image.open(path))
    mask = np.zeros((512,512,1))

    for i in range(0,512) :
        for j in range(0,512) :

            r = img[i][j][0]
            g = img[i][j][1]
            b = img[i][j][2]

            if(r==255 and g == 0 and b == 0) :
                mask[i][j][0] = 1
            if(r==255 and g == 255 and b == 0) :
                mask[i][j][0] = 2


    Ydata.append(mask)
  
Xdata = np.array(Xdata)
Ydata = np.array(Ydata)
Ydata = to_categorical(Ydata)


print(Xdata.shape)
print(Ydata.shape)

model = unet(num_classes = 3)
model.summary()

history = model.fit(Xdata, Ydata, epochs=200, batch_size=1, callbacks=[
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)
] ,validation_split = 1/20)

fig, ax = plt.subplots(2, 2, figsize=(10, 7))

ax[0, 0].set_title('loss')
ax[0, 0].plot(history.history['loss'], 'r')
ax[0, 1].set_title('acc')
ax[0, 1].plot(history.history['acc'], 'b')

ax[1, 0].set_title('val_loss')
ax[1, 0].plot(history.history['val_loss'], 'r--')
ax[1, 1].set_title('val_acc')
ax[1, 1].plot(history.history['val_acc'], 'b--')


model.save('weight200.h5')






