from model import unet
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
from processdata import dataGenerator
from keras.callbacks import ModelCheckpoint


myGenerator = dataGenerator(batch_size = 1)

#model = load_model('weight200.h5')

model = unet(num_classes = 3)

model.summary()
history = model.fit_generator(myGenerator,steps_per_epoch=2000,epochs=20)



# fig, ax = plt.subplots(2, 2, figsize=(10, 7))

# ax[0, 0].set_title('loss')
# ax[0, 0].plot(history.history['loss'], 'r')
# ax[0, 1].set_title('acc')
# ax[0, 1].plot(history.history['acc'], 'b')

# ax[1, 0].set_title('val_loss')
# ax[1, 0].plot(history.history['val_loss'], 'r--')
# ax[1, 1].set_title('val_acc')
# ax[1, 1].plot(history.history['val_acc'], 'b--')

model.save('allnight.h5')

plt.show()



