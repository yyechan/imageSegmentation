import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from keras.utils.data_utils import Sequence
from tensorflow.keras.utils import to_categorical
import glob




ypath_list = glob.glob('./data/trainY/1/*.png')


ypath_list = [filename.replace('\\','/') for filename in ypath_list]

ypath_list.sort()

print(ypath_list)

# Xdata = np.array(Xdata.append(np.array(Image.open(path)) for path in xpath_list))

for idx,path in enumerate(ypath_list):
    img = np.array(Image.open(path))

    newmask = np.zeros((img.shape))

    for i in range(img.shape[0]) :
        for j in range(img.shape[1]) :
            r = img[i][j][0]
            g = img[i][j][1]
            b = img[i][j][2]
            if(r==255 and g == 0 and b == 0) :
                newmask[i][j][1] = 1
            elif(r==255 and g == 255 and b == 0):
                newmask[i][j][2] = 1
            else :
                newmask[i][j][0] = 1
            
    data = Image.fromarray(newmask.astype('uint8'))
    data.save('./'+path.replace('./data/trainY/1/',''))
   



