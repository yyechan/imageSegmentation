import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt



for i in range(1,21):

    path = './resize/'
    path = path + str(i)
    image = np.array(Image.open(path + '.png'))
    resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((512, 512)))
    img = Image.fromarray(resized_image, 'RGB')
    img.save('./data/trainY/' + str(i) + '.png')
    

