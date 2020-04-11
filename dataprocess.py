import numpy as np
from PIL import Image
from matplotlib import pyplot as plt



# Generates labels using most basic setup.  Supports various image sizes.  Returns image labels in same format
# as original image.  Normalization matches MobileNetV2

trained_image_width=512 
mean_subtraction_value=127.5
image = np.array(Image.open('./개별연구/data/14.jpg'))

# resize to max dimension of images from training dataset

resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((512, 512)))

img = Image.fromarray(resized_image, 'RGB')

img.save('test.png')
img.show()
