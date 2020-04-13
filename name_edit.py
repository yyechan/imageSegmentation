from model import unet
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model

from keras_preprocessing import image
import os

xpath_list = glob.glob('./data/data_real/output/*')
# ypath_list = glob.glob('./data/realY/*.png')




# new_name = [filename.replace('_groundtruth_(1)_data_real_','') for filename in xpath_list]



# for filename in os.listdir('./data/data_real/'):
#      # 파일 확장자가 (properties)인 것만 처리 
#      # 파일명에서 AA를 BB로 변경하고 파일명 수정
#     new_filename = filename.replace("_groundtruth_(1)_data_real_", "")
#     os.rename(filename, new_filename)


new_name = [filename.replace('data_real_original_','') for filename in xpath_list]

for idx,name in enumerate(xpath_list):
    os.rename(name,new_name[idx])