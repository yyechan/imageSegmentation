import Augmentor
from PIL import Image
from matplotlib import pyplot as plt
from keras.utils.data_utils import Sequence


p = Augmentor.Pipeline("./data/data_real")
# Point to a directory containing ground truth data.
# Images with the same file names will be added as ground truth data
# and augmented in parallel to the original data.
p.ground_truth("./data/annotation_real/")
# Add operations to the pipeline as normal:

p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_top_bottom(probability=0.5)
p.crop_random(probability=1, percentage_area=0.5)
p.resize(probability=1.0, width=512, height=512)

p.sample(100)