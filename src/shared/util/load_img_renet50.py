import numpy as np
import cv2
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input    

def load_image_resnet50(img_path, input_shape):
    # img_path is the path to your image(s) and input shape is the height or width of your image (224)
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(input_shape, input_shape))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor (add number of images)
    x = np.expand_dims(x, axis=0)
    # convert RGB -> BGR, subtract mean ImageNet pixel, and return 4D tensor
	x = preprocess_input(x)
    return x