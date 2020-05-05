import numpy as np
import cv2

'''
I found that preprocessing your data while yours is a too different dataset vs the pre_trained model/dataset,
 then it may harm your accuracy somehow. If you do transfer learning and freezing some layers from a
 pre_trained model/ther weights, simply /255.0 your original dataset does the job just fine. In other terms
 use the load_img_cv2 function

'''

# Please make sure the input tensor is in the same shape your model takes 
def load_img_cv2(image_path, input_shape ):
    # img_path is the path to your image(s) and input shape is the height or width of your image (224)
    target_size=(input_shape,input_shape)
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img / 255 # Normalize the images to between 0 and 1
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img,(1,) + img.shape)
    return img	
	