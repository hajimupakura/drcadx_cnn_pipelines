import os
import numpy as np
import pandas as pd 
import random
import cv2from keras.preprocessing.image import ImageDataGenerator

'''
	Arguments:
	img_dims =  input size (256 or 224)
	batch_size =  batch_size you want
	train_dir = path to the training images
	validation_dir = path to the validation images
	test_dir = path to the test images used for our model prediction
	class_mode = either binary or categorical
	normal = directory name underneath test_dir where our normal images reside
	disease = directory name underneath test_dir where our abnormal/disease images reside
	
	Please make a variant of this train_datagen that fits your training needs. Not all the 
	augmentation parameters are needed.
	Below are the parameters you can include
	rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
	
		'''
	
def process_data(img_dims, batch_size, train_dir,validation_dir,test_dir, class_mode, normal, disease):
    # Data Generation Objects	
	train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
			vertical_flip=True)
     
	# Note that the validation data should not be augmented! 
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_dims, img_dims),
            batch_size=batch_size,
            class_mode=class_mode)
    
    validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(img_dims, img_dims),
            batch_size=batch_size,
            class_mode=class_mode)
			
    # I will be making predictions off of the test set in one batch size
    # This is useful to be able to get the confusion matrix
    test_data = []
    test_labels = []

    for cond in ['/normal/', '/disease/']:
        for img in (os.listdir(test_dir + cond)):
            img = plt.imread(input_path+cond+img)
            img = cv2.resize(img, (img_dims, img_dims))
            img = np.dstack([img, img, img])
            img = img.astype('float32') / 255
            if cond=='/normal/':
                label = 0
            elif cond=='/disease/':
                label = 1
            test_data.append(img)
            test_labels.append(label)
        
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)			
			
    return 	train_generator, validation_generator, test_data,test_labels

#How to use the function
# train_gen, test_gen, test_data, test_labels = process_data(img_dims, batch_size, train_dir,validation_dir,test_dir, class_mode, normal, disease)
# model.predict(test_data)
# acc = accuracy_score(test_labels, np.round(preds))*100  (Confusion Matrix)	