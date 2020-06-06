import numpy as np
import pandas as pd
import os
import sys
import keras
from keras.optimizers import *
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, BatchNormalization, Activation, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from resources.config import app_config as config
from imutils import paths
from keras.callbacks import LearningRateScheduler
from src.shared.util.learning_rate_scheduler import step_decay, poly_decay




	
# Actual Train

def main(model_name, train_path, val_path, test_path):
    
    if len(sys.argv) < 4:
        print("Number of arguments less than 4. Please provide 4 arguments")
        sys.exit()

    model_name = sys.argv[1]
    train_path = sys.argv[2]
    test_path = sys.argv[3]
    val_path = sys.argv[4]

    def mobilenet_architecture():
    """
    Pre-build architecture of mobilenet for our dataset.
    """
    # Imprting the model
    from keras.applications.mobilenet import MobileNet
    
	fine_tune = config['default']['mobilenet']['fine_tune']
	layer_trainable_idx_point = config['default']['mobilenet']['layer_trainable_idx_point']
    # Pre-build model
    base_model = MobileNet(include_top = False, weights = None, input_shape = (img_width, img_height, channels))

    # Adding output layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(units = output_classes, activation = 'softmax')(x)

    # Creating the whole model
    mobilenet_model = Model(base_model.input, output)
	
	#Finetune or not
	if fine_tune == 'True':
	    for layer in mobilenet_model.layers:
            layer.trainable=False
    else:
	    for layer in mobilenet_model.layers[:layer_trainable_idx_point]:
            layer.trainable=False
        for layer in mobilenet_model.layers[layer_trainable_idx_point:]:
            layer.trainable=True
	    
    # Getting the summary of architecture
    #mobilenet_model.summary()
    
    # Compiling the model
    mobilenet_model.compile(optimizer = keras.optimizers.Adam(lr = learning_rate), 
                            loss = loss, 
                            metrics = ['accuracy'])

    return mobilenet_model
	
	
    def inception_architecture():
        """
        Pre-build architecture of inception for our dataset.
        """
        # Imprting the model 
        from keras.applications.inception_v3 import InceptionV3
        
    	fine_tune = config['default']['inceptionv3']['fine_tune']
    	layer_trainable_idx_point = config['default']['inceptionv3']['layer_trainable_idx_point']
        # Pre-build model
        base_model = InceptionV3(include_top = False, weights = None, input_shape = (img_width, img_height, channels))
    
        # Adding output layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        output = Dense(units = output_classes, activation = 'softmax')(x)
    
        # Creating the whole model
        inception_model = Model(base_model.input, output)
        
    	#Finetune or not
    	if fine_tune == 'True':
    	    for layer in inception_model.layers:
                layer.trainable=False
        else:
    	    for layer in inception_model.layers[:layer_trainable_idx_point]:
                layer.trainable=False
            for layer in inception_model.layers[layer_trainable_idx_point:]:
                layer.trainable=True
    			
        # Summary of the model
        #inception_model.summary()
        
        # Compiling the model
        inception_model.compile(optimizer = keras.optimizers.Adam(lr = learning_rate), 
                                loss = loss, 
                                metrics = ['accuracy'])
        
        return inception_model
    
    def densenet_architecture(version):
        """
        Pre-build architecture of inception for our dataset.
        """
    	densenet_version = "Densenet" + version
        # Imprting the model 
        from keras.applications.densenet import densenet_version
        
    	fine_tune = config['default']['densenet']['fine_tune']
    	layer_trainable_idx_point = config['default']['densenet']['layer_trainable_idx_point']
        # Pre-build model
        base_model = densenet_version(include_top = False, weights = None, input_shape = (img_width, img_height, channels))
    
        # Adding output layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        output = Dense(units = output_classes, activation = 'softmax')(x)
    
        # Creating the whole model
        densenet_model = Model(base_model.input, output)
        
    	#Finetune or not
    	if fine_tune == 'True':
    	    for layer in densenet_model.layers:
                layer.trainable=False
        else:
    	    for layer in densenet_model.layers[:layer_trainable_idx_point]:
                layer.trainable=False
            for layer in densenet_model.layers[layer_trainable_idx_point:]:
                layer.trainable=True
    			
        # Summary of the model
        #inception_model.summary()
        
        # Compiling the model
        densenet_model.compile(optimizer = keras.optimizers.Adam(lr = learning_rate), 
                                loss = loss, 
                                metrics = ['accuracy'])
        
        return densenet_model
    	
    def resnet_architecture(version):
        """
        Pre-build architecture of inception for our dataset.
        """
    	resnet_version = "Resnet" + version
        # Imprting the model 
        from keras.applications.densenet import resnet_version
        
    	fine_tune = config['default']['resnet']['fine_tune']
    	layer_trainable_idx_point = config['default']['resnet']['layer_trainable_idx_point']
        # Pre-build model
        model = resnet_version(include_top = False, weights = None, input_shape = (img_width, img_height, channels))
    
        # Adding output layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        output = Dense(units = output_classes, activation = 'softmax')(x)
    
        # Creating the whole model
        resnet_model = Model(base_model.input, output)
        
    	#Finetune or not
    	if fine_tune == 'True':
    	    for layer in resnet_model.layers:
                layer.trainable=False
        else:
    	    for layer in resnet_model.layers[:layer_trainable_idx_point]:
                layer.trainable=False
            for layer in resnet_model.layers[layer_trainable_idx_point:]:
                layer.trainable=True
    			
        # Summary of the model
        #inception_model.summary()
        
        # Compiling the model
        resnet_model.compile(optimizer = keras.optimizers.Adam(lr = learning_rate), 
                                loss = loss, 
                                metrics = ['accuracy'])
        
        return densenet_model	
    
    def xception_architecture():
        """
        Pre-build architecture of inception for our dataset.
        """
        # Imprting the model
        from keras.applications.xception import Xception
        
    	fine_tune = config['default']['xception']['fine_tune']
    	layer_trainable_idx_point = config['default']['xception']['layer_trainable_idx_point']
        # Pre-build model
        base_model = Xception(include_top = False, weights = None, input_shape = (img_width, img_height, channels))
    
        # Adding output layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        output = Dense(units = output_classes, activation = 'softmax')(x)
    
        # Creating the whole model
        xception_model = Model(base_model.input, output)
        
    	#Finetune or not
    	if fine_tune == 'True':
    	    for layer in xception_model.layers:
                layer.trainable=False
        else:
    	    for layer in xception_model.layers[:layer_trainable_idx_point]:
                layer.trainable=False
            for layer in xception_model.layers[layer_trainable_idx_point:]:
                layer.trainable=True
    			
        # Summary of the model
        #xception_model.summary()
        
        # Compiling the model
        xception_model.compile(optimizer = keras.optimizers.Adam(lr = learning_rate), 
                               loss = loss, 
                               metrics = ['accuracy'])
    
        return xception_model  
    
	# Extract / Load dictionary
	if model_name == 'inceptionv3':
	    base_mod_dict = config['default']['inceptionv3']
	elif model_name == 'resnet':
	    base_mod_dict = config['default']['resnet']	
	elif model_name == 'densenet':
	    base_mod_dict = config['default']['densenet']    
	elif model_name == 'mobilenet':
	    base_mod_dict = config['default']['mobilenet']
    elif model_name == 'xception':
	    base_mod_dict = config['default']['xception']
		
		
    # defining constants and variables
    img_width = base_mod_dict['img_width']
    img_height = base_mod_dict['img_height']
    train_data_dir = train_path
    validation_data_dir = val_path
    test_data_dir = test_path
    output_classes = base_mod_dict['output_classes']
    batch_size = base_mod_dict['batch_size']
    num_epochs = base_mod_dict['num_epochs']
	# This is for Resnet or Densenet
    version = base_mod_dict['version']
	weights_model_name = base_mod_dict['weights_model_name']
	model_save_path = base_mod_dict['model_weights_save_path']
	step_decay_lr = base_mod_dict['step_decay_lr']
	poly_decay_lr = base_mod_dict['poly_decay_lr']
	earlychkpt_patience = base_mod_dict['earlychkpt_patience']
    class_mode = base_mod_dict['class_mode']
    channels = base_mod_dict['channels']
    learning_rate = base_mod_dict['learning_rate']
    loss = base_mod_dict['loss']
	
	# determine the # of image paths in training/validation/testing directories
    totalTrain = len(list(paths.list_images(train_path)))
    totalVal = len(list(paths.list_images(val_path)))
    totalTest = len(list(paths.list_images(test_path)))
    
    # initialize the training data augmentation object
    # randomly shifts, translats, and flips each training sample
    # trainAug = ImageDataGenerator(
    # 	rescale=1 / 255.0,
    # 	rotation_range=20,
    # 	zoom_range=0.05,
    # 	width_shift_range=0.05,
    # 	height_shift_range=0.05,
    # 	shear_range=0.05,
    # 	horizontal_flip=True,
    # 	fill_mode="nearest")

    trainAug = ImageDataGenerator(
    	rescale=1 / 255.0,
    	fill_mode="nearest")
     
    # initialize the validation (and testing) data augmentation object
    valAug = ImageDataGenerator(rescale=1 / 255.0)
    
    # initialize the training generator
    trainGen = trainAug.flow_from_directory(
    	train_path,
    	class_mode=class_mode,
    	target_size=(img_width,  img_height),
    	shuffle=True,
    	batch_size=batch_size)
     
    # initialize the validation generator
    valGen = valAug.flow_from_directory(
    	val_path,
    	class_mode=class_mode,
    	target_size=(img_width,  img_height),
    	shuffle=False,
    	batch_size=batch_size)
     
    # initialize the testing generator
    testGen = valAug.flow_from_directory(
    	test_path,
    	class_mode=class_mode,
    	target_size=(img_width,  img_height),
    	shuffle=False,
    	batch_size=batch_size)
    
    # initialize our ResNet model and compile it
	if model_name == 'inceptionv3':
	    model = inception_architecture()
	elif model_name == 'resnet':
	    model = resnet_architecture(version)	
	elif model_name == 'densenet':
	    model = resnet_architecture(version)    
	elif model_name == 'mobilenet':
	    model = mobilenet_architecture()
    elif model_name == 'xception':
	    model = xception_architecture()
	
   # Callbacks	
	earlycheckpoint = EarlyStopping(monitor = 'val_acc', min_delta = 0, 
                      patience = earlychkpt_patience, verbose= 1 , mode = 'auto')	
    
	checkpointer = ModelCheckpoint(filepath= model_save_path + weights_model_name, 
                               verbose=1, 
                               save_best_only=True)
							   
    
        trainAug = ImageDataGenerator(
        	rescale=1 / 255.0,
        	fill_mode="nearest")
         
        # initialize the validation (and testing) data augmentation object
        valAug = ImageDataGenerator(rescale=1 / 255.0)
        
        # initialize the training generator
        trainGen = trainAug.flow_from_directory(
        	train_path,
        	class_mode=class_mode,
        	target_size=(img_width,  img_height),
        	shuffle=True,
        	batch_size=batch_size)
         
        # initialize the validation generator
        valGen = valAug.flow_from_directory(
        	val_path,
        	class_mode=class_mode,
        	target_size=(img_width,  img_height),
        	shuffle=False,
        	batch_size=batch_size)
         
        # initialize the testing generator
        testGen = valAug.flow_from_directory(
        	test_path,
        	class_mode=class_mode,
        	target_size=(img_width,  img_height),
        	shuffle=False,
        	batch_size=batch_size)
        
        # initialize our ResNet model and compile it
	    if model_name == 'inceptionv3':
	        model = inception_architecture()
	    elif model_name == 'resnet':
	        model = resnet_architecture(version)	
	    elif model_name == 'densenet':
	        model = resnet_architecture(version)    
	    elif model_name == 'mobilenet':
	        model = mobilenet_architecture()
        elif model_name == 'xception':
	        model = xception_architecture()
	    
   #     Callbacks	
	    earlycheckpoint = EarlyStopping(monitor = 'val_acc', min_delta = 0, 
                          patience = earlychkpt_patience, verbose= 1 , mode = 'auto')	
        
	    checkpointer = ModelCheckpoint(filepath= model_save_path + weights_model_name, 
                                   verbose=1, 
                                   save_best_only=True)
	    						   
        
	    # LearningRateScheduler
	    if poly_decay_lr == 'True' and step_decay_lr == 'True':
	        raise Exception("Sorry, you can only use one LearningRateScheduler!!! Make sure only one is set to True in the config file")   
	    if poly_decay_lr == 'True':
	        callbacks = [LearningRateScheduler(poly_decay),earlycheckpoint, checkpointer]
        elif step_decay_lr == 'True':
	        callbacks = [LearningRateScheduler(step_decay),earlycheckpoint, checkpointer]		
        else:
	        callbacks = [checkpointer]
	    	
	    # define our set of callbacks and fit the model	
        model.fit_generator(
        	trainGen,
        	steps_per_epoch=totalTrain // batch_size,
        	validation_data=valGen,
        	validation_steps=totalVal // batch_size,
        	epochs=num_epochs,
        	callbacks=callbacks,
	    	verbose = 2)
        	
        