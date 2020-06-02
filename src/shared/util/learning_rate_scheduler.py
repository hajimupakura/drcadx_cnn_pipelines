import numpy
import math
from keras.callbacks import LearningRateScheduler

#STEP DECAY LEARNING RATE (LEARNING RATE SCHEDULER)
def step_decay(initial_lrate,drop, epochs_drop):
   ''' This is a function to implement a step decaying learning rate (Learning rate
        drops by whatever percentage every x amount of epochs)
       initial_lrate = the initial learning rate where you want to start at e.g 0.0001
       drop = A fraction between 0 and 1. This indicates how much you want the learning rate to drop e.g 0.5 indicates a 50% drop in learning rate
       epochs_drop = the number of epochs after which you waant the learning rate to drop e.g 10.0 indicates an lrate drop after every 10 epochs 
   ''' 
   initial_lrate = initial_lrate
   drop = drop
   epochs_drop = epochs_drop
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

'''This is how the function is called below

lrate = LearningRateScheduler(step_decay(0.0001,0.5,10.0))
callbacks_list = [lrate]
 OR
initial_lrate = 0.0001
drop = 0.5
epochs_drop = 10.0

lrate = LearningRateScheduler(step_decay(initial_lrate,drop, epochs_drop))
callbacks_list = [lrate]
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=50,
      callbacks=callbacks_list,
      verbose=2)
'''

def poly_decay(num_epochs,learning_rate):
	# initialize the maximum # of epochs, base learning rate,
	# and power of the polynomial
	maxEpochs = num_epochs
	baseLR = learning_rate
	power = 1.0  # turns our polynomial decay into a linear decay
 
	# compute the new learning rate based on polynomial decay
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
 
	# return the new learning rate
	return alpha