app_config = {

  "_comment": "Example configuration document",  
  "local": {

  },
  "default":{
   "inceptionv3": {
       "img_width": 256;
       "img_height": 256,
       "number_of_epochs": 3,
       "batch_size": 32,
       "fc_size":1024,
       "activation":['relu', 'sigmoid', 'tanh'],
       "loss": ['crossentropy', 'binary_crossentropy', 'categorical_crossentropy'],
       "output_classes": 2,
       "optimizer": ['sgd', 'adam', 'rmsprop', 'adagrad'],
       "learning_rate": 0.0001,
       

   },
   "paths":{
        "chnmcu_train_imgs": '/mnt/data/datasets/keras_trnsf_data/chnmcu256train/',
         "chnmcu_validation_imgs": '/mnt/data/datasets/keras_trnsf_data/chnmcu256validate/',
   }
      
  } 
}