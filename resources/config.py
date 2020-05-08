app_config = {

  "_comment": "Example configuration document",  
  "local": {

  },
  "default":{
   "inceptionv3": {
       "img_width": 256;
       "img_height": 256,
       "channels": 3,
       "stride": 1,
       "kernel": 3,
       "weights_model_name": '',
       "num_epochs": 20,
       "batch_size": 32,
       "fc_size":1024,
       "activation":['relu', 'sigmoid', 'tanh'],
       "loss": ['crossentropy', 'binary_crossentropy', 'categorical_crossentropy'],
       "output_classes": 2,
       "optimizer": ['sgd', 'adam', 'rmsprop', 'adagrad'],
       "class_mode": ['binary', 'categorical'],
       "learning_rate": 0.0001,
       "base_path": '',
       "train_split": 0.8,
       "val_split": 0.1,
       "loss_accuracy_plot_path": ''  

   },
    "resnet34": {
       "img_width": 64;
       "img_height": 64,
       "channels": 3,
       "stride": 1,
       "kernel": 3,
       "weights_model_name": 'resnet34_mdm.h5',
       "num_epochs": 50,
       "batch_size": 32,
       "fc_size":1024,
       "activation":['relu', 'sigmoid', 'tanh'],
       "loss": ['crossentropy', 'binary_crossentropy', 'categorical_crossentropy', 'l1','l2'],
       "output_classes": 2,
       "optimizer": ['sgd', 'adam', 'rmsprop', 'adagrad'],
       "class_mode": ['binary', 'categorical'],
       "learning_rate": 1e-1,
       "stages": [3, 4, 6],
       "filters":[64, 128, 256, 512],
       "orig_input_dataset": '',
       "base_path": '',
       "train_split": 0.8,
       "val_split": 0.1,
       "loss_accuracy_plot_path": ''        

   },
   "paths":{
        "chnmcu_train_imgs": '/mnt/data/datasets/keras_trnsf_data/chnmcu256train/',
         "chnmcu_validation_imgs": '/mnt/data/datasets/keras_trnsf_data/chnmcu256validate/',
   }
      
  } 
}