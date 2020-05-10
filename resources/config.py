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
       "weights_model_name": 'tb_inceptionv3_best.hd5',
       "model_weights_save_path": '',
       "num_epochs": 20,
       "batch_size": 32,
       "fc_size":1024,
       "activation":['relu', 'sigmoid', 'tanh'],
       "loss": ['crossentropy', 'binary_crossentropy', 'categorical_crossentropy'],
       "output_classes": 2,
       "optimizer": ['sgd', 'adam', 'rmsprop', 'adagrad'],
       "class_mode": ['binary', 'categorical'],
       "learning_rate": 0.0001,
       'earlychkpt_patience': 10
       "step_decay_lr": '', # Please make sure the value is empty string if you don't want to use this. Otherwise set it to True
       "poly_decay_lr": '', # Please make sure the value is empty string if you don't want to use this. Otherwise set it to True
       "base_path": '',
       "train_split": 0.8,
       "val_split": 0.1,
       "loss_accuracy_plot_path": '' # This is for resnet34
       "version": 121, # if resnet or densenet specify version (121, 169, 201) 
       "fine_tune": True,
       "layer_trainable_idx_point": 16,
       "disease_label": 'TB',
       "non_disease_label": 'Normal'
   },
    "resnet34": {
       "img_width": 64;
       "img_height": 64,
       "channels": 3,
       "stride": 1,
       "kernel": 3,
       "weights_model_name": 'resnet34_mdm.h5',
       "model_weights_save_path": '',
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
       "loss_accuracy_plot_path": '',
       "train_path": '',
       "val_path": '',
       "test_path": ''        

   },
   "paths":{
        "chnmcu_train_imgs": '/mnt/data/datasets/keras_trnsf_data/chnmcu256train/',
        "chnmcu_validation_imgs": '/mnt/data/datasets/keras_trnsf_data/chnmcu256validate/',
   }
      
  } 
}