import numpy as np
import tensorflow as tf
import os
import argparse
import time
import sys
import cv2
from pathlib import Path
from datetime import datetime
import json
import tqdm

import utils_mininet.Loader as Loader
from utils_mininet.utils import get_params, preprocess, lr_decay, convert_to_tensors, restore_state, apply_augmentation, get_metrics,init_model
import nets.MiniNetv2 as MiniNetv2
import nets.ResNet50 as ResNet50

def main():

  parser = argparse.ArgumentParser()
  arg = parser.add_argument
  arg('--path',type=str)
  arg('--n_classes',type=int)
  arg('--batch_size',type=int)
  arg('--epochs',type=int)
  arg('--init_lr',type=float)
  arg('--crop_factor_x',type=float)
  arg('--crop_factor_y',type=float)
  arg('--zoom_augmentation',type=float)

  args = parser.parse_args()
  device_name = tf.test.gpu_device_name()
  print(device_name)
  if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
  print('Found GPU at: {}'.format(device_name))
  
  def train_step(model, x, y, mask, loss_function, optimizer, size_input, zoom_factor):
    with tf.GradientTape() as tape: 

      [x, y, mask] = convert_to_tensors([x, y, mask]) # convert numpy data to tensors
      mask = tf.expand_dims(mask, axis=-1) 

      x, y, mask = apply_augmentation(x, y, mask, size_input, zoom_factor) # Do data augmentation
      y_ = model(x, training=True)  # get output of the model, prediction

      y *= mask # Apply mask to ignore labels

      loss = loss_function(y, y_) # apply loss function
      # print(loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))# Apply gradientes

    return loss
  
  # Trains the model for certains epochs on a dataset
  def train(loader, optimizer, loss_function, model, config=None, lr=None,
          evaluation=True, name_best_model='weights/best', preprocess_mode=None,step_num=0):
  
    # Parameters for training
    training_samples = len(loader.image_train_list)
    steps_per_epoch = int(training_samples / config['batch_size']) + 1
    best_miou = 0
    for epoch in range(config['epochs']):  # for each epoch
        lr_decay(lr, config['init_lr'], 1e-9, epoch, config['epochs'] - 1)  # compute the new learning rate
        print('epoch: ' + str(epoch+1) + '. Learning rate: ' + str(lr.numpy()))
        tq = tqdm.tqdm(total=(steps_per_epoch))
        tq.set_description('Epoch {}, lr {}'.format(epoch+1, lr.numpy()))
        for step in range(steps_per_epoch):  # for every batch
            # get batch
            x, y, mask, path = loader.get_batch(size=config['batch_size'], train=True)

            x = preprocess(x, mode=preprocess_mode) # preprocess data

            # do a train step 
            loss = train_step(model, x, y, mask, loss_function, optimizer, (config['height_train'], config['width_train']), config['zoom_augmentation'])
            tq.update(1)
            step_num += 1
            tq.set_postfix(loss='{:.5f}'.format(loss.numpy()))
            write_event(log,step_num,loss = float(loss.numpy()))
        print('done steps')
        tq.close()
        train_acc, train_miou, train_loss = get_metrics(loader, model, loss_function, loader.n_classes, train=True, flip_inference=False, preprocess_mode=preprocess_mode, optimizer=optimizer, scales=[1])
        print('done metrics')
        train_metrics = {'train_acc': float(train_acc.numpy()), 'train_miou': float(train_miou.numpy()), 'train_loss' : float(train_loss)}
        write_event(log, step_num, **train_metrics)   
        # When the epoch finishes, evaluate the model
        if evaluation: 
            print('evaluation')
            test_acc, test_miou, valid_loss = get_metrics(loader, model, loss_function, loader.n_classes, train=False, flip_inference=False, preprocess_mode=preprocess_mode, optimizer=optimizer, scales=[1])
            print('Test accuracy: ' + str(test_acc.numpy()))
            print('Test miou: ' + str(test_miou.numpy()))
            valid_metrics = {'test_acc': float(test_acc.numpy()), 'test_miou': float(test_miou.numpy()),'test_loss' : float(valid_loss)}
            write_event(log, step_num, **valid_metrics)
            # save model if best model
            if test_miou.numpy() > best_miou:
                best_miou = test_miou.numpy()
                model.save_weights(name_best_model)

            print('Current Best model miou: ' + str(best_miou))
            print('')

        else:
            model.save_weights(name_best_model)

  def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


  # Configuration for the training
  CONFIG = {}
  CONFIG['n_classes'] = args.n_classes # number of the classes of the dataset
  CONFIG['batch_size'] = args.batch_size # batch size
  CONFIG['epochs'] = args.epochs # Number of epochs to train

  # Training loop
  #datafiles = ['video6','video9','video39']
  datafiles = ['toolSeg']
  widths = [1280]
  heights = [1056]
  #widths = [640]
  #heights = [480]

  model_name = 'model_0'
  log = Path(args.path).joinpath('train_{}.log'.format(model_name)).open('at', encoding='utf8') 

  step_num = 0

  for i in range(len(datafiles)):
    #print(datafiles[i])
    '''
    width and height to read the images (it can be the same or different. the higher, the slower it will train and the more memory it will need)
    '''
    CONFIG['width'] = widths[i]
    CONFIG['height'] = heights[i]
    CONFIG['init_lr'] = args.init_lr # Initial learning rate
    '''
    Data augmentation parameters:
    crop rate on the X (width) and height (y)
    Zoom in/out apply_augmentation
    '''
    CONFIG['crop_factor_x'] = args.crop_factor_x
    CONFIG['crop_factor_y'] = args.crop_factor_y
    CONFIG['zoom_augmentation'] = args.zoom_augmentation

    CONFIG['width_train'] = int(CONFIG['width'] / CONFIG['crop_factor_x']) # will be cropped from width_test size
    CONFIG['height_train']  = int(CONFIG['height'] / CONFIG['crop_factor_y'])  # will be cropped from height_test size

    assert CONFIG['width'] * (1 - CONFIG['zoom_augmentation'] ) >= CONFIG['width_train']
    assert CONFIG['height'] * (1 - CONFIG['zoom_augmentation'] ) >= CONFIG['height_train']

    # GPU to use
    n_gpu = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(n_gpu)

    # Data Loader
    loader = Loader.Loader(dataFolderPath= args.path + '/datasets/'+datafiles[i], n_classes=CONFIG['n_classes'], width=CONFIG['width'], height=CONFIG['height'], median_frequency=0.)

    # Define your model
    #if i==0 :
      # build model
    model = MiniNetv2.MiniNetv2(num_classes=CONFIG['n_classes'])
    restore_state(model, args.path + '/weights_mininet/endovis_50/model_0')
    #else :
      # restore if model saved and show number of params
      #restore_state(model, colab_path + '/weights_mininet/tool_seg/'+model_name)
  
    # optimizer
    learning_rate = tf.Variable(CONFIG['init_lr'])
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_function = tf.keras.losses.CategoricalCrossentropy()

    # Train and evaluate
    
    train(loader=loader, optimizer=optimizer, loss_function=loss_function, model=model, config=CONFIG,
          lr=learning_rate,  name_best_model=args.path+'/weights_mininet/tool_seg/'+model_name, evaluation=True, preprocess_mode='normalize',step_num=step_num)


    print('Testing model')
    test_acc, test_miou,_ = get_metrics(loader, model, loss_function, loader.n_classes, train=False,  flip_inference=True, scales=[0.75,1,1.25,1.5,1.75],
                                      write_images=True, preprocess_mode='normalize', time_exect=True)
    print('Test accuracy: ' + str(test_acc.numpy()))
    print('Test miou: ' + str(test_miou.numpy()))

    log.close()
    '''
    you can change the parameters 
    flip_inference=True, scales=[0.75,1,1.25,1.5,1.75]
    For better results but it will be slower
    '''

if __name__ == '__main__':
    main()
