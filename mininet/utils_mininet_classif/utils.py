import numpy as np
import tensorflow as tf
import math
import os
import cv2
import time
import random
import tqdm

# Prints the number of parameters of a model
def get_params(model):
    # Init models (variables and input shape)
    total_parameters = 0
    for variable in model.variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters
    print("Total parameters of the net: " + str(total_parameters) + " == " + str(total_parameters / 1000000.0) + "M")

# preprocess a batch of images
def preprocess(x, mode='imagenet'):
    if mode:
        if 'imagenet' in mode:
            return tf.keras.applications.xception.preprocess_input(x)
        elif 'normalize' in mode:
            #print(np.min(x),np.max(x))
            x = x.astype(np.float32) / 127.5 - 1
            #print(np.min(x),np.max(x))
            return x
            
    else:
        return x

# applies to a lerarning rate tensor (lr) a decay schedule, the polynomial decay
def lr_decay(lr, init_learning_rate, end_learning_rate, epoch, total_epochs, power=0.9):
    lr.assign(
        (init_learning_rate - end_learning_rate) * math.pow(1 - epoch / 1. / total_epochs, power) + end_learning_rate)


# converts a list of arrays into a list of tensors
def convert_to_tensors(list_to_convert):
    if list_to_convert != []:
        return [tf.convert_to_tensor(list_to_convert[0])] + convert_to_tensors(list_to_convert[1:])
    else:
        return []

# restores a checkpoint model
def restore_state(model, checkpoint):
    try:
        model.load_weights(checkpoint)
        print('Model loaded')
    except Exception as e:
        print('Model not loaded: ' + str(e))

# inits a models (set input)
def init_model(model, input_shape):
    model._set_inputs(np.zeros(input_shape))

 

def inference(model, x, y, n_classes, flip_inference=True, scales=[1], preprocess_mode=None, time_exect=False, train=True):
    x = preprocess(x, mode=preprocess_mode)
    [x] = convert_to_tensors([x])
    #print(x)
    #print(x.shape)
    # creates the variable to store the scores
    y_ = convert_to_tensors([np.zeros((1, 1), dtype=np.float32)])[0]

    for scale in scales:
        # scale the image
        x_scaled = tf.image.resize(x, (int(x.shape[1] * scale), int(x.shape[2] * scale)),
                                              method=tf.image.ResizeMethod.BILINEAR)

        pre = time.time()
        #print(x_scaled.shape) 
        y_scaled = model(x_scaled, training=train)
        if time_exect and scale == 1:
            print("seconds to inference: " + str((time.time()-pre)*1000) + " ms")

        
        # get scores
        y_scaled = np.array(y_scaled>0.5).astype(np.uint8)

        y_ += y_scaled

    return y_

# Apply some augmentations
def apply_augmentation(image, labels, mask, size_crop, zoom_factor):
    # image, labels and masks are tensors of shape [b, w, h, c]

    dim_img = image.shape[-1]


    if random.random() > 0.5:
        #size to resize
        r_factor = (random.random() * zoom_factor * 2 - zoom_factor) + 1
        resize_size = (int(image.shape[1]* r_factor), int(image.shape[2]* r_factor))
        image = tf.image.resize(image, resize_size, method=tf.image.ResizeMethod.BILINEAR)


    size_crop = (image.shape[0], size_crop[0], size_crop[1], image.shape[3])
    image = tf.image.random_flip_left_right(image)
    #all = tf.image.random_flip_up_down(all)
    image = tf.image.random_crop(image, size_crop)


    if random.random() > 0.5:
        image = tf.image.random_brightness(image, max_delta=20. / 255.)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.15)



    return image, labels, mask


# get accuracy and miou from a model
def get_metrics(loader, model, loss_function, n_classes,val=False, train=True, flip_inference=False, scales=[1], write_images=False, preprocess_mode=None, time_exect=False,  optimizer=None):
  if val :
    loader.index_val = 0
    accuracy = tf.metrics.BinaryAccuracy()
    samples = len(loader.image_val_list)
    tq = tqdm.tqdm(total=(samples))
    tq.set_description('Evaluation Val')
    losses = []

    for step in range(samples):  # for every batch
        x, y, label, path = loader.get_batch(size=1, train=train, val=val)
        #print(path)
        #print(y)
        [imgs, y] = convert_to_tensors([x.copy(), y])
        #print(y)
        y_ = inference(model, x, y, n_classes, flip_inference, scales, preprocess_mode=preprocess_mode, time_exect=time_exect, train=train)
        #print(y_)
        
        accuracy.update_state(y, y_)
        losses.append(loss_function(y, y_))
        tq.update(1)
    
    tq.close()
    acc_result = accuracy.result()
    valid_loss = np.mean(losses)
    if optimizer != None:         # tensorboard
        #tf.summary.scalar('mIoU', miou_result, step=optimizer.iterations)
        tf.summary.scalar('accuracy', acc_result, step=optimizer.iterations)
    #print(acc_result)
    #print(valid_loss)
  else :
    if train:
        loader.index_train = 0
    else:
        loader.index_test = 0

    accuracy = tf.metrics.BinaryAccuracy()
    #mIoU = tf.metrics.MeanIoU(num_classes=n_classes)
    
    if train:
        samples = len(loader.image_train_list)
    else:
        samples = len(loader.image_test_list)
        tq = tqdm.tqdm(total=(samples))
        tq.set_description('Evaluation')
    losses = []
    
    for step in range(samples):  # for every batch
        x, y, label, path = loader.get_batch(size=1, train=train)

        [imgs, y] = convert_to_tensors([x.copy(), y])

        y_ = inference(model, x, y, n_classes, flip_inference, scales, preprocess_mode=preprocess_mode, time_exect=time_exect, train=train)

        #print(path)
        #print(y)
        #print(y_)
        # tnsorboard
        """
        if optimizer != None and random.random() < 0.05:
            tf.summary.image('input', tf.cast(imgs, tf.uint8), step=optimizer.iterations + step)
            tf.summary.image('labels', tf.expand_dims( tf.cast(tf.argmax(y, 3)*int(255/n_classes), tf.uint8), -1), step=optimizer.iterations + step)
            tf.summary.image('predicions', tf.expand_dims(tf.cast(tf.argmax(y_, 3)*int(255/n_classes), tf.uint8), -1), step=optimizer.iterations + step)
        """
        # generate images
        """
        if write_images:
            generate_image(y_[0,:,:,:], 'images_out', loader.dataFolderPath, loader, train, path[0])
        """
        # Rephape

        #labels, predictions = erase_ignore_pixels(labels=tf.argmax(y, 1), predictions=tf.argmax(y_, 1), mask=mask)
        accuracy.update_state(y, y_)
        #mIoU.update_state(labels, predictions)

        losses.append(loss_function(y, y_))
        if train != True:
          tq.update(1)
    if train != True:
      tq.close()
    # get the train and test accuracy from the model
    #miou_result = mIoU.result()
    acc_result = accuracy.result()
    valid_loss = np.mean(losses)
    if optimizer != None:         # tensorboard
        #tf.summary.scalar('mIoU', miou_result, step=optimizer.iterations)
        tf.summary.scalar('accuracy', acc_result, step=optimizer.iterations)
    #print(acc_result)
    #print(valid_loss)
  return acc_result, valid_loss

