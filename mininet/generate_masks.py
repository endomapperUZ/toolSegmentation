import utils_mininet.Loader as Loader
from utils_mininet.utils import get_params, preprocess, lr_decay, convert_to_tensors, restore_state, apply_augmentation, get_metrics,init_model
import gc
import nets.MiniNetModif as MiniNetModif
import tensorflow as tf
import sys
import time
import glob
import numpy as np
#import tensorflow as tf
import os
import argparse
import time
import sys
import cv2
#from utils_mininet.utils import inference
import glob
#import tensorflow as tf
#from utils_mininet.utils import preprocess
from dataset_ft import load_image
from dataset_ft import load_mask
import torch
#from utils import cuda
#from generate_masks import get_model
from albumentations import Compose, Normalize
from albumentations.pytorch.functional import img_to_tensor
import tensorflow as tf
import matplotlib.pyplot as plt
from torchsummary import summary
from models import UNet16, LinkNet34, UNet11, UNet, AlbuNet

colab_path = '/home/clara/Documentos/TFM/toolSegmentation/mininet'
sys.path.append(colab_path)

print(tf.__version__)

def img_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)
def val_transform(p=1):
    return Compose([
        #PadIfNeeded(min_height=val_crop_height, min_width=val_crop_width, p=1),
        CenterCrop(height=val_crop_height, width=val_crop_width, p=1),
        Normalize(p=1)
    ], p=p)

def create_clasifier(base_model):
    base_model.trainable = False
    input_shape = (1056,1280,3)
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)(x)
    model = tf.keras.Model(inputs,outputs)
    restore_state(model,colab_path+'/weights_mininet/clasif_mininet_endomapper/model1_0')
    model_encod = tf.keras.Model(inputs = inputs, outputs = model.layers[-4].output)
    input_2 = tf.keras.Input(shape=(132, 160, 128))
    y = input_2
    for layer in model.layers[2:]:
        y = layer(y)
    output_2 = y
    model_class = tf.keras.Model(inputs = input_2, outputs = output_2)
    return model_encod, model_class
    
def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)
    
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model_encod = MiniNetModif.MiniNetv2(num_classes=2,include_top=True)
    model_decod = MiniNetModif.MiniNetDecod(num_classes=2,include_top=True)

    restore_state(model_decod, colab_path + '/weights_mininet/mininet_endomapper/model_0')

    model_encod, model_class = create_clasifier(model_encod)

    print(model_encod.summary())
    print(model_class.summary())

    iou = [[]]
    
    files = [glob.glob('/home/clara/Documentos/TFM/mininet/datasets/conf2/tool_notool/video118_mixed/images/*.png')]
    print(len(files[0]))
    for i in range(len(files)):
        files[i].sort()

        for j in range(len(files[i])):
            img_path = files[i][j]
            gt_path = img_path.replace('images','labels').replace('frame','mask')
	
            img = tf.keras.preprocessing.image.load_img(img_path,0)
            img0 = tf.keras.preprocessing.image.img_to_array(img).astype(np.float32)
            img1 = img0/127.5 - 1 
            input_img = np.expand_dims(img1,0)

            if (input_img.shape == (1,1080,1440,3)):
                image = input_img[:,24:,160:,:]
            elif (input_img.shape == (1,1056,1920,3)):
                image = input_img[:,:,640:,:]
            else:
                image = input_img
            
            gt0 = tf.keras.preprocessing.image.load_img(gt_path,0)
            gt0 = gt0.convert(mode="L")
            gt0 = np.asarray(gt0)
            gt = gt0
            
            if (gt.shape == (1080,1440)):
                mask = gt[24:,160:]
            elif (gt.shape == (1056,1920)):
                mask = gt[:,640:]
            else:
                mask = gt

            features = model_encod(image,training=False)
            
            tool = model_class(features,training = False)
           
            tool = np.array(tool>0.5).astype(np.uint8)
            tool = int(tool)
            
            if tool == 1:
                mask_filt = model_decod(image, features, training = False)
                mask_filt = np.asarray(mask_filt)
                mask_filt = mask_filt.squeeze(0)
                mask_filt = mask_filt[:,:,1]
            else : 
                mask_filt = np.zeros((image.shape[1],image.shape[2]))
            """
            mask_filt = model_decod(image,features)
            mask_filt = np.asarray(mask_filt)
            mask_filt = mask_filt.squeeze(0)
            mask_filt = mask_filt[:,:,1]
            """
            iou_p = jaccard(mask>0,mask_filt>0.3)
            iou[i].append(iou_p)
	
            print(j)    
    print(np.mean(iou[i]))
if __name__ == '__main__':
    main()

