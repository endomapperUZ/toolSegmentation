from dataset_ft import load_image
from dataset_ft import load_mask
from pylab import *
import matplotlib.pyplot as plt
import cv2
import glob
import os

def main():
  files_train = glob.glob('/workspace/ctomasin/toolSegmentation/data/train_data/train_masks/video39*.png')
  files_train.sort()
  print(len(files_train))
  for i in range(3168):
    mask_file_name39 = files_train[i]
    img_file_name39 = mask_file_name39.replace('masks','raw').replace('mask','frame')
    image = load_image(mask_file_name39)
    mask = load_mask(mask_file_name39,'binary')
    print(i)
    print('shape image before resize : '+ str(image.shape))
    if (image.shape==(1080,1440,3)):
      image = image[24:,:,:]
      print('shape image after resize : '+ str(image.shape))

      print('shape mask : '+ str(mask.shape))

      image = image[:,:,::-1]
      img_file_name39 = mask_file_name39.replace('masks','raw').replace('mask','frame')
      print(img_file_name39)
      os.remove(img_file_name39)
      os.remove(mask_file_name39)
      cv2.imwrite(mask_file_name39, mask)
      cv2.imwrite(img_file_name39, image)
    else :     
      print('No resize necessary')

if __name__ == '__main__':
    main()