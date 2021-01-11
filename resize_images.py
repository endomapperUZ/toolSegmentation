from dataset_ft import load_image
from dataset_ft import load_mask
from pylab import *
import matplotlib.pyplot as plt
import cv2
import glob
import os

def main():
  files_train = glob.glob('/workspace/ctomasin/toolSegmentation/data/train_data/train_masks/*.png')
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

      image = image[:,:,::-1]
      img_file_name39 = mask_file_name39.replace('masks','raw').replace('mask','frame')
      print(img_file_name39)
      os.remove(img_file_name39)
      cv2.imwrite(img_file_name39, image)
    else :     
      print('No image resize necessary')

    print('shape mask before resize : '+ str(mask.shape))
    if (mask.shape==(1080,1440)):
      mask = mask[24:,:]
      print('shape mask after resize : '+ str(mask.shape))

      os.remove(mask_file_name39)
      cv2.imwrite(mask_file_name39, mask)
    else:
      print('No mask resize necessary')


  for i in range(3168,14267):
    mask_file_name6 = files_train[i]
    img_file_name6 = mask_file_name6.replace('masks','raw').replace('mask','frame')
    image = load_image(mask_file_name6)
    mask = load_mask(mask_file_name6,'binary')
    print(i)
    print('shape image before resize : '+ str(image.shape))
    if (image.shape==(1080, 1920, 3)):
      image = image[24:, 640:,:]
      print('shape image after resize : '+ str(image.shape))

      image = image[:,:,::-1]
      img_file_name6 = mask_file_name6.replace('masks','raw').replace('mask','frame')
      print(img_file_name6)
      os.remove(img_file_name6)
      cv2.imwrite(img_file_name6, image)  
    else :
      print('No resize necessary')

    print('shape mask before resize : '+ str(mask.shape))
    if (mask.shape==(1080,1920)):
      mask = mask[24:,640:]
      print('shape mask after resize : '+ str(mask.shape))

      os.remove(mask_file_name6)
      cv2.imwrite(mask_file_name6, mask)
    else:
      print('No mask resize necessary')

if __name__ == '__main__':
    main()