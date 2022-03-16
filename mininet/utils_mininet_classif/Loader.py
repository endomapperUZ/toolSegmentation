from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import glob
import cv2
from utils_mininet_classif import LoaderQueue
import time
import threading

class Loader:
    def __init__(self, dataFolderPath, width=224, height=224, n_classes=21, median_frequency=0.):
        '''
        Initializes the Loader

        :param dataFolderPath: Path to the dataset
        :param width: with to load the images
        :param height: height to load the images
        :param n_classes: number of classes of the dataset
        :param median_frequency: factor to power the median frequency balancing (0 to none effect, 1 to full efect)
        '''

        self.dataFolderPath = dataFolderPath
        self.height = height
        self.width = width
        self.dim = 3
        self.freq = np.zeros(n_classes)  # vector for calculating the class frequency
        self.index_train = 0  # indexes for iterating while training
        self.index_test = 0  # indexes for iterating while testing
        self.index_val = 0
        self.median_frequency_soft = median_frequency  # softener value for the median frequency balancing (if median_frequency==0, nothing is applied, if median_frequency==1, the common formula is applied)
        self.lock = threading.Lock()
        print('Reading files...')

        # Load filepaths
        files = glob.glob(os.path.join(dataFolderPath, '*', '*','*'))
        print(os.path.join(dataFolderPath, '*', '*','*'))
        #print(files)
        #print(file for file in files if 'train/tool' in file)
        print('Structuring test and train files...')
        self.test_tool_list = [file for file in files if 'val/tool' in file]
        self.train_tool_list = [file for file in files if 'train/tool' in file]
        self.test_notool_list = [file for file in files if 'val/notool' in file]
        self.train_notool_list = [file for file in files if 'train/notool' in file]
        self.image_train_list = self.train_notool_list[::2] + self.train_tool_list[::2]
        self.image_test_list = self.test_notool_list[::2] + self.test_tool_list[::2]
        #self.image_test_list = self.test_notool_list + self.test_tool_list
        self.val_tool_list = [file for file in files if 'test/tool' in file]
        self.val_notool_list = [file for file in files if 'test/notool' in file]
        #print(len(self.val_tool_list))
        #print(len(self.val_notool_list))
        self.image_val_list = self.val_notool_list + self.val_tool_list
        #self.image_val_list = self.image_val_list
        #print(self.train_tool_list)
        self.label_val_list = []
        for file in self.image_val_list:
            #print(file)
            if 'test/notool' in file:
                #print(0)
                self.label_val_list.append(0.)
            else:
                #print(1)
                self.label_val_list.append(1.)
        '''
        The structure has to be dataset/train/images/image.png
        The structure has to be dataset/train/labels/label.png
        Separate image and label lists
        Sort them to align labels and images
        '''
        self.label_train_list = []
        self.label_test_list = []
        for file in self.image_train_list:
            #print(file)
            if 'train/notool' in file:
                #print(0)
                self.label_train_list.append(0.)
            else:
                #print(1)
                self.label_train_list.append(1.)
        for file in self.image_test_list:
            #print(file)
            if 'val/notool' in file:
                #print(0)
                self.label_test_list.append(0.)
            else:
                #print(1)
                self.label_test_list.append(1.)
        """
        self.label_test_list = []
        for file in self.image_test_list:
            if 'notool' in file:
                self.label_test_list.append(0)
            else:
                self.label_test_list.append(1)
        """
        
        #self.image_train_list = [file for file in self.train_list if 'frame' in file]
        #self.image_test_list = [file for file in self.test_list if 'images' in file]
        #self.label_train_list = [0 for file in self.train_list if 'notool' in file else 1]
        #self.label_test_list = [0 for file in self.test_list if 'notool' in file else 1]

        #self.label_test_list.sort()
        #self.image_test_list.sort()
        self.label_train_list.sort()
        self.image_train_list.sort()
        self.image_val_list.sort()
        self.label_val_list.sort()
        self.image_test_list.sort()
        self.label_test_list.sort()
        # Shuffle train
        self.suffle_segmentation()
        #self.image_test_list = self.image_train_list
        #self.label_test_list = self.label_train_list[:392]
        #self.image_train_list = self.image_train_list[392:]
        #self.label_train_list = self.label_train_list[392:]
        #self.image_val_list = self.image_val_list
        #self.label_val_list = self.label_val_list
        print('Loaded ' + str(len(self.image_train_list)) + ' training samples')
        print('Loaded ' + str(len(self.image_test_list)) + ' testing samples')
        print('Loaded ' + str(len(self.image_val_list)) + ' val samples')
        self.n_classes = n_classes
        #print(self.image_val_list)
        if self.median_frequency_soft != 0:
            self.median_freq = self.median_frequency_balancing_sof(soft=self.median_frequency_soft)
        
        for i in range(len(self.image_val_list)):
            print(i)
            print('test/tool' in self.image_val_list[i])
            print(self.label_val_list[i])
        # Creates test and train queues
        self.test_queue = LoaderQueue.LoaderQueue(700, self, train=False,val=False, workers=1)
        self.train_queue = LoaderQueue.LoaderQueue(700, self, train=True,val=False, workers=8)
        self.val_queue = LoaderQueue.LoaderQueue(700, self, train=False,val=True, workers=1)

        print('Dataset contains ' + str(self.n_classes) + ' classes')

    def suffle_segmentation(self):
        '''
        Shuffles the training files
        :return:
        '''
        s = np.arange(len(self.image_train_list))
        np.random.shuffle(s)
        self.image_train_list = np.array(self.image_train_list)[s]
        self.label_train_list = np.array(self.label_train_list)[s]


    def get_data_list_and_index(self, train=True, val=False):
        '''

        :param train: whether to get training samples of testing samples
        :param size: size of the batch

        :return: image and label lists from where to load the images
        '''

        self.lock.acquire()

        if train:
            image_list = self.image_train_list
            label_list = self.label_train_list

            # Get [size] indexes
            index = self.index_train
            self.index_train = (index + 1) % len(image_list)
        else:
            if val :
                image_list = self.image_val_list
                label_list = self.label_val_list

                # Get [size] random numbers
                index = self.index_val
                self.index_val = (index + 1) % len(image_list)
            else :

                image_list = self.image_test_list
                label_list = self.label_test_list

                # Get [size] random numbers
                index = self.index_test
                self.index_test = (index + 1) % len(image_list)

        self.lock.release()

        return image_list, label_list, index


    def load_image(self, file_path):
        '''

        :param file_path: path to the image
        :return: return the loaded image
        '''

        if self.dim == 1:
            img = cv2.imread(file_path, 0)
        else:
            # img = cv2.imread(random_images[index])
            img = tf.keras.preprocessing.image.load_img(file_path)
            img = tf.keras.preprocessing.image.img_to_array(img).astype(np.uint8)

        return img



    def get_sample(self, train=True, val=False):
        '''
        Get a sample of the segmentation dataset

        :param train: whether to get training samples of testing samples
        :param labels_resize_factor: (downsampling) factor to resize the label images

        :return: sample of segmentation images: X, labels: Y and, masks: mask, and path
        '''


        # init numpy arrays
        image_list, label_list, index = self.get_data_list_and_index(train,val)

        # for every image, get the image, label and mask.
        # the augmentation has to be done separately due to augmentation

        img = self.load_image(image_list[index])
        label = label_list[index]
        label_image = 1

        # Reshape images if its needed
        if img.shape[1] != self.width or img.shape[0] != self.height:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if self.dim == 1:
            img = np.reshape(img, (img.shape[0], img.shape[1], self.dim))

        y = label

        return img, y, label_image, image_list[index]

    def get_queue(self, train=True, val=False):
        '''

        :param train: wheter to get the training or testing queue
        :return: LoaderQueue
        '''
        if train:
            return self.train_queue
        else:
            if val :
                return self.val_queue
            else :
                return self.test_queue


    def get_batch(self, size=32, train=True, val=False):
        '''
        Get a batch of the segmentation dataset

        :param size: size of the batch
        :param train: whether to get training samples of testing samples

        :return: batch of segmentation images: X, labels: Y and, masks: mask
        '''

        queue = self.get_queue(train,val)
        #print(queue)
        # init numpy arrays
        x = np.zeros([size, self.height, self.width, self.dim], dtype=np.float32)
        y = np.zeros([size, 1], dtype=np.float32)
        labels = np.zeros([size, 1], dtype=np.float32)
        paths = []
        for index in range(size):
            img, label, label_image, path = queue.__next__()
            #print(img, label, label_image, path)
            x[index, :, :, :] = img.astype(np.float32)
            y[index] = label
            labels[index] = label_image
            paths.append(path)
        return x, y, labels, paths



    '''
    # Called when iteration is initialized
    def __iter__(self):
        self.index_train = 0  # indexes for iterating while training
        self.index_test = 0  # indexes for iterating while testing
        return self
        
    def __getitem__(self, item):
        pass

    def __next__(self):
        # obtener el siguiente usando get item (dado self.index_train) y haciendo ++

        pass
    '''

if __name__ == "__main__":

    loader = Loader('./Datasets/camvid', n_classes=11, width=480, height=360, median_frequency=0.12)
    # print(loader.median_frequency_exp())
    x, y, mask, paths = loader.get_batch(size=2)

    for i in range(2):
        cv2.imshow('x', ((x[i, :, :, :] + 1) * 127.5).astype(np.uint8))
        cv2.imshow('y', (np.argmax(y, 3)[i, :, :] * 25).astype(np.uint8))
        print(mask.shape)
        cv2.imshow('mask', (mask[i, :, :] * 255).astype(np.uint8))
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    x, y, mask, path = loader.get_batch(size=3, train=False)
