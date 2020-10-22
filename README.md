# toolSegmentation
This repository is built on a fork of project **robot-surgery-segmentation** (at https://github.com/ternaus/robot-surgery-segmentation), the official implementation of the paper *Automatic Instrument Segmentation in Robot-Assisted Surgery using Deep Learning*. Shvets, Alexey A., et al. IEEE Int. Conf. on Machine Learning and Applications. 2018.

The main goal is to obtain a tool segmentation model adapted to the requirements of the project **EndoMapper**.

## Authors
Clara Tomasini, León Barbed, Ana Murillo, Luis Riazuelo, Pablo Azagra

## Results
All models were available pretrained on images similar to those of the Hamlyn dataset, and were then fine-tuned on more specific images from a different dataset (UCL).
File *training.ipynb* shows how to fine-tune the models.

Segmentations obtained for images from the Hamlyn dataset and from the fine-tuning dataset (UCL) using UNet, TernausNet-11 and LinkNet-34 models show how fine-tuning the models makes it possible to obtain better segmentations for images from the UCL dataset. 

![results](/images/results.png)



## How to run
Fine-tuned models are available at https://drive.google.com/drive/folders/1VOtD3U9jF4jPsPlZfU0LD3yXx0oI8d8-?usp=sharing

File *Demo.ipynb* provides an example of how to use the model in order to get a prediction for a given image using one of the models.
