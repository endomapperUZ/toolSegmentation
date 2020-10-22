# toolSegmentation
This repository is built on a fork of project **robot-surgery-segmentation** (at https://github.com/ternaus/robot-surgery-segmentation), the official implementation of the paper *Automatic Instrument Segmentation in Robot-Assisted Surgery using Deep Learning*. Shvets, Alexey A., et al. IEEE Int. Conf. on Machine Learning and Applications. 2018.

The main goal is to obtain a tool segmentation model adapted to the requirements of the project **EndoMapper**.

## Authors
Clara Tomasini, León Barbed, Ana Murillo, Luis Riazuelo, Pablo Azagra

## Results
All models were available pretrained on images similar to those of the Hamlyn dataset, and were then fine-tuned on more specific images from a different dataset (UCL).
Segmentations obtained for images from the Hamlyn dataset (1 & 2) and from the fine-tuning dataset (3) using UNet, TernausNet-11 and LinkNet-34 models show how fine-tuning the models makes it possible to obtain better segmentations for images from the UCL dataset. 

![results_2425](/images/results_2425.png) ![results_2425](/images/results_5801.png)



## How to run
Fine-tuned models are available at https://drive.google.com/drive/folders/1VOtD3U9jF4jPsPlZfU0LD3yXx0oI8d8-?usp=sharing

File *Demo.ipynb* provides an example of how to use the model
