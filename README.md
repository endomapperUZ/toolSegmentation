# toolSegmentation
This repository is built on a fork of project **robot-surgery-segmentation** (at https://github.com/ternaus/robot-surgery-segmentation), the official implementation of the paper *Automatic Instrument Segmentation in Robot-Assisted Surgery using Deep Learning*. Shvets, Alexey A., et al. IEEE Int. Conf. on Machine Learning and Applications. 2018.

The main goal is to obtain a tool segmentation model adapted to the requirements of the project **EndoMapper**.

## Authors
Clara Tomasini, León Barbed, Ana Murillo, Luis Riazuelo, Pablo Azagra

## Results
Segmentation obtained for an image from the Hamlyn dataset using UNet and TernausNet-11 models
![results_2425](/images/results_2425.png)

Model | IOU, % | Dice, % | Time, ms
------| ------ | ------- | -------
U-Net | 39.79 | 56.93 | 109
TernausNet-11 | 53.56 | 69.76 |  214
TernausNet-16 |  |  | 
LinkNet34 |  |  |

## How to run
Trained models are available at https://drive.google.com/drive/folders/1VOtD3U9jF4jPsPlZfU0LD3yXx0oI8d8-?usp=sharing

File *Demo.ipynb* provides an example of how to use the model
