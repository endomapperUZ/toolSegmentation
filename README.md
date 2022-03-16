# toolSegmentation
Code for 'Efficient tool segmentation for endoscopic videos in the wild' (MIDL 2022).

[Paper](https://openreview.net/pdf?id=DPkb7gxt6gZ) and [Video Demo](https://drive.google.com/file/d/1anOHK4h19EesMFc_drYFnbcYtOBeTuJb/view?usp=sharing).


This repository is built on a fork of projects **robot-surgery-segmentation** (at https://github.com/ternaus/robot-surgery-segmentation), the official implementation of the paper 

[1] *Automatic Instrument Segmentation in Robot-Assisted Surgery using Deep Learning*. Shvets, Alexey A., et al. IEEE Int. Conf. on Machine Learning and Applications. 2018.

and **MiniNet-v2** (at https://github.com/Shathe/MiniNet-v2), the official implementation of the paper

[2]*MiniNet: An Efficient Semantic Segmentation ConvNet for Real-time Robotic Applications*. Alonso, Iñigo et al. IEEE Transactions on Robotics. 2020.

The main goal is to obtain a tool segmentation model adapted to the requirements of the project **EndoMapper**.

## Authors
Clara Tomasini, León Barbed, Ana Murillo, Luis Riazuelo, Pablo Azagra

## How to run
Fine-tuned models are available at https://drive.google.com/drive/folders/1VOtD3U9jF4jPsPlZfU0LD3yXx0oI8d8-?usp=sharing. Best models obtained are Linknet models number 21 and 22 and Mininet.

XML files containing masks coordinates for UCL frames are available at https://drive.google.com/drive/folders/1bIv1w320DhIB3AW0UWRPtRgjlG4ephs5?usp=sharing

File *Demo.ipynb* provides an example of how to use the model in order to get a prediction for a given image using one of the models.

## Results
All models were available pretrained on images similar to those of the Hamlyn dataset, and were then fine-tuned on more specific images from a different dataset (UCL).
File *training.ipynb* shows how to fine-tune the models.

The following table shows several representative examples of the segmentations obtained for images both from the Hamlyn dataset [2] and from the project sequences (UCL and HCULB). The results use different models (UNet, TernausNet-11 and LinkNet-34) with the original and our fine-tuned versions. They show how the models fine-tuned with a few project labeled frames (just from one labeled sequence) adapt adequately to situations of our target domain (UCL and HCULB).

[3] *Three-dimensional tissue deformation recovery and tracking*. P. Mountney, D. Stoyanov, and G.-Z. Yang. IEEE Signal Processing Magazine, 27(4):14–24, 2010.

![results2](/images/resultados2_2.png)
