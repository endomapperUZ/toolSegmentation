# toolSegmentation
Code for 'Efficient tool segmentation for endoscopic videos in the wild' (MIDL 2022).

[Paper](https://openreview.net/pdf?id=DPkb7gxt6gZ) and [Video Demo](https://drive.google.com/file/d/1anOHK4h19EesMFc_drYFnbcYtOBeTuJb/view?usp=sharing).


This repository is built on a fork of projects **robot-surgery-segmentation** (at https://github.com/ternaus/robot-surgery-segmentation), the official implementation of the paper 

[1] *Automatic Instrument Segmentation in Robot-Assisted Surgery using Deep Learning*. Shvets, Alexey A., et al. IEEE Int. Conf. on Machine Learning and Applications. 2018.

and **MiniNet-v2** (at https://github.com/Shathe/MiniNet-v2), the official implementation of the paper

[2] *MiniNet: An Efficient Semantic Segmentation ConvNet for Real-time Robotic Applications*. Alonso, Iñigo et al. IEEE Transactions on Robotics. 2020.

The main goal is to obtain a tool segmentation model adapted to the requirements of the project **EndoMapper**.

## Authors
Clara Tomasini, Iñigo Alonso, Ana Murillo, Luis Riazuelo

## How to run
Folder **endovis_challenge** contains files adapted from **robot-surgery-segmentation** for models LinkNet and UNet. 
Folder **mininet** contains files adapted from **MiniNet-v2** as well as the implementation of our clasifier. 

Fine-tuned models and trained clasifier are available [here](https://drive.google.com/drive/folders/1BYyfUek6arVhpgChWuhD6JVQ9-RS4ZNm?usp=sharing). 
HCULB frames and masks : TO BE RELEASED 

File *mininet/generate_masks.py* provides an example of how to use the full segmentation pipeline including MiniNet model and our clasifier in order to get a prediction for a given image.

## Results 
Models UNet and LinkNet were available in **robot-surgery-segmentation** pretrained on images from the EndoVis17 dataset, and were then fine-tuned on more specific images from HCULB dataset. Mininet was trained from scratch on EndoVis17 ndataset and then fine-tuned on HCULB dataset.
File *endovis_challenge/train_ft.sh* performs training of LinkNet and UNet models. File *mininet/train.sh* performs training of MiniNet model. File *mininet/train_classif.sh* performs training of our clasifier. 
The following figure shows examples of binary segmentations from HCULB dataset obtained using different models without applying our pre-filtering clasifier. 
The last column shows an example of a frame without tool in which all segmentation models introduce False Positives. The effect of our pre-filtering classifier can be seen in the [Video Demo](https://drive.google.com/file/d/1anOHK4h19EesMFc_drYFnbcYtOBeTuJb/view?usp=sharing).


![results](/images/results_seg_hculb.png)
