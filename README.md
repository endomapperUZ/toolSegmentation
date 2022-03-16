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
Folder **endovis_challenge** contains files adapted from **robot-surgery-segmentation** for models LinkNet and UNet. 
Folder **mininet** contains files adapted from **MiniNet-v2** as well as the implementation of our clasifier. 

Fine-tuned models are available at https://drive.google.com/drive/folders/1BYyfUek6arVhpgChWuhD6JVQ9-RS4ZNm?usp=sharing. 
HCULB frames and masks are available at https://drive.google.com/drive/folders/1_zzQr82Vv5du99Mid6mt3wPoOYTWTJt8?usp=sharing.

File *mininet/generate_masks.py* provides an example of how to use the full segmentation pipeline including MiniNet model and our clasifier in order to get a prediction for a given image.

## Results
All models were available pretrained on images similar to those of the Hamlyn dataset, and were then fine-tuned on more specific images from a different dataset (HCULB).
File *endovis_challenge/train_ft.sh* performs training of LinkNet and UNet models. File *mininet/train.sh* performs training of MiniNet model. File *mininet/train_classif.sh* performs training of our clasifier. 

The following table shows several representative examples of the segmentations obtained for images from the HCULB dataset. The results use different models (UNet, TLinkNet-34) with the original and our fine-tuned versions. They show how the models fine-tuned with a few project labeled frames (just from one labeled sequence) adapt adequately to situations of our target domain (UCL and HCULB).

[3] *Three-dimensional tissue deformation recovery and tracking*. P. Mountney, D. Stoyanov, and G.-Z. Yang. IEEE Signal Processing Magazine, 27(4):14–24, 2010.

![results2](/images/resultados2_2.png)
