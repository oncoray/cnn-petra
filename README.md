# cnn-petra
Code for training and testing 3D convolutional neural networks for outcome prediction in Glioblastoma, as used in MIDL 2022 short paper submission.

### Submission title: 3D convolutional neural networks for outcome prediction in glioblastoma using methionine PET and T1w MRI

The repository contains jupyter notebooks for
  * End-to-end training and internal validation pipeline using 3D convolutional neural networks
  * Model testing pipleine
  
Train_valid_CV_3DCNN.ipynb is training notebook with cross-validation settings
Test_3D_CNN.ipynb is notebook for testing the trained models.

Three models can be used for training 

1. 3D-DenseNet, model available under DenseNet3D.py
2. 3D-ResNet, model available under ResNet3D.py
3. 3D-Vgg, model available under Vgg3D.py

Data Augmentation is performed using batchgenerators library, augmentation function is implemented in Aug3D.py
Other helper functions used for this work are available under util.py

# Requirements:

```
Keras 2.3.1
Tensorflow 2.1.0
Python 3.7.10
```
