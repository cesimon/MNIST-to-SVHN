## MNIST to SVHN

This is an academic project completed while studying at Telecom Paris/ENSTA in 2021-2022.

- The aim was to manage to transform images from the MNIST dataset to images of the SVHN dataset (unpaired image-to-image translation).
This repository thus essentially consists in an implementation of a solution based on CycleGAN (`cycle_gan.py`) to perform the translation.
The code is partly adapted from [this](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/cyclegan) implementation.

- We then trained a classifier on SVHN (`train_task_classifier.py`) by using *generated* SVHN images (from the MNIST train dataset),
and the associated source MNIST labels *only*. The goal was for the trained classifier to display 
a performance (accuracy-wise) as good as possible on the SVHN test dataset, without ever
using labels from the SVHN dataset during the training process.

