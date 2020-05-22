# GTSRB Classifier

This repository contains my work as part of my ongoing research efforts with the [Ethos Lab at Stony Brook University](https://github.com/Ethos-lab), supervised by [Prof. Amir Rahmati](https://amir.rahmati.com) and [Pratik Vaishnavi](https://www3.cs.stonybrook.edu/~pvaishnavi/). 

## Dataset

The dataset used for the tasks was the German Traffic Sign Data Set (GTSRB) which can be downloaded from Kaggle [here](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

## Python Packages

This code uses TensorFlow version 1.15 and makes use of image libraries, such as `opencv` and `pillow`. Python packages are in the `requirements.txt` file. 

Create a virtual environment and install the needed dependencies using pip
```bash
$ virtualenv venv
$ source venv/bin/activate
$ (venv) pip install -r requirements.txt
```

## Tasks Completed
- Multipurpose data loader with random data augmentation applied on each training batch, which can be found in [`data_loader.py`](https://github.com/asarj/GTSRB_Classifier/blob/master/data_loader.py)
- Two convolutional neural network classifiers:
  - A simplified version, which can be found in [`gtsrb_cnn_classifier1.py`](https://github.com/asarj/GTSRB_Classifier/blob/master/gtsrb_cnn_classifier1.py)
  - LeNet architecture-based version, which can be found in [`gtsrb_cnn_classifier2.py`](https://github.com/asarj/GTSRB_Classifier/blob/master/gtsrb_cnn_classifier2.py)
- Early stopping to prevent overfitting during training
- Learning rate estimator script that uses a Gradient Descent Optimizer to find the optimal learning rate before training
