This code is suport the publicatio **Varun Belagali, Achuth Rao M V, Pebbili Gopikishore, Rahul Krishnamurthy, and Prasanta Kumar Ghosh, “Two step convolutional neural network for automatic glottis localization and segmentation in stroboscopic videos”, Biomed. Opt. Express 11, 4695-4713 (2020)**

Steps to follow to train and test the models:  
    1) Run the Init_folders.py to create the folders  
    2) Download the vgg weights file to ./weights folder   
       link - https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5  
    3) Copy the images of of training, validation and test to ./Data/NO/CNN1/imagetrain,./Data/NO/CNN1/imageval,./Data/NO/CNN1/imagetest respectively  
    4) Copy corresponding annotations to ./Data/NO/CNN1/annotrain,./Data/NO/CNN1/annoval,./Data/NO/CNN1/annotest  
          annotations are in the form of .png files, with glottis region pixels labelled as '1' and non glottal regions as '0'  
    5) Train CNN1 - run train1.py with mname=CNN1  
    6) Get CNN1 output - run predict.py with mname=CNN1   
    7) Run Boundingbox.m to generate data for CNN2  
    8) Train CNN2 - run train1.py with mname=CNN2  
    9) Test the two step CNN method - Run test.py to get the dice scores  
