# two-step-cnn
Two step convolutional neural network__ 

Steps to follow:__
    1) Run the Init_folders.py to create the folders__
    2) Download the vgg weights file to ./weights folder__ 
          link - https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5__
    3) Copy the images of of training, validation and test to ./Data/NO/CNN1/imagetrain,./Data/NO/CNN1/imageval,./Data/NO/CNN1/imagetest respectively__
    4) the copy corresponding annotations to ./Data/NO/CNN1/annotrain,./Data/NO/CNN1/annoval,./Data/NO/CNN1/annotest__
          annotations are in the form of .png files, with glottis region pixels labelled as '1' and non glottal regions as '0'__
    5) Train CNN1 - run train1.py with mname=CNN1__
    6) Get CNN1 output - run predict.py with mname=CNN1__ 
    7) Train CNN2 - run train1.py with mname=CNN2__
    8) Test the two step CNN approach - Run test.py to get the dice scores__
