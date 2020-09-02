import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import argparse
import Models , LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import random



weightfile = "wseg1100f1.h5"
n_classes = 1
mname = "CNN1"
input_width = 224
input_height = 224

weight_path = "./weights/"+weightfile

modelFns = {'CNN1':Models.seg1100.seg1100,'CNN2':Models.seg4.seg4  }
modelFN = modelFns[ mname ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
m.load_weights( weight_path )
m.compile(loss='binary_crossentropy',
      optimizer= 'Adam' ,
      metrics=['accuracy'])


output_height = m.outputHeight
output_width = m.outputWidth


test_folders = ['test']
if mname == 'CNN1':
      test_folders.append('train')
      test_folders.append('val')

for f in test_folders:
      images_path = ".\\Data\\NO\\"+mname+"\\image"+f+"\\"
      output_path= ".\\Data\\NO\\"+mname+"\\pred"+f+"\\"
      images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
      images.sort()
      for imgName in images:
            outName = imgName.replace( images_path ,  output_path )
            X = LoadBatches.getImageArr(imgName , input_width  , input_height  )
            pr = m.predict( np.array([X]) )[0]
            seg_img = np.zeros( ( output_height , output_width , 3  ) )
            print(outName)
            if mname == "CNN1":
                  thre = np.max(pr) - 0.05
            else:
                  thre = 0.5
            pr = pr > thre
            pr = pr.astype(int)
            pr = np.reshape(pr,(output_height,output_width))
            seg_img[:,:,0] = pr
            seg_img[:,:,1] = pr*255
            seg_img[:,:,2] = pr
            cv2.imwrite(outName, seg_img)
              
