import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session  
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
set_session(tf.Session(config=config))

import sys
import glob
import Models , LoadBatches
import numpy as np
import keras
from keras import backend as K
from  keras import losses
import math


##################################
weightfile = "temp.h5"
mname = "CNN1"

##################################
if mname == "CNN1": #200 for CNN1, 2 for CNN2
     l2 = 200
else:
     l2 = 2                    
##################################

train_images_path = "./Data/NO/"+mname+"/imagetrain/"
train_segs_path = "./Data/NO/"+mname+"/annotrain/"
val_images_path= "./Data/NO/"+mname+"/imageval/"
val_segs_path= "./Data/NO/"+mname+"/annoval/"

train_batch_size = 2
val_batch_size=1

n_classes = 1
input_height =224
input_width = 224

save_weights_path = "./weights/"+weightfile

ima1 = glob.glob( train_images_path + "*.png"  )
ima2 = glob.glob( val_images_path + "*.png"  )

train_sel = math.ceil(len(ima1)/train_batch_size)
val_sel = math.ceil(len(ima2)/val_batch_size)

print("train_steps",train_sel)
print("validation_steps",val_sel)



modelFns = {'CNN1':Models.seg1100.seg1100,'CNN2':Models.seg4.seg4  }
modelFN = modelFns[ mname ]

def loss_fun(Y_True,Y_pred):
     def cur_loss(Y_True,Y_pred):
          global l2
          Y_pred1 = K.clip(Y_pred,K.epsilon(),None)
          Y_pred2 = K.clip(1-Y_pred,K.epsilon(),None)
          w_c = l2 * Y_True + (1-Y_True)*1 
          new = -w_c *(Y_True * K.log(Y_pred1) + (1-Y_True) * K.log(Y_pred2)) 
          new1 = K.mean(new)
          return new1
     return cur_loss(Y_True,Y_pred)

m = modelFN( n_classes=1 , input_height=input_height, input_width=input_width)
m.compile(loss=loss_fun,optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.0, nesterov=False) , metrics=['acc'])
print(m.summary())



print ("Model output shape" ,  m.output_shape)


early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
checkpoint_callback = keras.callbacks.ModelCheckpoint(save_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
output_height = m.outputHeight
output_width = m.outputWidth
print(output_height)
print(output_width)

G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

m.fit_generator( G , train_sel , validation_data=G2 , validation_steps=val_sel ,  epochs=500, callbacks=[early_stopping_callback,checkpoint_callback])
