import os
import cv2
import glob 
import numpy as np
from skimage.measure import regionprops,label
import matplotlib.pyplot as plt 
from statistics import mean

def dice_fun(gt_Img,pred_Img):
    pred_Img = cv2.resize(pred_Img,(224,224))
    gt_Img = cv2.resize(gt_Img,(224,224))
    pred_Img = pred_Img[:,:,0]
    gt_Img = gt_Img[:,:,0]
    labels_mask = label(pred_Img)
    regions = regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1
    mask = labels_mask
    dice_val = 2.0 * np.sum(np.multiply(mask,gt_Img))/(np.sum(mask)+np.sum(gt_Img)+0.0000000001)
    print(dice_val)
    return dice_val

mname = "CNN2"

pred_path = './Data/NO/'+mname+"/predtest/"
gt_path = './Data/NO/'+mname+"/annotest/"

images = glob.glob( pred_path + "*.png"  ) 
images.sort()
dice_scores =[]

for imgpaths in images:
    _,imgName = os.path.split(imgpaths)
    print(imgName)
    pred_img = cv2.imread(pred_path+imgName)
    gt_img = cv2.imread(gt_path+imgName)
    temp = dice_fun(gt_img,pred_img)
    dice_scores.append(temp)

print('--------')
print("mean_dice",mean(dice_scores))