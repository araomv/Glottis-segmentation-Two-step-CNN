
import numpy as np
import cv2
import glob
import itertools


def getImageArr( path , width , height , odering='channels_last' ):

	try:
		img = cv2.imread(path, 1)
		img = cv2.resize(img, ( width , height ))
		if odering == 'channels_first':
			img = np.rollaxis(img, 2, 0)
		return img

	except (Exception,e) as e:
		print (path , e)
		img = np.zeros((  height , width  , 3 ))
		if odering == 'channels_first':
			img = np.rollaxis(img, 2, 0)
		return img



def getSegmentationArr( path , nClasses ,  width , height  ):
	img = cv2.imread(path,1)
	img = cv2.resize(img,(width,height))
	img = img[:,:,0]                           #channel 0 has binary labels for pixels label:1 glottis, label0:non glottal area
	im = np.reshape(img,(width*height))
	return im

def imageSegmentationGenerator( images_path , segs_path ,  batch_size,  n_classes , input_height , input_width , output_height , output_width   ):
	
	assert images_path[-1] == '/'
	assert segs_path[-1] == '/'

	images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
	images.sort()
	segmentations  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
	segmentations.sort()

	assert len( images ) == len(segmentations)
	zipped = itertools.cycle( zip(images,segmentations) )

	while True:
		X = []
		Y = []
		for _ in range( batch_size) :
			im , seg = next(zipped)
			X.append( getImageArr(im , input_width , input_height )  )
			Y.append( getSegmentationArr( seg , n_classes , output_width , output_height ) )

		yield np.array(X) , np.array(Y)




