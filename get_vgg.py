from keras.applications.vgg19 import VGG19
from keras import backend as K
from keras.models import Model
from utils.misc import ImageLoader
from keras.preprocessing import image
import pandas as pd
import numpy as np
import csv

def get_feats():

	train_caption_file = 'D:/download/art_desc/train/ann.csv'
	eval_caption_file = 'D:/download/art_desc/val/ann.csv'
	train_features_dir = 'D:/download/art_desc/train/images_vgg/'
	eval_features_dir = 'D:/download/art_desc/val/images_vgg/'
	image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')

	net = VGG19(weights='imagenet')
	model = Model(input= net.input, output= net.get_layer('fc2').output)

	with open(eval_caption_file, 'r') as f:
		reader = csv.reader(f)
		for id, file_name, caption in reader:
			
			try: 
				img = image_loader.load_image(file_name)
				fc2 = model.predict(img)
				reshaped = np.reshape(fc2, (4096))
				np.save(eval_features_dir + 'art_desc'+ id, reshaped)
			except Exception:
				print ("cannot identify image file:" + file_name)
				pass

get_feats()