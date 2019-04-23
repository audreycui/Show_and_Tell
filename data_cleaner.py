#!/usr/bin/python
import tensorflow as tf

from scipy.misc import imread, imresize
from imagenet_classes import class_names
import numpy as np
import pandas as pd
import pprint
import json 
import csv
import sys
import os

from config import Config
from keras.preprocessing import image
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data

from utils.coco.coco import COCO
from utils.vocabulary import Vocabulary
from utils.misc import ImageLoader

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'train',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', None,
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

tf.flags.DEFINE_string('image_file','./man.jpg','The file to test the CNN')

tf.flags.DEFINE_string('model', 'SeqGAN',
                       'model type, can be SeqGAN, Show&Tell, and others to be added')


## Start token is not required, Stop Tokens are given via "." at the end of each sentence.
## TODO : Early stop functionality by considering validation error. We should first split the validation data.

def main():
    config = Config()

    #load_ignore_file(config)
    #prepare_train_data(config)
    cleanup_data(config)

def load_ignore_file(config):
  df = pd.read_csv(config.ignore_file).values
  df = [idx for seqno, idx in df]
  print(df)

def cleanup_data(config):
    bad_ids = []
    try:
        with open(config.train_caption_file, 'r') as f:
            self.dataset = json.load(f)
    except Exception:
        #try:
        with open(config.train_caption_file, 'r') as f:
            reader = csv.reader(f)
            for id, file_name, caption in reader:
                try:
                    img = image.load_img(file_name, target_size=(224, 224))
                except Exception:
                    print ("cannot identify image file:" + file_name)
                    bad_ids.append(id)
                    pass
    
        #except Exception:
            #print ("Unsupported caption file format other than json or csv")
            #return
        
    print("Total bad image files:%d" % len(bad_ids))
    data = pd.DataFrame({'index': bad_ids})
    data.to_csv(config.ignore_file)

if __name__ == '__main__':
    main()