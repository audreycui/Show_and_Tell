
import numpy as np
import cv2
from keras.preprocessing import image
from keras import backend as K
from keras.models import Model
from scipy.linalg import norm
from keras.applications.vgg19 import VGG19
import mxnet as mx
from collections import namedtuple
from utils.caffe_io import Transformer

import os
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize

Batch = namedtuple('Batch', ['data'])

#added highway for discriminator
class ImageLoader(object):
    def __init__(self, mean_file):
        self.bgr = True
        self.scale_shape = np.array([224, 224], np.int32)
        self.crop_shape = np.array([224, 224], np.int32)
        self.mean = np.load(mean_file).mean(1).mean(1)
        self.Batch = namedtuple('Batch', ['data'])

    def load_image(self, file_name):
        """
        Load and preprocess an image
        """
        #print(file_name)
        img = image.load_img(file_name, target_size=(224, 224))
        im = image.img_to_array(img)
        im = np.expand_dims(im, axis=0)

        im = self.preprocess_input(im)
        return im

    def preprocess_input(self, x, dim_ordering='default'):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        assert dim_ordering in {'tf', 'th'}

        if dim_ordering == 'th':
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
            # 'RGB'->'BGR'
            x = x[:, ::-1, :, :]
        else:
            x[:, :, :, 0] -= 103.939
            x[:, :, :, 1] -= 116.779
            x[:, :, :, 2] -= 123.68
            # 'RGB'->'BGR'
            x = x[:, :, :, ::-1]
        return x

    def load_image_obsolete(self, image_file):
        """ Load and preprocess an image. """
        #print (image_file)
        image = cv2.imread(image_file)
        if (image is None):
            print ('IMAGE IS NONE :( %s' % image_file)
            return None
        if self.bgr:
            temp = image.swapaxes(0, 2)
            temp = temp[::-1]
            image = temp.swapaxes(0, 2)

        image = cv2.resize(image, (self.scale_shape[0], self.scale_shape[1]))
        offset = (self.scale_shape - self.crop_shape) / 2
        offset = offset.astype(np.int32)
        image = image[offset[0]:offset[0]+self.crop_shape[0],
                      offset[1]:offset[1]+self.crop_shape[1]]
        image = image - self.mean
        return image

    def load_images_vgg19(self, image_files):
        """ Load and preprocess a list of images. """
        images = []
        for image_file in image_files:
            images.append(self.load_image(image_file))
        images = np.array(images, np.float32)
        return images

    def load_images_mxnet(self, image_files):
        """ Load and preprocess a list of images. """
        images = []
        for image_file in image_files:
            img = cv2.imread(image_file)
            assert img is not None, IOError(
                    'The file `{}` may be not an image'.format(image_file))
            # img.shape: H, W, T
            if img.ndim == 2:
                # gray image
                img = np.stack([img, img, img], axis=2)
            else:
                if img.ndim == 3 and img.shape[2] in [3, 4]:
                    if img.shape[2] == 4:
                        # remove alpha channel
                        img = img[:, :, :3]
                else:
                    raise Exception('Invalid Image `{}` whose shape is {}'.format(image_file, img.shape))
            images.append(img)
        #images = np.array(images, np.float32)
        return images

    def get_obj_feature(self, object_model, img):
        mu = np.array([104,117,123])
        transformed_img = self.crop_lit_centor(img, mu)
        transformed_img = transformed_img[None]
        object_model.forward(Batch([mx.nd.array(transformed_img)]), is_train = False)
        outputs = object_model.get_outputs()[0].asnumpy()
        return outputs

    def get_scene_feature(self, scene_model, img):
        mu = np.array([105.487823486,113.741088867,116.060394287])
        transformed_img = self.crop_lit_centor(img, mu)
        transformed_img = transformed_img[None]
        scene_model.forward(Batch([mx.nd.array(transformed_img)]), is_train = False)
        outputs = scene_model.get_outputs()[0].asnumpy()
        return outputs

    def get_sentiment_feature(self, sentiment_model, img):
        mu = np.array([97.0411,105.423,111.677])
        transformed_img = self.crop_lit_centor(img, mu, img_len = 227)
        transformed_img = transformed_img[None]
        sentiment_model.forward(Batch([mx.nd.array(transformed_img)]), is_train = False)
        outputs = sentiment_model.get_outputs()[0].asnumpy()
        return outputs

    def extract_features_vgg19(self, vgg19_model, images, feature_files, batch_size):
        features = []
        for i in range(batch_size):
            try:
                feature.append(np.load(feature_files[i]))
            except:
                fc2 = vgg19_model.predict(self.load_image(images[i]))
                reshaped = np.reshape(fc2, (4096))  
                features.append(reshaped)
        
        return features #shape: (batch_size, 4096)

    #extracts sentiment and object features
    def extract_features_mxnet(self, object_model, sentiment_model, scene_model, images, batch_size):
        #model = vgg19
        features = []
        for img in images:

            obj_feat = self.get_obj_feature(object_model, img)
            #sentiment_feat = self.get_sentiment_feature(sentiment_model, img)
            #scene_feat = self.get_scene_feature(scene_model, img)
            #feat = np.hstack([obj_feat, sentiment_feat, scene_feat])
            feat = np.hstack([obj_feat])
            #features.append(np.reshape(feat, (4096*3)))
            features.append(np.reshape(feat, (4096)))
        
        #print ("feature shape:" +str(np.shape(features)))  
        return features  #shape: (batch_size, 4096*3)

    def crop_lit_centor(self, img, mu, img_len = 224):
        [n,m,_]=img.shape
        if m>n:
            m = int(m*256/n)
            n = 256
        else:
            n = int(n*256/m)
            m = 256
        return self.data_trans(cv2.resize(img,(m,n))/255.0,(1,3,n,m), mu)[:,int((n-img_len)/2):int((n+img_len)/2),int((m-img_len)/2):int((m+img_len)/2)]

    def data_trans(self, img, shape, mu):
        transformer = Transformer({'data': shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', mu)
        transformer.set_raw_scale('data', 255)
        transformed_image = transformer.preprocess('data', img)
        return transformed_image

    """
    def extract_features(self, trained_model, images_files, batch_size):
        #print(images_files)
        images = self.load_images(images_files)
        #model = vgg19
        features = []
        for i in range(batch_size):
            fc2 = trained_model.predict(images[i])
            reshaped = np.reshape(fc2, (4096))  
            features.append(reshaped)
        
        #print ("feature shape:" +str(np.shape(features)))  
        return features #shape: (batch_size, 4096)
    """
    def mytest(self, image = 'D:/download/COCO/train/images/COCO_train2014_000000318556.jpg'):
        #model = self.vgg19

        features = []
        for i in range(1):
            #images[i] = np.expand_dims(images[i], axis=0)
            #print ('shape ' + str(images[i].shape))
            fc2 = self.model.predict(self.load_image(image))
            #reshaped = tf.reshape(fc2, [8, 512])
            features.append(reshaped)
        
        return features

