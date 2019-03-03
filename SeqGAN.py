import tensorflow as tf
import numpy as np
from tqdm import tqdm
import json

from keras.applications.vgg19 import VGG19
from keras import backend as K
from keras.models import Model

from baseModel import BaseModel
from discriminator import Discriminator
#from discriminator_t import Discriminator
from generator import Generator
from utils.coco.pycocoevalcap.eval import COCOEvalCap

import os
import shutil

#from Shaofan Lai's tensorflow implementation of SeqGAN (Yu et. al, 2017)
#https://github.com/Shaofanl/SeqGAN-Tensorflow
#SeqGAN: https://arxiv.org/abs/1609.05473

#Audrey's modifications:
#removed parameters for seqgan constructor, generator, disciminator (parameters are already in config)
class SeqGAN(BaseModel):
    def __init__(self, config):

        '''

        # generator related
        g_emb_dim: the size of the embedding space
        g_hidden_dim: the number of the hidden cells in LSTM 

        # discriminator related
        d_emb_dim: the size of the embedding space
        d_filter_sizes: the size of the filters (CNN)
        d_num_filters: the number of the filters (CNN)

        # others
        log_generation: whether to log the generation as
            an image in the tensorboard'''
        super().__init__(config)

        self.generator = Generator(self, config)
        self.discriminator = Discriminator(self, config) 
        self.log_generation = False
        
        """
    def build(self):
        config = self.config
        self.net = VGG19(weights='imagenet')
        self.model = Model(input= self.net.input, output= self.net.get_layer('fc2').output)

        self.num_ctx = 8
        self.dim_ctx = 512
    """
    def get_nn(self):
        return self.nn

    def get_imagefeatures(self, image_files):
        images = self.image_loader.load_images(image_files)
        return self.image_loader.extract_features(self.trained_model, images, self.config.batch_size) #extract image features using vgg19

    def next_batch(self, data):
        batch = data.next_batch()
        image_files, sentences, masks = batch
        conv_features = self.get_imagefeatures(image_files)

        return sentences, conv_features

    def train(self, sess, train_data):
        '''
        sampler: a function to sample given batch_size

        evaluator: a function to evaluate given the
            generation and the index of epoch
        evaluate: a bool function whether to evaluate
            while training
        '''

        # if os.path.exists(tensorboard_dir):
        #     shutil.rmtree(tensorboard_dir)
        # os.mkdir(tensorboard_dir)

        config = self.config
        pretrain_g_epochs = config.pretrain_g_epochs
        #pretrain_g_epochs = 1 #make this 1 for debugging purposes #TODO: when training delete this line
        pretrain_d_epochs = config.pretrain_d_epochs
        #pretrain_d_epochs = 1 #make this 1 for debugging purposes
        #tensorboard_dir = config.log_dir #TODO check what this is
        
        gen = self.generator
        dis = self.discriminator
        batch_size = config.batch_size

        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)
        
        fake_samples = []
        for epoch in tqdm(list(range(min(pretrain_g_epochs, train_data.num_batches))), desc='Pretraining Generator'):
            sentences, conv_features = self.next_batch(train_data)
            #print ('pretrain g epoch', epoch) #sampler gets coco data
            summary, fake_samples = gen.pretrain(sess, sentences, conv_features) #changed sampler to next_batch
            #n print(np.shape(fake_samples))
            #next_batch consists of images, captions, and masks
            writer.add_summary(summary, epoch)
            '''
            if evaluate and evaluator is not None: #TODO add eval
                evaluator(gen.generate(sess), epoch)
            '''
            #TODO evaluator
        train_data.reset()

        for epoch in tqdm(list(range(min(pretrain_g_epochs, train_data.num_batches))), desc='Pretraining Discriminator'):
            #fake_samples = gen.generate(sess)
            real_samples, conv_features = self.next_batch(train_data)
            samples = np.concatenate([fake_samples, real_samples])
            labels = np.concatenate([np.zeros((batch_size,)),
                                     np.ones((batch_size,))])
            for _ in range(3):
                indices = np.random.choice(
                    len(samples), size=(batch_size,), replace=False)
                dis.train(sess, samples[indices], labels[indices])
        train_data.reset()
        
        for epoch in tqdm(list(range(config.total_epochs)), desc='Adversarial training'):
            for _ in range(1):
                real_samples, conv_features = self.next_batch(train_data)
                #fake_samples = gen.generate(sess) #generator generates fake samples
                rewards = gen.get_reward(sess, real_samples, conv_features, 16, dis) 
                summary, fake_samples = gen.train(sess, real_samples, conv_features, rewards) #generate new fake samples and reward
                # np.set_printoptions(linewidth=np.inf,
                #                     precision=3)
                # print rewards.mean(0)
            writer.add_summary(summary, epoch)

            for _ in tqdm(list(range(5)), desc='Discriminator batch'):
                #fake_samples = gen.generate(sess) #generator generates fake samples after being trained
                real_samples, conv_features = self.next_batch(train_data)
                #TODO pass in image condition for conditional gan
                samples = np.concatenate([fake_samples, real_samples])
                labels = np.concatenate([np.zeros((batch_size,)),
                                         np.ones((batch_size,))])
                for _ in range(3):
                    indices = np.random.choice(
                        len(samples), size=(batch_size,), replace=False)
                    summary = dis.train(sess, samples[indices],
                                        labels[indices]) #discriminator trains on the fake and real samples
            writer.add_summary(summary, epoch)

            if self.log_generation:
                summary = sess.run(
                    gen.image_summary,
                    feed_dict={gen.given_tokens: real_samples})
                writer.add_summary(summary, epoch)

            '''
            if evaluate and evaluator is not None:
                evaluator(gen.generate(sess), pretrain_g_epochs+epoch)
            '''
            real_samples, conv_features = self.next_batch(train_data)
            np.save('generation', gen.generate(sess, real_samples, conv_features))

        self.save()
        writer.close()
        print("Training complete.")

    def eval(self, sess, eval_data):
        """ Evaluate the model using the COCO val2014 data. """
        print("Evaluating the model ...")
        config = self.config

        results = []
        if not os.path.exists(config.eval_result_dir):
            os.mkdir(config.eval_result_dir)

        # Generate the captions for the images
        idx = 0
        eval_epochs = 1000
        for k in tqdm(list(range(min(eval_epochs, eval_data.num_batches))), desc='batch'):
        #for k in range(1):
            image_files = eval_data.next_batch()
            conv_features = self.get_imagefeatures(image_files)
            caption_data, scores = self.generator.eval(sess, conv_features)

            fake_cnt = 0 if k<eval_data.num_batches-1 \
                         else eval_data.fake_count
            for l in range(eval_data.batch_size-fake_cnt):
                ## self.predictions will return the indexes of words, we need to find the corresponding word from it.
                word_idxs = caption_data[l]
                ## get_sentence will return a sentence till there is a end delimiter which is '.'
                caption = str(eval_data.vocabulary.get_sentence(word_idxs))
                results.append({'image_id': int(eval_data.image_ids[idx]),
                                'caption': caption})
                #print(results)
                idx += 1

                # Save the result in an image file, if requested
                if config.save_eval_result_as_image:
                    image_file = batch[l]
                    image_name = image_file.split(os.sep)[-1]
                    image_name = os.path.splitext(image_name)[0]
                    img = mpimg.imread(image_file)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(caption)
                    plt.savefig(os.path.join(config.eval_result_dir,
                                             image_name+'_result.jpg'))

        fp = open(config.eval_result_file, 'w')
        json.dump(results, fp)
        fp.close()

        # Evaluate these captions
        eval_result_coco = eval_data.coco.loadRes(config.eval_result_file)
        scorer = COCOEvalCap(eval_data.coco, eval_result_coco)
        scorer.evaluate()
        print("Evaluation complete.")