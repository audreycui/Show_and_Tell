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
from seq2seq import BasicS2SModel
import os
import shutil
import re
import pprint
import time
from collections import Counter

from test_model import TestS2SModel

#from Shaofan Lai's tensorflow implementation of SeqGAN (Yu et. al, 2017)
#https://github.com/Shaofanl/SeqGAN-Tensorflow
#SeqGAN: https://arxiv.org/abs/1609.05473

#Audrey's modifications:
#removed parameters for seqgan constructor, generator, disciminator (parameters are already in config)

enc_sentence_length = 4096
dec_sentence_length = 50

vgg_dir = 'D:/download/art_desc/train/images_vgg/'
# Batch_size: 1
input_batches = [
    [np.load(vgg_dir+'art_desc1.npy')], 
    [np.load(vgg_dir+'art_desc2.npy')], 
    [np.load(vgg_dir+'art_desc3.npy')], 
    [np.load(vgg_dir+'art_desc4.npy')], 
    [np.load(vgg_dir+'art_desc6.npy')], 
    [np.load(vgg_dir+'art_desc7.npy')], 
    [np.load(vgg_dir+'art_desc8.npy')], 
    [np.load(vgg_dir+'art_desc9.npy')],
    [np.load(vgg_dir+'art_desc11.npy')], 
    [np.load(vgg_dir+'art_desc12.npy')],
    [np.load(vgg_dir+'art_desc13.npy')], 
    [np.load(vgg_dir+'art_desc14.npy')], 
    [np.load(vgg_dir+'art_desc16.npy')], 
    [np.load(vgg_dir+'art_desc17.npy')],
    [np.load(vgg_dir+'art_desc18.npy')], 
    [np.load(vgg_dir+'art_desc19.npy')]] 

target_batches = [
    ['The close range of this photograph of peeling paint precludes the viewer from gaining any foothold into the space of the picture, emphasizing its ultimate flatness. Siskind was especially drawn to surfaces that resembled the canvases of the Abstract Expressionist painters, with whom he was friends.'],
    ['Metal Hook is one of Siskind\'s first photographs that truly focuses on the abstract visual language of ordinary objects.  The flatness of the image as a whole also serves to assert the graphic quality of the metal hook itself as a sign/symbol for male and female, thus suggesting a level of content in addition to that of form.'],
    ['One of Siskind\'s later works, Recife (Olinda) 8 was taken during his travels in Northeastern Brazil.  The result is that we are forced to remain as viewers attached to the abstract surface - noting with pleasure the additional details of age, texture, misaligned lines, and accidental drips.'],
    ['Siskind\'s first pictures show a decidedly more straightforward approach to picture making than the later work for which he became known. Although the male figure is a specific individual and technically the focal point, he is flattened in his own reflection against the back wall, pressed into the service of the overall design of the photograph.'],
    ['The Blue Series followed the Red Series of paintings and this is one of its most successful examples. The rectangular shapes of various shades of blue and green are suspended within a resplendent azure surface.'],
    ['This is one of the paintings belonging to the Red Series. Here the artist immersed himself completely into the exploration of the color red, one of the most expressive among the primary colors.'],
    ['In this famous cartoon of 1946 Ad Reinhardt tried to encapsulate the essence of the artistic modernism with its history and inherent conflicts within the American context. The tree of modern art has its roots deep in history - the Greeks are here, and so are Persian miniatures and Japanese prints.'],
    ['This early composition by Ad Reinhardt exhibits the artist\'s profound interest and understanding of the Cubist art of Pablo Picasso and George Braque. The palette is typical of the style and is comprised of four colors essential for a Cubist painting: black, white, brown, and gray.'],
    ['Here he is obviously quoting Stuart Davis, the American artist who was a key influence on young Reinhardt. Painted in the same year as the Cubist gouache, this canvas presents quite a stark contrast with Reinhardt\'s earlier artistic pursuits'],
    ['In 1962, the date of this painting, Gottlieb spoke about the emotional quality of color in his work. Beginning in 1956, Gottlieb\'s monumental Burst paintings developed from the Imaginary Landscapes, focusing on a simplification of space and color from the earlier series.'],
    ['Painted just a year before Gottlieb\'s death, this is one of the last in this series of Burst paintings. The picture\'s elongated form echoes the vertical composition of his earlier paintings, emphasizing the empty space between the lower and upper portions of the picture.'],
    ['Gottlieb began his Imaginary Landscapes series in 1951. This stylistic shift is reinforced by the Imaginary Landscapes\' addition of brighter tones and colors than the earlier Pictographs.'],
    ['The first of Gottlieb\'s Unstill Life series dates from 1948, but this is a larger and later work from the same series. Here, Gottlieb employs an uncompromising degree of abstraction and modern slickness with its palette of blacks, grays, and reds.'],
    ['Vigil confronts us directly with several mask-like faces that suggest Nonwestern sources (African, Sepik e.g.) against a dark background that suggests night, and perhaps the need to be watchful. This painting is typical of Gottlieb\'s Pictograph paintings with the geometric compartmentalization of the flat space and its use of seemingly mythic signs and symbols.'],
    ['After leaving New York permanently and traveling through America and Canada, Martin returned to New Mexico to live in isolation.  Although Martin did not activate her filmmaking career, Gabriel was another effort in exploring landscape, prompting an understanding of humans through their reaction to nature.'],
    ['Around 1964, Martin began using acrylic paint rather than oil and simultaneously replaced colored pencils with graphite. Martin in fact claimed that the idea of a grid first entered her mind when she was thinking about the innocence of trees.']]
    
all_target_sentences = []
for target_batch in target_batches:
    all_target_sentences.extend(target_batch)
    
def tokenizer(sentence):
    tokens = re.findall(r"[\w]+|[^\s\w]", sentence)
    return tokens

_START_ = "_GO_"
_PAD_ = "_PAD_"
_END_ = "_END_"

def build_vocab(sentences, max_vocab_size=None):
    word_counter = Counter()
    vocab = dict()
    reverse_vocab = dict()
    
    for sentence in sentences:
        tokens = tokenizer(sentence)
        word_counter.update(tokens)
        
    if max_vocab_size is None:
        max_vocab_size = len(word_counter)
    
    vocab[_START_] = 0
    vocab[_PAD_] = 1
    vocab[_END_] = 2
    vocab_idx = 3
    for key, value in word_counter.most_common(max_vocab_size):
            vocab[key] = vocab_idx
            vocab_idx += 1
            
    for key, value in vocab.items():
        reverse_vocab[value] = key
            
    return vocab, reverse_vocab, max_vocab_size


vocab, reverse_vocab, max_vocab_size = build_vocab(all_target_sentences)

def token2idx(word, vocab):
    return vocab[word]

def sent2idx(sent, vocab, max_sentence_length=enc_sentence_length, is_target=False):
    tokens = tokenizer(sent)
    current_length = len(tokens)
    pad_length = max_sentence_length - current_length
    if is_target:
        return [0] + [token2idx(token, vocab) for token in tokens] + [2] + [1] * (pad_length-1), current_length + 1
    else:
        return [token2idx(token, vocab) for token in tokens] + [1] * pad_length, current_length

def idx2token(idx, reverse_vocab):
    if idx in reverse_vocab:
        return reverse_vocab[idx]
    else:
        return '_UNK_'

def idx2sent(indices, reverse_vocab):
    return " ".join([idx2token(idx, reverse_vocab) for idx in indices])

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

        #sentences = np.array(sentences)
        #self.generator = Generator(self, config)
        self.generator = None
        #self.generator = BasicS2SModel(self, config)
        #self.discriminator = Discriminator(self, config) 
        self.discriminator = None
        #self.log_generation = False

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

    #TODO fix this 
    def get_imagefeatures_mxnet(self, image_files): #returns (batch_size, 4096*3)
        images = self.image_loader.load_images_mxnet(image_files)
        return self.image_loader.extract_features_mxnet(self.object_model, self.sentiment_model, self.scene_model, images, self.config.batch_size) #extract image features using vgg19

    
    def get_imagefeatures_vgg19(self, image_files):
        #images = self.image_loader.load_images_vgg19(image_files)

        return self.image_loader.extract_features_vgg19(self.trained_model, image_files, self.config.batch_size) #extract image features using vgg19
    
    def next_batch(self, data, extract=False):
        #print("HERE!!! next batch")
        batch = data.next_batch()
        image_files, sentences, masks, sent_lens = batch
        if (extract):
            conv_features = self.get_imagefeatures_vgg19(image_files)
        else:
            conv_features = []
        return sentences, conv_features, sent_lens


    def train(self, sess_unused, train_data):
        test_model = TestS2SModel(vocab=vocab, num_layers=self.config.num_decode_layers)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            num_epoch=200
            loss_history = []
            t0 = time.time()
            for epoch in tqdm(list(range(num_epoch)), desc='Training Generator'):
                all_preds = []
                epoch_loss = 0
                for input_batch, target_batch in zip(input_batches, target_batches):
                    input_batch_tokens = []
                    target_batch_tokens = []
                    enc_sentence_lengths = []
                    dec_sentence_lengths = []

                    for input_sent in input_batch:
                        tokens = input_sent
                        sent_len = len(input_sent)
                        input_batch_tokens.append(tokens)
                        enc_sentence_lengths.append(sent_len)

                    for target_sent in target_batch:
                        tokens, sent_len = sent2idx(target_sent,
                                     vocab=vocab,
                                     max_sentence_length=dec_sentence_length,
                                     is_target=True)
                        target_batch_tokens.append(tokens)
                        dec_sentence_lengths.append(sent_len)
            
                    batch_preds, batch_loss = test_model.train_one_step(sess,input_batch_tokens,enc_sentence_lengths,target_batch_tokens,dec_sentence_lengths)
                    epoch_loss += batch_loss
                    #loss_history.append(batch_loss)
                    all_preds.append(batch_preds)

                #loss_history.append(epoch_loss)
                if epoch % 10 == 0:
                    print('Epoch', epoch)
                    print('epoch loss: ', epoch_loss )
                    for input_sent, target_sent, pred in zip(input_batch, target_batch, batch_preds):
                        print('Input:', input_sent)
                        print('Prediction:', idx2sent(pred, reverse_vocab=reverse_vocab))
                        print('Target:', target_sent)
            
            test_model.save_model(sess, './models')

    def train_old(self, sess, train_data):
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

        vgg_dir = 'D:/download/art_desc/train/images_vgg/'
        # Batch_size: 1
        '''
        conv_features = [
            np.load(vgg_dir+'art_desc1.npy'), 
            np.load(vgg_dir+'art_desc2.npy'), 
            np.load(vgg_dir+'art_desc3.npy'), 
            np.load(vgg_dir+'art_desc4.npy')] 
        target_batches = [
            ['The close range of this photograph of peeling paint precludes the viewer from gaining any foothold into the space of the picture, emphasizing its ultimate flatness. Siskind was especially drawn to surfaces that resembled the canvases of the Abstract Expressionist painters, with whom he was friends.'],
            ['Metal Hook is one of Siskind\'s first photographs that truly focuses on the abstract visual language of ordinary objects.  The flatness of the image as a whole also serves to assert the graphic quality of the metal hook itself as a sign/symbol for male and female, thus suggesting a level of content in addition to that of form.'],
            ['One of Siskind\'s later works, Recife (Olinda) 8 was taken during his travels in Northeastern Brazil.  The result is that we are forced to remain as viewers attached to the abstract surface - noting with pleasure the additional details of age, texture, misaligned lines, and accidental drips.'],
            ['Siskind\'s first pictures show a decidedly more straightforward approach to picture making than the later work for which he became known. Although the male figure is a specific individual and technically the focal point, he is flattened in his own reflection against the back wall, pressed into the service of the overall design of the photograph.']]
        sentences = np.squeeze(target_batches)
        print(sentences)
        '''
        vgg_dir = 'D:/download/art_desc/train/images_vgg/'
# Batch_size: 1
        input_batches = [
            [np.load(vgg_dir+'art_desc1.npy')], 
            [np.load(vgg_dir+'art_desc2.npy')], 
            [np.load(vgg_dir+'art_desc3.npy')], 
            [np.load(vgg_dir+'art_desc4.npy')], 
            [np.load(vgg_dir+'art_desc6.npy')], 
            [np.load(vgg_dir+'art_desc7.npy')], 
            [np.load(vgg_dir+'art_desc8.npy')], 
            [np.load(vgg_dir+'art_desc9.npy')]] 

        target_batches = [
            ['The close range of this photograph of peeling paint precludes the viewer from gaining any foothold into the space of the picture, emphasizing its ultimate flatness. Siskind was especially drawn to surfaces that resembled the canvases of the Abstract Expressionist painters, with whom he was friends.'],
            ['Metal Hook is one of Siskind\'s first photographs that truly focuses on the abstract visual language of ordinary objects.  The flatness of the image as a whole also serves to assert the graphic quality of the metal hook itself as a sign/symbol for male and female, thus suggesting a level of content in addition to that of form.'],
            ['One of Siskind\'s later works, Recife (Olinda) 8 was taken during his travels in Northeastern Brazil.  The result is that we are forced to remain as viewers attached to the abstract surface - noting with pleasure the additional details of age, texture, misaligned lines, and accidental drips.'],
            ['Siskind\'s first pictures show a decidedly more straightforward approach to picture making than the later work for which he became known. Although the male figure is a specific individual and technically the focal point, he is flattened in his own reflection against the back wall, pressed into the service of the overall design of the photograph.'],
            ['The Blue Series followed the Red Series of paintings and this is one of its most successful examples. The rectangular shapes of various shades of blue and green are suspended within a resplendent azure surface.'],
            ['This is one of the paintings belonging to the Red Series. Here the artist immersed himself completely into the exploration of the color red, one of the most expressive among the primary colors.'],
            ['In this famous cartoon of 1946 Ad Reinhardt tried to encapsulate the essence of the artistic modernism with its history and inherent conflicts within the American context. The tree of modern art has its roots deep in history - the Greeks are here, and so are Persian miniatures and Japanese prints.'],
            ['This early composition by Ad Reinhardt exhibits the artist\'s profound interest and understanding of the Cubist art of Pablo Picasso and George Braque. The palette is typical of the style and is comprised of four colors essential for a Cubist painting: black, white, brown, and gray.']]
            
        #sentences, conv_features, sent_lens = self.next_batch(train_data, True)
        conv_features = np.squeeze(input_batches)
        sentences = []
        sent_lens = []
        target_batches = np.squeeze(target_batches)
        for sent in target_batches:
            current_word_idxs, current_length = (train_data.vocabulary.process_sentence(sent))
            current_num_words = min(config.max_caption_length-2, current_length)

            current_word_idxs = [config._START_] + current_word_idxs[:current_num_words] + [config._END_]
            pad_length = config.max_caption_length - current_num_words -2
            if pad_length > 0:
                current_word_idxs += [config._PAD_] * (pad_length)

            sentences.append(current_word_idxs)
            sent_lens.append(config.max_caption_length)
        sentences = np.array(sentences)
        sent_lens = np.array(sent_lens)

        print('sentences shape ' + str(sentences.shape))
        print('sent kebs shape ' + str(sent_lens.shape))

        for epoch in tqdm(list(range(pretrain_g_epochs)), desc='Pretraining Generator'):
        #for epoch in range(1):
            #sentences, conv_features, sent_lens = self.next_batch(train_data, True)
            
            summary, fake_samples, loss = gen.pretrain(sess, sentences, conv_features, sent_lens) #changed sampler to next_batch
            #print(np.shape(fake_samples))
            #next_batch consists of images, captions, and masks
            if (epoch%10 == 0):
                for sent, sample in zip(sentences, fake_samples):
                    print("TARGET: " + train_data.vocabulary.get_sentence(sent))
                    print("PREDICTED" + train_data.vocabulary.get_sentence(sample))
                print(">>>>> LOSS " + str(loss))
            writer.add_summary(summary, epoch)
            '''
            if evaluate and evaluator is not None: #TODO add eval
                evaluator(gen.generate(sess), epoch)
            '''
            #TODO evaluator
        if config.debug:
            saver = tf.train.Saver(tf.global_variables())
            saver.save(sess, self.config.checkpoint_dir+"model.ckpt", global_step=self.generator.global_step)
        else:
            #print("")
            self.save(sess)
        return

        train_data.reset()

        for epoch in tqdm(list(range(pretrain_d_epochs)), desc='Pretraining Discriminator'):
        #for epoch in range(1):
            sentences, conv_features, sent_lens = self.next_batch(train_data, True)
            fake_samples = gen.generate(sess, sentences, conv_features, sent_lens)
            fake_samples = np.squeeze(fake_samples)
            if (epoch%50 == 0):
                print("TARGET: " + train_data.vocabulary.get_sentence(sentences[0]))
                print("PREDICTED: " + train_data.vocabulary.get_sentence(fake_samples[0]))
            
            real_samples, conv_features, sample_lens = self.next_batch(train_data)

           #print("fake samples shape: " + str(fake_samples.shape))
            #print("real samples shape: " + str(real_samples.shape))
            samples = np.concatenate([fake_samples, real_samples])
            labels = np.concatenate([np.zeros((batch_size,)),
                                     np.ones((batch_size,))])
            for _ in range(3):
                indices = np.random.choice(
                    len(samples), size=(batch_size,), replace=False)
                dis.train(sess, samples[indices], labels[indices])
        train_data.reset()
        
        for epoch in tqdm(list(range(config.total_epochs)), desc='Adversarial training'):
        #for epoch in range(1):
            for _ in range(1):
                sentences, conv_features, sent_lens = self.next_batch(train_data, True)
                #real_samples = sentences
                fake_samples = gen.generate(sess, sentences, conv_features, sent_lens)
                fake_samples = np.squeeze(fake_samples) #generator generates fake samples
                if (epoch%50 == 0):
                    print("TARGET: " + train_data.vocabulary.get_sentence(sentences[0]))
                    print("PREDICTED: " + train_data.vocabulary.get_sentence(fake_samples[0]))
        
                rewards = gen.get_reward(sess, sentences, conv_features, config.num_rollout, dis) 

                #debug: changed fake samples to sentences (real samples)
                summary, fake_samples = gen.train(sess, fake_samples, conv_features, rewards) #generate new fake samples and reward
                
                # np.set_printoptions(linewidth=np.inf,
                #                     precision=3)
                # print rewards.mean(0)
            writer.add_summary(summary, epoch)

            for _ in tqdm(list(range(5)), desc='Discriminator batch'):
                #print("conv features shape " + str(np.array(conv_features).shape))
                sentences, conv_features, sent_lens = self.next_batch(train_data, True)

                fake_samples = gen.generate(sess, sentences, conv_features, sent_lens)
                fake_samples = np.squeeze(fake_samples) #generator generates fake samples after being trained
                #real_samples = sentences

                real_samples, conv_features, sample_lens = self.next_batch(train_data, True)
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
            real_samples, conv_features, sample_lens = self.next_batch(train_data, True)
            np.save('generation', gen.generate(sess, real_samples, conv_features, sample_lens))
        if not os.path.exists(config.save_dir):
            try:  
                os.mkdir(config.save_dir)
            except OSError:  
                print ("Creation of the directory %s failed" % path)
            else:  
                print ("Successfully created the directory %s " % path)
        self.save(sess)
        writer.close()
        print("Training complete.")

    def eval(self, sess_unused, eval_data):
        vocabulary, reverse_vocab, max_vocab_size = build_vocab(all_target_sentences)

        test_model = TestS2SModel(vocab=vocabulary, mode="inference",use_beam_search=False, num_layers=self.config.num_decode_layers)

        with tf.Session() as sess:
            test_model.restore_model(sess, './')
            batch_preds = []
            batch_tokens = []
            batch_sent_lens = []
            input_batches = [
                [np.load(vgg_dir+'art_desc1.npy')], 
                [np.load(vgg_dir+'art_desc2.npy')], 
                [np.load(vgg_dir+'art_desc3.npy')], 
                [np.load(vgg_dir+'art_desc4.npy')], 
                [np.load(vgg_dir+'art_desc6.npy')], 
                [np.load(vgg_dir+'art_desc7.npy')], 
                [np.load(vgg_dir+'art_desc8.npy')], 
                [np.load(vgg_dir+'art_desc9.npy')],
                [np.load(vgg_dir+'art_desc21.npy')], 
                [np.load(vgg_dir+'art_desc22.npy')],
                [np.load(vgg_dir+'art_desc23.npy')], 
                [np.load(vgg_dir+'art_desc24.npy')], 
                [np.load(vgg_dir+'art_desc26.npy')], 
                [np.load(vgg_dir+'art_desc31.npy')],
                [np.load(vgg_dir+'art_desc28.npy')], 
                [np.load(vgg_dir+'art_desc29.npy')]] 
            

            target_batches = [
                ['The close range of this photograph of peeling paint precludes the viewer from gaining any foothold into the space of the picture, emphasizing its ultimate flatness. Siskind was especially drawn to surfaces that resembled the canvases of the Abstract Expressionist painters, with whom he was friends.'],
                ['Metal Hook is one of Siskind\'s first photographs that truly focuses on the abstract visual language of ordinary objects.  The flatness of the image as a whole also serves to assert the graphic quality of the metal hook itself as a sign/symbol for male and female, thus suggesting a level of content in addition to that of form.'],
                ['One of Siskind\'s later works, Recife (Olinda) 8 was taken during his travels in Northeastern Brazil.  The result is that we are forced to remain as viewers attached to the abstract surface - noting with pleasure the additional details of age, texture, misaligned lines, and accidental drips.'],
                ['Siskind\'s first pictures show a decidedly more straightforward approach to picture making than the later work for which he became known. Although the male figure is a specific individual and technically the focal point, he is flattened in his own reflection against the back wall, pressed into the service of the overall design of the photograph.'],
                ['The Blue Series followed the Red Series of paintings and this is one of its most successful examples. The rectangular shapes of various shades of blue and green are suspended within a resplendent azure surface.'],
                ['This is one of the paintings belonging to the Red Series. Here the artist immersed himself completely into the exploration of the color red, one of the most expressive among the primary colors.'],
                ['In this famous cartoon of 1946 Ad Reinhardt tried to encapsulate the essence of the artistic modernism with its history and inherent conflicts within the American context. The tree of modern art has its roots deep in history - the Greeks are here, and so are Persian miniatures and Japanese prints.'],
                ['This early composition by Ad Reinhardt exhibits the artist\'s profound interest and understanding of the Cubist art of Pablo Picasso and George Braque. The palette is typical of the style and is comprised of four colors essential for a Cubist painting: black, white, brown, and gray.'],
                ['Martin destroyed much of her work made before the late 1950s when she shifted to a grid format, so works from this period of her oeuvre are scarce. Her early style has been compared to that of Arshile Gorky and, like his works, Untitled displays Martin\'s debt to Surrealism and Abstract Expressionism.'],
                ['Untitled XXI is an example of Martin\'s work after the mid-1970s. Though Untitled XXI is not explicitly designated as a landscape, by name or representation, Martin throughout her artistic life attempted to capture the sublime of everyday nature through her continued variation on the square format.'],
                ['With Window, Martin\'s forms became less organic and more rigid as she experimented with rectangular forms, anticipating the later introduction of the grid\'s mathematical precision in her work. Although this work was created during the first years of Martin\'s final return to New York, Window still incorporates a Southwestern palette, while abandoning the curved line of earlier work.'],
                ['Dropping a Han Dynasty Urn, an early work by the artist, demonstrates his show-stopping conceptual brilliance, and desire to provoke controversy. The Han dynasty is considered a defining moment in Chinese civilization.'],
                ['Study of Perspective Tiananmen Square was part of a series begun in 1995 and completed in 2003. In what first appears to be a classic tourist snapshot, Ai sticks his middle finger up at Tiananmen Square Gate.'],
                ['Ryder\'s moon both illuminates and obscures through the shadows it casts. The painting is one of Ryder\'s most abstract, until closer observation draws us past its compositionally powerful surface and we appreciate the represented scene more fully.'],
                ['Surveillance Camera, an austere and quite beautiful marble sculpture, reminds us that the artist is watching those who watch him. The artist, in turn, tracks the surveillance cameras, vans, and plain-clothes police officers that monitor his gates.'],
                ['This work compresses a ton of traditional pu\'er tea leaves into the space of one cubic meter. While in the West, drinking tea (especially from Chinese porcelain) has historically been a status symbol, tea is the everyday drink in China.']]
            

            input_batches = np.squeeze(input_batches)
            for input_sent in input_batches:
                #tokens, sent_len = sent2idx(input_sent)
                batch_tokens.append(input_sent)
                batch_sent_lens.append(len(input_sent))

            #print('batch_tokens shape: ' + str(np.array(batch_tokens).shape))
            #print('batch_sent_lens shape: ' + str(np.array(batch_sent_lens).shape))
            batch_preds = test_model.inference(sess, batch_tokens, batch_sent_lens)
            
            batch_preds = np.squeeze(np.array(batch_preds))
            input_batches = np.squeeze(np.array(input_batches))
            target_batches = np.squeeze(np.array(target_batches))

            print('batch preds shape: ' + str(np.array(batch_preds).shape))
            print('input batches shape: ' + str(np.array(input_batches).shape))
            print('target batches shape: ' + str(np.array(target_batches).shape))

            for input_sent, target_sent, pred in zip(input_batches, target_batches, batch_preds):
                print('Input:', input_sent)
                #print(pred)
                print('Prediction:', idx2sent(pred, reverse_vocab=reverse_vocab))
                print('Target:', target_sent)

    def eval_old(self, sess, eval_data):
        """ Evaluate the model using the COCO val2014 data. """
        print("Evaluating the model ...")
        config = self.config

        results = []
        if not os.path.exists(config.eval_result_dir):
            os.mkdir(config.eval_result_dir)

        #if config.debug:
        self.restore_model(sess)
        vgg_dir = 'D:/download/art_desc/train/images_vgg/'
        # Generate the captions for the images
        vgg_dir = 'D:/download/art_desc/train/images_vgg/'
# Batch_size: 1
        input_batches = [
            [np.load(vgg_dir+'art_desc1.npy')], 
            [np.load(vgg_dir+'art_desc2.npy')], 
            [np.load(vgg_dir+'art_desc3.npy')], 
            [np.load(vgg_dir+'art_desc4.npy')], 
            [np.load(vgg_dir+'art_desc6.npy')], 
            [np.load(vgg_dir+'art_desc7.npy')], 
            [np.load(vgg_dir+'art_desc8.npy')], 
            [np.load(vgg_dir+'art_desc9.npy')]] 

        conv_features = np.squeeze(input_batches)

        target_batches = [
            ['The close range of this photograph of peeling paint precludes the viewer from gaining any foothold into the space of the picture, emphasizing its ultimate flatness. Siskind was especially drawn to surfaces that resembled the canvases of the Abstract Expressionist painters, with whom he was friends.'],
            ['Metal Hook is one of Siskind\'s first photographs that truly focuses on the abstract visual language of ordinary objects.  The flatness of the image as a whole also serves to assert the graphic quality of the metal hook itself as a sign/symbol for male and female, thus suggesting a level of content in addition to that of form.'],
            ['One of Siskind\'s later works, Recife (Olinda) 8 was taken during his travels in Northeastern Brazil.  The result is that we are forced to remain as viewers attached to the abstract surface - noting with pleasure the additional details of age, texture, misaligned lines, and accidental drips.'],
            ['Siskind\'s first pictures show a decidedly more straightforward approach to picture making than the later work for which he became known. Although the male figure is a specific individual and technically the focal point, he is flattened in his own reflection against the back wall, pressed into the service of the overall design of the photograph.'],
            ['The Blue Series followed the Red Series of paintings and this is one of its most successful examples. The rectangular shapes of various shades of blue and green are suspended within a resplendent azure surface.'],
            ['This is one of the paintings belonging to the Red Series. Here the artist immersed himself completely into the exploration of the color red, one of the most expressive among the primary colors.'],
            ['In this famous cartoon of 1946 Ad Reinhardt tried to encapsulate the essence of the artistic modernism with its history and inherent conflicts within the American context. The tree of modern art has its roots deep in history - the Greeks are here, and so are Persian miniatures and Japanese prints.'],
            ['This early composition by Ad Reinhardt exhibits the artist\'s profound interest and understanding of the Cubist art of Pablo Picasso and George Braque. The palette is typical of the style and is comprised of four colors essential for a Cubist painting: black, white, brown, and gray.']]
            

        idx = 0
        eval_epochs = 500
        for k in tqdm(list(range(min(eval_epochs, eval_data.num_batches))), desc='batch'):
        #for k in range(1):
            image_files = eval_data.next_batch()
            #print("len image files: " + str(len(image_files)))
            #conv_features = [np.load(vgg_dir+'art_desc2047.npy'), np.load(vgg_dir+'art_desc2048.npy')]
            #conv_features = self.get_imagefeatures(image_files, config.batch_size)
            caption_data = self.generator.eval(sess, conv_features)
            caption_data = np.squeeze(caption_data)
            
            #print('caption data shape ' + str(caption_data.shape))

            fake_cnt = 0 if k<eval_data.num_batches-1 \
                         else eval_data.fake_count
            for l in range(eval_data.batch_size-fake_cnt):
                ## self.predictions will return the indexes of words, we need to find the corresponding word from it.
                word_idxs = caption_data[l]
                ## get_sentence will return a sentence till there is a end delimiter which is '.'
                caption = str(eval_data.vocabulary.get_sentence(word_idxs))
                print(caption)
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

    def restore_model(self,sess):
        checkpoint_dir = self.config.save_dir
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir))

    def save(self,sess):
        checkpoint_dir = self.config.save_dir
        writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess,checkpoint_dir + "model.ckpt",global_step=self.global_step)
        