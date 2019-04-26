import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
from nltk.tokenize import word_tokenize
import math
class Vocabulary(object):
    def __init__(self, size, ctrl_symbols, save_file=None):
        self.size = size
        self.words = []
        self.word2idx = {}
        self.word_frequencies = []
        self.ctrl_symbols = ctrl_symbols
        self.idx_head = len(self.ctrl_symbols)
        if save_file is not None:
            self.load(save_file)

    def build(self, sentences):
        """ Build the vocabulary and compute the frequency of each word. """
        word_counts = {}
        for sentence in tqdm(sentences):
            for w in word_tokenize(sentence.lower()):
                word_counts[w] = word_counts.get(w, 0) + 1.0

        assert self.size <= len(word_counts.keys())

        for i, pred in enumerate(self.ctrl_symbols):
            self.words.append(pred)
            self.word2idx[pred] = i
            self.word_frequencies.append(1.0)

        word_counts = sorted(list(word_counts.items()),
                            key=lambda x: x[1],
                            reverse=True)

        for idx in range(self.size):
            word, frequency = word_counts[idx]
            self.words.append(word)
            self.word2idx[word] = idx + self.idx_head
            self.word_frequencies.append(frequency)

        self.word_frequencies = np.array(self.word_frequencies)
        self.word_frequencies /= np.sum(self.word_frequencies)
        self.word_frequencies = np.log(self.word_frequencies)
        self.word_frequencies -= np.max(self.word_frequencies)

    def process_sentence(self, sentence):
        """ Tokenize a sentence, and translate each token into its index
            in the vocabulary. """
        words = word_tokenize(sentence.lower())
        current_length = len(words)

        word_idxs = []
        for w in words:
            if w in self.word2idx.keys():
                word_idxs.append(self.word2idx[w])
            else:
                word_idxs.append(self.word2idx['<UNK>'])

        return word_idxs, current_length

    def get_sentence_bad(self, idxs):
        """ Translate a vector of indicies into a sentence. """
        return  [self.words[i] for i in idxs]

    def get_sentence(self, idxs):
        """ Translate a vector of indicies into a sentence. """
        words = [self.words[i] for i in idxs]

        #print(words)
        #print('words shape ' + str(np.array(words).shape))
        
        if (words[-1] != '.'):
            words.append('.')
        length = np.argmax(np.array(words)=='.') + 1
        words = words[:length]

        for i in range (len(words)): 
            if not isinstance(words[i], str):
                words[i] = ""
        #print(words)
        
        sentence = "".join([" "+w if not w.startswith("'") \
                            and w not in string.punctuation \
                            else w for w in words]).strip()
        return sentence


    def save(self, save_file):
        """ Save the vocabulary to a file. """
        #print("words:"+str(len(self.words)))
        #print("index:"+str(len(list(range(self.size)))))
        #print("freq:"+str(len(self.word_frequencies)))
        data = pd.DataFrame({'word': self.words,
                             'index': list(range(self.size + self.idx_head)),
                             'frequency': self.word_frequencies})
        data.to_csv(save_file)

    def load(self, save_file):
        """ Load the vocabulary from a file. """
        assert os.path.exists(save_file)
        data = pd.read_csv(save_file)
        self.words = data['word'].values
        self.word2idx = {self.words[i]:i for i in range(self.size + self.idx_head)}
        self.word_frequencies = data['frequency'].values
