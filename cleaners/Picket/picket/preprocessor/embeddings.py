from __future__ import unicode_literals, division
import os
import logging
import numpy as np
import pandas as pd
from picket.globalvar import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from gensim.models.fasttext import FastText

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AvgModel(object):

    def __init__(self, config):
        self.config = config

    def load_vocab(self, data, path, wv, df_all):
        if not self.config['load']:
            # build vocab
            vec, vocab = self.build_vocab(data, path, wv, df_all)
        else:
            vec = np.load(path + 'vec.npy', allow_pickle=True)
            unique_cells = np.load(path + 'vocab.npy', allow_pickle=True)
            vocab = pd.DataFrame(data=unique_cells, columns=['word']).reset_index().set_index('word')
        return vec, vocab

    #TODO: resolve the problem that fasttext model is computed multiple times even if the model is the same
    def build_vocab(self, data, path, wv, df_all):

        '''



        def get_cell_vector(cell):
            cell = self.config['tokenizer'](cell)
            idx = word_vocab.loc[cell, 'idx'].values
            v = vec[idx]
            if len(cell) == 1:
                return v
            w = weight[idx].reshape(1, -1)
            return list(np.matmul(w, v)/np.sum(w))
        # compute embedding for each cell
        if max_length == 1:
            unique_cells = unique
        else:
            unique_cells = np.unique(data)
            vec = np.array(list(map(get_cell_vector, unique_cells))).squeeze()

        # add the '_mask_' token in the vocabulary of TEXT category
        unique_cells = np.append(unique_cells, ['_mask_'], axis=0)

        vocab = pd.DataFrame(data=unique_cells, columns=['word']).reset_index().set_index('word')

        # add the '_mask_' embedding in the vec array: all zeros vector
        vec = np.append(vec, [np.zeros(self.config['dim'])], axis=0)

        # (optional) save model
        if self.config['save']:
            np.save(path+'vec', vec)
            np.save(path+'vocab', unique_cells)
        return vec, vocab
        '''

        # tokenize cell
        if self.config['separate']:
            corpus = [self.config['tokenizer'](i) for i in data]
        else:
            corpus = [self.config['tokenizer'](i) for i in df_all.values]
        
        max_length = max([len(s) for s in corpus])

        # find unique words in the corpus of the attribute
        all_words = np.hstack(corpus)
        unique, counts = np.unique(all_words, return_counts=True)
        freq = counts / len(all_words)
        weight = self.config['a'] / (self.config['a'] + freq)

        # train language model
        if wv is None:
            wv = LocalFasttextModel(self.config, corpus)

        # obtain word vector
        vec = wv.get_array_vectors(unique)
        word_vocab = pd.DataFrame(list(zip(unique, list(range(len(unique))))),
                                  columns=['word', 'idx']).set_index('word')

        def get_cell_vector(cell):
            if self.config['SIF'] is True:
                cell = self.config['tokenizer'](cell)
                idx = word_vocab.loc[cell, 'idx'].values
                v = vec[idx]
                if len(cell) == 1:
                    return v
                w = weight[idx].reshape(1, -1)
                return list(np.matmul(w, v) / np.sum(w))

            elif self.config['SIF'] is False:
                cell = self.config['tokenizer'](cell)
                idx = word_vocab.loc[cell, 'idx'].values
                v = vec[idx]
                if len(cell) == 1:
                    return v[0]
                # take the average sum of the vectors of all words in the cell
                v = [sum(x) for x in zip(*v)]
                v = [x / len(cell) for x in v]
                return v

        # compute embedding for each cell
        if max_length == 1:
            unique_cells = unique
        else:
            unique_cells = np.unique(data)
            vec = np.array(list(map(get_cell_vector, unique_cells))).squeeze()

        vocab = pd.DataFrame(data=unique_cells, columns=['word']).reset_index().set_index('word')

        # (optional) save model
        if self.config['save']:
            np.save(path + 'vec', vec)
            np.save(path + 'vocab', unique_cells)
        return vec, vocab


class LocalFasttextModel(object):

    def __init__(self, config, data):
        self.model = FastText(size=config['dim'],
                              window=config['window'],
                              min_count=config['min_count'],
                              batch_words=config['batch_words'],
                              seed=config['seed'],
                              iter=config['epochs'])
        self.model.build_vocab(sentences=data)
        self.model.train(sentences=data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        self.dim = config['dim']

    def get_word_vector(self, word):
        return self.model.wv[word]

    def get_array_vectors(self, array):
        return self.model.wv[array]

    def get_wv(self):
        return self.model.wv


class OneHotEncoderModel(object):

    def __init__(self, data, config):
        self.label_encoder = LabelEncoder()
        integer_encoded = self.label_encoder.fit_transform(data)
        self.encoder = OneHotEncoder(sparse=False, categories='auto', handle_unknown='ignore')
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        self.encoder.fit(integer_encoded)
        self.shape = self.get_word_vector(data[0]).shape
        self.dim = self.shape[1]
        self.vec = None
        self.vocab = np.unique(data)
        self.config = config
        logger.info("Attr Embedding Size: {}".format(self.dim))

    def get_word_vector(self, word):
        # contains a single sample
        labeled = self.label_encoder.transform([word]).reshape(1, -1)
        encoded = self.encoder.transform(labeled)
        return encoded

    def get_array_vectors(self, array):
        labeled = self.label_encoder.transform(array).reshape(len(array), -1)
        encoded = self.encoder.transform(labeled)
        final_encoded = None
        for i in range(0, self.config['dim']-self.dim):
            if final_encoded is None:
                final_encoded = np.pad(encoded, [(0, 0), (0, 1)], mode='constant')
            else:
                final_encoded = np.pad(final_encoded, [(0, 0), (0, 1)], mode='constant')
        return final_encoded

    def load_vocab(self):
        self.vec = self.get_array_vectors(self.vocab)
        self.vocab = pd.DataFrame(data=self.vocab, columns=['word']).reset_index().set_index('word')
        return self.vec, self.vocab

    def get_wv(self):
        return pd.Series(self.encoder.categories_[0],
                         index=[x.encode('UTF8') for x in self.label_encoder.classes_]).to_dict()


class PaddingModel(object):

    def __init__(self, data, config):
        self.dim = config['dim']
        self.constant = config['padding_constant']
        self.vocab = np.unique(data)
        self.vocab = self.vocab[~np.isnan(self.vocab)]
        self.vocab = np.append(self.vocab, [float('nan')])
        self.config = config
        self.vec = None

    def get_array_vectors(self, r):
        encoded = None
        r = r[~np.isnan(r)].reshape(-1, 1)
        #r = (r - r.min())/(r.max() - r.min())
        for i in range(0, self.dim-1):
            if encoded is None:
                encoded = np.pad(r, [(0, 0), (0, 1)], mode='constant')
            else:
                encoded = np.pad(encoded, [(0, 0), (0, 1)], mode='constant')
        '''
        if self.dim-1 > 0:
            bins = np.arange(self.dim-1)/(self.dim-1)
            bins[0] -= 0.1

            inds = np.digitize(r.reshape(-1,), bins)
            encoded[np.arange(r.shape[0]), inds] = 1
            print(encoded.shape)
        else:
            encoded = r
        '''

        encoded = np.concatenate((encoded.reshape(-1, self.dim), np.zeros((1, self.dim))), axis=0)
        encoded[-1, 0] = 0
        #encoded = encoded / abs(encoded).max()
        
        return encoded

    def load_vocab(self):

        self.vec = self.get_array_vectors(self.vocab.reshape((len(self.vocab), 1)))
        self.vocab = pd.DataFrame(data=self.vocab, columns=['word']).reset_index().set_index('word')
        return self.vec, self.vocab


def load_embedding(attr, config, source_data, dtype, wv, df_all):
    """
    :param attr: the attribute index in the data-frame
    :param config: the configurations
    :param source_data: the values of the attribute stored in a data-frame
    :param dtype: the type of the data the attribute has
    :param wv: the vector in which the embeddings are going to be stored
    :return: the vocabulary of the attribute (unique cells) and the corresponding embedding
    stored in the vec table
    """
    vec, vocab = None, None
    if dtype == TEXT:
        # take the average of the vectors in each cell
        embed_model = AvgModel(config)
        vec, vocab = embed_model.load_vocab(source_data.values, "", wv, df_all)
    elif dtype == CATEGORICAL:
        # one hot encoding with hash table
        embed_model = OneHotEncoderModel(source_data.values, config)
        vec, vocab = embed_model.load_vocab()
    elif dtype == NUMERIC:
        # pad with zeros and normalize
        embed_model = PaddingModel(source_data.values, config)
        vec, vocab = embed_model.load_vocab()
    return vec, vocab
