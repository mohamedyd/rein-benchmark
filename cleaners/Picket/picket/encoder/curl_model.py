import torch
import numpy as np
import sys
import os
picket_path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(picket_path))
from encoder.example_encoder import AttrEmbedding
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LookUpModelSimple(torch.nn.Module):
    def __init__(self, dataset, env, attribute_info=None):
        super(LookUpModelSimple, self).__init__()
        self.ds = dataset
        self.env = env
        # create embedding layer
        if attribute_info is None:
            self.attr_embed = AttrEmbedding(self.ds.attributes)
        else:
            self.attr_embed = AttrEmbedding(attribute_info)

    def get_encodings(self, tuples):
        tuple_embed = None
        for idx in self.ds.attributes:
            if tuple_embed is None:
                tuple_embed = self.attr_embed.get_embeddings(tuples[:, idx], attr=idx, dtype='tuple')
            else:
                tuple_embed = torch.cat((tuple_embed, self.attr_embed.get_embeddings(tuples[:, idx], attr=idx, dtype='tuple')), dim=1)
        return tuple_embed

    def get_encodings_idx(self, tuples):
        tuple_embed = None
        vec_idx = None
        for idx in self.ds.attributes:
            if tuple_embed is None:
                tuple_embed, vec_idx = self.attr_embed.get_embeddings_idx(tuples[:, idx], attr=idx, dtype='tuple')
                vec_idx = vec_idx.view(-1, 1)
            else:
                tuple_embed_tmp, vec_idx_tmp = self.attr_embed.get_embeddings_idx(tuples[:, idx], attr=idx, dtype='tuple')
                tuple_embed = torch.cat((tuple_embed, tuple_embed_tmp), dim=1)
                vec_idx = torch.cat((vec_idx, vec_idx_tmp.view(-1, 1)), dim=1)
        return tuple_embed, vec_idx



class EncodingSimple(torch.nn.Module):
    def __init__(self, dataset, env):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(EncodingSimple, self).__init__()
        self.ds = dataset
        self.env = env
        self.model = None
        self.nsteps = int(self.ds.numAttr * self.ds.numTuple / self.env['batch_size'])
        self.dataset = self.ds.df.groupby(np.arange(self.ds.numTuple) // env['batch_size'])

    def train(self, attribute_info=None):
        self.model = LookUpModelSimple(self.ds, self.env, attribute_info)
        tuple_embed = None
        step = 0

        start = time.time()

        for _, data in self.dataset:
            data = data.values
            mask_tuples = data.copy()

            if tuple_embed is None:
                tuple_embed = self.model.get_encodings(mask_tuples)
            else:
                tuple_embed_buf = self.model.get_encodings(mask_tuples)
                tuple_embed = torch.cat((tuple_embed, tuple_embed_buf), dim=0)
            step += 1

            if self.ds.numTuple//5//self.env['batch_size'] != 0:
                if step % (self.ds.numTuple//5//self.env['batch_size']) == 0:
                    logger.info('%d / %d has been loaded.' % (step*self.env['batch_size'], self.ds.numTuple))

        end = time.time()
        logger.info('Time spent: %f s' % (end-start))

        return tuple_embed

    def train_idx(self, attribute_info=None):
        self.model = LookUpModelSimple(self.ds, self.env, attribute_info)
        tuple_embed = None
        tuple_idx = None
        step = 0

        start = time.time()

        for _, data in self.dataset:
            data = data.values
            mask_tuples = data.copy()

            if tuple_embed is None:
                tuple_embed, tuple_idx = self.model.get_encodings_idx(mask_tuples)
            else:
                tuple_embed_buf, tuple_idx_buf = self.model.get_encodings_idx(mask_tuples)
                tuple_embed = torch.cat((tuple_embed, tuple_embed_buf), dim=0)
                tuple_idx = torch.cat((tuple_idx, tuple_idx_buf), dim=0)
            step += 1

            if self.ds.numTuple//5//self.env['batch_size'] != 0:
                if step % (self.ds.numTuple//5//self.env['batch_size']) == 0:
                    logger.info('%d / %d has been loaded.' % (step*self.env['batch_size'], self.ds.numTuple))

        end = time.time()
        logger.info('Time spent: %f s' % (end-start))

        return tuple_embed, tuple_idx


class LookUpModel(torch.nn.Module):

    def __init__(self, dataset, env):
        super(LookUpModel, self).__init__()
        self.ds = dataset
        self.env = env
        # create embedding layer
        self.attr_embed = AttrEmbedding(self.ds.attributes)

    def get_encodings(self, inputs):
        """
        Get encodings for the tuples and pos and neg examples.
        :param inputs:
        :return:
        """
        # the masked tuple
        mask_tuple = inputs[0]
        # the index of the masked attribute
        mask_idx = inputs[1][0]
        # the positive and negative examples
        mask_attrs = inputs[2]
        # get example embeddings
        attr_embeds = self.attr_embed.get_embeddings(mask_attrs, attr=mask_idx, dtype='example')
        # get tuple embeddings

        tuple_embed = None
        for idx in self.ds.attributes:
            if tuple_embed is None:
                tuple_embed = self.attr_embed.get_embeddings(mask_tuple[:, idx], attr=idx, dtype='tuple')
            else:
                tuple_embed = torch.cat((tuple_embed, self.attr_embed.get_embeddings(mask_tuple[:, idx], attr=idx, dtype='tuple')), dim=1)
        return tuple_embed, attr_embeds


class Encoding(torch.nn.Module):
    def __init__(self, dataset, env):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Encoding, self).__init__()
        self.ds = dataset
        self.env = env
        self.model = None
        self.nsteps = int(self.ds.numAttr * self.ds.numTuple / self.env['batch_size'])
        self.dataset = self.ds.df.groupby(np.arange(self.ds.numTuple) // env['batch_size'])

    def train(self):

        self.model = LookUpModel(self.ds, self.env)
        # create the original labels which correspond
        # to the positive example
        y = np.zeros((self.env['num_neg'] + 1, 1))
        y[0] = 1

        whole_tuples = None
        whole_examples = None
        i = 0
        mask = np.zeros(0)
        for mask_idx in range(self.ds.numAttr):
            attr_embeds = None
            tuple_embed = None
            for _, data in self.dataset:
                i = i + 1
                data = data.values
                mask_idxs = [mask_idx] * data.shape[0]  # self.env['batch_size']
                pos = data[:, mask_idx].reshape(-1, 1)
                neg = np.random.choice(self.ds.attributes[mask_idx].vocab.index.values,
                                           (data.shape[0], self.env['num_neg']))

                # we only need the negative examples for now
                # mask_attrs = np.concatenate([pos, neg], axis=1)
                mask_attrs = neg

                mask_tuples = data.copy()
                # we don't need to mask the tuples for now
                # mask_tuples[:, mask_idx] = "_mask_"

                x = (mask_tuples, np.asarray(mask_idxs), mask_attrs)

                if attr_embeds is None and tuple_embed is None:
                    tuple_embed, attr_embeds = self.model.get_encodings(x)
                    s = attr_embeds.size()
                    t = s[0]/self.env['num_neg']
                    d = int(s[1])
                    attr_embeds = attr_embeds.view(int(t), self.env['num_neg'], d)
                else:
                    tuple_embed_buf, attr_embeds_buf = self.model.get_encodings(x)
                    tuple_embed = torch.cat((tuple_embed, tuple_embed_buf), dim=0)
                    s = attr_embeds_buf.size()
                    t = s[0] / self.env['num_neg']
                    d = int(s[1])
                    attr_embeds_buf = attr_embeds_buf.view(int(t), self.env['num_neg'], d)
                    attr_embeds = torch.cat((attr_embeds, attr_embeds_buf), dim=0)

                mask = np.append(mask, mask_idxs, axis=0)
                # [mask_idx] * self.env['batch_size']

            if whole_tuples is None and whole_examples is None:
                whole_tuples = tuple_embed
                whole_examples = attr_embeds
            else:
                whole_tuples = torch.cat((whole_tuples, tuple_embed), dim=0)
                whole_examples = torch.cat((whole_examples, attr_embeds), dim=0)

        mask = torch.from_numpy(mask).int()

        return tuple_embed, whole_examples, mask