import sys
import os
picket_path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(picket_path))
from globalvar import *
from preprocessor.embeddings import load_embedding as preprocessor_embedding
from collections import OrderedDict
from tqdm import tqdm
import pandas as pd
import numpy as np
import json, logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Dataset(object):

    def __init__(self, env, dirtydf=None):
        self.env = env
        self.config = self.env['dataset_config']
        # the data-frame that holds the data
        self.df = dirtydf
        # the functional dependencies
        self.fds = None
        self.attributes = OrderedDict({})
        self.attr_to_idx = {}
        self.numAttr = -1
        self.numTuple = -1

    def load_dataset(self):
        """
        Loads the data to a data-frame. Pre-processing of the data-frame,
        drop columns etc.. Create dictionary with name of attribute
        to the index of the attribute. Load the attributes
        """
        # setup header:
        if self.config['header'].lower() == 'none':
            self.config['header'] = None
        
        """
        # load preprocessor from file
        self.df = pd.read_csv(self.env['dataset_path'],
                              encoding='utf8',
                              header=self.config['header'],
                              sep=self.config['sep'],
                              na_values=self.config['na_values'])
        """
        #self.df = dirtydf
        # replace null values
        self.df = self.df.fillna('Nan')
        # pre-processing the dataset
        self.preprocess_df(self.config['dropna'], self.config['dropcol'])
        # numTuple: number of total instances/tuples in data-set
        # numAttr: number of attributes in the data-set
        self.numTuple, self.numAttr = self.df.shape[0], self.df.shape[1]
        # dictionary with name of attribute as index to number of attribute
        self.attr_to_idx = dict(zip(self.df.columns.values, list(range(self.numAttr))))
        # change column types based on the user input
        self.change_column_type()
        # load attributes to a dictionary: num of attribute to Attribute object
        self.load_attributes(self.config['dtypes'])

    def load_fds(self, path):
        logger.info("Load FDs...")
        self.fds = json.load(open(path))
        return self.fds

    def preprocess_df(self, dropna, dropcol):

        logger.info("Preprocessing Data...")

        # (optional) drop specified columns
        if dropcol is not None:
            self.df = self.df.drop(dropcol, axis=1)

        # (optional) drop rows with empty values
        if dropna:
            self.df.dropna(axis=0, how='any', inplace=True)

        # (optional) replace empty cells
        self.df = self.df.replace(np.nan, self.config['nan'], regex=True)

        # drop empty columns
        self.df.dropna(axis=1, how='all', inplace=True)

    def infer_column_type(self, c, data):
        if np.issubdtype(c, np.number):
            return NUMERIC
        if data.unique().shape[0] >= self.config['min_categories_for_text']:
            return TEXT
        return CATEGORICAL

    def change_column_type(self):
        for idx, attr in enumerate(self.df):
            if self.config['dtypes'][idx] == CATEGORICAL or self.config['dtypes'][idx] == TEXT:
                self.df[attr] = self.df[attr].astype(str)
                logger.info("change column type from {} to '{}'".format('Numeric', 'String'))
            elif self.config['dtypes'][idx] == NUMERIC:
                self.df[attr] = pd.to_numeric(self.df[attr], errors='coerce')
                logger.info("change column type from {} to '{}'".format('String', 'Numeric'))

    def load_attributes(self, dtypes):
        for idx, attr in enumerate(self.df):
            # infer column type
            # inferred_type = self.infer_column_type(self.df[attr].dtype, self.df[attr])
            # self.attributes[idx] = Attribute(self, idx, attr, inferred_type)
            self.attributes[idx] = Attribute(self, idx, attr, dtypes[idx])

    def load_embedding(self, wv=None):
        first = True
        df_all = None
        for attr in tqdm(self.attributes.values()):
            if attr.dtype == TEXT:
                if first:
                    df_all = self.df[attr.name]
                    first = False
                else:
                    df_all = pd.concat([df_all, self.df[attr.name]])

        for attr in tqdm(self.attributes.values()):
            attr.load_embedding(wv, df_all)


class Attribute(object):

        def __init__(self, ds, idx, name, dtype):
            self.ds = ds
            self.idx = idx
            self.name = name
            self.dtype = dtype
            # dimension of the embedding
            self.dim = 0
            # the unique cells of the specified attribute
            self.vocab = None
            # the embeddings of the unique cells
            self.vec = None

        def load_embedding(self, wv, df_all):

            # replace null values with a specific value based on the type of the data
            '''
            :param wv:
            :return:
            '''
            '''
            if self.dtype == TEXT:
                self.ds.df = self.ds.df.replace(np.nan, self.ds.env[self.dtype]['nan'], regex=True)
            elif self.dtype == NUMERIC:
                self.ds.df = self.ds.df.replace(np.nan, self.ds.env[self.dtype]['nan'], regex=True)
            elif self.dtype == CATEGORICAL:
                self.ds.df = self.ds.df.replace(np.nan, self.ds.env[self.dtype]['nan'], regex=True)
            '''
            # for the specific attribute take all the values and create vocab and lookup table with embeddings
            self.vec, self.vocab = preprocessor_embedding(self.idx, self.ds.env['embed_config'][self.dtype],
                                                  self.ds.df[self.name], self.dtype, wv, df_all)
            # set the number of the dimension of the embedding
            # print(self.vec.shape)
            self.dim = self.vec.shape[1]







