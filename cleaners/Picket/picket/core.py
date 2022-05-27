
import sys
import os
picket_path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(picket_path))
from preprocessor.dataset import Dataset
from helper import GlobalTimer
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DeepTable(object):

    def __init__(self, env={}, dirtydf = None):
        self.env = {}
        self.env.update(env)
        self.ds = Dataset(self.env, dirtydf)
        # start timer
        self.timer = GlobalTimer()
        self.timer.time_point("START")
        self.dirtydf = dirtydf

    def load_dataset(self):
        self.timer.time_start("Load Dataset")
        self.ds.load_dataset()
        self.timer.time_end("Load Dataset")

    def load_embedding(self, wv=None):
        """
        :param wv: (optional) pre-trained embedding model
        :return: none
        """
        self.timer.time_start("Load Attribute Embedding")
        self.ds.load_embedding(wv)
        self.timer.time_end("Load Attribute Embedding")

    def prepare_training_data(self):
        return self.ds.prepare_training_data()



