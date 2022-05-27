import torch
import numpy as np


class AttrEmbedding(torch.nn.Module):
    """
    The g encoders for each different attribute.
    """
    def __init__(self, attributes):
        super(AttrEmbedding, self).__init__()
        self.word_to_idx = {}
        self.idx_to_vec = {}
        # create the look-up table for the embeddings
        for idx, attr in attributes.items():
            self.word_to_idx[idx] = attr.vocab
            embeddings = torch.Tensor(np.array(attr.vec))
            embed = torch.nn.Embedding.from_pretrained(embeddings)
            self.idx_to_vec[idx] = embed

    def get_embeddings(self, inputs, attr=None, dtype=None):
        """
        :param inputs: a batch of tuples that contains the masked tuples or the pos/neg examples in that batch
        :param attr: the attribute that is masked
        :param dtype: the type of the data, tuple or examples
        :return: the representation of the batch
        """

        vec = []
        if dtype is 'tuple':
            idx = []
            for row in inputs:
                idx.append(self.word_to_idx[attr].loc[row, 'index'])
            new_list = list(map(int, idx))
            idx = torch.tensor(new_list)
            for i in idx:
                vec.append(self.idx_to_vec[attr](torch.tensor(i)))
            vec = torch.stack(vec)
        elif dtype is 'example':
            idx = []
            for row in inputs:
                for r in row:
                    idx.append(self.word_to_idx[attr].loc[r, 'index'])
            new_list = list(map(int, idx))
            idx = torch.tensor(new_list)
            for i in idx:
                vec.append(self.idx_to_vec[attr](torch.tensor(i)))
            vec = torch.stack(vec)
        return vec

    def get_embeddings_idx(self, inputs, attr=None, dtype=None):
        """
        :param inputs: a batch of tuples that contains the masked tuples or the pos/neg examples in that batch
        :param attr: the attribute that is masked
        :param dtype: the type of the data, tuple or examples
        :return: the representation of the batch
        """

        vec = []
        if dtype is 'tuple':
            idx = []
            for row in inputs:
                idx.append(self.word_to_idx[attr].loc[row, 'index'])
            new_list = list(map(int, idx))
            idx = torch.tensor(new_list)
            for i in idx:
                vec.append(self.idx_to_vec[attr](i.clone().detach()))
            vec = torch.stack(vec)
        elif dtype is 'example':
            idx = []
            for row in inputs:
                for r in row:
                    idx.append(self.word_to_idx[attr].loc[r, 'index'])
            new_list = list(map(int, idx))
            idx = torch.tensor(new_list)
            for i in idx:
                vec.append(self.idx_to_vec[attr](i.clone().detach()))
            vec = torch.stack(vec)
        return vec, idx
