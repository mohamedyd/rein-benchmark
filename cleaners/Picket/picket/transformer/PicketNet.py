'''
h and g streams are aggregated and only positional encoding are provided at the beginning of each layer.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from .sparsemax import Sparsemax


global_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TwoStreamTransformer(nn.Module):
    """
    Two stream transformer
    """
    def __init__(self, encoder, generator, preprocessor, shape):
        super(TwoStreamTransformer, self).__init__()
        self.encoder = encoder
        self.generator = generator
        self.preprocessor = preprocessor
        layer_num = self.encoder.N
        self.W = []
        for _ in range(layer_num):
            w_init = torch.nn.Parameter(torch.randn(shape))
            w_init.requires_grad = True
            self.W.append(w_init)
        self.W = torch.nn.ParameterList(self.W)

    def forward(self, src):
        "Take in and process masked src sequences."
        src = src.to(global_device)
        G = self.preprocessor(src)
        return self.encode(G, self.W)

    def encode(self, src, w):
        return self.encoder(src, w)

class Preprocessor(nn.Module):
    "The G functions"
    def __init__(self, attribute_num, d_input, d_model, d_hidden, useEncoding=False, categorical_ids=[]):
        super(Preprocessor, self).__init__()
        self.attribute_num = attribute_num
        self.useEncoding = useEncoding
        print('Use Encoding: %d' % useEncoding)
        self.categorical_ids = categorical_ids

        self.categorical_encode = []

        if useEncoding:
            for i in range(attribute_num):
                self.categorical_encode.append(nn.Linear(d_input, d_input))
        self.categorical_encode = torch.nn.ModuleList(self.categorical_encode)

    def forward(self, x):
        if self.useEncoding:
            x_hat = []
            for fid in range(self.attribute_num):
                if fid in self.categorical_ids:
                    x_hat_tmp = self.categorical_encode[fid](x[:, fid, :])
                else:
                    x_hat_tmp = x[:, fid, :]

                x_hat.append(x_hat_tmp)
            x_hat = torch.stack(x_hat, dim=1)
        else:
            x_hat = x
        return x_hat


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class Generator(nn.Module):
    "Different architectures for different attribute types"
    def __init__(self, d_model, d_hidden,preprocessor, numerical_ids):
        super(Generator, self).__init__()
        self.preprocessor = preprocessor
        self.numerical_ids = numerical_ids
        layer1 = nn.Linear(d_model, d_hidden)
        layer2 = nn.Linear(d_hidden, 1)
        self.layers1 = clones(layer1, len(numerical_ids))
        self.layers2 = clones(layer2, len(numerical_ids))

    def forward(self, x, posNeg, mask_id):
        posNeg = posNeg.to(global_device)
        sample_num = posNeg.shape[1]

        inner_product = torch.matmul(posNeg, x[:, mask_id, :].unsqueeze(-1)).squeeze(-1)
        score = inner_product.to(global_device)

        recovered_value = torch.zeros(x.shape[0]).double().to(global_device)

        for layer_id, num_id in enumerate(self.numerical_ids):
            if num_id == mask_id:
                hidden = F.relu(self.layers1[layer_id](x[:, num_id, :]))
                recovered_value_tmp = self.layers2[layer_id](hidden).squeeze()
                recovered_value = recovered_value_tmp

                fake_inner_product = torch.ones(sample_num).double()*(-1e9)
                fake_inner_product[0] = 0
                score = fake_inner_product.repeat([x.shape[0], 1])
        return score, recovered_value

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# To be modified
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N, mode='both'):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.mode = mode

    def forward(self, x_h_init, x_g_init):
        "Pass the input (and mask) through each layer in turn."
        batch_size = x_h_init.shape[0]
        layer_id = 0

        for layer in self.layers:
            if layer_id == 0:
                x_g_init_use = x_g_init[layer_id].repeat([batch_size, 1, 1]).double().to(global_device)
                x_h, x_g = layer(x_h_init, x_g_init_use)
                if self.mode == 'onlyh':
                    x_sum = x_h
                if self.mode == 'onlyg':
                    x_sum = x_g
                if self.mode == 'both': 
                    x_sum = x_h+x_g
            else:
                x_g_init_use = x_g_init[layer_id].repeat([batch_size, 1, 1]).double().to(global_device)
                x_h, x_g = layer(x_sum, x_g_init_use)
                if self.mode == 'onlyh':
                    x_sum = x_h
                if self.mode == 'onlyg':
                    x_sum = x_g
                if self.mode == 'both': 
                    x_sum = x_h+x_g
            layer_id += 1

        return self.norm(x_sum)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayerDiff(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, attr_num, self_attn_h, self_attn_g, feed_forward_h, feed_forward_g, dropout):
        super(EncoderLayerDiff, self).__init__()
        self.attr_num = attr_num
        self.self_attn_h = self_attn_h
        self.self_attn_g = self_attn_g
        self.feed_forward_h = clones(feed_forward_h, attr_num)
        self.feed_forward_g = clones(feed_forward_g, attr_num)
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.sublayer_g = clones(SublayerConnection(size, dropout), attr_num)
        self.sublayer_h = clones(SublayerConnection(size, dropout), attr_num)
        self.norm = LayerNorm(size)
        self.size = size

    def forward(self, x_h, x_g):
        "Two stream connection"
        x_h_new = self.sublayer[0](x_h, lambda x: self.self_attn_h(x, x, x, False))
        x_g_new = self.sublayer[1](x_g, lambda x: self.self_attn_g(x, self.norm(x_h), self.norm(x_h), False))

        g_group = []
        h_group = []

        for attr_id in range(self.attr_num):
            h = self.sublayer_h[attr_id](x_h_new[:, attr_id, :], self.feed_forward_h[attr_id])
            h_group.append(h)
            g = self.sublayer_g[attr_id](x_g_new[:, attr_id, :], self.feed_forward_g[attr_id])
            g_group.append(g)

        return torch.stack(h_group, dim=1), torch.stack(g_group, dim=1)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn_h, self_attn_g, feed_forward_h, feed_forward_g, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn_h = self_attn_h
        self.self_attn_g = self_attn_g
        self.feed_forward_h = feed_forward_h
        self.feed_forward_g = feed_forward_g
        self.sublayer = clones(SublayerConnection(size, dropout), 4)
        self.norm = LayerNorm(size)
        self.size = size

    def forward(self, x_h, x_g):
        "Two stream connection"
        x_h_new = self.sublayer[0](x_h, lambda x: self.self_attn_h(x, x, x, False))
        x_g_new = self.sublayer[1](x_g, lambda x: self.self_attn_g(x, self.norm(x_h), self.norm(x_h), False))

        return self.sublayer[2](x_h_new, self.feed_forward_h), self.sublayer[3](x_g_new, self.feed_forward_g)

def attention(query, key, value, mask=False, dropout=None, hard_mask=None, sparse=False):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k)

    if mask:
        diagonalMask = torch.Tensor(np.eye(scores.shape[-1])).to(global_device)
        scores = scores.masked_fill(diagonalMask == 1, -1e9)

    if hard_mask is not None:
        hard_mask = torch.Tensor(hard_mask).to(global_device)
        scores = scores.masked_fill(hard_mask == 0, -1e9)

    if sparse:
        sparsemax = Sparsemax()
        p_attn = sparsemax(scores)
    else:
        p_attn = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, hard_mask=None, sparse=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.hard_mask = hard_mask
        self.sparse = sparse

    def forward(self, query, key, value, mask=False):
        "Implements Figure 2"
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                dropout=self.dropout, hard_mask=self.hard_mask, sparse=self.sparse)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class MultiHeadedAttentionDiff(nn.Module):
    def __init__(self, h, d_model, attr_num, dropout=0.1, hard_mask=None, sparse=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionDiff, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.attr_num = attr_num
        self.linears = clones(nn.Linear(d_model, d_model), 4*attr_num)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.hard_mask = hard_mask
        self.sparse = sparse

    def forward(self, query, key, value, mask=False):
        "Implements Figure 2"
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query_group = []
        key_group = []
        value_group = []
        for attr_id in range(self.attr_num):
            query_tmp, key_tmp, value_tmp = \
                [l(x).view(nbatches, self.h, self.d_k)
                 for l, x in zip((self.linears[attr_id*4], self.linears[attr_id*4+1], self.linears[attr_id*4+2]), 
                    (query[:, attr_id, :], key[:, attr_id, :], value[:, attr_id, :]))]
            query_group.append(query_tmp)
            key_group.append(key_tmp)
            value_group.append(value_tmp)
        query = torch.stack(query_group, dim=2)
        key = torch.stack(key_group, dim=2)
        value = torch.stack(value_group, dim=2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                dropout=self.dropout, hard_mask=self.hard_mask, sparse=self.sparse)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        res_group = []
        for attr_id in range(self.attr_num):
            res = self.linears[attr_id*4+3](x[:, attr_id, :])
            res_group.append(res)
        return torch.stack(res_group, dim=1)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

def make_model(N=6, attribute_num=10, d_input=512, d_model=512, d_ff=2048,
    d_preprocessor=256, d_generator=256, h=8, dropout=0.1, numerical_ids=[], 
    hard_mask=None, sparse=False, useEncoding=False, categorical_ids=[], ghmode='both',
    sameTransform=True, sameFF=True):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    if sameTransform:
        attn = MultiHeadedAttention(h=h, d_model=d_model, hard_mask=hard_mask, sparse=sparse)
    else:
        attn = MultiHeadedAttentionDiff(h=h, d_model=d_model, attr_num=attribute_num, hard_mask=hard_mask, sparse=sparse)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    pp = Preprocessor(attribute_num, d_input, d_model, d_preprocessor,
                    useEncoding=useEncoding, categorical_ids=categorical_ids)

    if sameFF:
        model = TwoStreamTransformer(
           Encoder(EncoderLayer(d_model, c(attn), c(attn), c(ff), c(ff), dropout), N, ghmode),
           Generator(d_model, d_generator, pp, numerical_ids),
           pp,
           (attribute_num, d_model))
    else:
        model = TwoStreamTransformer(
           Encoder(EncoderLayerDiff(d_model, attribute_num, c(attn), c(attn), c(ff), c(ff), dropout), N, ghmode),
           Generator(d_model, d_generator, pp, numerical_ids),
           pp,
           (attribute_num, d_model))       

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, opt=None, soft_mask=None, alpha=0, beta=0):
        self.generator = generator
        self.CrossEntropy = torch.nn.CrossEntropyLoss()
        self.MSE = torch.nn.MSELoss()
        self.CrossEntropyNoReduction = torch.nn.CrossEntropyLoss(reduction='none')
        self.MSENoReduction = torch.nn.MSELoss(reduction='none')
        self.opt = opt
        self.soft_mask = soft_mask
        self.alpha = alpha
        self.beta = beta

        if self.soft_mask is not None:
            self.soft_mask = torch.Tensor(self.soft_mask).to(global_device).double()

    def __call__(self, x, samples, y_class, y_value, layers, attribute_id_mask, bp=True):
        y_class = y_class.to(global_device)
        y_value = y_value.to(global_device)
        out_score, out_value = self.generator(x, samples, attribute_id_mask)
        out_score = out_score.to(global_device)
        out_value = out_value.to(global_device)

        loss1 = self.CrossEntropy(out_score.view(-1, out_score.shape[-1]), y_class.view(-1))
        loss2 = self.MSE(out_value, y_value)
        loss1_no_reduction = self.CrossEntropyNoReduction(out_score.view(-1, out_score.shape[-1]), y_class.view(-1))
        loss2_no_reduction = self.MSENoReduction(out_value, y_value)
        loss1_no_reduction = loss1_no_reduction.view(-1, 1)
        loss3 = 0
        loss4 = 0
        if self.soft_mask is not None:
            for layer in layers:
                attn_h = layer.self_attn_h.attn
                attn_g = layer.self_attn_g.attn
                attn_h = attn_h.view(-1, attn_h.shape[-2], attn_h.shape[-1])
                attn_g = attn_g.view(-1, attn_g.shape[-2], attn_g.shape[-1])
                tmp_h = torch.mul(torch.mean(attn_h**2, dim=0), 1-self.soft_mask)
                tmp_g = torch.mul(torch.mean(attn_g**2, dim=0), 1-self.soft_mask)

                loss3 += torch.mean(tmp_h.view(-1))
                loss4 += torch.mean(tmp_g.view(-1))

        loss = loss1 + loss2 + self.alpha*loss3 + self.beta*loss4

        if bp:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
        return loss1.data.item(), loss2.data.item(), loss1_no_reduction, loss2_no_reduction

    def zero_grad(self):
        if self.opt is not None:
            self.opt.optimizer.zero_grad()

def run_epoch(data_iter, model, loss_compute, device):
    "Standard Training and Logging Function"
    total_loss1 = 0
    total_loss2 = 0
    num_of_batch = 0
    start = time.time()
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src.to(device))
        loss1, loss2, _, _ = loss_compute(out, batch.samples, batch.target_class, batch.target_value, model.encoder.layers, batch.attribute_id_mask)
        loss_compute.zero_grad()
        total_loss1 += loss1
        total_loss2 += loss2
        num_of_batch += 1
    elapsed = time.time() - start
    print("CrossEntropy Loss: %f MSE Loss: %f Time elapsed: %f s" %
        (total_loss1/num_of_batch, total_loss2/num_of_batch, elapsed))
    return total_loss1/num_of_batch, total_loss2/num_of_batch

def run_epoch_with_loss(data_iter, model, loss_compute, device, bp=True):
    "Standard Training and Logging Function"
    total_loss1 = 0
    total_loss2 = 0
    loss1_no_reduction_all = None
    loss2_no_reduction_all = None
    loss1_tmp = None
    loss2_tmp = None
    num_of_batch = 0
    start = time.time()
    attr_list = []
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src.to(device))
        loss1, loss2, loss1_no_reduction, loss2_no_reduction = loss_compute(out, batch.samples, batch.target_class, batch.target_value, model.encoder.layers,
            batch.attribute_id_mask, bp=bp)

        loss1_no_reduction = loss1_no_reduction.view(-1, 1)
        loss2_no_reduction = loss2_no_reduction.view(-1, 1)

        #print(attr_list)
        #print(batch.attribute_id_mask)

        if batch.attribute_id_mask in attr_list:

            if loss1_no_reduction_all is None:
                loss1_no_reduction_all = loss1_tmp
                loss2_no_reduction_all = loss2_tmp
            else:
                loss1_no_reduction_all = torch.cat((loss1_no_reduction_all, loss1_tmp), dim=0)
                loss2_no_reduction_all = torch.cat((loss2_no_reduction_all, loss2_tmp), dim=0)

            loss1_tmp = None
            loss2_tmp = None
            attr_list = []

        if loss1_tmp is None:
            loss1_tmp = loss1_no_reduction
            loss2_tmp = loss2_no_reduction
        else:
            loss1_tmp = torch.cat((loss1_tmp, loss1_no_reduction), dim=1)
            loss2_tmp = torch.cat((loss2_tmp, loss2_no_reduction), dim=1)
        attr_list.append(batch.attribute_id_mask)

        loss_compute.zero_grad()
        total_loss1 += loss1
        total_loss2 += loss2
        num_of_batch += 1

    if loss1_no_reduction_all is None:
        loss1_no_reduction_all = loss1_tmp
        loss2_no_reduction_all = loss2_tmp
    else:
        loss1_no_reduction_all = torch.cat((loss1_no_reduction_all, loss1_tmp), dim=0)
        loss2_no_reduction_all = torch.cat((loss2_no_reduction_all, loss2_tmp), dim=0)

    elapsed = time.time() - start
    print("CrossEntropy Loss: %f MSE Loss: %f Time elapsed: %f s" %
        (total_loss1/num_of_batch, total_loss2/num_of_batch, elapsed))

    return loss1_no_reduction_all, loss2_no_reduction_all

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, target_class, samples, target_value, attribute_id_mask):
        self.src = src
        self.target_class = target_class
        self.target_value = target_value
        self.samples = samples
        self.attribute_id_mask = attribute_id_mask

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

class PicketNetModel:
    def __init__(self, input_param):
        self.param = {
            'description': 'PicketNet Model',
            'model_dim': 256,
            'preprocessor_dim': 256,
            'generator_dim': 256,
            'input_dim': 10,
            'attribute_num': 5,
            'transformer_layer': 2,
            'head_num': 8,
            'hidden_dim': 2048,
            'dropout': 0.1,
            'numerical_ids': [],
            'batch_size': 30,
            'epochs': 200,
            'opt_factor': 0.5,
            'warmup': 300,
            'adam_lr': 0,
            'adam_betas': (0.9, 0.98),
            'adam_eps': 1e-9,
            'neg_sample_num': 4,
            'random_mask': True,
            'structure_mask_type': None, # could be hard, soft, sample
            'structure_mask': None,
            'structure_alpha': 0.01,
            'structure_beta': 0.01,
            'loss_warm_up_epochs': 10,
            'loss_trim_epochs': 20,
            'loss_trim_p': 0.2,
            'fast': True,
            'sparse': False,
            'num_of_std': 3,
            'mask_ratio': 0.15,
            'holdout_ratio': 0.2,
            'useEncoding': False,
            'categorical_ids': [],
            'ghmode': 'both',
            'sameFF': True,
            'sameTransform': True,
        }

        self.param.update(input_param)

        self.description = self.param['description']
        self.model_dim = self.param['model_dim']
        self.preprocessor_dim = self.param['preprocessor_dim']
        self.generator_dim = self.param['generator_dim']
        self.input_dim = self.param['input_dim']
        self.attribute_num = self.param['attribute_num']
        self.N = self.param['transformer_layer']
        self.h = self.param['head_num']
        self.hidden_dim = self.param['hidden_dim']
        self.dropout = self.param['dropout']
        self.numerical_ids = self.param['numerical_ids']
        self.batch_size = self.param['batch_size']
        self.epochs = self.param['epochs']
        self.opt_factor = self.param['opt_factor']
        self.warmup = self.param['warmup']
        self.adam_lr = self.param['adam_lr']
        self.adam_betas = self.param['adam_betas']
        self.adam_eps = self.param['adam_eps']
        self.neg_sample_num = self.param['neg_sample_num']
        self.random_mask = self.param['random_mask']
        self.structure_mask_type = self.param['structure_mask_type']
        self.structure_mask = self.param['structure_mask']
        self.structure_alpha = self.param['structure_alpha']
        self.structure_beta = self.param['structure_beta']
        self.loss_warm_up_epochs = self.param['loss_warm_up_epochs']
        self.loss_trim_epochs = self.param['loss_trim_epochs']
        self.loss_trim_p = self.param['loss_trim_p']
        self.fast = self.param['fast']
        self.sparse = self.param['sparse']
        self.num_of_std = self.param['num_of_std']
        self.mask_ratio = self.param['mask_ratio']
        self.holdout_ratio = self.param['holdout_ratio']
        self.useEncoding = self.param['useEncoding']
        self.categorical_ids = self.param['categorical_ids']
        self.ghmode = self.param['ghmode']
        self.sameFF =self.param['sameFF']
        self.sameTransform = self.param['sameTransform']


        self.current_epoch = 0

        hard_mask = None
        if self.structure_mask_type == 'hard':
            hard_mask = self.structure_mask

        self.model = make_model(
            N=self.N,
            attribute_num=self.attribute_num,
            d_input=self.input_dim,
            d_model=self.model_dim,
            d_ff=self.hidden_dim,
            d_preprocessor=self.preprocessor_dim,
            d_generator=self.generator_dim,
            h=self.h,
            dropout=self.dropout,
            numerical_ids=self.numerical_ids,
            hard_mask=hard_mask,
            sparse = self.sparse,
            useEncoding = self.useEncoding,
            categorical_ids = self.categorical_ids,
            ghmode = self.ghmode,
            sameFF = self.sameFF,
            sameTransform = self.sameTransform,
        )

        self.model_opt = NoamOpt(
            model_size=self.model_dim,
            factor=self.opt_factor,
            warmup=self.warmup,
            optimizer=torch.optim.Adam(self.model.parameters(), lr=self.adam_lr, betas=self.adam_betas, eps=self.adam_eps)
        )

        self.model.double()
        self.device = global_device
        self.model = self.model.to(self.device)


    def loadData(self, tuples, neg_samples, attribute_info=None, pos_mask=None, tuple_idx=None, iidx=[]):
        self.table = tuples
        self.neg_samples = neg_samples
        self.attribute_info = attribute_info
        self.pos_mask = pos_mask
        self.tuple_idx = tuple_idx
        self.iidx = iidx

    def getBatchFast(self, shuffle = True):
        tuple_num = self.table.shape[0]

        bs = self.batch_size

        if shuffle:
            indices = torch.randperm(tuple_num)
        else:
            indices = torch.arange(tuple_num)

        for batch_id in range(tuple_num//bs+1):
            if batch_id == tuple_num//bs:
                if tuple_num % bs == 0:
                    break;
                else:
                    batch_indices = indices[batch_id*bs:]
                    batch_data = self.table[batch_indices, :, :]
            else:
                batch_indices = indices[batch_id*bs:(batch_id+1)*bs]
                batch_data = self.table[batch_indices, :, :]

            batch = batch_data.shape[0]

            src = Variable(batch_data, requires_grad=False)

            # Generate positive and negative samples
            label = torch.from_numpy(np.random.randint(self.neg_sample_num+1, size=(batch, self.attribute_num)))
            tgt_class = Variable(label, requires_grad=False)
            samples = torch.zeros(batch, self.attribute_num, self.neg_sample_num+1, self.input_dim).double()
            tgt_value = torch.zeros(batch, self.attribute_num).double()

            if self.structure_mask_type == 'sample' and self.pos_mask is not None:
                print('Structure Sample')
                for attribute_id in range(self.attribute_num):
                    pos_one_hot = self.pos_mask[batch_indices, :, attribute_id]
                    neg_one_hot = 1-pos_one_hot
                    for i, pos_id in enumerate(label[:, attribute_id]):
                        rand_id = np.random.choice(np.nonzero(pos_one_hot[i, :])[0], 1)
                        samples[i, attribute_id, pos_id, :] = self.table[rand_id, attribute_id, :]
                        for sample_id in range(self.neg_sample_num+1):
                            if sample_id != pos_id:
                                rand_id = np.random.choice(np.nonzero(neg_one_hot[i, :])[0], 1)
                                samples[i, attribute_id, sample_id, :] = self.table[rand_id, attribute_id, :]

                    # Make CrossEntropyLoss 0 for numerical attributes
                    # and MSELoss 0 for non-numerical attributes
                    if attribute_id in self.numerical_ids:
                        tgt_value[:, attribute_id] = batch_data[:, attribute_id, 0]
                        tgt_class[:, attribute_id].fill_(0)

            else:
                if self.attribute_info is None:
                    print("Please provide attribute info")
                if self.tuple_idx is not None:
                    tuple_idx_batch = self.tuple_idx[batch_indices, :]
                    have_idx = True
                else:
                    have_idx = False

                for attribute_id in range(self.attribute_num):
                    vecs = self.attribute_info[attribute_id].vec
                    vec_size = vecs.shape[0]
                    vec_idx = np.random.randint(vec_size, size=batch*(self.neg_sample_num+1))

                    if have_idx:
                        tuple_idx_tmp = np.repeat(tuple_idx_batch[:, attribute_id].numpy(), self.neg_sample_num+1)
                        vec_idx[vec_idx==tuple_idx_tmp] = -1

                    rand_samples = vecs[vec_idx, :].reshape((batch, self.neg_sample_num+1, -1))
                    samples[:, attribute_id, :, :] = torch.Tensor(rand_samples)

                    pos_idx = label[:, attribute_id].numpy()
                    mask_of_pos = np.zeros((batch, self.neg_sample_num+1, self.input_dim))
                    mask_of_pos[np.arange(batch), pos_idx] = 1

                    samples[:, attribute_id, :, :][torch.Tensor(mask_of_pos)==1] = batch_data[:, attribute_id, :].contiguous().view(-1,)

                    # Make CrossEntropyLoss 0 for numerical attributes
                    # and MSELoss 0 for non-numerical attributes
                    if attribute_id in self.numerical_ids:
                        tgt_value[:, attribute_id] = batch_data[:, attribute_id, 0]
                        tgt_class[:, attribute_id].fill_(0)

            if self.random_mask:
                random_mask = torch.rand(src.shape) < self.mask_ratio
                src.masked_fill_(random_mask, 0)

            for attribute_id_mask in range(self.attribute_num):
                if attribute_id_mask not in self.iidx:
                    masked_src = src.clone()
                    masked_src[:, attribute_id_mask, :].fill_(0)
                    yield Batch(masked_src, tgt_class[:, attribute_id_mask], samples[:, attribute_id_mask, :, :], tgt_value[:, attribute_id_mask], attribute_id_mask)

    def getBatchTest(self, table, tuple_idx = None, shuffle = False):
        tuple_num = table.shape[0]
        bs = self.batch_size

        if shuffle:
            indices = torch.randperm(tuple_num)
        else:
            indices = torch.arange(tuple_num)

        for batch_id in range(tuple_num//bs+1):
            if batch_id == tuple_num//bs:
                if tuple_num % bs == 0:
                    break;
                else:
                    batch_indices = indices[batch_id*bs:]
                    batch_data = table[batch_indices, :, :]
            else:
                batch_indices = indices[batch_id*bs:(batch_id+1)*bs]
                batch_data = table[batch_indices, :, :]

            batch = batch_data.shape[0]

            src = Variable(batch_data, requires_grad=False)

            # Generate positive and negative samples
            label = torch.from_numpy(np.random.randint(self.neg_sample_num+1, size=(batch, self.attribute_num)))
            tgt_class = Variable(label, requires_grad=False)
            samples = torch.zeros(batch, self.attribute_num, self.neg_sample_num+1, self.input_dim).double()
            tgt_value = torch.zeros(batch, self.attribute_num).double()

            if self.attribute_info is None:
                print("Please provide attribute info")
            if tuple_idx is not None:
                tuple_idx_batch = tuple_idx[batch_indices, :]
                have_idx = True
            else:
                have_idx = False

            for attribute_id in range(self.attribute_num):
                vecs = self.attribute_info[attribute_id].vec
                vec_size = vecs.shape[0]
                vec_idx = np.random.randint(vec_size, size=batch*(self.neg_sample_num+1))

                if have_idx:
                    tuple_idx_tmp = np.repeat(tuple_idx_batch[:, attribute_id].numpy(), self.neg_sample_num+1)
                    vec_idx[vec_idx==tuple_idx_tmp] = -1

                rand_samples = vecs[vec_idx, :].reshape((batch, self.neg_sample_num+1, -1))
                samples[:, attribute_id, :, :] = torch.Tensor(rand_samples)

                pos_idx = label[:, attribute_id].numpy()
                mask_of_pos = np.zeros((batch, self.neg_sample_num+1, self.input_dim))
                mask_of_pos[np.arange(batch), pos_idx] = 1

                samples[:, attribute_id, :, :][torch.Tensor(mask_of_pos)==1] = batch_data[:, attribute_id, :].contiguous().view(-1,)

                # Make CrossEntropyLoss 0 for numerical attributes
                # and MSELoss 0 for non-numerical attributes
                if attribute_id in self.numerical_ids:
                    tgt_value[:, attribute_id] = batch_data[:, attribute_id, 0]
                    tgt_class[:, attribute_id].fill_(0)

            for attribute_id_mask in range(self.attribute_num):
                if attribute_id_mask not in self.iidx:
                    masked_src = src.clone()
                    masked_src[:, attribute_id_mask, :].fill_(0)
                    yield Batch(masked_src, tgt_class[:, attribute_id_mask], samples[:, attribute_id_mask, :, :], tgt_value[:, attribute_id_mask], attribute_id_mask)


    def train(self):
        start_time = time.time()
        soft_mask = None
        if self.structure_mask_type == 'soft':
            soft_mask = self.structure_mask
        for epoch in range(self.epochs):
            print("Epoch: %d" % (self.current_epoch+epoch))
            self.model.train()
            if self.fast:
                batchLoader = self.getBatchFast()
            else:
                batchLoader = self.getBatch()
            CELoss, MSELoss = run_epoch(
                batchLoader,
                self.model,
                SimpleLossCompute(self.model.generator, self.model_opt, soft_mask, self.structure_alpha, self.structure_beta),
                self.device
            )
        self.current_epoch += self.epochs
        end_time = time.time()
        print("Total training time: %f s" % (end_time-start_time))
        if np.isnan(CELoss) or np.isnan(MSELoss):
            return 1
        else:
            return 0

    def loss_based_train(self, flip=False):
        start_time = time.time()
        soft_mask = None
        if self.structure_mask_type == 'soft':
            soft_mask = self.structure_mask

        print('----- Start warmup training... -----')
        loss1_warmup, loss2_warmup = self.getLossSeq(self.loss_warm_up_epochs)

        print('----- Start loss trim training... -----')
        loss1, loss2 = self.getAverageLoss(self.loss_trim_epochs)

        loss1_mean = torch.mean(loss1, dim=0, keepdim=True)
        loss2_mean = torch.mean(loss2, dim=0, keepdim=True)

        loss1_median = torch.median(loss1, dim=0, keepdim=True)[0]
        loss2_median = torch.median(loss2, dim=0, keepdim=True)[0]

        loss1 = loss1 / (loss1_median + 0.00000001)
        loss2 = loss2 / (loss2_median + 0.00000001)

        loss = loss1 + loss2
        self.outlierScoreCell = loss.cpu().detach().numpy()
        loss = torch.sum(loss, dim=1).cpu().detach()
        self.outlierScore = loss.numpy()

        if flip:
            indices_to_remove = torch.topk(-loss, int(self.loss_trim_p*self.table.shape[0]))[1].numpy()
        else:
            indices_to_remove = torch.topk(loss, int(self.loss_trim_p*self.table.shape[0]))[1].numpy()
        indices_left = np.delete(np.arange(self.table.shape[0]), indices_to_remove)
        self.indices_left = indices_left
        self.indices_to_remove = indices_to_remove
        '''
        random_idx = np.random.permutation(indices_left.shape[0])
        holdout_size = int(indices_left.shape[0]*self.holdout_ratio)
        self.indices_holdout = indices_left[random_idx[:holdout_size]]
        indices_left = indices_left[random_idx[holdout_size:]]
        '''
        self.table = self.table[indices_left, :, :]

        print('----- Start normal training... -----')
        for epoch in range(self.epochs):
            print("Epoch: %d" % (self.current_epoch+epoch))
            self.model.train()

            if self.fast:
                batchLoader = self.getBatchFast()
            else:
                batchLoader = self.getBatch()

            run_epoch(
                batchLoader,
                self.model,
                SimpleLossCompute(self.model.generator, self.model_opt, soft_mask, self.structure_alpha, self.structure_beta),
                self.device
            )
        self.current_epoch += self.epochs
        end_time = time.time()
        print("Total training time: %f s" % (end_time-start_time))

        return loss1_warmup.cpu().numpy(), loss2_warmup.cpu().numpy()


    def getAverageLoss(self, epoch_num = 1):
        loss1_all = None
        loss2_all = None

        if self.structure_mask_type == 'soft':
            soft_mask = self.structure_mask
        else:
            soft_mask = None

        for epoch in range(epoch_num):
            print("Epoch: %d" % (self.current_epoch+epoch))
            self.model.train()
            if self.fast:
                batchLoader = self.getBatchFast(shuffle=False)
            else:
                batchLoader = self.getBatch(shuffle=False)
            loss1, loss2 = run_epoch_with_loss(
                batchLoader,
                self.model,
                SimpleLossCompute(self.model.generator, self.model_opt, soft_mask, self.structure_alpha, self.structure_beta),
                self.device
            )
            if loss1_all is None:
                loss1_all = loss1.detach()
                loss2_all = loss2.detach()
            else:
                loss1_all += loss1.detach()
                loss2_all += loss2.detach()
        self.current_epoch += epoch_num
        return loss1_all/epoch_num, loss2_all/epoch_num

    def getLossSeq(self, epoch_num = 1):
        loss1_all = None
        loss2_all = None

        if self.structure_mask_type == 'soft':
            soft_mask = self.structure_mask
        else:
            soft_mask = None

        for epoch in range(epoch_num):
            print("Epoch: %d" % (self.current_epoch+epoch))
            self.model.train()
            if self.fast:
                batchLoader = self.getBatchFast(shuffle=False)
            else:
                batchLoader = self.getBatch(shuffle=False)
            loss1, loss2 = run_epoch_with_loss(
                batchLoader,
                self.model,
                SimpleLossCompute(self.model.generator, self.model_opt, soft_mask, self.structure_alpha, self.structure_beta),
                self.device
            )
            if loss1_all is None:
                loss1_all = loss1.detach().unsqueeze(0)
                loss2_all = loss2.detach().unsqueeze(0)
            else:
                loss1_all = torch.cat((loss1_all, loss1.detach().unsqueeze(0)), dim = 0)
                loss2_all = torch.cat((loss2_all, loss2.detach().unsqueeze(0)), dim = 0)
        self.current_epoch += epoch_num
        return loss1_all, loss2_all

    def getLossTest(self, table, tuple_idx):
        loss1_all = None
        loss2_all = None

        if self.structure_mask_type == 'soft':
            soft_mask = self.structure_mask
        else:
            soft_mask = None

        self.model.eval()        
        batchLoader = self.getBatchTest(table, tuple_idx, shuffle = False)
        
        with torch.no_grad():
            loss1, loss2 = run_epoch_with_loss(
                batchLoader,
                self.model,
                SimpleLossCompute(self.model.generator, self.model_opt, soft_mask, self.structure_alpha, self.structure_beta),
                self.device,
                bp = False
            )

        if loss1_all is None:
            loss1_all = loss1.detach().cpu().data.numpy()
            loss2_all = loss2.detach().cpu().data.numpy()
        else:
            loss1_all = np.concatenate((loss1_all, loss1.detach().cpu().data.numpy()), axis = 0)
            loss2_all = np.concatenate((loss2_all, loss2.detach().cpu().data.numpy()), axis = 0)

        return loss1_all, loss2_all

    def getTupleEmbedding(self, input, batch_size=1000, dropout = True):
        self.model.eval()
        if dropout:
            enable_dropout(self.model)
        res_all = None
        total_size = input.shape[0]
        for i in range(total_size//batch_size):
            input_to_be_used = input[i*batch_size:(i+1)*batch_size,:,:]
            res = self.get_tuple_embedding_(input_to_be_used).cpu().data.numpy()
            if res_all is None:
                res_all = res
            else:
                res_all = np.concatenate((res_all, res), axis=0)

        if total_size % batch_size != 0:
            input_to_be_used = input[(total_size//batch_size)*batch_size:,:,:]
            res = self.get_tuple_embedding_(input_to_be_used).cpu().data.numpy()
            if res_all is None:
                res_all = res
            else:
                res_all = np.concatenate((res_all, res), axis=0)

        return res_all

    def get_tuple_embedding_(self, input_to_be_used):
        input_device = input_to_be_used.to(global_device)
        res = self.model.forward(input_device)
        return res

    def getRecoveredNumeric(self, input, batch_size=1000, dropout = True):
        self.model.eval()
        if dropout:
            enable_dropout(self.model)

        recovered_value = np.zeros((input.shape[0], input.shape[1]))

        total_size = input.shape[0]
        for i in range(total_size//batch_size):
            input_to_be_used = input[i*batch_size:(i+1)*batch_size,:,:]
            res = self.get_tuple_embedding_(input_to_be_used)
            for layer_id, num_id in enumerate(self.numerical_ids):
                hidden = F.relu(self.model.generator.layers1[layer_id](res[:, num_id, :]))
                recovered_value_tmp = self.model.generator.layers2[layer_id](hidden).squeeze()
                recovered_value[i*batch_size:(i+1)*batch_size, num_id] = recovered_value_tmp.cpu().data.numpy()

        if total_size % batch_size != 0:
            input_to_be_used = input[(total_size//batch_size)*batch_size:,:,:]
            res = self.get_tuple_embedding_(input_to_be_used)
            for layer_id, num_id in enumerate(self.numerical_ids):
                hidden = F.relu(self.model.generator.layers1[layer_id](res[:, num_id, :]))
                recovered_value_tmp = self.model.generator.layers2[layer_id](hidden).squeeze()
                recovered_value[(total_size//batch_size)*batch_size:, num_id] = recovered_value_tmp.cpu().data.numpy()

        return recovered_value

    def gFunction(self, input):
        input = input.to(global_device)
        self.model.eval()
        return self.model.preprocessor(input)

    def gFunctionSingle(self, input, attribute_id):
        self.model.eval()
        input = input.view(-1)
        fake_input = torch.zeros(1, self.attribute_num, self.input_dim).double()
        fake_input[:, attribute_id, :] = input
        fake_input = fake_input.to(global_device)
        return self.model.preprocessor(fake_input)[:, attribute_id, :].view(-1)

    def saveModel(self, path):
        torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.model_opt.optimizer.state_dict(),
                'step': self.model_opt._step
            }, path)

    def loadModel(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.model_opt._step = checkpoint['step']

    def getAttentionMatrix(self):
        res = {}
        layer_id = 0
        for layer in self.model.encoder.layers:
            res['attention_h'+str(layer_id)] = torch.mean(layer.self_attn_h.attn, dim=0).detach().cpu().data.numpy()
            res['attention_g'+str(layer_id)] = torch.mean(layer.self_attn_g.attn, dim=0).detach().cpu().data.numpy()
            layer_id += 1

        return res






