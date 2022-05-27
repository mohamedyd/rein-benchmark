import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import OrderedDict
import pandas as pd


# TODO: remove the first dim of numeric
# For each cell, get positive examples based on context
def getPositiveMask(tuples, structure_mask, thres=0.8, num_ids=[]):
    attr_num =     structure_mask.shape[0]
    tuple_num = tuples.shape[0]

    norm = torch.rsqrt(torch.sum(tuples**2, dim=-1, keepdim=True)).expand(-1, -1, tuples.shape[-1])
    normed_tuples = tuples*norm

    structure_mask_no_diag = structure_mask*(1-np.eye(attr_num))

    a = normed_tuples.unsqueeze(0).repeat(tuple_num, 1, 1, 1)
    b = normed_tuples.unsqueeze(1).repeat(1, tuple_num, 1, 1)

    cos_sim = torch.sum(a*b, dim=-1)

    res_mask = None

    for i in range(attr_num):
        mask = torch.Tensor(structure_mask_no_diag[i]).unsqueeze(0).unsqueeze(0).repeat(tuple_num, tuple_num, 1)
        masked_cos_sim = cos_sim*mask
        if np.sum(structure_mask_no_diag[i]) > 0:
            masked_cos_avg = torch.sum(masked_cos_sim, dim=-1)/np.sum(structure_mask_no_diag[i])
        else:
            masked_cos_avg = torch.sum(masked_cos_sim, dim=-1)

        masked_cos_avg = masked_cos_avg*(1-torch.eye(tuple_num))+torch.eye(tuple_num)

        if res_mask is None:
            res_mask = (masked_cos_avg > thres).unsqueeze(-1)
        else:
            res_mask = torch.cat((res_mask, (masked_cos_avg > thres).unsqueeze(-1)), dim=-1)

    return res_mask.data.numpy()

def generateMask(attr_num, groups):
    mask = np.eye(attr_num)
    for group in groups:
        for i in group:
            for j in group:
                mask[i][j] = 1

    return mask

def generateMaskFromName(names, groups):
    attr_num = len(names)
    ids = []
    for group in groups:
        ids.append([names.index(name) for name in group])

    return generateMask(attr_num, ids)

def getIndependentIdx(names, groups):
    attr_num = len(names)
    iidx = []
    for i in range(attr_num):
        inRelation = False
        for group in groups:
            for name in group:
                idx = names.index(name)
                if idx == i:
                    inRelation = True
        if not inRelation:
            iidx.append(i)

    return iidx

def hideValues(table, attr, attr_list, ratio, seed = 0):
    np.random.seed(seed)
    attr_id = attr_list.index(attr)
    num_of_tuples = table.shape[0]
    num_to_hide = int(num_of_tuples*ratio)
    indices = np.random.permutation(num_of_tuples)
    table[indices[:num_to_hide], attr_id, :] = 0
    return table

def permuteValues(table, attr, attr_list, ratio, seed = 0):
    np.random.seed(seed)
    attr_id = attr_list.index(attr)
    num_of_tuples = table.shape[0]
    num_to_permute = int(num_of_tuples*ratio)
    indices = np.random.permutation(num_of_tuples)
    permute_indices = np.random.permutation(num_to_permute)
    part_to_be_permuted = table[indices[:num_to_permute], attr_id, :]
    table[indices[:num_to_permute], attr_id, :] = part_to_be_permuted[permute_indices, :]
    return table

def flipValues(table, attr, attr_list, ratio, attribute_info, seed = 0):
    np.random.seed(seed)
    attr_id = attr_list.index(attr)
    attr_vec = attribute_info[attr_id].vec
    num_of_vecs = attr_vec.shape[0]
    num_of_tuples = table.shape[0]
    num_to_flip = int(num_of_tuples*ratio)
    indices = np.random.permutation(num_of_tuples)

    for i in range(num_to_flip):
        index = indices[i]
        vec_id = np.random.randint(num_of_vecs)
        wrong_vec = attr_vec[vec_id, :]
        correct_vec = table[index, attr_id, :]
        while np.allclose(wrong_vec, correct_vec):
            vec_id = np.random.randint(num_of_vecs)
            wrong_vec = attr_vec[vec_id, :]

        table[index, attr_id, :] = torch.Tensor(wrong_vec)

    return table, indices[:num_to_flip]


def trainTest(clf, X_train, X_test, y_train, y_test, f1=False):
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    predictions = [round(value) for value in y_pred]
    acc_test = accuracy_score(y_test, predictions)

    y_pred_train = clf.predict(X_train)
    predictions_train = [round(value) for value in y_pred_train]
    acc_train = accuracy_score(y_train, predictions_train)

    if f1:
        precision_test = precision_score(y_test, predictions)
        recall_test = recall_score(y_test, predictions)
        f1_test = f1_score(y_test, predictions)

        precision_train = precision_score(y_train, predictions_train)
        recall_train = recall_score(y_train, predictions_train)
        f1_train = f1_score(y_train, predictions_train)

        return ({'acc': acc_train, 'precision': precision_train,
                 'recall': recall_train, 'f1': f1_train},
                {'acc': acc_test, 'precision': precision_test,
                 'recall': recall_test, 'f1': f1_test}
               )

    else:
        return acc_train, acc_test

def computePR(predict, correct):
    TP = np.intersect1d(predict, correct).shape[0]
    P = TP/predict.shape[0]
    R = TP/correct.shape[0]
    return P, R

def oneHotFilter(embedding, original, threshold, attribute_ids, attribute_info):
    print('--------- Filter for onehot ---------')
    for attribtue_idx in attribute_ids:
        lookup = attribute_info[attribtue_idx].vec
        lookup = lookup[:, :lookup.shape[0]]
        torecover = original[:, attribtue_idx, :lookup.shape[0]]
        tocheck = embedding[:, attribtue_idx, :lookup.shape[0]]
        tocheck_norm = np.sqrt(np.sum(tocheck**2, axis=1))
        tocheck = tocheck/tocheck_norm.reshape(-1, 1)
        maximum = np.max(tocheck, axis=1)
        argmax = np.argmax(tocheck, axis=1)
        tocheck[range(tocheck.shape[0]), argmax] = -1
        second_maximum = np.max(tocheck, axis=1)

        diff = maximum - second_maximum
        tocheck[diff > threshold] = lookup[argmax[diff > threshold]]
        # do not impute if not confident
        tocheck[diff <= threshold] = torecover[diff <= threshold]
        embedding[:, attribtue_idx, :lookup.shape[0]] = tocheck
        embedding[:, attribtue_idx, lookup.shape[0]:] = 0
    return embedding

def NLL(T, maxZ, allZ):
    NLL_all = -np.log(np.exp(maxZ/T[0])/np.sum(np.exp(allZ/T[0]), axis=-1))
    return np.sum(NLL_all)


def independentRecover(embedding, original, iidx):
    print('--------- Recover independent attributes ---------')
    for attribtue_idx in iidx:
        embedding[:, attribtue_idx, :] = original[:, attribtue_idx, :]
    return embedding

def saveAttributeInfo(attribute_info, DIR):
    for i in attribute_info:
        vec = attribute_info[i].vec
        vocab = attribute_info[i].vocab

        vocab.to_pickle(DIR+'vocab'+str(i)+'.pkl')
        np.save(DIR + 'vec%d.npy'%i, vec)

class AttributeInfo:
    def __init__(self, vec, vocab):
        self.vec = vec
        self.vocab = vocab

def loadAttributeInfo(num, DIR):
    od = OrderedDict()

    for i in range(num):
        vocab = pd.read_pickle(DIR+'vocab'+str(i)+'.pkl')
        vec = np.load(DIR + 'vec%d.npy'%i)
        ai = AttributeInfo(vec, vocab)
        od[i] = ai

    return od












