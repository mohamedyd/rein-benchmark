import torch
import numpy as np
import sys
import os
picket_path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(picket_path))
from encoder.curl_model import LookUpModelSimple, EncodingSimple
from core import DeepTable
from globalvar import *
import pandas as pd
from prepare.dataInfo import *
from sklearn.svm import LinearSVC
#from art.attacks.evasion import ProjectedGradientDescent, FastGradientMethod
#from art.classifiers import SklearnClassifier
from transformer.utils import *

from wrappers.pytorchNN import torchNN

from sklearn.linear_model import LogisticRegression


def standardize(X, vecs, dtypes, mean=None, std=None):
    X_res = np.copy(X)
    vecs_res = np.copy(vecs)
    if mean is None:
        mean = np.mean(X[:, :, 0], axis=0)
        std = np.std(X[:, :, 0], axis=0)
        
    for i in range(X_res.shape[1]):
        if dtypes[i] == 'numeric':
            X_res[:, i, 0] = (X_res[:, i, 0]-mean[i])/std[i]
            vecs[i][:, 0] = (vecs[i][:, 0]-mean[i])/std[i]
    return X_res, vecs_res, mean, std

def flattern(X, first_dim=False):
    if first_dim:
        return X[:, :, 0]
    else:
        return X.reshape(X.shape[0], -1)

def random_flip(x, idx, eps, vec, dtypes, stds, noise_std):
    x_res = np.copy(x)
    idx_res = np.copy(idx)
    for i in range(x.shape[0]):
        if np.random.rand() < eps:
            if dtypes[i] != 'numeric':
                rand_idx = np.random.randint(vec[i].shape[0])
                idx_res[i] = rand_idx
                x_res[i, :] = vec[i][rand_idx]
            else:
                noise = np.random.normal(0, noise_std)
                x_res[i, 0] = x_res[i, 0] + noise
    return x_res, idx_res

def random_flip_ds(X_input, idx_input, eps, vec, dtypes, stds, noise_level=5):
    X = np.copy(X_input)
    idx = np.copy(idx_input)
    
    for i in range(X.shape[0]):
        x_flipped, idx_flipped = random_flip(X[i], idx[i], eps, vec, dtypes, stds, noise_level)
        X[i] = x_flipped
        idx[i] = idx_flipped
        
    return {
                'X_dirty': X,
                'idx_dirty': idx,
           }

def system_flip(x, idx, eps, vec, dtypes, stds, depends, noise_level=1):
    x_res = np.copy(x)
    idx_res = np.copy(idx)
    for i in range(x.shape[0]):
        if np.random.rand() < eps:
            if dtypes[i] != 'numeric':
                sys_idx = (idx[i] + idx[depends[i]]) % vec[i].shape[0]
                idx_res[i] = sys_idx
                x_res[i, :] = vec[i][sys_idx]
            else:
                noise = noise_level
                x_res[i, 0] = x_res[i, 0] + noise
    return x_res, idx_res

def system_flip_ds(X_input, idx_input, eps, vec, dtypes, stds, depends, noise_level=1):
    X = np.copy(X_input)
    idx = np.copy(idx_input)
    
    for i in range(X.shape[0]):
        x_flipped, idx_flipped = system_flip(X[i], idx[i], eps, vec, dtypes, stds, depends, noise_level)
        X[i] = x_flipped
        idx[i] = idx_flipped
        
    return {
                'X_dirty': X,
                'idx_dirty': idx,
           }

def toOneHot(y):
    return np.concatenate((y.reshape(-1, 1)==0, y.reshape(-1, 1)==1), axis=-1).astype(int)

def pad_to_3d(X, last_dim):
    res = np.zeros((X.shape[0], X.shape[1], last_dim), dtype=np.float32)
    res[:, :, 0] = X
    return res

def getAttackSuccessId(X, X_adv, y, classifier):
    y_pred = classifier.predict(X)
    y_pred_adv = classifier.predict(X_adv)

    y = np.argmax(y, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred_adv = np.argmax(y_pred_adv, axis=1)
 
    return np.logical_and(y_pred != y_pred_adv, y_pred == y)

def getAttackFailId(X, X_adv, y, classifier):
    y_pred = classifier.predict(X)
    y_pred_adv = classifier.predict(X_adv)

    y = np.argmax(y, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred_adv = np.argmax(y_pred_adv, axis=1)
 
    return np.logical_and(y_pred == y_pred_adv, y_pred == y) 

def oneHotToZeroOne(y):
    return np.argmax(y, axis=1)

def evasionAttack(X, y, classifier, encoding_dim=8, method='PGD'):
    #Evaluate the ART classifier on benign test examples
    predictions = classifier.predict(X)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)) / len(y)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    
    #Generate adversarial test examples
    if method == 'PGD':
        attack = ProjectedGradientDescent(classifier=classifier, eps=0.2)
    if method == 'FGM':
        attack = FastGradientMethod(classifier=classifier, eps=0.1)
    X_adv = attack.generate(x=X)
    
    # Step 7: Evaluate the ART classifier on adversarial test examples
    predictions = classifier.predict(X_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)) / len(y)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
    
    success_idx_test = getAttackSuccessId(X, X_adv, y, classifier)
    fail_idx_test = getAttackFailId(X, X_adv, y, classifier)
    
    print('Number of samples: %d' % np.sum(success_idx_test))
    
    return {
        'X': pad_to_3d(X[success_idx_test], encoding_dim),
        'X_attack': pad_to_3d(X_adv[success_idx_test], encoding_dim),
        'X_fail': pad_to_3d(X_adv[fail_idx_test], encoding_dim),
        'y': oneHotToZeroOne(y[success_idx_test]),
        'y_fail': oneHotToZeroOne(y[fail_idx_test])
    }

def flip_correct_wrong(X_t, idx_t, y_t, y_t_pred, clf, dtypes, noise_std, eps_col, vec, depends=None, mode='random'):
    is_all_num = (dtypes.count('numeric') == len(dtypes))
    stds = np.std(X_t[:, :, 0], axis=0)
    means = np.mean(X_t[:, :, 0], axis=0)
    #print(stds)
    #print(means)
    
    X_t_correct = X_t[y_t_pred == y_t]
    y_t_correct = y_t[y_t_pred == y_t]
    idx_t_correct = idx_t[y_t_pred == y_t]

    target_size = X_t_correct.shape[0]

    X_dirty_correct = []
    idx_dirty_correct = []
    y_dirty_correct = []

    X_dirty_wrong = []
    idx_dirty_wrong = []
    y_dirty_wrong = []

    while len(X_dirty_correct) < target_size or len(X_dirty_wrong) < target_size*2:
        rand_sid = np.random.randint(X_t_correct.shape[0])

        if mode == 'random':
            x_flipped, idx_flipped = random_flip(X_t_correct[rand_sid], idx_t_correct[rand_sid], 
                eps_col, vec, dtypes, stds, noise_std)
        if mode == 'system':
            x_flipped, idx_flipped = system_flip(X_t_correct[rand_sid], idx_t_correct[rand_sid], 
                eps_col, vec, dtypes, stds, depends)
        
        if is_all_num:
            y_flipped_pred = clf.predict(x_flipped[:, 0].reshape(1, -1))[0]
        else:
            y_flipped_pred = clf.predict(x_flipped.reshape(1, -1))[0]
        if y_flipped_pred == y_t_correct[rand_sid] and len(X_dirty_correct) < target_size:
            X_dirty_correct.append(x_flipped)
            idx_dirty_correct.append(idx_flipped)
            y_dirty_correct.append(y_t_correct[rand_sid])
        if y_flipped_pred != y_t_correct[rand_sid] and len(X_dirty_wrong) < target_size*2:
            X_dirty_wrong.append(x_flipped)
            idx_dirty_wrong.append(idx_flipped)
            y_dirty_wrong.append(y_t_correct[rand_sid])

    X_dirty_correct = np.stack(X_dirty_correct, axis=0)
    idx_dirty_correct = np.stack(idx_dirty_correct, axis=0)
    y_dirty_correct = np.array(y_dirty_correct)

    X_dirty_wrong = np.stack(X_dirty_wrong, axis=0)
    idx_dirty_wrong = np.stack(idx_dirty_wrong, axis=0)
    y_dirty_wrong = np.array(y_dirty_wrong)
    
    return {
                'X_dirty_correct': X_dirty_correct,
                'idx_dirty_correct': idx_dirty_correct,
                'y_dirty_correct': y_dirty_correct,
                'X_dirty_wrong': X_dirty_wrong,
                'idx_dirty_wrong': idx_dirty_wrong,
                'y_dirty_wrong': y_dirty_wrong,
                'X_clean_correct': X_t_correct,
                'idx_clean_correct': idx_t_correct,
                'y_clean_correct': y_t_correct,
           }  

def TrainTestSplit(name, dirtydf, label_col,train_size=0.8, random_id=0, dtypes=None, embed_dim=None, resample=None, save=False):

    #DATA_DIR = '../data/%s/' % name
    #FILE_NAME = 'features.csv'

    if dtypes is None:
        dtypes = dtype_dict[name]

    if embed_dim is None:
        input_dim = input_dim_dict[name]
    else:
        input_dim = embed_dim
    load_embedding = False

    if resample is None:
        resample = resample_dict[name]

    #if not os.path.exists(DATA_DIR+'models/'):
    #    os.makedirs(DATA_DIR+'models/')

    env = {'workers': 1,
        'seed': 123,
        #'dataset_path': "not available", #DATA_DIR+FILE_NAME,
        'dataset_config': {
            'na_values': {"?", "", "None", "none", "nan", "NaN"},
            'sep': ',',
            'header': 'infer',
            'dropna': False,
            'dropcol': None,
            'fillna': True,
            'min_categories_for_text': 10,
            'nan': "_unknown_",
            # set the type of the attributes here.
            # different options: 'categorical', 'text', 'numeric'
            'dtypes': dtypes
        },
        'embed_config': {
            'dim': input_dim,
            TEXT: {
                'dim': input_dim,
                'tokenizer': lambda x: x.split(),
                'a': 1e-3,
                'save': True,
                #'path': DATA_DIR+'models/',
                'load': load_embedding,
                'wv': None,
                'window': 3,
                'min_count': 1,
                'batch_words': 100,
                'epochs': 100,
                'separate': False, #if set False, the text attr would be trained with the same corpus
                'SIF': True,
                'seed': 123
            },
            NUMERIC: {
                'dim': input_dim,
                'padding_constant': 0,
                'seed': 123
            },
            CATEGORICAL: {
                'dim': input_dim,
                'seed': 123
            },
        },
        'general_dim': 256,
        'hidden_dim_1': 100,
        'hidden_dim_2': 50,
        'out_dim': 20,
        'num_neg': 2,
        'batch_size': 3,
        'mask': '_mask_',
        'num_epochs': 1
    }
    
    dirtydf_no_labels = dirtydf.drop(columns=[label_col])

    deeptable = DeepTable(env, dirtydf_no_labels)
    deeptable.load_dataset()
    deeptable.load_embedding()
    model = EncodingSimple(deeptable.ds, deeptable.env)

    whole_tuples, whole_idx = model.train_idx()
    tuples = whole_tuples.size()[0]
    attrib = len(deeptable.ds.df.iloc[0,:])
    dim = int(whole_tuples.size()[1]/attrib)
    whole_tuples = whole_tuples.view(int(tuples), attrib, dim).numpy()
    whole_idx = whole_idx.numpy()

    #label = pd.read_csv(DATA_DIR+'class.csv').values
    label = dirtydf[label_col].values
    label = label.reshape(-1,)

    if resample:
        num_of_pos = np.sum(label)
        num_of_neg = label.size - num_of_pos
        print('Before Resample: pos %d neg %d' % (num_of_pos, num_of_neg))
        if num_of_pos > num_of_neg:
            pos_idx = np.where(label == 1)[0]
            rand_idx = np.random.permutation(pos_idx.size)
            idx_to_delete = pos_idx[rand_idx[:num_of_pos-num_of_neg]]
        else:
            neg_idx = np.where(label == 0)[0]
            rand_idx = np.random.permutation(neg_idx.size)
            idx_to_delete = neg_idx[rand_idx[:num_of_neg-num_of_pos]]

        idx_remain = np.delete(np.arange(label.size), idx_to_delete)
        whole_tuples = whole_tuples[idx_remain]
        whole_idx = whole_idx[idx_remain]
        label = label[idx_remain]

        num_of_pos = np.sum(label)
        num_of_neg = label.size - num_of_pos
        print('Before Resample: pos %d neg %d' % (num_of_pos, num_of_neg))

    vecs = [deeptable.ds.attributes[i].vec for i in range(whole_tuples.shape[1])]

    train_size = int(train_size*label.shape[0])
    indices = np.random.permutation(label.shape[0])

    X_train, vecs_standard, mean, std = standardize(whole_tuples[indices[:train_size]], vecs, dtypes, mean=None, std=None)
    X_test, _, _, _ = standardize(whole_tuples[indices[train_size:]], vecs, dtypes, mean=mean, std=std)

    ds = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': label[indices[:train_size]],
        'y_test': label[indices[train_size:]],
        'idx_train': whole_idx[indices[:train_size]],
        'idx_test': whole_idx[indices[train_size:]],
        'vec': vecs_standard,
        'mean': mean,
        'std': std,
    }

    if not save:
        return ds
    else:
        np.savez(DATA_DIR + '%s_train_test_%d.npz' % (name, random_id), **ds)

def randomFlipTest(name, random_id=0, model_type='lr', ds=None, dtypes=None, save=True, level='medium'):
    DATA_DIR = '../data/%s/' % name

    eps_col_dict = {'low': 0.2, 'medium': 0.3, 'high': 0.5, 'alow': 0.15, 'amedium': 0.25, 'ahigh': 0.4, 'atiny': 1}
    noise_level_dict = {'low': 1, 'medium': 3, 'high': 5, 'alow': 1.5, 'amedium': 2, 'ahigh': 4, 'atiny': 0.25}

    eps_col = eps_col_dict[level]
    noise_std = noise_level_dict[level]

    if ds is None:
        ds = np.load(DATA_DIR + '%s_train_test_%d.npz' % (name, random_id), allow_pickle=True, encoding="latin1")
    X_train = ds['X_train']
    X_test = ds['X_test']

    if model_type == 'svm':
        y_train = 2*ds['y_train'] - 1
        y_test = 2*ds['y_test'] - 1
    else:
        y_train = ds['y_train']
        y_test = ds['y_test']

    vec = ds['vec']
    idx_train = ds['idx_train']
    idx_test = ds['idx_test']
    mean_ds = ds['mean']
    std_ds = ds['std']

    if dtypes is None:
        dtypes = dtype_dict[name]

    is_all_num = (dtypes.count('numeric') == len(dtypes))

    if model_type == 'svm':
        clf = LinearSVC()
        clf.fit(flattern(X_train, is_all_num), y_train)
        #dump(clf, DATA_DIR+'svm%d.joblib'%(random_id))
    if model_type == 'lr':
        clf = LogisticRegression(random_state=0)
        clf.fit(flattern(X_train, is_all_num), y_train)
    if model_type == 'nn':
        if is_all_num:
            input_size = X_train.shape[1]
        else:
            input_size = flattern(X_train).shape[-1]
        clf = torchNN(input_size=input_size)

        if os.path.exists(DATA_DIR+name+'_'+str(random_id)+'_nn.model'):
            print('Loading Saved Model...')
            clf.load(DATA_DIR, name+'_'+str(random_id))
        else:
            clf.fit(flattern(X_train, is_all_num), y_train)
            clf.save(DATA_DIR, name+'_'+str(random_id))

    
    print('Acc of the Classifier: %f' % clf.score(flattern(X_test, is_all_num), y_test))
    y_train_pred = clf.predict(flattern(X_train, is_all_num))
    y_test_pred = clf.predict(flattern(X_test, is_all_num))
        

    train_dirty_correct_wrong = flip_correct_wrong(X_train, idx_train, y_train, 
        y_train_pred, clf, dtypes, noise_std, eps_col, vec)
    test_dirty_correct_wrong = flip_correct_wrong(X_test, idx_test, y_test, 
        y_test_pred, clf, dtypes, noise_std, eps_col, vec)

    # Validate
    '''
    print('Expected: 110110')
    print(clf.score(flattern(train_dirty_correct_wrong['X_dirty_correct'], is_all_num),
              train_dirty_correct_wrong['y_dirty_correct']))
    print(clf.score(flattern(train_dirty_correct_wrong['X_clean_correct'], is_all_num),
              train_dirty_correct_wrong['y_clean_correct']))
    print(clf.score(flattern(train_dirty_correct_wrong['X_dirty_wrong'], is_all_num),
              train_dirty_correct_wrong['y_dirty_wrong']))
    print(clf.score(flattern(test_dirty_correct_wrong['X_dirty_correct'], is_all_num),
              test_dirty_correct_wrong['y_dirty_correct']))
    print(clf.score(flattern(test_dirty_correct_wrong['X_clean_correct'], is_all_num),
              test_dirty_correct_wrong['y_clean_correct']))
    print(clf.score(flattern(test_dirty_correct_wrong['X_dirty_wrong'], is_all_num),
              test_dirty_correct_wrong['y_dirty_wrong']))
    '''
    if save:
        np.savez(DATA_DIR + '%s_%s_random_flip_train_%s_%d.npz' % (name, model_type, level, random_id), **train_dirty_correct_wrong)
        np.savez(DATA_DIR + '%s_%s_random_flip_test_%s_%d.npz' % (name, model_type, level, random_id), **test_dirty_correct_wrong)
    else:
        return train_dirty_correct_wrong, test_dirty_correct_wrong, clf

def systematicFlipTest(name, random_id=0, model_type='lr', ds=None, dtypes=None, save=True, level='medium'):
    DATA_DIR = '../data/%s/' % name

    eps_col_dict = {'low': 0.2, 'medium': 0.3, 'high': 0.5, 'alow': 0.15, 'amedium': 0.25, 'ahigh': 0.4, 'atiny': 1}
    noise_level_dict = {'low': 1, 'medium': 3, 'high': 5, 'alow': 1.5, 'amedium': 2, 'ahigh': 4, 'atiny': 0.25}

    eps_col = eps_col_dict[level]
    noise_std = noise_level_dict[level]

    if ds is None:
        ds = np.load(DATA_DIR + '%s_train_test_%d.npz' % (name, random_id), allow_pickle=True, encoding="latin1")
    X_train = ds['X_train']
    X_test = ds['X_test']
    if model_type == 'svm':
        y_train = 2*ds['y_train'] - 1
        y_test = 2*ds['y_test'] - 1
    else:
        y_train = ds['y_train']
        y_test = ds['y_test']
    vec = ds['vec']
    idx_train = ds['idx_train']
    idx_test = ds['idx_test']
    mean_ds = ds['mean']
    std_ds = ds['std']

    if dtypes is None:
        dtypes = dtype_dict[name]
    is_all_num = (dtypes.count('numeric') == len(dtypes))

    nonnums = [i for i, x in enumerate(dtypes) if x != "numeric"]
    if len(nonnums) > 0:
        dependencies = [nonnums[np.random.randint(len(nonnums))] for i in range(len(dtypes))]
    else:
        dependencies = None

    if model_type == 'svm':
        clf = LinearSVC()
        clf.fit(flattern(X_train, is_all_num), y_train)
    if model_type == 'lr':
        clf = LogisticRegression(random_state=0)
        clf.fit(flattern(X_train, is_all_num), y_train)
    if model_type == 'nn':
        if is_all_num:
            input_size = X_train.shape[1]
        else:
            input_size = flattern(X_train).shape[-1]
        clf = torchNN(input_size=input_size)
        clf.load(DATA_DIR, name+'_'+str(random_id))
    
    print('Acc of the classifier: %f' % clf.score(flattern(X_test, is_all_num), y_test))
    y_train_pred = clf.predict(flattern(X_train, is_all_num))
    y_test_pred = clf.predict(flattern(X_test, is_all_num))
        

    train_dirty_correct_wrong = flip_correct_wrong(X_train, idx_train, y_train, 
        y_train_pred, clf, dtypes, noise_std, eps_col, vec, depends=dependencies, mode='system')
    test_dirty_correct_wrong = flip_correct_wrong(X_test, idx_test, y_test, 
        y_test_pred, clf, dtypes, noise_std, eps_col, vec, depends=dependencies, mode='system')

    # Validate
    '''
    print('Expected: 110110')
    print(clf.score(flattern(train_dirty_correct_wrong['X_dirty_correct'], is_all_num),
              train_dirty_correct_wrong['y_dirty_correct']))
    print(clf.score(flattern(train_dirty_correct_wrong['X_clean_correct'], is_all_num),
              train_dirty_correct_wrong['y_clean_correct']))
    print(clf.score(flattern(train_dirty_correct_wrong['X_dirty_wrong'], is_all_num),
              train_dirty_correct_wrong['y_dirty_wrong']))
    print(clf.score(flattern(test_dirty_correct_wrong['X_dirty_correct'], is_all_num),
              test_dirty_correct_wrong['y_dirty_correct']))
    print(clf.score(flattern(test_dirty_correct_wrong['X_clean_correct'], is_all_num),
              test_dirty_correct_wrong['y_clean_correct']))
    print(clf.score(flattern(test_dirty_correct_wrong['X_dirty_wrong'], is_all_num),
              test_dirty_correct_wrong['y_dirty_wrong']))
    '''
    if save:
        np.savez(DATA_DIR + '%s_%s_system_flip_train_%s_%d.npz' % (name, model_type, level, random_id), **train_dirty_correct_wrong)
        np.savez(DATA_DIR + '%s_%s_system_flip_test_%s_%d.npz' % (name, model_type, level, random_id), **test_dirty_correct_wrong)
    else:
        return train_dirty_correct_wrong, test_dirty_correct_wrong, clf

def randomFlipTrain(name, random_id=0, ds=None, dtypes=None, save=True, noise_level=5, eps_col = 0.5):
    DATA_DIR = '../data/%s/' % name

    if ds is None:
        ds = np.load(DATA_DIR + '%s_train_test_%d.npz' % (name, random_id), allow_pickle=True, encoding="latin1")
    X_train = ds['X_train']
    X_test = ds['X_test']
    vec = ds['vec']
    idx_train = ds['idx_train']
    idx_test = ds['idx_test']
    mean_ds = ds['mean']
    std_ds = ds['std']

    if dtypes is None:
        dtypes = dtype_dict[name]

    train_flipped = random_flip_ds(X_train, idx_train, eps_col, vec, dtypes, std_ds, noise_level)
    test_flipped = random_flip_ds(X_test, idx_test, eps_col, vec, dtypes, std_ds, noise_level)

    if save:
        np.savez(DATA_DIR + '%s_random_train_%d_%d.npz' % (name, int(100*eps_col), random_id), **train_flipped)
        np.savez(DATA_DIR + '%s_random_test_%d_%d.npz' % (name, int(100*eps_col), random_id), **test_flipped)
    else:
        return train_flipped

def systematicFlipTrain(name, random_id=0, ds=None, dtypes=None, save=True, noise_level=1, eps_col = 0.5):
    DATA_DIR = '../data/%s/' % name

    if ds is None:
        ds = np.load(DATA_DIR + '%s_train_test_%d.npz' % (name, random_id), allow_pickle=True, encoding="latin1")
    X_train = ds['X_train']
    X_test = ds['X_test']
    vec = ds['vec']
    idx_train = ds['idx_train']
    idx_test = ds['idx_test']
    mean_ds = ds['mean']
    std_ds = ds['std']

    if dtypes is None:
        dtypes = dtype_dict[name]

    nonnums = [i for i, x in enumerate(dtypes) if x != "numeric"]
    if len(nonnums) > 0:
        dependencies = [nonnums[np.random.randint(len(nonnums))] for i in range(len(dtypes))]
    else:
        dependencies = None

    train_flipped = system_flip_ds(X_train, idx_train, eps_col, vec, dtypes, std_ds, dependencies, noise_level=noise_level)
    test_flipped = system_flip_ds(X_test, idx_test, eps_col, vec, dtypes, std_ds, dependencies, noise_level=noise_level)

    if save:
        np.savez(DATA_DIR + '%s_system_train_%d_%d.npz' % (name, int(100*eps_col), random_id), **train_flipped)
        np.savez(DATA_DIR + '%s_system_test_%d_%d.npz' % (name, int(100*eps_col), random_id), **test_flipped)
    else:
        return train_flipped

def PGDAttack(name, random_id=0, model_type='lr', ds=None, save=True):
    DATA_DIR = '../data/%s/' % name
    is_all_num = True

    if ds is None:
        ds = np.load(DATA_DIR + '%s_train_test_%d.npz' % (name, random_id), allow_pickle=True, encoding="latin1")
    X_train = flattern(ds['X_train'], is_all_num)
    y_train = toOneHot(ds['y_train'])
    idx_train = ds['idx_train']

    vec = ds['vec']

    X_test = flattern(ds['X_test'], is_all_num)
    y_test = toOneHot(ds['y_test'])
    idx_test = ds['idx_test']

    encoding_dim = ds['X_train'].shape[-1]

    if model_type == 'svm':
        model = LinearSVC()
        classifier = SklearnClassifier(model=model)
        classifier.fit(X_train, y_train)
    if model_type == 'lr':
        model = LogisticRegression(random_state=0)
        classifier = SklearnClassifier(model=model)
        classifier.fit(X_train, y_train)
    if model_type == 'nn':
        input_size = X_train.shape[-1]
        clf = torchNN(input_size=input_size)
        clf.load(DATA_DIR, name+'_'+str(random_id))
        classifier = clf.classifier

    res_train = evasionAttack(X_train, y_train, classifier, encoding_dim)
    res_test = evasionAttack(X_test, y_test, classifier, encoding_dim)

    if save:
        np.savez(DATA_DIR + '%s_%s_adv_flip_train_pgd_%d.npz' % (name, model_type, random_id), **res_train)
        np.savez(DATA_DIR + '%s_%s_adv_flip_test_pgd_%d.npz' % (name, model_type, random_id), **res_test)
    else:
        return res_train, res_test

    '''
    print('Should be 1010')

    if model_type == 'nn':
        print(clf.score(flattern(res_train['X'], is_all_num), res_train['y']))
        print(clf.score(flattern(res_train['X_attack'], is_all_num), res_train['y']))
        print(clf.score(flattern(res_test['X'], is_all_num), res_test['y']))
        print(clf.score(flattern(res_test['X_attack'], is_all_num), res_test['y']))
    if model_type == 'svm':
        clf = LinearSVC()
        clf.fit(X_train, oneHotToZeroOne(y_train))
        print(clf.score(flattern(res_train['X'], is_all_num), res_train['y']))
        print(clf.score(flattern(res_train['X_attack'], is_all_num), res_train['y']))
        print(clf.score(flattern(res_test['X'], is_all_num), res_test['y']))
        print(clf.score(flattern(res_test['X_attack'], is_all_num), res_test['y']))
    if model_type == 'lr':
        clf = LogisticRegression(random_state=0)
        clf.fit(X_train, oneHotToZeroOne(y_train))
        print(clf.score(flattern(res_train['X'], is_all_num), res_train['y']))
        print(clf.score(flattern(res_train['X_attack'], is_all_num), res_train['y']))
        print(clf.score(flattern(res_test['X'], is_all_num), res_test['y']))
        print(clf.score(flattern(res_test['X_attack'], is_all_num), res_test['y']))
    '''
def FGMAttack(name, random_id=0, model_type='lr', ds=None, save=True):
    DATA_DIR = '../data/%s/' % name
    is_all_num = True

    if ds is None:
        ds = np.load(DATA_DIR + '%s_train_test_%d.npz' % (name, random_id), allow_pickle=True, encoding="latin1")
    X_train = flattern(ds['X_train'], is_all_num)
    y_train = toOneHot(ds['y_train'])
    idx_train = ds['idx_train']

    vec = ds['vec']

    X_test = flattern(ds['X_test'], is_all_num)
    y_test = toOneHot(ds['y_test'])
    idx_test = ds['idx_test']

    encoding_dim = ds['X_train'].shape[-1]

    if model_type == 'svm':
        model = LinearSVC()
        classifier = SklearnClassifier(model=model)
        classifier.fit(X_train, y_train)
    if model_type == 'lr':
        model = LogisticRegression(random_state=0)
        classifier = SklearnClassifier(model=model)
        classifier.fit(X_train, y_train)
    if model_type == 'nn':
        input_size = X_train.shape[-1]
        clf = torchNN(input_size=input_size)
        clf.load(DATA_DIR, name+'_'+str(random_id))
        classifier = clf.classifier

    res_train = evasionAttack(X_train, y_train, classifier, encoding_dim, method='FGM')
    res_test = evasionAttack(X_test, y_test, classifier, encoding_dim, method='FGM')

    if save:
        np.savez(DATA_DIR + '%s_%s_adv_flip_train_fgm_%d.npz' % (name, model_type, random_id), **res_train)
        np.savez(DATA_DIR + '%s_%s_adv_flip_test_fgm_%d.npz' % (name, model_type, random_id), **res_test)
    else:
        return res_train, res_test

def dataPrepareTest(name, adv=False, level='medium', count_group=range(5), artificial=False):
    for count in count_group:
        print('############################################')
        print('#                    %d                     #' % count)
        print('############################################')
        randomFlipTest(name, random_id=count, model_type='svm', level=level)
        randomFlipTest(name, random_id=count, model_type='nn', level=level)
        randomFlipTest(name, random_id=count, model_type='lr', level=level)
        if artificial:
            randomFlipTest(name, random_id=count, model_type='svm', level='alow')
            randomFlipTest(name, random_id=count, model_type='svm', level='amedium')
            randomFlipTest(name, random_id=count, model_type='svm', level='ahigh')

        
            randomFlipTest(name, random_id=count, model_type='nn', level='alow')
            randomFlipTest(name, random_id=count, model_type='nn', level='amedium')
            randomFlipTest(name, random_id=count, model_type='nn', level='ahigh')

        
            randomFlipTest(name, random_id=count, model_type='lr', level='alow')
            randomFlipTest(name, random_id=count, model_type='lr', level='amedium')
            randomFlipTest(name, random_id=count, model_type='lr', level='ahigh')

        systematicFlipTest(name, random_id=count, model_type='svm', level=level)
        systematicFlipTest(name, random_id=count, model_type='nn', level=level)
        systematicFlipTest(name, random_id=count, model_type='lr', level=level)

        if adv:
            PGDAttack(name, random_id=count, model_type='svm')
            PGDAttack(name, random_id=count, model_type='nn')
            PGDAttack(name, random_id=count, model_type='lr')
            FGMAttack(name, random_id=count, model_type='svm')
            FGMAttack(name, random_id=count, model_type='nn')
            FGMAttack(name, random_id=count, model_type='lr')
            randomFlipTest(name, random_id=count, model_type='svm', level='atiny')
            randomFlipTest(name, random_id=count, model_type='nn', level='atiny')
            randomFlipTest(name, random_id=count, model_type='lr', level='atiny')

def dataPrepareTestAdv(name, count_group=range(5)):
    for count in count_group:
        print('############################################')
        print('#                    %d                     #' % count)
        print('############################################')
        PGDAttack(name, random_id=count, model_type='svm')
        PGDAttack(name, random_id=count, model_type='nn')
        PGDAttack(name, random_id=count, model_type='lr')
        FGMAttack(name, random_id=count, model_type='svm')
        FGMAttack(name, random_id=count, model_type='nn')
        FGMAttack(name, random_id=count, model_type='lr')

def dataPrepareTrain(name, level='medium', split=True):
    DATA_DIR = '../data/%s/' % name
    eps_col_dict = {'low': 0.2, 'medium': 0.3, 'high': 0.5}
    noise_level_dict = {'low': 1, 'medium': 3, 'high': 5}

    for count in range(5):
        print('############################################')
        print('#                    %d                     #' % count)
        print('############################################')
        if split:
            TrainTestSplit(name, count)
        train_flipped = randomFlipTrain(name, count, eps_col=eps_col_dict[level], noise_level=noise_level_dict[level], save=False)
        np.savez(DATA_DIR + '%s_random_train_%s_%d.npz' % (name, level, count), **train_flipped)
        train_flipped = systematicFlipTrain(name, count, eps_col=eps_col_dict[level], noise_level=noise_level_dict[level], save=False)
        np.savez(DATA_DIR + '%s_system_train_%s_%d.npz' % (name, level, count), **train_flipped)

def mixCleanDirtyData(ds, dirty_ds, eps_row):
    X_dirty = dirty_ds['X_dirty']
    idx_dirty = dirty_ds['idx_dirty']
    X = ds['X_train']
    y = ds['y_train']
    idx = ds['idx_train']
    X_mix = np.copy(X)
    idx_mix = np.copy(idx)
    rand = np.random.rand(X.shape[0])
    X_mix[rand < eps_row] = X_dirty[rand < eps_row]
    idx_mix[rand < eps_row] = idx_dirty[rand < eps_row]
    return X_mix, y, idx_mix, (rand < eps_row)

def pad_to_3d(X, last_dim):
    res = np.zeros((X.shape[0], X.shape[1], last_dim), dtype=np.float32)
    res[:, :, 0] = X
    return res

def mixCleanPoisonData(ds, poison_ds, eps_row):
    X = ds['X_train']
    y = ds['y_train']
    idx = ds['idx_train']

    X_dirty = pad_to_3d(poison_ds['X_poison'].astype(np.float32), X.shape[-1])
    y_dirty = poison_ds['y_poison'].astype(np.float32).reshape(-1,)

    N = int(np.floor(X.shape[0]*(eps_row/(1-eps_row))))

    X_mix = np.concatenate((X, X_dirty[:N]), axis=0)
    idx_mix = np.concatenate((idx, idx[:N]))
    y_mix = np.concatenate((y, y_dirty[:N]), axis=0)
    label = np.concatenate((np.zeros(X.shape[0]), np.ones(N)), axis=0)

    random_perm = np.random.permutation(X_mix.shape[0])

    return X_mix[random_perm], y_mix[random_perm], idx_mix[random_perm], label[random_perm] 














