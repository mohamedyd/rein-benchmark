import numpy as np
from picket.transformer.utils import *
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
import torch
import os
from picket.transformer.PicketNet import PicketNetModel
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from picket.rvae.run import get_outlier_scores
from picket.prepare.dataInfo import *
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from picket.wrappers.pytorchNN import torchNN
import scipy.io

def getCleanDirtyMix(X, X_dirty, idx, idx_dirty, eps_row):
    X_mix = np.copy(X)
    idx_mix = np.copy(idx)
    rand = np.random.rand(X.shape[0])
    X_mix[rand < eps_row] = X_dirty[rand < eps_row]
    idx_mix[rand < eps_row] = idx_dirty[rand < eps_row]
    return X_mix, idx_mix, (rand < eps_row)

def getCleanPoisonMix(X, X_dirty, y, y_dirty, idx, eps_row):
    N = int(np.floor(X.shape[0]*(eps_row/(1-eps_row))))

    #N = int(np.floor(X.shape[0]*eps_row))

    X_mix = np.concatenate((X, X_dirty[:N]), axis=0)
    idx_mix = np.concatenate((idx, idx[:N]))
    y_mix = np.concatenate((y, y_dirty[:N]), axis=0)
    label = np.concatenate((np.zeros(X.shape[0]), np.ones(N)), axis=0)

    random_perm = np.random.permutation(X_mix.shape[0])

    return X_mix[random_perm], y_mix[random_perm], idx_mix[random_perm], label[random_perm] 


def flattern(X, first_dim=False):
    if first_dim:
        return X[:, :, 0]
    else:
        return X.reshape(X.shape[0], -1)

def pad_to_3d(X, last_dim):
    res = np.zeros((X.shape[0], X.shape[1], last_dim), dtype=np.float32)
    res[:, :, 0] = X
    return res

class Attribute:
    def __init__(self, vec):
        self.vec = vec

def computeMetric(score, label, flip=False):
    if flip:
        score = -score
    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    #print('AUROC')
    #print(auc)

    AVPR = metrics.average_precision_score(label, score, pos_label=1)
    #print('AVPR')
    #print(AVPR)

    return auc, AVPR

class RVAE_ds:
    def __init__(self, X_raw, idx, vecs, dtypes, means=None, stds=None, dataset_type='mixed'):
        self.cat_cols = []
        self.num_cols = []
        self.feat_info = []
        self.dataset_type = dataset_type
        self.X = []
        self.means_ = []
        self.stds_ = []
        
        for i in range(X_raw.shape[1]):
            if dtypes[i] in ['categorical', 'text']:
                self.X.append(idx[:, i])
                self.cat_cols.append(str(i))
                self.feat_info.append((str(i), 'categ', vecs[i].shape[0]))
                self.means_.append(0)
                self.stds_.append(0)
            elif dtypes[i] == 'numeric':
                feature_tmp = X_raw[:, i, 0]
                if means is None:
                    mean = np.mean(feature_tmp)
                    std = np.std(feature_tmp)
                else:
                    mean = means[i]
                    std = stds[i]
                #Don't need this since the data have been standardized
                #feature_tmp = (feature_tmp-mean)/std
                self.means_.append(mean)
                self.stds_.append(std)
                
                self.X.append(feature_tmp)
                self.num_cols.append(str(i))
                self.feat_info.append((str(i), 'real', 1))
                
            else:
                print('Something is wrong.')
        self.X = np.stack(self.X, axis=-1)
        print(self.X.shape)
        
        self.X = torch.Tensor(self.X)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        ret_list = [torch.Tensor(self.X[idx])]
        # Just a placeholder
        ret_list += [torch.Tensor([0, 0, 0])]
        return ret_list

class Metadata:
    def __init__(self):
        self.AVI=False
        self.activation='relu'
        self.alpha_prior=0.95
        self.batch_size=150 
        self.cuda_on=torch.cuda.is_available() 
        self.embedding_size=50
        self.inference_type='vae' 
        self.is_one_hot=False 
        self.l2_reg=0.0 
        self.latent_dim=20
        self.layer_size=400
        self.log_interval=50
        self.lr=0.001
        self.number_epochs=100
        self.outlier_model='RVAE'
        self.seqvae_bprop=False
        self.seqvae_steps=4
        self.seqvae_two_stage=False 
        self.std_gauss_nll=2.0
        self.steps_2stage=4

class FiltersTrain:
    def __init__(self, name, dtypes, input_dim, level='medium', eps_row=0.2, ntype='random', modelname='' , 
        random_id=0, loss_warm_up=50, loss_trim=20, sparse=False, remove_factor=1, load_log=False):
        self.name = name
        self.DATA_DIR = '../data/%s/' % (name)
        self.dtypes = dtypes
        self.input_dim = input_dim
        self.level = level
        self.eps_row = eps_row
        self.ntype = ntype
        self.random_id = random_id
        self.loss_warm_up = loss_warm_up
        self.loss_trim = loss_trim
        self.sparse = sparse
        self.modelname = modelname
        self.remove_factor = remove_factor

        if load_log:
            self.load_clean_data()
        else:
            self.load_clean_data()
            self.load_mix_data()

    def load_clean_data(self):
        if self.random_id is not None:
            ds = np.load(self.DATA_DIR + '%s_train_test_%d.npz' % (self.name, self.random_id), 
                allow_pickle=True, encoding="latin1")
        else:
            ds = np.load(self.DATA_DIR + '%s_train_test.npz' % self.name, 
                allow_pickle=True, encoding="latin1")

        self.vec = ds['vec']
        self.attribute_info = [Attribute(self.vec[i]) for i in range(len(self.vec))]

        self.X_train = ds['X_train']
        self.y_train = ds['y_train']
        self.idx_train = ds['idx_train']

        self.X_test = ds['X_test']
        self.y_test = ds['y_test']
        self.idx_test = ds['idx_test']

    def load_mix_data(self):
        if self.ntype != 'poison':
            ds_train_dirty = np.load(self.DATA_DIR + '%s_%s_train_%s_%d.npz' % (self.name, self.ntype, self.level, self.random_id), 
                            allow_pickle=True, encoding="latin1")
            X_train_dirty = ds_train_dirty['X_dirty']
            idx_train_dirty = ds_train_dirty['idx_dirty']

            self.X_train_mix, self.idx_train_mix, self.label = getCleanDirtyMix(self.X_train, X_train_dirty, self.idx_train, idx_train_dirty, self.eps_row)
        else:
            poison_ds = scipy.io.loadmat('../data/dataPoisonSet/%s/%s_poison_%d.mat' % (self.modelname, self.name, self.random_id))
            X_poison = pad_to_3d(poison_ds['X_poison'].astype(np.float32), self.X_train.shape[-1])
            y_poison = poison_ds['y_poison'].astype(np.float32).reshape(-1,)
            self.X_train_mix, self.y_train_mix, self.idx_train_mix, self.label = getCleanPoisonMix(self.X_train, X_poison, 
                self.y_train, y_poison, self.idx_train, self.eps_row)

    def PicketFilter(self, ghmode='both', transformer_layer=6, head_num=2, hidden_dim=64, dropout=0.1, sameFF=True, sameTransform=True,
        useEncoding=False):
        param = {
            'description': 'Contrastive Transformer',
            'model_dim': self.input_dim,
            'input_dim': self.input_dim,
            'attribute_num': self.X_train.shape[1],
            'transformer_layer': transformer_layer,
            'head_num': head_num,
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'numerical_ids': [i for i, x in enumerate(self.dtypes) if x == "numeric"],
            'batch_size': 500,
            'epochs': 1,
            'opt_factor': 0.5,
            'warmup': 300,
            'adam_lr': 0,
            'adam_betas': (0.9, 0.98),
            'adam_eps': 1e-9,
            'neg_sample_num': 4,
            'random_mask': False,
            'fast': True,
            'structure_mask_type': 'none', # could be 'hard', 'soft', 'sample'
            'loss_warm_up_epochs': self.loss_warm_up,
            'loss_trim_epochs': self.loss_trim,
            'sparse': self.sparse,
            'loss_trim_p': self.eps_row,
            'ghmode': ghmode,
            'useEncoding': useEncoding,
            'categorical_ids': [i for i, x in enumerate(self.dtypes) if x == 'categorical'],
            'sameFF': sameFF,
            'sameTransform': sameTransform
        }

        PicketN = PicketNetModel(param)
        PicketN.loadData(torch.Tensor(self.X_train_mix).double(), None, self.attribute_info, 
                    iidx = [], tuple_idx = torch.Tensor(self.idx_train_mix))
        PicketN.loss_based_train()
        self.PK_score = PicketN.outlierScore

        if np.isnan(self.PK_score).any():
            return 1

        computeMetric(self.PK_score, self.label, flip=(self.ntype=='poison'))
        return 0


    def IFFilter(self, hptune=False, hp=None):
        if not hptune:
            if hp is None:
                if self.ntype == 'poison':
                    clf = IsolationForest(random_state=0, contamination=1-self.eps_row).fit(flattern(self.X_train_mix))
                else:
                    clf = IsolationForest(random_state=0, contamination=self.eps_row).fit(flattern(self.X_train_mix))
            else:
                clf = IsolationForest(random_state=0, contamination=hp).fit(flattern(self.X_train_mix))
            self.IF_score = np.zeros(self.X_train_mix.shape[0]) - clf.score_samples(flattern(self.X_train_mix))
            computeMetric(self.IF_score, self.label, flip=(self.ntype=='poison'))
        else:
            for contamination in [0.1*i for i in range(10)]:
                clf = IsolationForest(random_state=0, contamination=self.eps_row).fit(flattern(self.X_train_mix))
                IF_score = np.zeros(self.X_train_mix.shape[0]) - clf.score_samples(flattern(self.X_train_mix))
                print('contamination=%f' % contamination)
                computeMetric(IF_score, self.label, flip=(self.ntype=='poison'))

    def OCSVMFilter(self, hptune=False, hp=None):
        if not hptune:
            if hp is None:
                if self.ntype == 'poison':
                    clf = OneClassSVM(gamma='auto', nu=1-self.eps_row).fit(flattern(self.X_train_mix))
                else:
                    clf = OneClassSVM(gamma='auto', nu=self.eps_row).fit(flattern(self.X_train_mix))
            else:
                clf = OneClassSVM(gamma='auto', nu=hp).fit(flattern(self.X_train_mix))
            self.OCSVM_score = np.zeros(self.X_train_mix.shape[0]) - clf.score_samples(flattern(self.X_train_mix))
            computeMetric(self.OCSVM_score, self.label, flip=(self.ntype=='poison'))
        else:
            for nu in [0.1*i+0.1 for i in range(9)]:
                clf = OneClassSVM(gamma='auto', nu=nu).fit(flattern(self.X_train_mix))
                OCSVM_score = np.zeros(self.X_train_mix.shape[0]) - clf.score_samples(flattern(self.X_train_mix))
                print('nu=%f' % nu)
                computeMetric(OCSVM_score, self.label, flip=(self.ntype=='poison'))    

    def RVAEFilter(self):
        RVAE_mix_set = RVAE_ds(self.X_train_mix, self.idx_train_mix, self.vec, self.dtypes)
        mix_loader = torch.utils.data.DataLoader(RVAE_mix_set, batch_size=150, shuffle=True)
        md = Metadata()

        _, scores_mix = get_outlier_scores(md, mix_loader, RVAE_mix_set.X, mix_loader, RVAE_mix_set.X, RVAE_mix_set)
        self.RVAE_score = -np.sum(scores_mix, axis=-1)

        computeMetric(self.RVAE_score, self.label, flip=(self.ntype=='poison'))

    def effectOnDownstream(self, clf, debug=False):
        scores = {
            'PK': self.PK_score,
            'IF': self.IF_score,
            'OCSVM': self.OCSVM_score,
            'RVAE':    self.RVAE_score
        }

        methods = ['PK', 'IF', 'OCSVM', 'RVAE']

        res = {}
        filtered_data = {}

        print('On clean data')
        clf.fit(flattern(self.X_train), self.y_train)
        print(clf.score(flattern(self.X_test), self.y_test))

        res['dsacc_CLEAN'] = clf.score(flattern(self.X_test), self.y_test)

        print('Do nothing')
        if self.ntype == 'poison':
            y_train_to_use = self.y_train_mix
        else:
            y_train_to_use = self.y_train

        clf.fit(flattern(self.X_train_mix), y_train_to_use)

        print(clf.score(flattern(self.X_test), self.y_test))

        res['dsacc_NOTHING'] = clf.score(flattern(self.X_test), self.y_test)

        for mt in methods:
            print(mt)
            if self.ntype == 'poison':
                score = -scores[mt]
            else:
                score = scores[mt]

            sidx = np.argsort(score)
            split_idx = int((1-self.eps_row*self.remove_factor)*sidx.shape[0])
            sidx_remove = sidx[split_idx:]
            sidx_left = sidx[:split_idx]
            if debug:
                print('Mean score removed: %f, left: %f' % (np.mean(score[sidx_remove], np.mean(score[sidx_left]))))
            clf.fit(flattern(self.X_train_mix[sidx_left]), y_train_to_use[sidx_left])
            print(clf.score(flattern(self.X_test), self.y_test))

            res['dsacc_'+mt] = clf.score(flattern(self.X_test), self.y_test)

            filtered_data['X_'+mt] = flattern(self.X_train_mix[sidx_left], True)
            filtered_data['y_'+mt] = y_train_to_use[sidx_left]

        print('Remove all dirty samples')
        clf.fit(flattern(self.X_train_mix[self.label==0]), y_train_to_use[self.label==0])
        print(clf.score(flattern(self.X_test), self.y_test))
        res['dsacc_PERFECT'] = clf.score(flattern(self.X_test), self.y_test)

        if self.ntype == 'poison':
            scipy.io.savemat('../data/' + 'res/train/%s_%s_data_left_%d.mat' % (self.name, self.modelname, self.random_id), filtered_data)

        return res

    def effectOnDownstreamDiffThreshold(self, clf, debug=False):
        scores = {
            'PK': self.PK_score,
        }

        methods = ['PK']

        res = {}
        filtered_data = {}

        if self.ntype == 'poison':
            y_train_to_use = self.y_train_mix
        else:
            y_train_to_use = self.y_train

        for mt in methods:
            print(mt)
            if self.ntype == 'poison':
                score = -scores[mt]
            else:
                score = scores[mt]

            for remove_percent in range(20):
                print(remove_percent)

                sidx = np.argsort(score)
                split_idx = int((1-float(remove_percent)/100)*sidx.shape[0])

                sidx_remove = sidx[split_idx:]
                sidx_left = sidx[:split_idx]
                clf.fit(flattern(self.X_train_mix[sidx_left]), y_train_to_use[sidx_left])
                print(clf.score(flattern(self.X_test), self.y_test))

                res['dsacc_'+mt+'_%d' % remove_percent] = clf.score(flattern(self.X_test), self.y_test)
        return res

    def aggregateStatistics(self):
        res = {}

        res['label'] = self.label

        res['score_PK'] = self.PK_score
        res['score_IF'] = self.IF_score
        res['score_OCSVM'] = self.OCSVM_score
        res['score_RVAE'] = self.RVAE_score

        PK_AUROC, PK_AVPR = computeMetric(self.PK_score, self.label, flip=(self.ntype=='poison'))
        IF_AUROC, IF_AVPR = computeMetric(self.IF_score, self.label, flip=(self.ntype=='poison'))
        OCSVM_AUROC, OCSVM_AVPR = computeMetric(self.OCSVM_score, self.label, flip=(self.ntype=='poison'))
        RVAE_AUROC, RVAE_AVPR = computeMetric(self.RVAE_score, self.label, flip=(self.ntype=='poison'))

        res['AUROC_PK'] = PK_AUROC 
        res['AUROC_IF'] = IF_AUROC
        res['AUROC_OCSVM'] = OCSVM_AUROC
        res['AUROC_RVAE'] = RVAE_AUROC

        res['AVPR_PK'] = PK_AVPR 
        res['AVPR_IF'] = IF_AVPR
        res['AVPR_OCSVM'] = OCSVM_AVPR
        res['AVPR_RVAE'] = RVAE_AVPR

        return res

    def aggregateStatisticsPK(self):
        res = {}

        res['label'] = self.label

        res['score_PK'] = self.PK_score

        PK_AUROC, PK_AVPR = computeMetric(self.PK_score, self.label, flip=(self.ntype=='poison'))

        res['AUROC_PK'] = PK_AUROC 

        res['AVPR_PK'] = PK_AVPR 

        return res

def pack_res(res):
    packed_res = {}
    for key in res[0]:
        packed_res[key] = np.stack([sub_res[key] for sub_res in res])

    return packed_res

def picketOnly(name, ntype='random', level='medium', eps_row=0.2, loss_warm_up=50, loss_trim=20, 
    sparse=False, transformer_layer=6, head_num=2, hidden_dim=64, dropout=0.1, modelname=None, remove_factor=1, 
    ghmodeTest=False, encoding=True, count_group=range(5)):

    DATA_DIR = '../data/%s/' % name
    dtypes = dtype_dict[name]
    input_dim = input_dim_dict[name]

    for count in count_group:
        print('############################################')
        print('#                    %d                     #' % count)
        print('############################################')

        trainFilter = FiltersTrain(name, dtypes, input_dim, level=level, eps_row=eps_row, ntype=ntype, random_id=count, 
            loss_warm_up=loss_warm_up, loss_trim=loss_trim, sparse=sparse, modelname=modelname, remove_factor=remove_factor)
        flag = 1
        while flag == 1:
            flag = trainFilter.PicketFilter(transformer_layer=transformer_layer, head_num=head_num, hidden_dim=hidden_dim, dropout=dropout,
                useEncoding=encoding)
        res = trainFilter.aggregateStatisticsPK()
        print(res['AUROC_PK'])

def evaluateTrainTime(name, ntype='random', level='medium', eps_row=0.2, loss_warm_up=50, loss_trim=20, 
    sparse=False, transformer_layer=6, head_num=2, hidden_dim=64, dropout=0.1, modelname=None, remove_factor=1, 
    ghmodeTest=False, encoding=True):
    DATA_DIR = '../data/%s/' % name
    dtypes = dtype_dict[name]
    input_dim = input_dim_dict[name]

    if not os.path.exists('../data/res/'):
        os.makedirs('../data/res/')
    if not os.path.exists('../data/res/train/'):
        os.makedirs('../data/res/train/')

    if modelname is None:
        modelname = 'na'

    if name == 'HTRU2':
        loss_warm_up = 20
        loss_trim = 20 
        hidden_dim = 16
        transformer_layer = 1

    if not ghmodeTest:
        main_res = []
        svm_acc = []
        nn_acc = []
        lr_acc = []
        for count in range(5):
            print('############################################')
            print('#                    %d                     #' % count)
            print('############################################')

            trainFilter = FiltersTrain(name, dtypes, input_dim, level=level, eps_row=eps_row, ntype=ntype, random_id=count, 
                loss_warm_up=loss_warm_up, loss_trim=loss_trim, sparse=sparse, modelname=modelname, remove_factor=remove_factor)
            flag = 1
            while flag == 1:
                flag = trainFilter.PicketFilter(transformer_layer=transformer_layer, head_num=head_num, hidden_dim=hidden_dim, dropout=dropout,
                    useEncoding=encoding)
            trainFilter.IFFilter()
            trainFilter.OCSVMFilter()
            trainFilter.RVAEFilter()

            main_res.append(trainFilter.aggregateStatistics())
            svm_acc.append(trainFilter.effectOnDownstream(LinearSVC()))
            nn_acc.append(trainFilter.effectOnDownstream(torchNN(input_size=len(dtypes)*input_dim)))
            lr_acc.append(trainFilter.effectOnDownstream(LogisticRegression(random_state=0)))

        main_res_packed = pack_res(main_res)
        svm_acc_packed = pack_res(svm_acc)
        nn_acc_packed = pack_res(nn_acc)
        lr_acc_packed = pack_res(lr_acc)

        np.savez('../data/' + 'res/train/%s_%s_%s_error_detect_res_%s.npz' % (name, ntype, modelname, level), **main_res_packed)
        np.savez('../data/' + 'res/train/%s_%s_%s_error_detect_svm_%s.npz' % (name, ntype, modelname, level), **svm_acc_packed)
        np.savez('../data/' + 'res/train/%s_%s_%s_error_detect_nn_%s.npz' % (name, ntype, modelname, level), **nn_acc_packed)
        np.savez('../data/' + 'res/train/%s_%s_%s_error_detect_lr_%s.npz' % (name, ntype, modelname, level), **lr_acc_packed)

    else:
        g_res = []
        h_res = []
        both_res = []

        print('ghmode')
        for count in range(5):
            print('############################################')
            print('#                    %d                     #' % count)
            print('############################################')

            trainFilter = FiltersTrain(name, dtypes, input_dim, level=level, eps_row=eps_row, ntype=ntype, random_id=count, 
                loss_warm_up=loss_warm_up, loss_trim=loss_trim, sparse=sparse, modelname=modelname, remove_factor=remove_factor)
            flag = 1
            while flag == 1:
                flag = trainFilter.PicketFilter(transformer_layer=transformer_layer, head_num=head_num, hidden_dim=hidden_dim)
            both_res.append(trainFilter.aggregateStatisticsPK())
            flag = 1
            while flag == 1:
                flag = trainFilter.PicketFilter(transformer_layer=transformer_layer, head_num=head_num, hidden_dim=hidden_dim, 
                    ghmode='onlyg')
            g_res.append(trainFilter.aggregateStatisticsPK())
            flag = 1
            while flag == 1:
                flag = trainFilter.PicketFilter(transformer_layer=transformer_layer, head_num=head_num, hidden_dim=hidden_dim, 
                    ghmode='onlyh')
            h_res.append(trainFilter.aggregateStatisticsPK())
 
        both_res_packed = pack_res(both_res)
        g_res_packed = pack_res(g_res)
        h_res_packed = pack_res(h_res)
        
        np.savez('../data/' + 'res/train/%s_%s_%s_both_error_detect_res_%s.npz' % (name, ntype, modelname, level), **both_res_packed)
        np.savez('../data/' + 'res/train/%s_%s_%s_onlyg_error_detect_res_%s.npz' % (name, ntype, modelname, level), **g_res_packed)
        np.savez('../data/' + 'res/train/%s_%s_%s_onlyh_error_detect_res_%s.npz' % (name, ntype, modelname, level), **h_res_packed)

def printResTrainTime(name, level='medium', fix_ntypes=None):
    methods = ['IF', 'OCSVM', 'RVAE', 'PK']
    methodnames = {'IF': 'IF', 'OCSVM': 'OCSVM', 'RVAE': 'RVAE', 'PK': 'Picket'}
    if name in ['wine', 'HTRU2']:
        ntypes = ['random', 'system', 'poison']
    else:
        ntypes = ['random', 'system']

    if fix_ntypes is not None:
        ntypes = fix_ntypes
    models = ['lr', 'svm', 'nn']

    print('AUROC of Outlier Detection')
    for ntype in ntypes:
        print('---------------------------------------------')
        print('Noise Type: %s' % ntype)
        if ntype == 'poison':
            for modelname in models:
                print('Downstream Model: %s' % modelname)
                ds = np.load('../data/' + 'res/train/%s_%s_%s_error_detect_res_%s.npz' % (name, ntype, modelname, level), 
                             allow_pickle=True, encoding="latin1")
                for mt in methods:
                    print('%s: %.4f' % (methodnames[mt], np.mean(ds['AUROC_%s'%mt])), end=' ')
                print(' ')
        else:         
            modelname = 'na'
            ds = np.load('../data/' + 'res/train/%s_%s_%s_error_detect_res_%s.npz' % (name, ntype, modelname, level), 
                         allow_pickle=True, encoding="latin1")
            for mt in methods:
                print('%s: %.4f' % (methodnames[mt], np.mean(ds['AUROC_%s'%mt])), end=' ')
            print(' ')
            for mt in methods:
                print('& %.4f' % (np.mean(ds['AUROC_%s'%mt])), end=' ')
            print(' ')

    print('======================================================')

    methods.append('CLEAN')
    methods.append('NOTHING')
    methodnames['CLEAN'] = 'Clean'
    methodnames['NOTHING'] = 'No Filtering'

    print('Downstream Accuracy')
    for ntype in ntypes:
        print('---------------------------------------------')
        print('Noise Type: %s' % ntype)
        if ntype == 'poison':
            for modelname in models:
                print('Downstream Model: %s' % modelname)
                if modelname == 'nn':
                    print('Please Use the MATLAB Code to Evaluate')
                else:
                    ds = np.load('../data/' + 'res/train/%s_%s_%s_error_detect_%s_%s.npz' % (name, ntype, modelname, modelname, level), 
                                 allow_pickle=True, encoding="latin1")
                    for mt in methods:
                        print('%s: %.4f' % (methodnames[mt], np.mean(ds['dsacc_%s'%mt])), end=' ')
                    print(' ')
        else:
            for modelname in models:
                print('Downstream Model: %s' % modelname)         
                ds = np.load('../data/' + 'res/train/%s_%s_na_error_detect_%s_%s.npz' % (name, ntype, modelname, level), 
                             allow_pickle=True, encoding="latin1")
                for mt in methods:
                    print('%s: %.4f' % (methodnames[mt], np.mean(ds['dsacc_%s'%mt])), end=' ')
                print(' ')
                for mt in methods:
                    print('& %.4f' % (np.mean(ds['dsacc_%s'%mt])), end=' ')
                print(' ')   

def printResTwoStream(name, ntype, modelname=None, level='medium'):
    if modelname is None:
        modelname = 'na'
    ds = np.load('../data/' + 'res/train/%s_%s_%s_both_error_detect_res_%s.npz' % (name, ntype, modelname, level), 
                 allow_pickle=True, encoding="latin1")
    print('AUROC (both streams): %f' % (np.mean(ds['AUROC_PK'])))

    ds = np.load('../data/' + 'res/train/%s_%s_%s_onlyg_error_detect_res_%s.npz' % (name, ntype, modelname, level), 
                 allow_pickle=True, encoding="latin1")
    print('AUROC (schema stream only): %f' % (np.mean(ds['AUROC_PK'])))

    ds = np.load('../data/' + 'res/train/%s_%s_%s_onlyh_error_detect_res_%s.npz' % (name, ntype, modelname, level), 
                 allow_pickle=True, encoding="latin1")
    print('AUROC (value stream only): %f' % (np.mean(ds['AUROC_PK'])))

def validate_early_filtering(name, ntype='random', level='medium', eps_row = 0.2, loss_warm_up=50, loss_trim=20, 
    sparse=False, transformer_layer=6, head_num=2, hidden_dim=64, dropout=0.1, modelname=None, remove_factor=1, count_group=range(5)):
    DATA_DIR = '../data/%s/' % name
    dtypes = dtype_dict[name]
    input_dim = input_dim_dict[name]

    if not os.path.exists('../data/res/'):
        os.makedirs('../data/res/')
    if not os.path.exists('../data/res/train/'):
        os.makedirs('../data/res/train/')

    if modelname is None:
        modelname = 'na'

    if name == 'HTRU2':
        loss_warm_up = 20
        loss_trim = 20 
        hidden_dim = 16
        transformer_layer = 1


    early_res = []
    late_res = []

    for count in count_group:
        print('############################################')
        print('#                    %d                     #' % count)
        print('############################################')

        trainFilter = FiltersTrain(name, dtypes, input_dim, level=level, eps_row=eps_row, ntype=ntype, random_id=count, 
            loss_warm_up=loss_warm_up, loss_trim=loss_trim, sparse=sparse, modelname=modelname, remove_factor=remove_factor)
        flag = 1
        while flag==1:
            flag = trainFilter.PicketFilter(transformer_layer=transformer_layer, head_num=head_num, hidden_dim=hidden_dim, dropout=dropout)
        early_res.append(trainFilter.aggregateStatisticsPK())

        trainFilter.loss_warm_up = 500
        flag = 1
        while flag==1:        
            flag = trainFilter.PicketFilter(transformer_layer=transformer_layer, head_num=head_num, hidden_dim=hidden_dim, dropout=dropout)
        late_res.append(trainFilter.aggregateStatisticsPK())

    early_res_packed = pack_res(early_res)
    late_res_packed = pack_res(late_res)


    np.savez('../data/' + 'res/train/%s_%s_%s_early_error_detect_res_%s.npz' % (name, ntype, modelname, level), **early_res_packed)
    np.savez('../data/' + 'res/train/%s_%s_%s_late_error_detect_res_%s.npz' % (name, ntype, modelname, level), **late_res_packed)

def printResEarlyFiltering(name, ntype, modelname=None, level='medium'):
    if modelname is None:
        modelname = 'na'

    ds = np.load('../data/' + 'res/train/%s_%s_%s_early_error_detect_res_%s.npz' % (name, ntype, modelname, level), 
                 allow_pickle=True, encoding="latin1")
    print('AUROC (filtering at early stage): %f' % (np.mean(ds['AUROC_PK'])))

    ds = np.load('../data/' + 'res/train/%s_%s_%s_late_error_detect_res_%s.npz' % (name, ntype, modelname, level), 
                 allow_pickle=True, encoding="latin1")
    print('AUROC (filtering after convergence): %f' % (np.mean(ds['AUROC_PK'])))

def ds_acc_diff_thres(name, ntype='random', level='medium', eps_row = 0.2, loss_warm_up=50, loss_trim=20, 
    sparse=False, transformer_layer=6, head_num=2, hidden_dim=64, dropout=0.1, modelname='', remove_factor=1, count_group=range(5)):
    DATA_DIR = '../data/%s/' % name
    dtypes = dtype_dict[name]
    input_dim = input_dim_dict[name]

    nn_acc = []

    for count in count_group:
        print('############################################')
        print('#                    %d                     #' % count)
        print('############################################')

        trainFilter = FiltersTrain(name, dtypes, input_dim, level=level, eps_row=eps_row, ntype=ntype, random_id=count, 
            loss_warm_up=loss_warm_up, loss_trim=loss_trim, sparse=sparse, modelname=modelname, remove_factor=remove_factor, load_log=False)

        trainFilter.PicketFilter()

        nn_acc.append(trainFilter.effectOnDownstreamDiffThreshold(torchNN(input_size=len(dtypes)*input_dim, batch_size=trainFilter.X_train.shape[0]//5)))

    nn_acc_packed = pack_res(nn_acc)

    np.savez('../data/' + 'res/train/%s_%s_%s_error_detect_accTT_%s.npz' % (name, ntype, modelname, level), **nn_acc_packed)

def printAccDiffThres(name, ntype='random', modelname='', level='medium'):
    ds = np.load('../data/' + 'res/train/%s_%s_%s_error_detect_accTT_%s.npz' % (name, ntype, modelname, level), 
                 allow_pickle=True, encoding="latin1")
    
    for remove_percent in range(20):
        print('%.4f' % np.mean(ds['dsacc_'+'PK'+'_%d' % remove_percent]), end=' ') 










