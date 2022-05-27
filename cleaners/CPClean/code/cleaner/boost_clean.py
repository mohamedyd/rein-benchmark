from sklearn.linear_model import LogisticRegression
import numpy as np
from copy import deepcopy
from ..training.train import train, train_evaluate
from sklearn.metrics import precision_recall_fscore_support

def train_classifiers(X_train_list, y_train, model):
    C_list = []
    for X_train in X_train_list:
        C = train(X_train, y_train, model)
        C_list.append(C)
    return C_list

def transform_y(y, c):
    y_c = deepcopy(y)
    mask = y == c
    y_c[mask] = 1
    y_c[mask == False] = -1 
    return y_c

# X_train_list is a list of repairs of different imputers that are used
# for stacking
def boost_clean(model, X_train_list, y_train, X_val, y_val, X_test, y_test, T=1):
    y_train = transform_y(y_train, 1)
    y_val = transform_y(y_val, 1)
    y_test = transform_y(y_test, 1)

    # train for each repaired version of X_train a classifier
    C_list = train_classifiers(X_train_list, y_train, model)
    N = len(y_val)
    W = np.ones((1, N)) / N # initial weights for samples of validation set

    preds_val = np.array([C.predict(X_val) for C in C_list]).T
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    acc_list = (preds_val == y_val).astype(int)
    C_T = [] # holds order in which classifiers should be applied
    a_T = [] # holds amount of say of each classifier
    for t in range(T):
        acc_t = W.dot(acc_list) 
        c_t = np.argmax(acc_t) # get best classifier in current iteration

        e_c = 1 - acc_t[0, c_t]
        a_t = np.log((1-e_c)/(e_c+1e-8))
        
        C_T.append(c_t) # add classifier to list of classifiers
        a_T.append(a_t) # add amount of say to list
        
        for i in range(N):
            # update weights
            W[0, i] = W[0, i] * np.exp(-a_t * y_val[i, 0] * preds_val[i, c_t])

    
    a_T = np.array(a_T).reshape(1, -1)

    preds_test = [C.predict(X_test) for C in C_list]
    preds_test_T = np.array([preds_test[c_t] for c_t in C_T])
    test_scores = a_T.dot(preds_test_T).T
    
    preds_val = [C.predict(X_val) for C in C_list]
    preds_val_T = np.array([preds_val[c_t] for c_t in C_T])
    val_scores = a_T.dot(preds_val_T).T
    

    y_pred_test = np.sign(test_scores)
    y_pred_val = np.sign(val_scores)

    test_acc = (y_pred_test == y_test).mean()
    val_acc = (y_pred_val == y_val).mean()

    return test_acc, val_acc

def modified_boost_clean(model, X_train_list, y_train, X_val, y_val, X_test, y_test, T=5):
    '''
    model : which ml model should be used
    X_train_list : repaired version of dataset for each repairer. len(X_train_list) = #repairMethods
    y_train : true labels for training data
    X_val, y_val : Validation set to validate boosting steps
    T (int) : number of rounds for boosting (T=5 means boosting results in combination of 5 classifiers) 
    '''
    y_train = transform_y(y_train, 1)
    y_val = transform_y(y_val, 1)
    y_test = transform_y(y_test, 1)

    # train for each repaired version of X_train a classifier
    C_list = train_classifiers(X_train_list, y_train, model)
    N = len(y_val)
    W = np.ones((1, N)) / N # initial weights for samples of validation set

    preds_val = np.array([C.predict(X_val) for C in C_list]).T
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    acc_list = (preds_val == y_val).astype(int)
    C_T = [] # holds order in which classifiers should be applied
    a_T = [] # holds amount of say of each classifier
    for t in range(T):
        acc_t = W.dot(acc_list) 
        c_t = np.argmax(acc_t) # get best classifier in current iteration

        e_c = 1 - acc_t[0, c_t]
        a_t = np.log((1-e_c)/(e_c+1e-8))
        
        C_T.append(c_t) # add classifier to list of classifiers
        a_T.append(a_t) # add amount of say to list
        
        for i in range(N):
            # update weights
            W[0, i] = W[0, i] * np.exp(-a_t * y_val[i, 0] * preds_val[i, c_t])

    
    a_T = np.array(a_T).reshape(1, -1)

    preds_test = [C.predict(X_test) for C in C_list]
    preds_test_T = np.array([preds_test[c_t] for c_t in C_T])
    test_scores = a_T.dot(preds_test_T).T
    
    preds_val = [C.predict(X_val) for C in C_list]
    preds_val_T = np.array([preds_val[c_t] for c_t in C_T])
    val_scores = a_T.dot(preds_val_T).T
    

    y_pred_test = np.sign(test_scores)
    y_pred_val = np.sign(val_scores)

    test_acc = (y_pred_test == y_test).mean()
    val_acc = (y_pred_val == y_val).mean()
    
    y_pred_test = y_pred_test.flatten()
    y_test = y_test.flatten()
        
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='binary')
    
    return test_acc, val_acc, p, r, f1 