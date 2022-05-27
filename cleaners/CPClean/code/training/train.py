"""Train the model"""
import numpy as np

def train(X_train, y_train, model):
    """Train the model 
        
    Args:
        X_train (pd.DataFrame): features (train)
        y_train (pd.DataFrame): label (train)
        model (dict): model dict
    """
    params = model["params"]
    best_model = model["fn"](**params)
    best_model.fit(X_train, y_train)
    return best_model

def train_evaluate(model, X_train, y_train, X_test, y_test, return_preds=False):
    best_model = train(X_train, y_train, model)
    preds = best_model.predict(X_test)

    score = np.mean(preds == y_test)

    if len(set(preds)) == 1:
        print("!!!!bad pred", score)

    if return_preds:
        return score, preds
    else:
        return score