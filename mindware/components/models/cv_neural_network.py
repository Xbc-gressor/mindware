import numpy as np

class cv_neural_network(object):
    '''
    When we choose evaluationn as cv, we need to save all neural_networks. 
    '''
    def __init__(self):
        self.models = []
        self.origin_model = None
    
    def append(self, model):
        self.models.append(model)

    def predict_proba(self, X): 
        proba = None
        for model in self.models:
            if proba is None:
                proba = model.predict_proba(X)
            else:
                proba += model.predict_proba(X)
        return proba/ len(self.models)


    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba,axis=1)

    
