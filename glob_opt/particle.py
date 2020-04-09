import numpy as np

class Particle:
    def __init__(self,vector,fit,index):
        self.vector=vector
        self.fit=fit
        self.index=index


    def vector_to_pos(self,vector,bounds):
        return ((np.max(bounds)-np.min(bounds)) * vector)  + np.min(bounds)

    @property
    def return_fit(self):
        return self.fit

    @property
    def return_vec(self):
        return self.vector



