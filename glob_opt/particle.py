import numpy as np

class Particle:
    def __init__(self,vector,fit,index):
        self.vector=vector
        self.fit=fit
        self.index=index


    def vector_to_pos(self,vector,bounds):
        return ((self.bounds[:,1]-self.bounds[:,0])*vector)+self.bounds[:,0]

    @property
    def return_fit(self):
        return self.fit

    @property
    def return_vec(self):
        return self.vector



