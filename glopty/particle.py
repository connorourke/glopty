import numpy as np

class Particle:
    def __init__(self,vector,fit,index):
        self.vector=vector
        self.fit=fit
        self.index=index

    def vector_to_pos(self, bounds):
        return ((bounds[:,1]-bounds[:,0])*self.vector)+bounds[:,0]

    def pos_to_vector(self, pos, bounds):
        return (np.asarray(pos)-bounds[:,0])/(bounds[:,1]-bounds[:,0])

    @property
    def return_fit(self):
        return self.fit

    @property
    def return_vec(self):
        return self.vector


