from pyDOE import lhs
import math
import copy
import numpy as np
from scipy.optimize import OptimizeResult
from pathos.multiprocessing import ProcessingPool as Pool
import pathos.multiprocessing as mp
from glopty.particle import Particle
from glopty.io import output, VectorInOut
import sys


class SOS:

    def __init__(self, func, bounds, niter=100, population=10, ftol=0.001, workers=-1, restart= False):
        ''' 
        Initialise a symbiotic organisms search instance
        
        Args:
            func (callable): Function to be minimised. f(x, *args) - x is the argument to be minimised, args is a tuple of any additional  fixed parameters to specify the function
            bounds (list(Double)): list of pairs of (min,max) bounds for x
            niter (Int): number of iterations for optimiser
            population (Int): number of members in population
            ftol (Double) : convergence criteria for function
            workers (Int): number of multiprocessing workers to use. -1 sets workers to mp.cpu_count()
        '''
        
        self.function = func
        self.niter = niter
        self.population = population
        self.particles = []
        self.best_global_vec = None
        self.best_global_fit = math.inf
        self.ftol = ftol
        self.bounds = np.asarray(bounds)
        self.restart = restart
        self.vector_restart = VectorInOut(bounds,'sos.rst')

        if workers == -1:
            self.pool = Pool(mp.cpu_count())
        else:
            self.pool = Pool(workers)
 

    def vector_to_pot(self,vector):
        '''
        Converts sos vector to actual x values

        Args:
            vector (numpy array): vector position in parameter space

        '''
        return ((self.bounds[:,1]-self.bounds[:,0])*vector)+self.bounds[:,0]


    def initialise_particles(self):
        '''
        Initialises the population: sets particle vectors using latin hypercube, and sets global bests

        Args:
            None
            
        '''

        if self.restart:
            vec_fit = self.vector_restart.read_vectors()
            for i,vec in enumerate(vec):
                self.particles.append(Particle(vec,fit[i]))

        else:
            vectors =lhs(len(self.bounds),self.population)

            for i,vector in enumerate(vectors):
                self.particles.append(Particle(vector,self.function(self.vector_to_pot(vector),self.args),i))

            self.best_global_fit = copy.deepcopy(self.particles[0].return_fit)
            self.best_global_vec = copy.deepcopy(self.particles[0].return_vec)

    def set_global_best(self):
        '''
        Sets current global best fit for function, and corresponding vector
    
        Args:
            None
        '''

         
        for particle in self.particles:
            if particle.fit < self.best_global_fit:
               self.best_global_fit = copy.deepcopy(particle.return_fit)
               self.best_global_vec = copy.deepcopy(particle.return_vec)
        output("Current best fit:" + str(self.best_global_fit)+'\n')
            
    def mutualism(self,part):
        '''
        Performs mutualism step of sos

        Args:
            part (Particle): particle member of population on which to perform mutualism

        Returns:
            part.vector (np.array): vector position in paramter space
            part.fit    (Double): value of function at point in param space corresponding to part.vector
        '''

        b_ind = np.random.choice([i for i in range(self.population) if i != part.index],1,replace = False)[0]
        a = part.vector
        b = self.particles[b_ind].vector
        bf = np.random.randint(1, 3, 2)

        mutant = np.random.rand(len(self.bounds))
        mutual = (a+b)/2
        new_a = np.clip(a + (mutant * (self.best_global_vec - (mutual*bf[0]))),0,1)
        new_b = np.clip(b + (mutant * (self.best_global_vec - (mutual*bf[1]))),0,1)


        for i,vec in enumerate([[part.index,new_a],[b_ind,new_b]]):
            trial_pot = self.vector_to_pot(vec[1])
            error = self.function(trial_pot,self.args)
            if error < self.particles[vec[0]].fit:
                self.particles[vec[0]].fit = error
                self.particles[vec[0]].vector= vec[1]

        return part.vector,part.fit

    def run_mutualism(self):
        '''
        Wrapper for mutualism step, for multiprocessing

        Args:
            None
        '''

        res = self.pool.amap(self.mutualism,self.particles)
        for i,val in enumerate(res.get()):
            self.particles[i].vector, self.particles[i].fit=val
            

    def commensalism(self, part):
        '''
        Performs commensalism step of sos

        Args:
            part (Particle): particle member of population on which to perform commensalism

        Returns:
            part.vector (np.array): vector position in paramter space
            part.fit    (Double): value of function at point in param space corresponding to part.vector
        '''


        b_ind = np.random.choice([i for i in range(self.population) if i != part.index],1,replace = False)[0]

        a = part.vector
        b = self.particles[b_ind].vector

        mutant = np.random.uniform(-1,1,len(self.bounds))
        new_a = np.clip(a + (mutant[0] * (self.best_global_vec - b)),0,1)

        trial_pot = self.vector_to_pot(new_a)
        error = self.function(trial_pot,self.args)

        if error < part.fit:
            part.fit = error
            part.vector= new_a

        return part.vector,part.fit

    def run_commensalism(self):
        '''
        Wrapper for commensalism step, for multiprocessing

        Args:
            None
        '''

        res = self.pool.amap(self.commensalism,self.particles)
        for i,val in enumerate(res.get()):
            self.particles[i].vector, self.particles[i].fit=val


    def parasitism(self, part):
        '''
        Performs parasitism step of sos

        Args:
            part (Particle): particle member of population on which to perform parasitism

        Returns:
            part.vector (np.array): vector position in paramter space
            part.fit    (Double): value of function at point in param space corresponding to part.vector
        '''



        b_ind = np.random.choice([i for i in range(self.population) if i != part.index],1,replace = False)[0]
        parasite =  copy.deepcopy(part.vector)
        parasite[np.random.randint(0, len(self.bounds))] = np.random.rand()

        trial_pot = self.vector_to_pot(parasite)
        error = self.function(trial_pot,self.args)


        if error < self.particles[b_ind].fit:
            self.particles[b_ind].fit = error
            self.particles[b_ind].vector = parasite

        return b_ind,self.particles[b_ind].fit,self.particles[b_ind].vector


    def run_parasitism(self):
        '''
        Wrapper for parasitism step, for multiprocessing

        Args:
            None
        '''

        res = self.pool.amap(self.parasitism,self.particles)
        for i,val in enumerate(res.get()):
            self.particles[val[0]].vector, self.particles[val[0]].fit=val[2],val[1]

    def optimise(self, args):
        '''
        Optimise the function: run 

        Args:
            function (Function): function to optimise
            args (Optional): any further args required by function 

        '''

        self.args = args
        self.initialise_particles()

        for step in range(self.niter):
            output("Doing step: "+str(step)+'\n')
            self.run_mutualism()
            self.run_commensalism()
            self.run_parasitism()
            self.set_global_best()
            if self.best_global_fit < self.ftol:
                break

        results_min = OptimizeResult()
        results_min.x = self.vector_to_pot(self.best_global_vec)
        results_min.fun = self.best_global_fit
        
        self.vector_restart.write_vectors(self.particles)        
        
        return results_min


