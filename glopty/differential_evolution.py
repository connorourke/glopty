from pyDOE import lhs
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import pathos.multiprocessing as mp
import pathos.helpers as pathelp
from glopty.particle import Particle
from glopty.io import output, VectorInOut
import sys
import math
import copy

class DiffEvolution:

    def __init__(self, func, bounds, niter=100, population=100, ftol=0.001, workers=-1, vec_dump=10, restart=False, mut_fac=0.3, cross_prob=0.7): 
        """
        Initialise a differential evolution optimisation  instance.

        Args:
            func (callable): Function to be minimised. f(x, *args) - x is the argument to be minimised, args is a tuple of any additional  fixed parameters to specify the function
            bounds (list(Double)): list of pairs of (min,max) bounds for 
x
            niter (Int): number of iterations for optimiser
            population (Int): number of members in population
            ftol (Double) : convergence criteria for function
            workers (Int): number of multiprocessing workers to use. -1 sets workers to mp.cpu_count()
            vec_dump (Int): outputs restart file vec_dump number of steps  
            restart (Bool): restart the run from a restart file
            mut_fac (Double): mutation factor of diff evolution
            cross_prob (Double): cross over probability for mutant to generate trial

        """


        self.function = func
        self.bounds = bounds
        self.niter = niter
        self.population = population
        self.ftol = ftol
        self.vec_dump = vec_dump
        self.restart = restart
        self.mut_fac = mut_fac
        self.cross_prob = cross_prob
        self.particles = []
        self.best_global_vec = None               
        self.best_global_fit = math.inf           
        self.dim = len(self.bounds)
        self.vector_restart = VectorInOut(bounds, "sos.rst")

            
        if workers == -1:                    
            self.pool = Pool(mp.cpu_count()) 
        else:                                
            self.pool = Pool(workers)        

    def part_init(self, vector):
        """
        Wrapper for particle initialisation for multiprocess
        
        Args:
        
        vector (numpy array)

        Returns: 
        
        vector (numpy array)
        result of function(vector)
        """
        return vector, self.function(self.vector_to_pot(vector), self.args)

    def initialise_particles(self):
        """
        Initialises the population: sets particle vectors using latin hypercube, and sets global bests

        Args:
            None
            
        """

        if self.restart:
            vec, fit = self.vector_restart.read_vectors()
            for i, vec in enumerate(vec):
                self.particles.append(Particle(np.asarray(vec), fit[i], i))

            self.set_global_best()
        else:
            vectors = lhs(len(self.bounds), self.population)
            res = self.pool.amap(self.part_init, vectors)
            for i, val in enumerate(res.get()):
                self.particles.append(Particle(val[0], val[1], i))
            self.best_global_fit = copy.deepcopy(self.particles[0].return_fit)
            self.best_global_vec = copy.deepcopy(self.particles[0].return_vec)

    def vector_to_pot(self, vector):
        """
        Converts particle vector to actual x values
        
        Args:
            vector (numpy array): vector position in parameter space

        """

        return ((self.bounds[:, 1] - self.bounds[:, 0]) * vector) + self.bounds[:, 0]

    def set_global_best(self):
        """
        Sets current global best fit for function, and corresponding vector
    
        Args:
            None
        """

        for particle in self.particles:
            if particle.fit < self.best_global_fit:
                self.best_global_fit = copy.deepcopy(particle.return_fit)
                self.best_global_vec = copy.deepcopy(particle.return_vec)
        output("Current best fit:" + str(self.best_global_fit) + "\n")

    def evolve(self,part):

        np.random.seed() 
        ind = np.random.choice([i for i in range(self.population) if i != part.index], 3, replace=False)
        a = self.particles[ind[0]]
        b = self.particles[ind[1]]
        c = self.particles[ind[2]]

        mutant = a.vector + self.mut_fac * (b.vector-c.vector)
        mutant[mutant > 1.0] = np.random.uniform()
        mutant[mutant < 0.0] = np.random.uniform()
        cross_points = np.random.rand(self.dim) < self.cross_prob

        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points,mutant,part.vector)
        fit = self.function(self.vector_to_pot(trial),self.args)
        
        if fit < part.fit:
            return trial, fit 
        else:
            return part.vector, part.fit
             
            
    def run_evolution(self):

        res = self.pool.amap(self.evolve, self.particles)
        for i, val in enumerate(res.get()):
            self.particles[i].vector, self.particles[i].fit = val
 


    def optimise(self, args):                                       
        """                                                         
        Optimise the function: run                                  
                                                                    
        Args:                                                       
            function (Function): function to optimise               
            args (Optional): any further args required by function  
                                                                    
        """                                                         
                                                                    
        self.args = args
        self.initialise_particles()
        self.set_global_best()
     
        for step in range(self.niter):
            output("Doing step: " + str(step) + "\n")
            self.run_evolution()
            self.set_global_best()
            if self.best_global_fit < self.ftol:
                break
            if step % self.vec_dump == 0:
                output("Going to dump particle vectors\n")
                self.vector_restart.write_vectors(self.particles)
            

