def output( msg ):
    outfile = open('sos.out','a')
    outfile.write( msg )
    outfile.close()

class VectorInOut:

    def __init__( self, bounds, filename ):
        ''' 
        Initialise VectorInOut object

        Args:
            bounds (list(double):  list of pairs of (min,max) bounds for x
            filname (str): restart filename
        '''
        self.fit_params = bounds
        self.filename = filename
        

    def pot_to_vector( self, pot):
        '''
        convert read in x values to vector
        
        Args:
            pot (list(double)): list f x values

        '''

        return (np.asarray(pot)-self.bounds[:,0])/(self.bounds[:,1]-self.bounds[:,0])

    def vector_to_pot( self, vector):
        '''
        Converts sos vector to actual x values

        Args:
            vector (numpy array): vector position in parameter space

        '''

        return ((self.bounds[:,1]-self.bounds[:,0])*vector)+self.bounds[:,0]


    def write_vectors(self, vectors, fit):
        '''
        write x-values to restart file, given vectors

        Args:
            vector (NumPy array): sos vector
            fit (Double): current function value corresponding to vector

        '''


        with open( self.filename, 'w' ) as f:
            for i,vect in enumerate(vectors):
                pot = self.vector_to_pot(vect)
                f.write(" ".join([str(item) for item in pot]))
                f.write("  "+str(fit[i])+"\n")

    def read_vectors( self):
        '''
        read in vectors from restart file
        
        Args:
            None
        Requires:
            restart file 'self.filename'

        '''

        output("About to read in vector restart\n")
        with open( self.filename, 'r' ) as restart_file:
            temp = [[float(val) for val in line.split()] for line in restart_file]
        vectors=[]
        fit=[]
        for i in range(len(temp)):
            vector = self.pot_to_vector(temp[i][:-1])
            vectors.append(vector)
            fit.append(temp[i][-1])

        return vectors, fit

