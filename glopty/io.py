import numpy as np


def output(msg):
    outfile = open("sos.out", "a")
    outfile.write(msg)
    outfile.close()


class VectorInOut:
    def __init__(self, bounds, filename):
        """ 
        Initialise VectorInOut object

        Args:
            bounds (list(double):  list of pairs of (min,max) bounds for x
            filname (str): restart filename
        """
        self.bounds = np.asarray(bounds)
        self.filename = filename

    def pot_to_vector(self, pot):
        """
        convert read in x values to vector
        
        Args:
            pot (list(double)): list f x values

        """

        return (np.asarray(pot) - self.bounds[:, 0]) / (
            self.bounds[:, 1] - self.bounds[:, 0]
        )

    def vector_to_pot(self, vector):
        """
        Converts sos vector to actual x values

        Args:
            vector (numpy array): vector position in parameter space

        """

        return ((self.bounds[:, 1] - self.bounds[:, 0]) * vector) + self.bounds[:, 0]

    def write_vectors(self, particles):
        """
        write x-values to restart file, given vectors

        Args:
            particles (list(Particle)): list of particles
        """

        with open(self.filename, "w") as f:
            for part in particles:
                pot = self.vector_to_pot(part.vector)
                f.write(" ".join([str(item) for item in pot]))
                f.write("  " + str(part.fit) + "\n")

    def read_vectors(self):
        """
        read in vectors from restart file
        
        Args:
            None
        Returns:
            vectors (list(double)): list containing vectors
            fir (list(double)): list containing corresponding function values

        Requires:
            restart file 'self.filename'

        """

        output("About to read in vector restart\n")
        with open(self.filename, "r") as restart_file:
            temp = [[float(val) for val in line.split()] for line in restart_file]
        vectors = []
        fit = []
        for i in range(len(temp)):
            vector = self.pot_to_vector(temp[i][:-1])
            vectors.append(vector)
            fit.append(temp[i][-1])

        return vectors, fit
