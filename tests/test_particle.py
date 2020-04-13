import unittest
import copy
from glopty.tests.functions import ackley
from glopty.particle import Particle
from unittest.mock import Mock
import numpy as np


class ParticleTestCase(unittest.TestCase):
    """test for Particle class"""

    def setUp(self):

        self.vector = np.array([0.875, 0.625, 0.875, 0.125])
        self.fit = ackley(self.vector)
        self.index = 1
        self.particle = Particle(self.vector, self.fit, self.index)
        self.mock_particle = Mock(
            spec=Particle, vector=self.vector, fit=self.fit, index=self.index
        )

    def test_particle_is_initialised(self):
        self.assertEqual(self.particle.fit, self.mock_particle.fit)
        self.assertEqual(self.particle.vector.all(), self.mock_particle.vector.all())
        self.assertEqual(self.particle.index, self.mock_particle.index)

    def test_return_fit(self):
        self.assertEqual(self.particle.return_fit, self.fit)

    def test_return_vec(self):
        self.assertEqual(self.particle.return_vec.all(), self.vector.all())

    def test_vector_convert(self):
        bounds = np.asarray([(-5, 5)] * 4)
        self.assertEqual(
            self.particle.vector_to_pos(bounds).all(),
            self.particle.pos_to_vector(
                self.particle.vector_to_pos(bounds), bounds
            ).all(),
        )


if __name__ == "__main__":
    unittest.main()
