import unittest
import math
from vector import Vec

class Test_vector_operations(unittest.TestCase):

    def setUp(self):
        self.v1 = Vec([10, 20, 30])
        self.v_empty = Vec([])

    def test_mean(self):
        self.assertAlmostEqual(self.v1.mean(), 20.0)
        with self.assertRaises(ValueError):
            self.v_empty.mean()

    def test_demean(self):
        demeaned = self.v1.demean()
        expected_elements = (-10.0, 0, 10.0)

        for i, val in enumerate(demeaned.elements):
            self.assertAlmostEqual(val, expected_elements[i])

        self.assertAlmostEqual(sum(demeaned.elements), 0.0)
        self.assertAlmostEqual(demeaned.mean(), 0.0)

    def test_std(self):
        expected_std = math.sqrt(200 / 3)
        self.assertAlmostEqual(self.v1.std(), expected_std)
        with self.assertRaises(ValueError):
            self.v_empty.std()

if __name__ == '__main__':
    unittest.main()