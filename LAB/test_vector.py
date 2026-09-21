import unittest
import math
from vector import Vec

class TestVectorOperations(unittest.TestCase):

    def setUp(self):
        self.v_sample = Vec([1, 2, 3, 4, 5])
        self.v_constant = Vec([5, 5, 5, 5])
        self.v_empty = Vec([])

    def test_mean(self):
        self.assertAlmostEqual(self.v_sample.mean(), 3.0)
        self.assertAlmostEqual(self.v_constant.mean(), 5.0)
        with self.assertRaises(ValueError):
            self.v_empty.mean()

    def test_demean(self):
        demeaned = self.v_sample.demean()
        expected_elements = (-2.0, -1.0, 0.0, 1.0, 2.0)
        for i, val in enumerate(demeaned.elements):
            self.assertAlmostEqual(val, expected_elements[i])
        self.assertAlmostEqual(sum(demeaned.elements), 0.0)
        self.assertAlmostEqual(demeaned.mean(), 0.0)

    def test_std(self):
        self.assertAlmostEqual(self.v_sample.std(), math.sqrt(2.0))
        self.assertAlmostEqual(self.v_constant.std(), 0.0)
        with self.assertRaises(ValueError):
            self.v_empty.std()

if __name__ == '__main__':
    unittest.main()
