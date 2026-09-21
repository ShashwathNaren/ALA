import sys
import math
from typing import Self

class Vec:

    def __init__(self, src=None) -> None:
        if src is None:
            self.elements = ()
        else:
            self.elements = tuple(src)

    def scalar_mul2(self, alpha: float) -> Self:
        return Vec([alpha * i for i in self.elements])

    def mean(self) -> float:
        n = len(self.elements)
        if n == 0:
            raise ValueError("Cannot calculate the mean of an empty vector.")
        return sum(self.elements) / n

    def demean(self) -> Self:
        mu = self.mean()
        return Vec([x - mu for x in self.elements])

    def std(self) -> float:
        n = len(self.elements)
        if n == 0:
            raise ValueError("Cannot calculate the standard deviation of an empty vector.")
        
        demeaned_vector = self.demean()
        squared_deviations = [x**2 for x in demeaned_vector.elements]
        return math.sqrt(sum(squared_deviations) / n)

    def __repr__(self) -> str:
        return "Vec : " + repr(self.elements)


if sys.version_info < (3, 8):
    sys.exit("Error: This script requires Python 3.8 or higher.")

if __name__ == "__main__":
    v6 = Vec([100, 200, 300])
    v7 = v6.scalar_mul2(4)
    print("V7 : ", v7)

    print("Mean calculated:", v6.mean())
    print("Demeaned version:", v6.demean())
    print("Standard Deviation:", v6.std())