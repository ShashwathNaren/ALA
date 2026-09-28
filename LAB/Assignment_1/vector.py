import math
from typing import Self

class Vec:

    def __init__(self, src = None) -> None:
        if src is None:
            self.elements = ()
        else:
            self.elements = tuple(src)

    def scalar_mul(self, alpha : float) -> Self:
        return Vec([alpha * i for i in self.elements])

    def mean(self) -> float:
        n = len(self.elements)
        if n == 0:
            raise ValueError("Cannot calculate the mean of an empty vector.")
        return sum(self.elements) / n

    def demean(self) -> Self:
        mean = self.mean()
        return Vec([x - mean for x in self.elements])

    def std(self) -> float:
        n = len(self.elements)
        if n == 0:
            raise ValueError("Cannot calculate the standard deviation of an empty vector.")

        demeaned_vector = self.demean()
        squared_deviations = [x**2 for x in demeaned_vector.elements]
        return math.sqrt(sum(squared_deviations) / n)

    def __repr__(self) -> str:
        return "Vec : " + repr(self.elements)

if __name__ == "__main__":
    v1 = Vec([10, 20, 30])
    v2 = v1.scalar_mul(4)
    print("v1 : ", v1)
    print("v2 : ", v2)

    print("Mean : ", v1.mean())
    print("Demeaned Vector : ", v1.demean())
    print("Standard Deviation : ", v1.std())