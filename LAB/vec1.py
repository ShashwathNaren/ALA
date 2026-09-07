
import sys
from typing import Self


"""
A custom vector class implementation for educational purposes.
"""

class Vec:

    def __init__(self, src=None) -> Self:
        if src is None:
            self.elements = ()
        else:
            self.elements = tuple(src) #changed elements to self.elements   #Task : change list to tuple

    #def scalar_mul(self, alpha):
       # elements = list(self.elements)
       # for i in range(len(elements)):
              #elements[i] = alpha * elements[i]
        #result = Vec(elements)
       # return result

    def scalar_mul2(self, alpha):
            return Vec([alpha*i for i in self.elements])
    

    #Implementation scalar mult without creating a temp list

    def __repr__(self):
         return "Vec : " + repr(self.elements)
            
            #for x in elements:
                #if not isinstance(x, (int, float)):
                  #  raise TypeError(f"Scalar must be a number: {type(x)}")
               # else:
                #    self.elements = elements
            


"""
(1) Understand the basic design of the vector abstraction. Review the implementation.
(2) Document each function.
(3) Implement all unimplemented methods.
(4) Create appropriate tests for this implementation, increasing the confidence about its correctness.
(5) Test this implementation by importing the class in a sepatate python script.

(6) Measure the performance of each of these functions on vectors of varying lengths.
    Try 2k to 64k dimension vectors and time the results.
    How would you do the measurements?
(7) Measure the performance on your machine. Check it on colab.

(8) use numpy and compare the performance.
"""


if sys.version_info < (3, 8):
    sys.exit("Error: This script requires Python 3.8 or higher.")

if __name__ == "__main__":
    #z1 = Vec.zeros(10)
    #v1 = Vec([0, 1, 1.03])
    #print(v1)

    #v2 = Vec()
    #print(v2)

    #v3 = Vec(None)
    #print(v3)

   # v4 = Vec([100, 200, 300])
    #v5 = v4.scalar_mul(4)
    #print(v5)

    v6 = Vec([100, 200, 300])
    v7 = v6.scalar_mul2(4)
    print("V7 : ", v7)
    """
    v3 = 2.2 * v1
    v3 *= 5
    # v3 = 1 + v3
    print(v3)
    v2 = v1 + v3
    print(v1 + v3)
    #print(-(v1 + v3))
    """
