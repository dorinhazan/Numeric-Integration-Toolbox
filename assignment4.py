"""
In this assignment you should fit a model function of your choice to data
that you sample from a given function.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you take an iterative approach and know that
your iterations may take more than 1-2 seconds break out of any optimization
loops you have ahead of time.


"""

import numpy as np
import time
import random
import torch
# import tensorflow as tf



class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        f : callable.
            A function which returns an approximate (noisy) Y value given X.
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        x_points = np.linspace(a, b, int(d) * 40)
        # Create a matrix of x points in powers of the expected degree of the polynomial
        x_point_mat = []
        for x in x_points:
            x_point_mat.append([x ** i for i in range(d, -1, -1)])

        # Convert x point matrix to a PyTorch tensor
        mat = torch.tensor(x_point_mat,dtype=torch.float64)

        # Reshape the tensor so that it can be multiplied with its transpose
        mat = mat.reshape(-1, mat.shape[-1])

        # Transpose the tensor and convert to float64 data type
        matT = torch.transpose(mat, 0, 1).type(torch.float64)

        # Create a matrix of y points
        y_points_mat = []
        for x in x_points:
            y_points_mat.append(f(x))
        y_points= np.array(y_points_mat, dtype=np.float64)

        # Multiply the transpose of the x point matrix with the x point matrix
        mat_mul = torch.matmul(matT, mat)

        # Calculate the inverse of the multiplied matrix
        mat_inv = torch.inverse(mat_mul)

        # Multiply the inverse matrix with the transpose of the x point matrix
        mat_mul = torch.matmul(mat_inv, matT)

        # Multiply the result matrix with the y points matrix
        coef = torch.matmul(mat_mul, torch.tensor(y_points,dtype=torch.float64))

        # Define a function that will return the fitted polynomial equation
        def result(x):
            ret = 0.0
            for i in range(d, -1, -1):
                ret = ret * x + coef[d - i].item()
            return ret

        return result






##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1,1,1))
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    # def test_delay(self):
    #     f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))
    #
    #     ass4 = Assignment4()
    #     T = time.time()
    #     shape = ass4.fit(f=f, a=0, b=1, d=2, maxtime=5)
    #     T = time.time() - T
    #     self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1,1,1)
        nf = NOISY(1)(f)
        ass4 = Assignment4()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):
            self.assertNotEqual(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print(mse)






if __name__ == "__main__":
    unittest.main()
